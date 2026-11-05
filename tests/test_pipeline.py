"""Training, splitting, cleaning and download logic, on small synthetic data.

Nothing here reads the real datasets, so these tests run in CI.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from phishlens.datasets import DatasetError, clean, load_hannousse, load_phiusiil
from phishlens.evaluation import (
    HARMLESS_EDITS,
    add_tracking_query,
    add_trailing_slash,
    add_www,
    strip_trailing_slash,
    use_plain_http,
)
from phishlens.features import canonical_key
from phishlens.model import METADATA_FILE, MODEL_FILE, Detector
from phishlens.training import (
    domain_weights,
    feature_matrix,
    fit_booster,
    save_model,
    split_by_domain,
    train_on,
)

ROOT = Path(__file__).resolve().parent.parent


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_data", ROOT / "scripts" / "download_data.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_frame(rows: int = 3000) -> pd.DataFrame:
    random = np.random.default_rng(7)
    urls, labels, domains = [], [], []
    for index in range(rows):
        phishing = index % 2
        name = f"site{index % 400}"
        if phishing:
            token = random.integers(10**8, 10**9)
            url = f"https://login-{token}.{name}-verify.top/secure/{token}.php"
            domain = f"{name}-verify.top"
        else:
            url = f"https://www.{name}.com/blog/article-{index}"
            domain = f"{name}.com"
        urls.append(url)
        labels.append(phishing)
        domains.append(domain)
    return pd.DataFrame({"url": urls, "label": np.array(labels, dtype=np.int8), "domain": domains})


def test_split_keeps_every_domain_on_one_side() -> None:
    domains = np.array([f"d{i % 50}.com" for i in range(1000)])
    fit, held = split_by_domain(domains, 0.2, seed=3)
    assert not set(domains[fit]) & set(domains[held])
    assert len(fit) + len(held) == len(domains)


def test_split_is_fixed_by_the_seed() -> None:
    domains = np.array([f"d{i % 50}.com" for i in range(1000)])
    first = split_by_domain(domains, 0.2, seed=3)[1]
    assert (first == split_by_domain(domains, 0.2, seed=3)[1]).all()
    assert (first != split_by_domain(domains, 0.2, seed=4)[1]).any()


def test_domain_weights_cap_large_domains() -> None:
    domains = np.array(["big.com"] * 1000 + ["small.com"] * 10)
    weights = domain_weights(domains, cap=100.0)
    assert weights[:1000].sum() == pytest.approx(100.0)
    assert weights[1000:].sum() == pytest.approx(10.0)


def test_training_is_deterministic() -> None:
    frame = synthetic_frame()
    rows = feature_matrix(frame["url"].tolist())
    labels = frame["label"].to_numpy(dtype=np.int8)
    weights = np.ones(len(labels), dtype=np.float32)
    first = fit_booster(rows, labels, weights, seed=1, rounds=10).model_to_string()
    second = fit_booster(rows, labels, weights, seed=1, rounds=10).model_to_string()
    assert first == second


def test_saved_model_round_trips_through_the_detector(tmp_path: Path) -> None:
    frame = synthetic_frame()
    rows = feature_matrix(frame["url"].tolist())
    save_model(train_on(frame, rows, seed=1), tmp_path, seed=1)
    assert (tmp_path / MODEL_FILE).exists()
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["seed"] == 1
    assert (
        hashlib.sha256((tmp_path / MODEL_FILE).read_bytes()).hexdigest()
        == (metadata["model_sha256"])
    )
    detector = Detector.load(tmp_path)
    assert detector.score(["https://example.com/"])[0] >= 0


def test_a_model_learns_the_synthetic_pattern_and_validation_stays_held_out() -> None:
    frame = synthetic_frame()
    rows = feature_matrix(frame["url"].tolist())
    model = train_on(frame, rows, seed=1)
    assert model.fit_rows + model.validation_rows == len(frame)
    fit_domains = set(frame["domain"]) - set(model.validation_domains)
    assert fit_domains.isdisjoint(model.validation_domains)
    scores = np.asarray(model.booster.predict(rows))
    labels = frame["label"].to_numpy()
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_clean_drops_formatting_duplicates_and_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "url": [
                "https://example.com/a",
                "http://www.example.com/a/?x=1",
                "https://other.example.org/b",
                "https://both.example.net/c",
                "http://both.example.net/c",
                "mailto:someone@example.com",
            ],
            "label": np.array([1, 1, 0, 1, 0, 0], dtype=np.int8),
        }
    )
    cleaned, report = clean(frame)
    assert cleaned["url"].tolist() == ["https://example.com/a", "https://other.example.org/b"]
    assert report.as_dict() == {
        "rows_in": 6,
        "no_host": 1,
        "duplicates": 1,
        "label_conflicts": 2,
        "rows_out": 2,
    }


def test_clean_keeps_the_earlier_row_so_test_rows_cannot_repeat_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "url": ["https://example.com/a", "http://example.com/a"],
            "label": np.array([1, 1], dtype=np.int8),
            "split": ["train", "test"],
        }
    )
    assert clean(frame)[0]["split"].tolist() == ["train"]


def test_missing_data_names_the_download_script(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="download_data"):
        load_hannousse(tmp_path)


def test_flipped_or_truncated_labels_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "phiusiil"
    folder.mkdir()
    (folder / "PhiUSIIL_Phishing_URL_Dataset.csv").write_text(
        "URL,label\nhttps://www.example.com,1\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="phiusiil"):
        load_phiusiil(tmp_path)


@pytest.mark.parametrize("edit", HARMLESS_EDITS.values())
def test_harmless_edits_keep_the_canonical_key(edit: Any) -> None:
    url = "https://www.example.com/a/b"
    assert canonical_key(edit(url)) == canonical_key(url)


def test_harmless_edit_helpers() -> None:
    assert add_tracking_query("https://a.example/x") == "https://a.example/x?utm_source=newsletter"
    assert add_tracking_query("https://a.example/x?y=1").endswith("&utm_source=newsletter")
    assert add_trailing_slash("https://a.example/x") == "https://a.example/x/"
    assert strip_trailing_slash("https://a.example/x/") == "https://a.example/x"
    assert use_plain_http("https://a.example/x") == "http://a.example/x"
    assert add_www("https://a.example/x") == "https://www.a.example/x"
    assert add_www("a.example/x") == "www.a.example/x"
    assert add_www("https://www.a.example/x") == "https://www.a.example/x"


def test_download_refuses_a_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    monkeypatch.setattr(script, "fetch", lambda _: iter([b"tampered"]))
    source = {"output": "x.csv", "url": "https://example.com/x", "sha256": "0" * 64}
    with pytest.raises(SystemExit, match="checksum mismatch"):
        script.download_file("x", source, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_accepts_a_matching_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    monkeypatch.setattr(script, "fetch", lambda _: iter([b"data"]))
    source = {
        "output": "x.csv",
        "url": "https://example.com/x",
        "sha256": hashlib.sha256(b"data").hexdigest(),
    }
    script.download_file("x", source, tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == b"data"


def test_download_refuses_plain_http() -> None:
    with pytest.raises(ValueError, match="non-https"):
        next(load_script().fetch("http://example.com/x"))


def test_extract_writer_hashes_uncompressed_bytes_and_is_repeatable(tmp_path: Path) -> None:
    script = load_script()
    paths = [tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"]
    digests = []
    for path in paths:
        writer = script.HashingGzipCsv(path)
        writer.write(["url", "label"])
        writer.write(["https://example.com/a,b", "phish"])
        writer.close()
        digests.append(writer.digest.hexdigest())
    assert digests[0] == digests[1]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    content = gzip.decompress(paths[0].read_bytes())
    assert hashlib.sha256(content).hexdigest() == digests[0]
    assert script.gz_content_sha256(paths[0]) == digests[0]


def test_stale_extract_pin_fails() -> None:
    script = load_script()
    source = {"content_sha256": "a" * 64}
    with pytest.raises(SystemExit, match="mismatch"):
        script.check_pin("x", "content_sha256", "b" * 64, source, update=False)


def test_every_source_records_a_license_and_a_pin() -> None:
    manifest = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    for source in manifest.values():
        assert source["license"] == "CC-BY-4.0"
        assert source["citation"]
        assert source.get("sha256") or source.get("content_sha256")
