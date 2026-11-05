from __future__ import annotations

import json
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phishlens.features import FEATURE_NAMES
from phishlens.model import (
    ARTIFACT_DIR,
    METADATA_FILE,
    MODEL_FILE,
    NOTE_IGNORED_PARTS,
    NOTE_SHARED_HOST,
    NOTE_SHORTENER,
    Detector,
    ModelError,
    display_value,
)

URLS = [
    "https://www.wikipedia.org/wiki/Phishing",
    "http://paypal.secure-login.example.com/verify/account.php",
    "https://user-a1b2c3.pages.dev/",
]


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector.load()


@pytest.fixture
def model_copy(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    shutil.copytree(ARTIFACT_DIR, target)
    return target


def read_metadata(directory: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    return data


def write_metadata(directory: Path, metadata: dict[str, Any]) -> None:
    (directory / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")


def test_the_shipped_model_loads_and_passes_its_self_check(detector: Detector) -> None:
    assert 0 < detector.threshold < 1
    assert detector.metadata["features"] == list(FEATURE_NAMES)


def test_model_is_not_a_pickle() -> None:
    assert (ARTIFACT_DIR / MODEL_FILE).read_bytes().startswith(b"tree\n")


def test_missing_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="cannot read"):
        Detector.load(tmp_path)


def test_tampered_model_is_refused(model_copy: Path) -> None:
    path = model_copy / MODEL_FILE
    path.write_bytes(path.read_bytes().replace(b"num_leaves=", b"num_leaves=9", 1))
    with pytest.raises(ModelError, match="checksum"):
        Detector.load(model_copy)


def test_corrupt_metadata_is_refused(model_copy: Path) -> None:
    (model_copy / METADATA_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelError, match="cannot read"):
        Detector.load(model_copy)


def test_feature_list_mismatch_is_refused(model_copy: Path) -> None:
    metadata = read_metadata(model_copy)
    metadata["features"] = metadata["features"][:-1]
    write_metadata(model_copy, metadata)
    with pytest.raises(ModelError, match="feature list"):
        Detector.load(model_copy)


def test_self_check_catches_drift_between_code_and_model(model_copy: Path) -> None:
    metadata = read_metadata(model_copy)
    metadata["self_check"][0]["score"] += 0.01
    write_metadata(model_copy, metadata)
    with pytest.raises(ModelError, match="self-check"):
        Detector.load(model_copy)


def test_missing_self_check_is_refused(model_copy: Path) -> None:
    metadata = read_metadata(model_copy)
    metadata["self_check"] = []
    write_metadata(model_copy, metadata)
    with pytest.raises(ModelError, match="self-check"):
        Detector.load(model_copy)


def test_scores_are_probabilities(detector: Detector) -> None:
    assert all(0.0 <= score <= 1.0 for score in detector.score(URLS))


def test_scores_are_repeatable(detector: Detector) -> None:
    assert detector.score(URLS) == detector.score(URLS)


def test_batch_and_single_scoring_agree(detector: Detector) -> None:
    batch = detector.score(URLS)
    single = [detector.score([url])[0] for url in URLS]
    assert batch == pytest.approx(single)


def test_assess_and_score_agree(detector: Detector) -> None:
    assessed = [item.score for item in detector.assess(URLS)]
    assert assessed == pytest.approx(detector.score(URLS), abs=1e-5)


def test_empty_input(detector: Detector) -> None:
    assert detector.score([]) == []
    assert detector.assess([]) == []


def test_explanation_is_additive(detector: Detector) -> None:
    """TreeSHAP contributions plus the bias must rebuild the raw score exactly."""
    parts, values = detector._contributions(URLS)
    assert len(parts) == len(URLS)
    raw = values.sum(axis=1)
    rebuilt = 1.0 / (1.0 + np.exp(-raw))
    assert rebuilt == pytest.approx(detector.score(URLS), abs=1e-9)


def test_bias_is_shared_by_all_rows(detector: Detector) -> None:
    _, values = detector._contributions(URLS)
    assert values[:, -1] == pytest.approx(detector.bias())


def test_explanations_are_sorted_and_limited(detector: Detector) -> None:
    result = detector.assess(URLS, top=3)[1]
    sizes = [abs(item.contribution) for item in result.contributions]
    assert len(sizes) <= 3
    assert sizes == sorted(sizes, reverse=True)
    assert all(item.description for item in result.contributions)


def test_top_zero_gives_no_explanations(detector: Detector) -> None:
    assert detector.assess(URLS, top=0)[0].contributions == ()


def test_verdict_uses_the_threshold_and_never_claims_safety(detector: Detector) -> None:
    assert detector.verdict(detector.threshold) == "likely_phishing"
    assert detector.verdict(detector.threshold - 0.001) == "no_strong_signal"


def test_shortener_gets_a_warning(detector: Detector) -> None:
    assert NOTE_SHORTENER in detector.assess(["https://bit.ly/3abcDEF"])[0].notes


def test_shared_host_gets_a_warning(detector: Detector) -> None:
    assert NOTE_SHARED_HOST in detector.assess(["https://docs.google.com/forms/d/abc"])[0].notes


def test_ignored_query_gets_a_note(detector: Detector) -> None:
    assert NOTE_IGNORED_PARTS in detector.assess(["https://example.com/a?b=c"])[0].notes
    assert NOTE_IGNORED_PARTS not in detector.assess(["https://example.com/a"])[0].notes


@pytest.mark.parametrize(
    "edit",
    [
        "https://example.org/about/team?utm_source=newsletter",
        "http://example.org/about/team",
        "https://www.example.org/about/team/",
        "example.org/about/team#top",
    ],
)
def test_harmless_edits_do_not_move_the_score(detector: Detector, edit: str) -> None:
    """The artefact that would flag every URL with a query string must stay out."""
    base = detector.score(["https://example.org/about/team"])[0]
    assert detector.score([edit])[0] == base


def test_scoring_never_opens_a_socket(detector: Detector, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("scoring tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert len(detector.assess(URLS)) == len(URLS)


def test_concurrent_scoring_gives_the_same_answers(detector: Detector) -> None:
    expected = detector.score(URLS)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: detector.score(URLS), range(64)))
    assert all(result == expected for result in results)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("tld_id", 0.0, "other"),
        ("file_kind", 2.0, "script"),
        ("is_ip_host", 1.0, "yes"),
        ("has_userinfo", 0.0, "no"),
        ("brand_in_path", 1.0, "yes"),
        ("host_length", 12.0, "12"),
        ("sld_entropy", 2.5, "2.500"),
    ],
)
def test_display_value(name: str, value: float, expected: str) -> None:
    assert display_value(name, value) == expected
