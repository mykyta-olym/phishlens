"""Load the downloaded datasets into one shape: url, label, and source columns.

`label` is always 1 for phishing and 0 for benign. The sources disagree: PhiUSIIL and
LegitPhish use 1 for legitimate. Each loader checks the class counts the publisher
documents, so a flipped mapping fails here instead of producing a nonsense model.

Only the URL string and the label are read. The other columns in these files hold
values derived from page content or external services, which this project must not
use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phishlens.features import canonical_key
from phishlens.urlparts import parse_url

DEFAULT_RAW_DIR = Path("data/raw")


class DatasetError(RuntimeError):
    """A dataset is missing or does not match what its publisher documents."""


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"{path} is missing. Run: uv run python scripts/download_data.py")
    return pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )


def _check_counts(name: str, frame: pd.DataFrame, phishing: int, benign: int) -> None:
    actual = (int(frame["label"].sum()), int((frame["label"] == 0).sum()))
    if actual != (phishing, benign):
        raise DatasetError(f"{name}: got {actual} phishing and benign, want {(phishing, benign)}")


def load_phreshphish(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    columns = ["split", "sha256", "url", "label", "target", "date"]
    frame = _read_csv(raw_dir / "phreshphish" / "rows.csv.gz", columns)
    frame["label"] = (frame["label"] == "phish").astype("int8")
    _check_counts("phreshphish", frame, 221526 + 76800, 276729 + 91260)
    return frame


def load_hannousse(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    frame = _read_csv(raw_dir / "hannousse" / "dataset_B_05_2020.csv", ["url", "status"])
    frame["label"] = (frame.pop("status") == "phishing").astype("int8")
    _check_counts("hannousse", frame, 5715, 5715)
    return frame


def load_phiusiil(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    frame = _read_csv(path, ["URL", "label"]).rename(columns={"URL": "url"})
    frame["label"] = (frame["label"] == "0").astype("int8")
    _check_counts("phiusiil", frame, 100945, 134850)
    return frame


def load_legitphish(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "legitphish" / "url_features_extracted1.csv"
    frame = _read_csv(path, ["URL", "ClassLabel"]).rename(columns={"URL": "url"})
    # One row has no label. 0 is phishing per the dataset description.
    frame = frame[frame["ClassLabel"] != ""].reset_index(drop=True)
    frame["label"] = (frame.pop("ClassLabel") == "0").astype("int8")
    _check_counts("legitphish", frame, 63678, 37540)
    return frame


LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "phreshphish": load_phreshphish,
    "hannousse": load_hannousse,
    "phiusiil": load_phiusiil,
    "legitphish": load_legitphish,
}


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """What `clean` removed, kept so the evaluation can state it."""

    rows_in: int
    no_host: int
    duplicates: int
    label_conflicts: int
    rows_out: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_in": self.rows_in,
            "no_host": self.no_host,
            "duplicates": self.duplicates,
            "label_conflicts": self.label_conflicts,
            "rows_out": self.rows_out,
        }


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Add `domain` and `key`, then drop rows the model could not tell apart.

    `key` is exactly what the model reads, so URLs that differ only by query string or
    scheme collapse to one row. Left in, they would sit on both sides of a split as
    identical feature vectors. A key seen with both labels is dropped entirely, since
    no string-only model can get it right and either label would be arbitrary.
    """
    rows_in = len(frame)
    frame = frame.assign(
        domain=[parse_url(url).registered_domain for url in frame["url"]],
        key=[canonical_key(url) for url in frame["url"]],
    )
    has_host = frame["domain"] != ""
    no_host = int((~has_host).sum())
    frame = frame[has_host]

    per_key = frame.groupby("key")["label"].nunique()
    conflicted = per_key[per_key > 1].index
    conflict_mask = frame["key"].isin(conflicted)
    label_conflicts = int(conflict_mask.sum())
    frame = frame[~conflict_mask]

    before = len(frame)
    frame = frame.drop_duplicates("key", keep="first").reset_index(drop=True)
    report = CleaningReport(rows_in, no_host, before - len(frame), label_conflicts, len(frame))
    return frame, report
