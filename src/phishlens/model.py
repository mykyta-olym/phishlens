"""Load the saved model, score URLs and explain each score.

The model is stored in LightGBM's text format, not as a pickle, so loading a model
file cannot run code. Explanations come from LightGBM's built-in TreeSHAP. They are
exact for this model: the contributions plus the bias sum to the raw score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from phishlens.features import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_TEXT,
    FILE_KINDS,
    extract_parts,
    is_shared_host,
    is_shortener,
)
from phishlens.lexicon import TLD_VOCABULARY
from phishlens.urlparts import UrlParts, parse_url

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = "model.txt"
METADATA_FILE = "metadata.json"
DEFAULT_TOP_FEATURES = 5
MIN_SHOWN_CONTRIBUTION = 0.01
SELF_CHECK_TOLERANCE = 1e-6

NOTE_SHORTENER = (
    "This is a URL shortener. The string does not show the real destination, "
    "so this score says almost nothing about where the link leads."
)
NOTE_SHARED_HOST = (
    "This host is a shared platform where anyone can publish. "
    "Its reputation does not transfer to one page, in either direction."
)
NOTE_IGNORED_PARTS = "The query string and fragment were ignored. See the README for why."


class ModelError(RuntimeError):
    """The saved model is missing, damaged or does not match this code."""


@dataclass(frozen=True, slots=True)
class Contribution:
    """How much one feature moved the score, in log-odds. Positive means more phishing."""

    feature: str
    description: str
    value: str
    contribution: float


@dataclass(frozen=True, slots=True)
class Assessment:
    url: str
    score: float
    verdict: str
    threshold: float
    contributions: tuple[Contribution, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def display_value(name: str, value: float) -> str:
    """Render a feature value for people. Categories become names, flags become yes or no."""
    if name == "tld_id":
        return TLD_VOCABULARY[int(value)]
    if name == "file_kind":
        return FILE_KINDS[int(value)]
    if name.startswith(("is_", "has_", "brand_in_", "brand_official", "path_has_")):
        return "yes" if value else "no"
    return f"{value:g}" if value == int(value) else f"{value:.3f}"


def _notes(parts: UrlParts) -> tuple[str, ...]:
    notes = []
    if is_shortener(parts):
        notes.append(NOTE_SHORTENER)
    if is_shared_host(parts):
        notes.append(NOTE_SHARED_HOST)
    if parts.query or parts.fragment:
        notes.append(NOTE_IGNORED_PARTS)
    return tuple(notes)


class Detector:
    """A loaded model with its decision threshold."""

    def __init__(self, booster: lgb.Booster, metadata: dict[str, Any]) -> None:
        self._booster = booster
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.version = str(metadata["version"])

    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR) -> Detector:
        """Load and verify a model. Raises `ModelError` rather than serving a bad one."""
        model_path = directory / MODEL_FILE
        metadata_path = directory / METADATA_FILE
        try:
            model_bytes = model_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ModelError(f"cannot read the model in {directory}: {error}") from error

        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != metadata.get("model_sha256"):
            raise ModelError(f"{model_path} does not match the checksum in {METADATA_FILE}")
        if metadata.get("features") != list(FEATURE_NAMES):
            raise ModelError("the saved model was trained with a different feature list")

        try:
            booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
        except (lgb.basic.LightGBMError, UnicodeDecodeError) as error:
            raise ModelError(f"cannot parse {model_path}: {error}") from error

        detector = cls(booster, metadata)
        detector._self_check()
        return detector

    def _self_check(self) -> None:
        """Re-score stored URLs, so changed parsing or word lists stop the service."""
        expected = self.metadata.get("self_check", [])
        if not expected:
            raise ModelError("the model metadata has no self-check URLs")
        actual = self.score([item["url"] for item in expected])
        for item, score in zip(expected, actual, strict=True):
            if abs(item["score"] - score) > SELF_CHECK_TOLERANCE:
                raise ModelError(
                    f"self-check failed for {item['url']}: "
                    f"stored {item['score']:.6f}, computed {score:.6f}"
                )

    def _contributions(self, urls: Sequence[str]) -> tuple[list[UrlParts], NDArray[np.float64]]:
        parts = [parse_url(url) for url in urls]
        rows = np.array([extract_parts(item) for item in parts], dtype=np.float64)
        # One thread: requests already run in a pool, and OpenMP inside it only adds contention.
        values = self._booster.predict(rows, pred_contrib=True, num_threads=1)
        return parts, np.asarray(values, dtype=np.float64)

    def score(self, urls: Sequence[str]) -> list[float]:
        """Scores only, for callers that do not need explanations."""
        if not urls:
            return []
        rows = np.array([extract_parts(parse_url(url)) for url in urls], dtype=np.float64)
        raw = self._booster.predict(rows, raw_score=True, num_threads=1)
        return [_sigmoid(float(value)) for value in raw]

    def assess(self, urls: Sequence[str], top: int = DEFAULT_TOP_FEATURES) -> list[Assessment]:
        """Score URLs and attach the features that moved each score the most."""
        if not urls:
            return []
        parts, contributions = self._contributions(urls)
        results = []
        for url, item, row in zip(urls, parts, contributions, strict=True):
            score = _sigmoid(float(row.sum()))
            features = extract_parts(item)
            order = np.argsort(-np.abs(row[:-1]))[:top]
            shown = tuple(
                Contribution(
                    feature=FEATURE_NAMES[index],
                    description=FEATURE_TEXT[FEATURE_NAMES[index]],
                    value=display_value(FEATURE_NAMES[index], features[index]),
                    contribution=round(float(row[index]), 4),
                )
                for index in order
                if abs(row[index]) >= MIN_SHOWN_CONTRIBUTION
            )
            results.append(
                Assessment(
                    url=url,
                    score=round(score, 6),
                    verdict=self.verdict(score),
                    threshold=self.threshold,
                    contributions=shown,
                    notes=_notes(item),
                )
            )
        return results

    def verdict(self, score: float) -> str:
        """Never `safe`: a low score only means this string gave no strong signal."""
        return "likely_phishing" if score >= self.threshold else "no_strong_signal"

    def bias(self) -> float:
        """The score before any feature is considered, in log-odds."""
        row = np.zeros((1, len(FEATURE_INDEX)), dtype=np.float64)
        values = self._booster.predict(row, pred_contrib=True, num_threads=1)
        return float(np.asarray(values)[0, -1])
