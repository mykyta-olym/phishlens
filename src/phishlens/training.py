"""Train the model with a fixed seed and save it.

Only rows from the PhreshPhish `train` split are read for fitting. Rows are split by
registered domain into fit and validation sets, and the decision threshold is chosen
on validation rows. The `test` split is left to `phishlens evaluate`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from phishlens import __version__
from phishlens.datasets import DEFAULT_RAW_DIR, clean, load_phreshphish
from phishlens.features import CATEGORICAL_FEATURES, FEATURE_NAMES, artefact_features, extract
from phishlens.metrics import threshold_for_fpr, threshold_for_max_f1
from phishlens.model import ARTIFACT_DIR, METADATA_FILE, MODEL_FILE, Detector

SEED = 42
VALIDATION_SHARE = 0.15
FEATURE_BLOCK = 20000
TRAIN_THREADS = 4
ROUNDS = 200
DOMAIN_WEIGHT_CAP = 100.0
TARGET_FALSE_POSITIVE_RATE = 0.01

LIMITS = [
    "Reads only the URL string, so it cannot see page content, certificates or reputation.",
    "A low score is not evidence of safety.",
    "URL shorteners hide the destination, so their scores carry almost no information.",
    "Compromised legitimate sites look like ordinary URLs and are often missed.",
    "It must not be the only control.",
]

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "cat_smooth": 50.0,
    "max_cat_to_onehot": 4,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": TRAIN_THREADS,
}

# Stored with the model and re-scored at startup. These hosts are reserved or made up.
SELF_CHECK_URLS = (
    "https://example.com/",
    "https://www.example.org/about/team",
    "http://paypal.secure-login.example.com/verify/account",
    "https://account-update.example.net/signin.php",
    "http://192.0.2.10/login",
    "https://tinyurl.com/abc123",
    "https://user-a1b2c3.pages.dev/",
    "https://docs.example.co.uk/guide/getting-started.html",
)


def _block(urls: Sequence[str]) -> NDArray[np.float32]:
    return np.array([extract(url) for url in urls], dtype=np.float32)


def _artefact_block(urls: Sequence[str]) -> NDArray[np.float32]:
    return np.array([artefact_features(url) for url in urls], dtype=np.float32)


def _in_parallel(function: Any, urls: Sequence[str]) -> NDArray[np.float32]:
    blocks = [urls[i : i + FEATURE_BLOCK] for i in range(0, len(urls), FEATURE_BLOCK)]
    return np.vstack(Parallel(n_jobs=-1)(delayed(function)(block) for block in blocks))


def feature_matrix(urls: Sequence[str]) -> NDArray[np.float32]:
    return _in_parallel(_block, urls)


def artefact_matrix(urls: Sequence[str]) -> NDArray[np.float32]:
    return _in_parallel(_artefact_block, urls)


def split_by_domain(
    domains: NDArray[np.str_], share: float, seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Put each registered domain wholly on one side. Returns fit and held-out row indices."""
    unique = np.unique(domains)
    random = np.random.default_rng(seed)
    held_out = random.permutation(unique)[: round(len(unique) * share)]
    mask = np.isin(domains, held_out)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def domain_weights(
    domains: NDArray[np.str_], cap: float = DOMAIN_WEIGHT_CAP
) -> NDArray[np.float32]:
    """Limit how much one domain can count, so a few big platforms do not define the model."""
    counts = pd.Series(domains).map(pd.Series(domains).value_counts()).to_numpy()
    return np.minimum(1.0, cap / counts).astype(np.float32)


def fit_booster(
    rows: NDArray[np.float32],
    labels: NDArray[np.int8],
    weights: NDArray[np.float32],
    seed: int,
    names: Sequence[str] = FEATURE_NAMES,
    rounds: int = ROUNDS,
) -> lgb.Booster:
    data = lgb.Dataset(
        rows,
        label=labels,
        weight=weights,
        feature_name=list(names),
        categorical_feature=[name for name in CATEGORICAL_FEATURES if name in names],
        free_raw_data=True,
    )
    return lgb.train({**PARAMS, "seed": seed}, data, num_boost_round=rounds)


@dataclass(frozen=True, slots=True)
class TrainedModel:
    booster: lgb.Booster
    threshold: float
    strict_threshold: float
    fit_rows: int
    validation_rows: int
    validation_domains: NDArray[np.str_]


def train_on(
    frame: pd.DataFrame,
    rows: NDArray[np.float32],
    seed: int,
    names: Sequence[str] = FEATURE_NAMES,
) -> TrainedModel:
    """Fit on domains outside the validation share, then pick thresholds on validation."""
    domains = frame["domain"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=np.int8)
    fit, validation = split_by_domain(domains, VALIDATION_SHARE, seed)
    booster = fit_booster(rows[fit], labels[fit], domain_weights(domains[fit]), seed, names)
    scores = np.asarray(booster.predict(rows[validation]), dtype=np.float64)
    return TrainedModel(
        booster=booster,
        threshold=threshold_for_max_f1(labels[validation], scores),
        strict_threshold=threshold_for_fpr(labels[validation], scores, TARGET_FALSE_POSITIVE_RATE),
        fit_rows=len(fit),
        validation_rows=len(validation),
        validation_domains=np.unique(domains[validation]),
    )


def save_model(model: TrainedModel, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = model.booster.model_to_string()
    (output_dir / MODEL_FILE).write_bytes(text.encode("utf-8"))
    raw = np.array(model.booster.predict(_block(SELF_CHECK_URLS), raw_score=True))
    metadata = {
        "version": __version__,
        "trained_at": dt.datetime.now(dt.UTC).date().isoformat(),
        "seed": seed,
        "model_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "training_rows": model.fit_rows,
        "validation_rows": model.validation_rows,
        "threshold": model.threshold,
        "strict_threshold": model.strict_threshold,
        "features": list(FEATURE_NAMES),
        "limits": LIMITS,
        "self_check": [
            {"url": url, "score": float(1.0 / (1.0 + np.exp(-value)))}
            for url, value in zip(SELF_CHECK_URLS, raw, strict=True)
        ],
    }
    text = json.dumps(metadata, indent=2) + "\n"
    (output_dir / METADATA_FILE).write_text(text, encoding="utf-8", newline="\n")


def load_training_frame(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return the clean train rows, the clean test rows and the cleaning report."""
    frame, report = clean(load_phreshphish(raw_dir))
    is_train = frame["split"] == "train"
    return (
        frame[is_train].reset_index(drop=True),
        frame[~is_train].reset_index(drop=True),
        report.as_dict(),
    )


def run_training(
    raw_dir: Path = DEFAULT_RAW_DIR, output_dir: Path | None = None, seed: int = SEED
) -> Detector:
    target = output_dir or ARTIFACT_DIR
    train, _, report = load_training_frame(raw_dir)
    print(f"cleaning: {report}")
    print(f"training rows: {len(train)}, phishing share {train['label'].mean():.4f}")
    rows = feature_matrix(train["url"].tolist())
    model = train_on(train, rows, seed)
    save_model(model, target, seed)
    print(f"fit rows {model.fit_rows}, validation rows {model.validation_rows}")
    print(f"threshold {model.threshold}, strict threshold {model.strict_threshold}")
    print(f"saved to {target}")
    return Detector.load(target)
