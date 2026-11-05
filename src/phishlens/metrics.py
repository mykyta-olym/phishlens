"""Metrics used by the evaluation. Everything here is measured on labelled rows."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

Floats = NDArray[np.float64]
Labels = NDArray[np.int8]

BOOTSTRAP_ROUNDS = 200
ROUND_TO = 4


def confusion(labels: Labels, predicted: NDArray[np.bool_]) -> dict[str, int]:
    positive = labels == 1
    return {
        "tp": int((predicted & positive).sum()),
        "fp": int((predicted & ~positive).sum()),
        "tn": int((~predicted & ~positive).sum()),
        "fn": int((~predicted & positive).sum()),
    }


def _ratio(top: float, bottom: float) -> float:
    return float(top / bottom) if bottom else 0.0


def threshold_metrics(labels: Labels, scores: Floats, threshold: float) -> dict[str, Any]:
    """Precision, recall, F1, false positive rate and the confusion matrix at one cut."""
    matrix = confusion(labels, scores >= threshold)
    precision = _ratio(matrix["tp"], matrix["tp"] + matrix["fp"])
    recall = _ratio(matrix["tp"], matrix["tp"] + matrix["fn"])
    return {
        "precision": round(precision, ROUND_TO),
        "recall": round(recall, ROUND_TO),
        "f1": round(_ratio(2 * precision * recall, precision + recall), ROUND_TO),
        "false_positive_rate": round(_ratio(matrix["fp"], matrix["fp"] + matrix["tn"]), ROUND_TO),
        "confusion": matrix,
    }


def full_metrics(labels: Labels, scores: Floats, threshold: float) -> dict[str, Any]:
    """Threshold metrics plus the ones that do not depend on a threshold."""
    both_classes = len(np.unique(labels)) == 2
    result = {
        "rows": len(labels),
        "phishing_share": round(float(labels.mean()), ROUND_TO),
        "roc_auc": round(float(roc_auc_score(labels, scores)), ROUND_TO) if both_classes else None,
        "average_precision": (
            round(float(average_precision_score(labels, scores)), ROUND_TO)
            if both_classes
            else None
        ),
        "brier": round(float(np.mean((scores - labels) ** 2)), ROUND_TO),
    }
    return result | threshold_metrics(labels, scores, threshold)


def slice_metrics(labels: Labels, scores: Floats, threshold: float) -> dict[str, Any]:
    """Figures that stay defined when a slice holds only one class."""
    flagged = scores >= threshold
    positive = labels == 1
    return {
        "rows": len(labels),
        "phishing_rows": int(positive.sum()),
        "benign_rows": int((~positive).sum()),
        "recall": round(float(flagged[positive].mean()), ROUND_TO) if positive.any() else None,
        "false_positive_rate": (
            round(float(flagged[~positive].mean()), ROUND_TO) if (~positive).any() else None
        ),
        "median_score": round(float(np.median(scores)), ROUND_TO) if len(scores) else None,
    }


def calibration_table(labels: Labels, scores: Floats, bins: int = 10) -> list[dict[str, Any]]:
    """Mean predicted score against the observed phishing share, per score band."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    table = []
    for band in range(bins):
        mask = index == band
        if not mask.any():
            continue
        table.append(
            {
                "band": f"{edges[band]:.1f}-{edges[band + 1]:.1f}",
                "rows": int(mask.sum()),
                "mean_score": round(float(scores[mask].mean()), ROUND_TO),
                "observed_phishing_share": round(float(labels[mask].mean()), ROUND_TO),
            }
        )
    return table


def domain_bootstrap(
    labels: Labels,
    scores: Floats,
    domains: NDArray[np.str_],
    threshold: float,
    seed: int,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> dict[str, list[float]]:
    """95 percent intervals from resampling whole domains.

    Rows on one domain are not independent, so resampling rows would give intervals
    that are too narrow. Resampling domains keeps each domain's rows together.
    """
    _, group = np.unique(domains, return_inverse=True)
    groups = int(group.max()) + 1
    random = np.random.default_rng(seed)
    predicted = scores >= threshold
    collected: dict[str, list[float]] = {"precision": [], "recall": [], "f1": [], "roc_auc": []}
    for _ in range(rounds):
        picks = np.bincount(random.integers(0, groups, groups), minlength=groups)
        weight = picks[group].astype(np.float64)
        tp = weight[predicted & (labels == 1)].sum()
        fp = weight[predicted & (labels == 0)].sum()
        fn = weight[~predicted & (labels == 1)].sum()
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        collected["precision"].append(precision)
        collected["recall"].append(recall)
        collected["f1"].append(_ratio(2 * precision * recall, precision + recall))
        collected["roc_auc"].append(float(roc_auc_score(labels, scores, sample_weight=weight)))
    return {
        name: [
            round(float(np.percentile(values, 2.5)), ROUND_TO),
            round(float(np.percentile(values, 97.5)), ROUND_TO),
        ]
        for name, values in collected.items()
    }


def threshold_for_max_f1(labels: Labels, scores: Floats) -> float:
    """The cut with the best F1. Call this on validation rows only."""
    grid = np.linspace(0.01, 0.99, 197)
    values = [float(threshold_metrics(labels, scores, cut)["f1"]) for cut in grid]
    return round(float(grid[int(np.argmax(values))]), ROUND_TO)


def threshold_for_fpr(labels: Labels, scores: Floats, target: float) -> float:
    """The lowest cut whose false positive rate stays within `target`, on validation rows."""
    benign = np.sort(scores[labels == 0])
    position = int(np.ceil(len(benign) * (1.0 - target)))
    return round(float(benign[min(position, len(benign) - 1)]), 6)
