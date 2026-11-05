from __future__ import annotations

import numpy as np
import pytest

from phishlens.metrics import (
    calibration_table,
    confusion,
    domain_bootstrap,
    full_metrics,
    slice_metrics,
    threshold_for_fpr,
    threshold_for_max_f1,
    threshold_metrics,
)

LABELS = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
SCORES = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.2, 0.1, 0.1])


def test_confusion_counts_by_hand() -> None:
    assert confusion(LABELS, SCORES >= 0.5) == {"tp": 2, "fp": 1, "tn": 4, "fn": 1}


def test_threshold_metrics_by_hand() -> None:
    result = threshold_metrics(LABELS, SCORES, 0.5)
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["false_positive_rate"] == pytest.approx(0.2, abs=1e-4)


def test_nothing_flagged_gives_zero_not_a_crash() -> None:
    result = threshold_metrics(LABELS, SCORES, 2.0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_perfect_ranking_has_auc_one() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert full_metrics(labels, scores, 0.5)["roc_auc"] == 1.0


def test_inverted_ranking_has_auc_zero() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert full_metrics(labels, scores, 0.5)["roc_auc"] == 0.0


def test_one_class_has_no_auc() -> None:
    labels = np.array([0, 0], dtype=np.int8)
    assert full_metrics(labels, np.array([0.1, 0.2]), 0.5)["roc_auc"] is None


def test_slice_metrics_with_one_class() -> None:
    labels = np.array([1, 1], dtype=np.int8)
    result = slice_metrics(labels, np.array([0.9, 0.1]), 0.5)
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] is None


def test_slice_metrics_when_empty() -> None:
    result = slice_metrics(np.array([], dtype=np.int8), np.array([]), 0.5)
    assert result["rows"] == 0
    assert result["median_score"] is None


def test_calibration_rows_add_up() -> None:
    table = calibration_table(LABELS, SCORES)
    assert sum(band["rows"] for band in table) == len(LABELS)


def test_max_f1_threshold_separates_clean_data() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    cut = threshold_for_max_f1(labels, scores)
    assert 0.3 < cut <= 0.7


def test_fpr_threshold_respects_the_target() -> None:
    random = np.random.default_rng(0)
    scores = random.random(10000)
    labels = np.zeros(10000, dtype=np.int8)
    cut = threshold_for_fpr(labels, scores, 0.01)
    assert (scores >= cut).mean() <= 0.0101


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    random = np.random.default_rng(0)
    labels = random.integers(0, 2, 600).astype(np.int8)
    scores = np.clip(labels * 0.5 + random.random(600) * 0.6, 0, 1)
    domains = np.array([f"d{i % 60}" for i in range(600)])
    result = domain_bootstrap(labels, scores, domains, 0.5, seed=1, rounds=50)
    point = threshold_metrics(labels, scores, 0.5)["f1"]
    assert result["f1"][0] <= point <= result["f1"][1]


def test_bootstrap_is_wider_when_a_domain_carries_the_result() -> None:
    """Resampling domains must see that one big domain is one observation."""
    random = np.random.default_rng(0)
    labels = random.integers(0, 2, 2000).astype(np.int8)
    scores = random.random(2000)
    scores[:1000] = labels[:1000] * 0.9 + 0.05
    spread = np.array([f"d{i}" for i in range(2000)])
    clumped = np.array(["big"] * 1000 + [f"d{i % 5}" for i in range(1000)])
    narrow = domain_bootstrap(labels, scores, spread, 0.5, seed=1, rounds=60)["f1"]
    wide = domain_bootstrap(labels, scores, clumped, 0.5, seed=1, rounds=60)["f1"]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
