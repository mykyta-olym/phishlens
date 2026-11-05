"""The documents must match what was measured."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from phishlens.model import Detector

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "metrics.json"


def test_reports_match_the_measurements() -> None:
    script = ROOT / "scripts" / "render_report.py"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script), "--check"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_metrics_describe_the_shipped_model() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    detector = Detector.load()
    assert report["main"]["threshold"] == detector.threshold
    assert report["main"]["strict_threshold"] == detector.metadata["strict_threshold"]
    assert report["seed"] == detector.metadata["seed"]
    assert report["quick_run"] is False


def test_shuffled_labels_remove_almost_all_of_the_score() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    check = report["checks"]["shuffled_labels"]
    assert len(check["roc_auc_values"]) >= 20
    assert abs(check["roc_auc_mean"] - 0.5) < 0.03


def test_no_train_test_overlap_was_recorded() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    assert report["data"]["test_keys_also_in_train"] == 0


def test_harmless_edits_changed_nothing_for_the_shipped_model() -> None:
    block = json.loads(METRICS.read_text(encoding="utf-8"))["artefacts"]
    before = block["benign_flagged_before_edit"]["shipped_flagged"]
    for edit in block["benign_flagged_after_edit"].values():
        assert edit["shipped_flagged"] == before


def test_the_readme_warns_that_this_is_not_a_sole_control() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "must not be the only control" in text


def test_prose_quotes_no_metric_outside_the_generated_block() -> None:
    """Numbers like 0.97 belong in the generated block, where they trace to metrics.json."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    prose = re.sub(r"<!-- results:start -->.*<!-- results:end -->", "", text, flags=re.DOTALL)
    prose = re.sub(r"```.*?```", "", prose, flags=re.DOTALL)
    assert not re.findall(r"\b0\.\d{2,}\b|\b\d{1,3}(?:\.\d+)?%", prose)
