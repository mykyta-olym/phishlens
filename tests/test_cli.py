from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from phishlens.cli import EXIT_FLAGGED, EXIT_OK, EXIT_USAGE, main
from phishlens.model import Detector

URL = "https://example.org/about/team"


def flagged_url() -> str:
    """Find a stored self-check URL over the threshold, so the test follows the model."""
    detector = Detector.load()
    items = detector.metadata["self_check"]
    return str(max(items, key=lambda item: item["score"])["url"])


def test_score_prints_score_and_explanation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", URL]) == EXIT_OK
    output = capsys.readouterr().out
    assert URL in output
    assert "phishing" in output


def test_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", URL, "--json", "--top", "2"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data[0]["url"] == URL
    assert len(data[0]["contributions"]) <= 2


def test_reads_a_file_and_skips_blank_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "urls.txt"
    path.write_text(f"{URL}\n\nhttps://example.com/\n", encoding="utf-8")
    assert main(["score", "--file", str(path), "--json"]) == EXIT_OK
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(URL + "\n"))
    assert main(["score", "--file", "-", "--json"]) == EXIT_OK
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_no_urls_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score"]) == EXIT_USAGE
    assert "at least one URL" in capsys.readouterr().err


def test_invalid_url_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", "javascript:alert(1)"]) == EXIT_USAGE
    assert "only http and https" in capsys.readouterr().err


def test_missing_file_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", "--file", "does-not-exist.txt"]) == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_bad_model_dir_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", URL, "--model-dir", str(tmp_path)]) == EXIT_USAGE
    assert "cannot read" in capsys.readouterr().err


def test_fail_on_flag_sets_the_exit_code() -> None:
    url = flagged_url()
    assert main(["score", url]) == EXIT_OK
    assert main(["score", url, "--fail-on-flag"]) == EXIT_FLAGGED


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip()
