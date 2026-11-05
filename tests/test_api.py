from __future__ import annotations

import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from phishlens.api import MAX_BODY_BYTES, BodyLimitMiddleware, create_app
from phishlens.model import ARTIFACT_DIR, METADATA_FILE, ModelError
from phishlens.validation import MAX_BATCH_SIZE, MAX_URL_LENGTH

GOOD = "https://example.org/about/team"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_states_its_limits(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["training_rows"] > 0
    assert any("only control" in limit for limit in body["limits"])


def test_score_returns_a_score_and_an_explanation(client: TestClient) -> None:
    response = client.post("/v1/score", json={"url": GOOD})
    assert response.status_code == 200
    body = response.json()
    assert body["url"] == GOOD
    assert 0 <= body["score"] <= 1
    assert body["verdict"] in ("likely_phishing", "no_strong_signal")
    assert body["contributions"]
    first = body["contributions"][0]
    assert set(first) == {"feature", "description", "value", "contribution"}


def test_top_limits_the_explanation(client: TestClient) -> None:
    body = client.post("/v1/score", json={"url": GOOD, "top": 2}).json()
    assert len(body["contributions"]) <= 2


def test_batch_keeps_order(client: TestClient) -> None:
    urls = [GOOD, "https://bit.ly/3abcDEF", "http://192.0.2.1/login"]
    response = client.post("/v1/score/batch", json={"urls": urls})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [item["url"] for item in body["results"]] == urls


def test_batch_matches_single_scoring(client: TestClient) -> None:
    single = client.post("/v1/score", json={"url": GOOD}).json()
    batch = client.post("/v1/score/batch", json={"urls": [GOOD]}).json()
    assert batch["results"][0] == single


def test_largest_allowed_batch(client: TestClient) -> None:
    response = client.post("/v1/score/batch", json={"urls": [GOOD] * MAX_BATCH_SIZE, "top": 0})
    assert response.status_code == 200
    assert response.json()["count"] == MAX_BATCH_SIZE


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"url": ""},
        {"url": "   "},
        {"url": 5},
        {"url": None},
        {"url": ["https://example.com"]},
        {"url": "javascript:alert(1)"},
        {"url": "file:///etc/passwd"},
        {"url": "https://"},
        {"url": "localhost"},
        {"url": "https://exa mple.com"},
        {"url": "https://example.com/\u0000"},
        {"url": "https://example.com/" + "a" * MAX_URL_LENGTH},
        {"url": GOOD, "extra": 1},
        {"url": GOOD, "top": -1},
        {"url": GOOD, "top": 21},
    ],
)
def test_bad_input_is_a_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/score", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"urls": []},
        {"urls": "https://example.com"},
        {"urls": [GOOD] * (MAX_BATCH_SIZE + 1)},
    ],
)
def test_bad_batch_is_a_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/score/batch", json=payload).status_code == 422


def test_one_bad_url_names_its_position(client: TestClient) -> None:
    response = client.post("/v1/score/batch", json={"urls": [GOOD, GOOD, "ftp://x.example"]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "urls", 2]


def test_error_text_is_plain(client: TestClient) -> None:
    body = client.post("/v1/score", json={"url": "ftp://x.example"}).json()
    assert "only http and https" in body["detail"][0]["msg"]


def test_malformed_json_is_a_422(client: TestClient) -> None:
    response = client.post("/v1/score", content="{", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_lone_surrogate_is_rejected_cleanly(client: TestClient) -> None:
    response = client.post(
        "/v1/score",
        content=b'{"url": "https://example.com/\\ud800"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_lone_surrogate_in_a_field_name_is_rejected_cleanly(client: TestClient) -> None:
    response = client.post(
        "/v1/score",
        content=b'{"url": "https://example.com", "\ud800": 1}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_error_bodies_do_not_echo_the_url(client: TestClient) -> None:
    secret = "ftp://x.example/reset?token=abc123"
    response = client.post("/v1/score", json={"url": secret})
    assert response.status_code == 422
    assert "abc123" not in response.text


def test_wrong_method(client: TestClient) -> None:
    assert client.get("/v1/score").status_code == 405


def test_oversized_declared_body_is_a_413(client: TestClient) -> None:
    headers = {"content-type": "application/json", "content-length": str(MAX_BODY_BYTES + 1)}
    response = client.post("/v1/score", content=b"{}", headers=headers)
    assert response.status_code == 413


def test_chunked_body_over_the_limit_is_a_413() -> None:
    """A body without Content-Length must hit the limit too."""
    app = create_app()
    app.user_middleware.clear()
    app.add_middleware(BodyLimitMiddleware, max_bytes=64)

    def chunks() -> Iterator[bytes]:
        for _ in range(10):
            yield b'{"url": "https://example.com/aaaaaaaaaaaaaaaaaaaaa"}'

    with TestClient(app) as test_client:
        response = test_client.post("/v1/score", content=chunks())
    assert response.status_code == 413


def test_a_bad_model_stops_startup(tmp_path: Path) -> None:
    target = tmp_path / "model"
    shutil.copytree(ARTIFACT_DIR, target)
    (target / METADATA_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(ModelError), TestClient(create_app(target)):
        pass


def test_model_dir_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHISHLENS_MODEL_DIR", str(tmp_path))
    with pytest.raises(ModelError, match="cannot read"), TestClient(create_app()):
        pass


def test_the_service_never_contacts_a_submitted_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("the service tried to resolve a submitted URL")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    response = client.post("/v1/score", json={"url": "http://169.254.169.254/latest/meta-data"})
    assert response.status_code == 200
