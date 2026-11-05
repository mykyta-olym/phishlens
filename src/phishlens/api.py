"""HTTP service. Scores strings only: it never resolves or requests the submitted URLs.

Request URLs are not logged, because real URLs often carry tokens and personal data.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from phishlens import __version__
from phishlens.model import ARTIFACT_DIR, DEFAULT_TOP_FEATURES, Detector
from phishlens.validation import MAX_BATCH_SIZE, MAX_URL_LENGTH, validate_url

# The largest valid batch, plus room for JSON syntax and escapes.
MAX_BODY_BYTES = MAX_BATCH_SIZE * MAX_URL_LENGTH * 2

Url = Annotated[str, AfterValidator(validate_url)]
Top = Annotated[int, Field(ge=0, le=20, description="How many features to explain.")]


class BodyLimitMiddleware:
    """Reject oversized bodies before they are parsed. FastAPI sets no limit by itself.

    The body is read here, up to the limit, and then replayed. Raising from inside
    `receive` would not work: FastAPI turns errors during a body read into a 400.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length", b"0")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse({"detail": "request body is too large"}, status_code=413)
        await response(scope, receive, send)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Url = Field(examples=["http://paypal.account-verify.example.com/login"])
    top: Top = DEFAULT_TOP_FEATURES


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    urls: list[Url] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    top: Top = DEFAULT_TOP_FEATURES


class ContributionOut(BaseModel):
    feature: str
    description: str
    value: str
    contribution: float = Field(description="Log-odds. Positive pushes towards phishing.")


class ScoreOut(BaseModel):
    url: str
    score: float = Field(ge=0, le=1)
    verdict: str = Field(description="likely_phishing or no_strong_signal. Never a safety claim.")
    threshold: float
    contributions: list[ContributionOut]
    notes: list[str]


class BatchOut(BaseModel):
    count: int
    results: list[ScoreOut]


class HealthOut(BaseModel):
    status: str
    version: str
    model_version: str


class ModelOut(BaseModel):
    version: str
    trained_at: str
    training_rows: int
    threshold: float
    features: list[str]
    limits: list[str]


def _detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


def create_app(model_dir: Path | None = None) -> FastAPI:
    """Build the app. The model loads once at startup, and a bad model stops startup."""
    directory = model_dir or Path(os.environ.get("PHISHLENS_MODEL_DIR") or ARTIFACT_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.detector = Detector.load(directory)
        yield

    app = FastAPI(
        title="phishlens",
        version=__version__,
        summary="Scores a URL string for phishing and explains the score.",
        description=(
            "Uses only the URL string. It never fetches the URL. A low score is not "
            "evidence that a URL is safe, and this must not be your only control."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> Response:
        """Report what is wrong without echoing the input.

        URLs can carry tokens, so they stay out of error bodies. Escaping to ASCII also
        keeps a lone surrogate in a field name from crashing the response.
        """
        details = [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        body = json.dumps({"detail": details})
        return Response(body, status_code=422, media_type="application/json")

    @app.get("/health")
    def health(request: Request) -> HealthOut:
        return HealthOut(status="ok", version=__version__, model_version=_detector(request).version)

    @app.get("/v1/model")
    def model_info(request: Request) -> ModelOut:
        detector = _detector(request)
        metadata: dict[str, Any] = detector.metadata
        return ModelOut(
            version=detector.version,
            trained_at=metadata["trained_at"],
            training_rows=metadata["training_rows"],
            threshold=detector.threshold,
            features=metadata["features"],
            limits=metadata["limits"],
        )

    @app.post("/v1/score")
    def score(body: ScoreRequest, request: Request) -> ScoreOut:
        result = _detector(request).assess([body.url], top=body.top)[0]
        return ScoreOut.model_validate(result.as_dict())

    @app.post("/v1/score/batch")
    def score_batch(body: BatchRequest, request: Request) -> BatchOut:
        results = _detector(request).assess(body.urls, top=body.top)
        return BatchOut(
            count=len(results),
            results=[ScoreOut.model_validate(item.as_dict()) for item in results],
        )

    return app
