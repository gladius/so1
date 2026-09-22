"""HTTP surface. POST /v1/systemone is wire-compatible with TypeSafe's Jev API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from so1 import __version__
from so1 import config as config_module
from so1.schemas import ModelMetadata, ModelMetadataList, SystemOneRequest, SystemOneResponse
from so1.service import Service, ServiceError

log = logging.getLogger("so1")

# Verbatim from the real API, so SDK error messages read the same against either service.
NO_KEY = "Must supply an API key! Check your request and try again."
BAD_KEY = "Cannot authenticate with the server. Please check your API key and try again."


def _error(status: int, detail: Any, retry_after: int | None = None) -> JSONResponse:
    """`detail` is passed through verbatim: a list for validation, an object or bare string otherwise."""
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)


def _auth_error(status: int, message: str) -> JSONResponse:
    return _error(status, {"error_type": "authentication_error", "message": message})


def create_app(config: config_module.Config, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=config.limits.max_upstream_concurrency + 8, max_keepalive_connections=32)
        async with httpx.AsyncClient(
            limits=limits, timeout=config.limits.upstream_timeout_s, transport=transport
        ) as client:
            service = Service(config, client)
            app.state.service = service
            await service.startup()
            yield

    app = FastAPI(
        title="so1",
        version=__version__,
        description="Jev-compatible System One API served by our own vLLM endpoints.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > config.limits.max_body_bytes:
            return _error(
                413,
                {
                    "error_type": "api_usage_error",
                    "message": f"Request body exceeds {config.limits.max_body_bytes} bytes.",
                },
            )
        if config.api_key and request.url.path.startswith("/v1/"):
            header = request.headers.get("authorization", "").strip()
            if not header:
                return _auth_error(403, NO_KEY)  # missing key: 403, same as the real API
            if header.removeprefix("Bearer ").strip() != config.api_key:
                return _auth_error(401, BAD_KEY)
        return await call_next(request)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.detail, exc.retry_after)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        # The real API masks an unknown question `type` as a plain 400 rather than leaking the
        # union internals, while a *missing* type stays a 422 union_tag_not_found. Match both.
        if any(error.get("type") == "union_tag_invalid" for error in errors):
            return _error(400, {"error_type": "api_usage_error", "message": "Invalid request."})
        return _error(422, jsonable_encoder(errors))

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
    async def systemone(request: SystemOneRequest, response: Response) -> SystemOneResponse:
        result = await app.state.service.answer(request)
        response.headers["Server-Timing"] = ", ".join(
            f"{name};dur={value:.1f}" for name, value in result.timings.items()
        )
        return result.response

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models() -> ModelMetadataList:
        return ModelMetadataList(
            models=[
                ModelMetadata(
                    name=name,
                    description=model.description or f"vLLM endpoint {model.upstream_model}.",
                    release_date=model.release_date,
                )
                for name, model in config.public_names()
            ]
        )

    @app.get("/v1/limits")
    async def limits() -> dict[str, Any]:
        service: Service = app.state.service
        return {
            "limits": vars(config.limits),
            "models": {
                name: {
                    "ready": up.ready,
                    "max_prompt_tokens": service._budget(up),
                    "max_choice_options": min(config.limits.max_choice_options, len(up.labels)),
                    "max_score_levels": config.limits.max_score_levels,
                }
                for name, up in service.upstreams.items()
            },
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        service: Service = app.state.service
        upstreams = {
            name: {"ready": up.ready, "detail": up.detail, "mode": up.mode, "strategy": up.strategy}
            for name, up in service.upstreams.items()
        }
        healthy = any(u["ready"] for u in upstreams.values())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "unavailable", "version": __version__, "upstreams": upstreams},
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    return app


def build() -> FastAPI:
    """Entry point for `uvicorn so1.app:build --factory`."""
    logging.basicConfig(level=os.environ.get("SO1_LOG_LEVEL", "INFO"))
    return create_app(config_module.load(os.environ.get("SO1_CONFIG", "config.yaml")))
