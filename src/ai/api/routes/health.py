"""Readiness and meta endpoints for health checks and operational transparency."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import get_async_session, require_internal_key
from ai.api.schemas.health import HealthResponse, MetaResponse, ReadyResponse
from ai.config import settings
from ai.core.errors import AppError

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up. No auth required. Used by Docker healthcheck."""
    return HealthResponse()


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    dependencies=[Depends(require_internal_key)],
    responses={503: {"description": "Service not ready"}},
)
async def readiness(
    db_session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> JSONResponse:
    """Readiness: database reachable and pgvector present.

    Returns 200 when all checks pass, 503 otherwise with a ``checks`` object
    showing which dependency failed.
    """
    checks: dict[str, Any] = {}

    # 1. Database connectivity
    try:
        await db_session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["database"] = f"error: {e}"

    # 2. pgvector extension present
    try:
        result = await db_session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.scalar_one_or_none()
        checks["pgvector"] = "ok" if row == "vector" else "not installed"
    except Exception as e:  # noqa: BLE001
        checks["pgvector"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    payload = ReadyResponse(
        status="ready" if all_ok else "degraded",
        checks=checks,
    )

    if not all_ok:
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
    return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))


@router.get(
    "/v1/meta",
    response_model=MetaResponse,
    dependencies=[Depends(require_internal_key)],
)
async def meta() -> MetaResponse:
    """Runtime metadata: model names, embedding config, prompt versions.

    Useful for debugging, audits, and verifying the running configuration.
    """
    return MetaResponse(
        api_version="0.1.0",
        build=settings.build,
        environment=settings.environment,
        models={
            "chat_fast": settings.llm_model_fast,
            "chat_main": settings.llm_model_main,
        },
        embedding={
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dim": settings.embedding_dim,
        },
        prompt_versions={
            "system": "system_v1.0",
            "answer": "answer_v1.0",
        },
        retrieval={
            "candidates": settings.retrieval_candidates,
            "top_k": settings.rerank_top_k,
            "abstain_threshold": settings.abstain_score_threshold,
        },
        languages=["en", "fr"],
    )
