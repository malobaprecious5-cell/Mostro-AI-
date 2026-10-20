"""Health-check and meta response schemas."""

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, Any]


class MetaResponse(BaseModel):
    api_version: str
    build: str
    environment: str
    models: dict[str, str]
    embedding: dict[str, Any]
    prompt_versions: dict[str, str]
    retrieval: dict[str, Any]
    languages: list[str]
