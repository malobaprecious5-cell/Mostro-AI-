"""FastAPI dependencies shared across route modules."""

import hmac
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import Header
from sqlalchemy.ext.asyncio import AsyncSession

from ai.config import settings
from ai.core.errors import AppError
from ai.db.session import get_db_session
from ai.llm.factory import get_chat_client, get_embedding_client
from ai.llm.interfaces import ChatClient, EmbeddingClient


async def require_internal_key(
    supplied_key: Annotated[str | None, Header(alias="X-Internal-Key")] = None,
) -> None:
    configured_key = settings.internal_api_key.get_secret_value()
    if not configured_key:
        raise AppError(
            status_code=503,
            code="NOT_READY",
            message="Internal API authentication is not configured.",
        )
    if supplied_key is None or not hmac.compare_digest(supplied_key, configured_key):
        raise AppError(
            status_code=401,
            code="UNAUTHENTICATED",
            message="A valid X-Internal-Key header is required.",
        )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    async for session in get_db_session():
        yield session


def get_chat_client_dep() -> ChatClient:
    """Dependency for chat client."""
    return get_chat_client()


def get_embedding_client_dep() -> EmbeddingClient:
    """Dependency for embedding client."""
    return get_embedding_client()


async def require_user_id(
    user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> UUID:
    """Require the gateway-supplied user id on user-facing endpoints."""
    return user_id


async def require_admin_key(
    supplied_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> None:
    """Enforce the admin API key for admin-only endpoints."""
    configured_key = settings.admin_api_key.get_secret_value()
    if not configured_key:
        raise AppError(
            status_code=503,
            code="NOT_READY",
            message="Admin API authentication is not configured.",
        )
    if supplied_key is None or not hmac.compare_digest(supplied_key, configured_key):
        raise AppError(
            status_code=401,
            code="UNAUTHENTICATED",
            message="A valid X-Admin-Key header is required.",
        )


