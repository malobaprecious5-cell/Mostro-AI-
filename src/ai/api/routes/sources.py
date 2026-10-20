"""Sources route: serve the passages behind citations so users can verify answers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import get_async_session, require_internal_key, require_user_id
from ai.api.schemas.sources import DocumentPageView, SourceChunk
from ai.core.errors import AppError
from ai.db.models import ChatSession, Chunk, Document

router = APIRouter(prefix="/v1", tags=["sources"])

_AUTH = [Depends(require_internal_key)]
_PDF_URL_TTL = timedelta(minutes=15)


async def _owned_session(
    session_id: UUID, user_id: UUID, db: AsyncSession
) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Session not found.")
    if session.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")
    return session


@router.get("/chunks/{chunk_id}", response_model=SourceChunk, dependencies=_AUTH)
async def get_chunk(
    chunk_id: Annotated[UUID, Path()],
    session_id: Annotated[UUID, Query()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> SourceChunk:
    """Return the exact passage behind a citation, scoped to the session IPO."""
    session = await _owned_session(session_id, user_id, db)
    chunk = await db.get(Chunk, chunk_id)
    if chunk is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Chunk not found.")
    if chunk.ipo_id != session.ipo_id:
        raise AppError(
            status_code=403,
            code="FORBIDDEN",
            message="Chunk is not part of this session's IPO.",
        )
    return SourceChunk(
        id=chunk.id,
        document_id=chunk.document_id,
        section_path=chunk.section_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_type=chunk.chunk_type,
        content=chunk.content,
    )


@router.get(
    "/documents/{document_id}/pages/{page}",
    response_model=DocumentPageView,
    dependencies=_AUTH,
)
async def get_document_page(
    document_id: Annotated[UUID, Path()],
    page: Annotated[int, Path(ge=1)],
    session_id: Annotated[UUID, Query()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> DocumentPageView:
    """Page-level view of an official document bound to the caller's session IPO."""
    session = await _owned_session(session_id, user_id, db)
    document = await db.get(Document, document_id)
    if document is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Document not found.")
    if document.ipo_id != session.ipo_id:
        raise AppError(
            status_code=403,
            code="FORBIDDEN",
            message="Document is not part of this session's IPO.",
        )

    chunks = (
        await db.execute(
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.page_start <= page,
                Chunk.page_end >= page,
            )
            .order_by(Chunk.chunk_index)
        )
    ).scalars().all()
    text = "\n\n".join(chunk.content for chunk in chunks)
    return DocumentPageView(
        document_id=document.id,
        page=page,
        text=text,
        pdf_url=document.storage_uri,
        expires_at=datetime.now(UTC) + _PDF_URL_TTL,
    )
