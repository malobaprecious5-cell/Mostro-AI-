"""Sessions, messages, and feedback routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import get_async_session, require_internal_key, require_user_id
from ai.api.schemas.sessions import (
    CalendarProposalOut,
    FeedbackCreate,
    MessageCitationOut,
    MessageOut,
    MessagePage,
    Reminder,
    Session,
    SessionCreate,
    SessionPage,
    SessionUpdate,
)
from ai.core.errors import AppError
from ai.db.models import CalendarProposal, ChatSession, Chunk, Document, IPO, Message
from ai.db.models import MessageCitation

router = APIRouter(prefix="/v1", tags=["sessions"])

_AUTH = [Depends(require_internal_key)]


def _session_to_schema(s: ChatSession) -> Session:
    return Session(
        id=s.id,
        user_id=s.user_id,
        ipo_id=s.ipo_id,
        language=s.language,  # type: ignore[arg-type]
        level=s.level,  # type: ignore[arg-type]
        created_at=s.created_at,
        last_active=s.last_active,
    )


async def _get_own_session(
    session_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> ChatSession:
    row = await db.get(ChatSession, session_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Session not found.")
    if row.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")
    return row


def _parse_cursor(cursor: str | None) -> datetime | None:
    if not cursor:
        return None
    try:
        parsed = datetime.fromisoformat(cursor)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


async def _message_to_schema(msg: Message, db: AsyncSession) -> MessageOut:
    citation_rows = (
        await db.execute(
            select(MessageCitation, Chunk)
            .join(Chunk, Chunk.id == MessageCitation.chunk_id)
            .where(MessageCitation.message_id == msg.id)
            .order_by(MessageCitation.id)
        )
    ).all()

    citations = [
        MessageCitationOut(
            index=i,
            chunk_id=citation.chunk_id,
            document_id=chunk.document_id,
            page_start=citation.page,
            page_end=citation.page,
            section_path=chunk.section_path,
            quote=citation.quote,
        )
        for i, (citation, chunk) in enumerate(citation_rows, start=1)
    ]

    proposal_rows = (
        await db.execute(
            select(CalendarProposal).where(CalendarProposal.message_id == msg.id)
        )
    ).scalars().all()

    return MessageOut(
        id=msg.id,
        session_id=msg.session_id,
        role=msg.role,  # type: ignore[arg-type]
        content=msg.content,
        outcome=msg.outcome,  # type: ignore[arg-type]
        citations=citations,
        proposals=[_proposal_to_schema(row) for row in proposal_rows],
        created_at=msg.created_at,
    )


def _proposal_to_schema(row: CalendarProposal) -> CalendarProposalOut:
    return CalendarProposalOut(
        id=row.id,
        key_fact_id=row.key_fact_id,
        title=row.title,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        timezone=row.timezone,
        reminders=[Reminder.model_validate(item) for item in row.reminders],
        status=row.status,
        source_page=row.source_page,
    )


@router.post("/sessions", response_model=Session, status_code=201, dependencies=_AUTH)
async def create_session(
    body: SessionCreate,
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Session:
    """Create a new session bound to one IPO."""
    ipo = await db.get(IPO, body.ipo_id)
    if ipo is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")
    if ipo.status != "published":
        raise AppError(
            status_code=409,
            code="IPO_NOT_PUBLISHED",
            message=f"IPO '{ipo.issuer_name}' is not published (status: {ipo.status}).",
        )

    indexed_count = (
        await db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.ipo_id == body.ipo_id, Document.status == "indexed")
        )
    ).scalar_one()
    if indexed_count == 0:
        raise AppError(
            status_code=409,
            code="PUBLISH_PRECONDITION_FAILED",
            message="This IPO has no indexed prospectus yet.",
            details={"missing": ["indexed_document"]},
        )

    now = datetime.now(UTC)
    session = ChatSession(
        id=uuid4(),
        user_id=user_id,
        ipo_id=body.ipo_id,
        language=body.language,
        level=body.level,
        created_at=now,
        last_active=now,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_to_schema(session)


@router.get("/sessions/{session_id}", response_model=Session, dependencies=_AUTH)
async def get_session(
    session_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Session:
    session = await _get_own_session(session_id, user_id, db)
    return _session_to_schema(session)


@router.get("/sessions", response_model=SessionPage, dependencies=_AUTH)
async def list_sessions(
    user_id: Annotated[UUID, Depends(require_user_id)],
    ipo_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> SessionPage:
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.last_active.desc())
        .limit(limit + 1)
    )
    if ipo_id is not None:
        stmt = stmt.where(ChatSession.ipo_id == ipo_id)
    cursor_dt = _parse_cursor(cursor)
    if cursor_dt is not None:
        stmt = stmt.where(ChatSession.last_active < cursor_dt)

    rows = (await db.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].last_active.isoformat() if has_more and items else None
    return SessionPage(
        items=[_session_to_schema(s) for s in items],
        next_cursor=next_cursor,
    )


@router.patch("/sessions/{session_id}", response_model=Session, dependencies=_AUTH)
async def update_session(
    session_id: Annotated[UUID, Path()],
    body: SessionUpdate,
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Session:
    session = await _get_own_session(session_id, user_id, db)
    if body.language is not None:
        session.language = body.language
    if body.level is not None:
        session.level = body.level
    await db.commit()
    await db.refresh(session)
    return _session_to_schema(session)


@router.delete("/sessions/{session_id}", status_code=204, dependencies=_AUTH)
async def delete_session(
    session_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Response:
    session = await _get_own_session(session_id, user_id, db)
    await db.delete(session)
    await db.commit()
    return Response(status_code=204)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessagePage,
    dependencies=_AUTH,
)
async def list_messages(
    session_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> MessagePage:
    await _get_own_session(session_id, user_id, db)

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(limit + 1)
    )
    cursor_dt = _parse_cursor(cursor)
    if cursor_dt is not None:
        stmt = stmt.where(Message.created_at > cursor_dt)

    rows = (await db.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].created_at.isoformat() if has_more and items else None
    messages = [await _message_to_schema(m, db) for m in items]
    return MessagePage(items=messages, next_cursor=next_cursor)


@router.get("/messages/{message_id}", response_model=MessageOut, dependencies=_AUTH)
async def get_message(
    message_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> MessageOut:
    msg = await db.get(Message, message_id)
    if msg is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Message not found.")

    session = await db.get(ChatSession, msg.session_id)
    if session is None or session.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")

    return await _message_to_schema(msg, db)


@router.post(
    "/messages/{message_id}/feedback",
    status_code=204,
    dependencies=_AUTH,
)
async def submit_feedback(
    message_id: Annotated[UUID, Path()],
    body: FeedbackCreate,
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Response:
    msg = await db.get(Message, message_id)
    if msg is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Message not found.")
    if msg.role != "assistant":
        raise AppError(
            status_code=400,
            code="VALIDATION_ERROR",
            message="Feedback can only be submitted on assistant messages.",
        )

    session = await db.get(ChatSession, msg.session_id)
    if session is None or session.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")

    msg.feedback = body.rating
    msg.feedback_reason = body.reason
    msg.feedback_comment = body.comment
    await db.commit()
    return Response(status_code=204)
