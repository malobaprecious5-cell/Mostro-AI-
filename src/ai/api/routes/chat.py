"""Chat endpoint: send messages and get prospectus-grounded streamed responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import TypeAdapter
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import (
    get_async_session,
    get_chat_client_dep,
    get_embedding_client_dep,
    require_internal_key,
    require_user_id,
)
from ai.api.schemas.chat import (
    CalendarProposal as ChatCalendarProposal,
)
from ai.api.schemas.chat import ChatMessage, ChatRequest, ChatStreamEvent, Citation, Reminder
from ai.core.errors import AppError
from ai.db.models import CalendarProposal, ChatSession, Chunk, Message, MessageCitation
from ai.llm.interfaces import ChatClient, EmbeddingClient
from ai.services.chat_service import ChatService

router = APIRouter(prefix="/v1", tags=["chat"])

_AUTH = [Depends(require_internal_key)]
_EVENT_ADAPTER: TypeAdapter[ChatStreamEvent] = TypeAdapter(ChatStreamEvent)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatMessage,
    dependencies=_AUTH,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "SSE stream: token, citation, proposal, done, or error events."
                        ),
                    }
                }
            }
        }
    },
)
async def send_message(
    request: ChatRequest,
    session_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db_session: AsyncSession = Depends(get_async_session),  # noqa: B008
    chat_client: ChatClient = Depends(get_chat_client_dep),  # noqa: B008
    embedding_client: EmbeddingClient = Depends(get_embedding_client_dep),  # noqa: B008
) -> Response:
    """Send a user message and get the assistant's grounded, streamed answer.

    The ``ipo_id`` is read from the session record — it cannot be supplied
    by the caller or inferred from model output.
    """
    session = await db_session.get(ChatSession, session_id)
    if session is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Session not found.")
    if session.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")

    existing = await _load_idempotent_assistant(
        db_session, session_id, request.client_message_id
    )
    if existing is not None:
        if request.stream:
            return StreamingResponse(
                _replay_sse(existing),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(content=existing.model_dump(mode="json"))

    now = datetime.now(UTC)
    try:
        await db_session.execute(
            insert(Message).values(
                id=uuid4(),
                session_id=session_id,
                client_message_id=request.client_message_id,
                role="user",
                content=request.content,
                outcome=None,
                trace=None,
                created_at=now,
            )
        )
        session.last_active = now
        await db_session.commit()
    except IntegrityError:
        await db_session.rollback()
        replay = await _load_idempotent_assistant(
            db_session, session_id, request.client_message_id
        )
        if replay is not None:
            if request.stream:
                return StreamingResponse(
                    _replay_sse(replay),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return JSONResponse(content=replay.model_dump(mode="json"))

    service = ChatService(db_session, chat_client, embedding_client)
    stream = service.stream_message(
        session_id=session_id,
        ipo_id=session.ipo_id,
        user_id=user_id,
        user_query=request.content,
        client_message_id=request.client_message_id,
        language=session.language,
        level=session.level,
    )

    if not request.stream:
        message = await _collect_chat_message(stream, session_id, now)
        return JSONResponse(content=message.model_dump(mode="json"))

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _load_idempotent_assistant(
    db_session: AsyncSession,
    session_id: UUID,
    client_message_id: UUID,
) -> ChatMessage | None:
    user_msg = (
        await db_session.execute(
            select(Message).where(
                Message.session_id == session_id,
                Message.client_message_id == client_message_id,
                Message.role == "user",
            )
        )
    ).scalar_one_or_none()
    if user_msg is None:
        return None

    assistant = (
        await db_session.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.role == "assistant",
                Message.created_at >= user_msg.created_at,
            )
            .order_by(Message.created_at.asc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if assistant is None:
        return None
    return await _assistant_to_chat_message(assistant, db_session)


async def _assistant_to_chat_message(
    msg: Message, db: AsyncSession
) -> ChatMessage:
    citation_rows = (
        await db.execute(
            select(MessageCitation, Chunk)
            .join(Chunk, Chunk.id == MessageCitation.chunk_id)
            .where(MessageCitation.message_id == msg.id)
            .order_by(MessageCitation.id)
        )
    ).all()
    citations = [
        Citation(
            index=i,
            chunk_id=citation.chunk_id,
            document_id=chunk.document_id,
            page_start=citation.page,
            page_end=citation.page,
            section_path=chunk.section_path,
            quote=citation.quote or "",
        )
        for i, (citation, chunk) in enumerate(citation_rows, start=1)
    ]
    proposal_rows = (
        await db.execute(
            select(CalendarProposal).where(CalendarProposal.message_id == msg.id)
        )
    ).scalars().all()
    return ChatMessage(
        id=msg.id,
        session_id=msg.session_id,
        role="assistant",
        content=msg.content,
        outcome=msg.outcome or "error",  # type: ignore[arg-type]
        citations=citations,
        proposals=[
            ChatCalendarProposal(
                id=p.id,
                key_fact_id=p.key_fact_id,
                title=p.title,
                starts_at=p.starts_at,
                ends_at=p.ends_at,
                timezone=p.timezone,
                reminders=[Reminder.model_validate(item) for item in p.reminders],
            )
            for p in proposal_rows
        ],
        created_at=msg.created_at,
    )


async def _collect_chat_message(
    stream: object,
    session_id: UUID,
    created_at: datetime,
) -> ChatMessage:
    tokens: list[str] = []
    citations: list[Citation] = []
    message_id = uuid4()
    outcome: str = "error"

    async for event_str in stream:  # type: ignore[union-attr]
        if not isinstance(event_str, str):
            continue
        try:
            lines = dict(
                line.split(": ", maxsplit=1)
                for line in event_str.strip().split("\n")
                if line.strip()
            )
            if "event" not in lines or "data" not in lines:
                continue
            event = _EVENT_ADAPTER.validate_python(
                {"event": lines["event"], "data": json.loads(lines["data"])}
            )
            if event.event == "token":
                tokens.append(event.data.delta)
            elif event.event == "citation":
                citations.append(
                    Citation(
                        index=event.data.index,
                        chunk_id=event.data.chunk_id,
                        document_id=event.data.document_id or event.data.chunk_id,
                        page_start=event.data.page_start,
                        page_end=event.data.page_end,
                        section_path=event.data.section_path,
                        quote=event.data.quote,
                    )
                )
            elif event.event == "done":
                message_id = event.data.message_id
                outcome = event.data.outcome
        except (ValueError, KeyError, json.JSONDecodeError, AttributeError):
            continue

    return ChatMessage(
        id=message_id,
        session_id=session_id,
        content="".join(tokens),
        outcome=outcome,  # type: ignore[arg-type]
        citations=citations,
        proposals=[],
        created_at=created_at,
    )


async def _replay_sse(message: ChatMessage) -> AsyncIterator[str]:
    if message.content:
        for word in message.content.split():
            yield f'event: token\ndata: {json.dumps({"delta": word + " "})}\n\n'
    for citation in message.citations:
        yield (
            "event: citation\ndata: "
            f"{json.dumps(citation.model_dump(mode='json'))}\n\n"
        )
    for proposal in message.proposals:
        payload = {
            "proposal_id": str(proposal.id),
            "title": proposal.title,
            "starts_at": proposal.starts_at.isoformat(),
            "ends_at": proposal.ends_at.isoformat() if proposal.ends_at else None,
            "timezone": proposal.timezone,
            "reminders": [item.model_dump() for item in proposal.reminders],
            "key_fact_id": str(proposal.key_fact_id),
        }
        yield f"event: proposal\ndata: {json.dumps(payload)}\n\n"
    yield (
        "event: done\ndata: "
        f"{json.dumps({'message_id': str(message.id), 'outcome': message.outcome, 'usage': {}})}\n\n"
    )
