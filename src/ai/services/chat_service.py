"""Chat service: orchestrates the LangGraph workflow, handles streaming, and persists messages."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent.graph import build_chat_graph
from ai.agent.state import ChatState
from ai.config import get_settings
from ai.core.logging import get_logger
from ai.db.models import Message, MessageCitation
from ai.guardrails.injection import check_injection
from ai.llm.interfaces import ChatClient, EmbeddingClient

logger = get_logger(__name__)


class ChatService:
    """Orchestrates the chat graph and handles SSE streaming."""

    def __init__(
        self,
        db_session: AsyncSession,
        chat_client: ChatClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.db_session = db_session
        self.chat_client = chat_client
        self.embedding_client = embedding_client

    async def stream_message(
        self,
        session_id: UUID,
        ipo_id: UUID,
        user_id: UUID,
        user_query: str,
        client_message_id: UUID | None = None,
        language: str = "en",
        level: str = "beginner",
    ) -> AsyncIterator[str]:
        """Stream a chat response as SSE events.

        Events emitted (in order):
        - ``token`` — one per word of the assistant's answer (word-split; real
          token streaming is a Phase 10 concern).
        - ``citation`` — one per retrieved chunk when the outcome is *answered*.
        - ``done`` — final outcome and token-usage summary.
        - ``error`` — replaces all of the above when something goes wrong.
        """
        if client_message_id is None:
            client_message_id = uuid4()

        # ------------------------------------------------------------------ #
        # Input guard — injection check before anything touches the LLM.      #
        # ------------------------------------------------------------------ #
        injection_check = check_injection(user_query)
        if injection_check.is_injection:
            logger.warning(
                "injection_detected",
                reason=injection_check.reason,
                session_id=str(session_id),
            )
            yield self._encode_event(
                "error",
                {
                    "code": "blocked",
                    "message": "Your request was flagged as potentially malicious.",
                },
            )
            await self._save_message(
                session_id=session_id,
                client_message_id=None,
                role="assistant",
                content="Your request was blocked due to security checks.",
                outcome="blocked",
                trace={"injection_reason": injection_check.reason},
            )
            return

        # ------------------------------------------------------------------ #
        # Build initial state and run the compiled graph.                      #
        # ------------------------------------------------------------------ #
        settings = get_settings()
        _ = settings  # accessed inside nodes via get_settings()

        initial_state = ChatState(
            session_id=session_id,
            ipo_id=ipo_id,
            user_id=user_id,
            user_query=user_query,
            language=language,
            level=level,
            trace={
                "client_message_id": str(client_message_id),
                "system_prompt_version": "system_v1.0",
            },
        )

        logger.info(
            "chat_start",
            session_id=str(session_id),
            ipo_id=str(ipo_id),
            query_len=len(user_query),
        )

        try:
            graph = build_chat_graph(
                self.db_session,
                self.chat_client,
                self.embedding_client,
            )
            result_dict = await graph.ainvoke(initial_state.model_dump())
            state = ChatState(**result_dict)

            # ---------------------------------------------------------------- #
            # Stream the answer word-by-word (real streaming: Phase 10).        #
            # ---------------------------------------------------------------- #
            if state.answer:
                for word in state.answer.split():
                    yield self._encode_event("token", {"delta": word + " "})

            # Stream citations only when we actually answered.
            for i, citation in enumerate(state.citations, start=1):
                yield self._encode_event(
                    "citation",
                    {
                        "index": i,
                        "chunk_id": citation["chunk_id"],
                        "document_id": citation.get("document_id"),
                        "page_start": citation["page_start"],
                        "page_end": citation["page_end"],
                        "section_path": citation["section_path"],
                        "quote": citation["quote"],
                    },
                )

            # Final done event.
            message_id = uuid4()
            yield self._encode_event(
                "done",
                {
                    "message_id": str(message_id),
                    "outcome": state.outcome,
                    "usage": state.trace.get("usage", {}),
                },
            )

            # ---------------------------------------------------------------- #
            # Persist the message + citations to the audit log.                 #
            # ---------------------------------------------------------------- #
            saved_id = await self._save_message(
                session_id=session_id,
                client_message_id=None,
                role="assistant",
                content=state.answer,
                outcome=state.outcome,
                trace=state.trace,
            )

            if state.citations and saved_id:
                for citation in state.citations:
                    await self._save_citation(
                        message_id=saved_id,
                        chunk_id=UUID(citation["chunk_id"]),
                        page=citation["page_start"],
                        quote=citation["quote"],
                    )

            logger.info(
                "chat_complete",
                session_id=str(session_id),
                outcome=state.outcome,
            )

        except Exception as e:
            logger.error(
                "chat_error",
                error=str(e),
                session_id=str(session_id),
            )
            yield self._encode_event(
                "error",
                {
                    "code": "internal_error",
                    "message": "An error occurred while processing your request.",
                },
            )
            await self._save_message(
                session_id=session_id,
                client_message_id=None,
                role="assistant",
                content="",
                outcome="error",
                trace={"error": str(e)},
            )

    # ---------------------------------------------------------------------- #
    # Persistence helpers                                                       #
    # ---------------------------------------------------------------------- #

    async def _save_message(
        self,
        session_id: UUID,
        client_message_id: UUID | None,
        role: str,
        content: str,
        outcome: str,
        trace: dict,
    ) -> UUID | None:
        """Persist a message to the database and return its generated ID."""
        try:
            message_id = uuid4()
            stmt = insert(Message).values(
                id=message_id,
                session_id=session_id,
                client_message_id=client_message_id,
                role=role,
                content=content,
                outcome=outcome,
                trace=trace,
                created_at=datetime.now(UTC),
            )
            await self.db_session.execute(stmt)
            await self.db_session.commit()
            return message_id
        except Exception as e:
            logger.error(
                "message_save_error",
                error=str(e),
                session_id=str(session_id),
            )
            return None

    async def _save_citation(
        self,
        message_id: UUID,
        chunk_id: UUID,
        page: int,
        quote: str | None,
    ) -> None:
        """Persist a citation record for the given message."""
        try:
            stmt = insert(MessageCitation).values(
                id=uuid4(),
                message_id=message_id,
                chunk_id=chunk_id,
                page=page,
                quote=quote,
            )
            await self.db_session.execute(stmt)
            await self.db_session.commit()
        except Exception as e:
            logger.error(
                "citation_save_error",
                error=str(e),
                message_id=str(message_id),
            )

    @staticmethod
    def _encode_event(event_type: str, data: dict) -> str:
        """Format a server-sent event string."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
