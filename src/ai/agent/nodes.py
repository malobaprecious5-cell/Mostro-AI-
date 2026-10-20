"""LangGraph nodes for the chat flow: understand → retrieve → generate → respond.

Each public function is a *factory* that closes over the I/O dependencies
(db session, LLM clients) and returns a plain ``async (state) -> dict``
callable.  LangGraph only sees the inner callable, which carries the correct
signature ``(ChatState) -> dict``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent.state import ChatState
from ai.config import get_settings
from ai.core.logging import get_logger
from ai.guardrails.citations import check_citations
from ai.llm.interfaces import ChatClient, EmbeddingClient
from ai.retrieval.hybrid import retrieve_chunks

logger = get_logger(__name__)

# Type alias for a LangGraph-compatible node callable.
NodeCallable = Callable[[ChatState], Coroutine[Any, Any, dict[str, Any]]]


def _read_prompt(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    path = Path(__file__).parent / "prompts" / filename
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# understand
# ---------------------------------------------------------------------------


async def understand_node(state: ChatState) -> dict[str, Any]:
    """Understand the query and record the fact in the trace.

    In a richer implementation this node would rewrite the query using
    conversation history and classify the intent.  For Phase 4 it simply
    passes through so the graph wiring and trace are correct.
    """
    logger.info(
        "understand",
        query=state.user_query[:50],
        session_id=str(state.session_id),
        ipo_id=str(state.ipo_id),
    )
    return {
        "trace": {
            **state.trace,
            "understand_complete": True,
        }
    }


# ---------------------------------------------------------------------------
# retrieve  (factory)
# ---------------------------------------------------------------------------


def make_retrieve_node(
    db_session: AsyncSession,
    embedding_client: EmbeddingClient,
) -> NodeCallable:
    """Return a retrieve node with I/O dependencies closed over."""

    async def _retrieve(state: ChatState) -> dict[str, Any]:
        settings = get_settings()
        logger.info(
            "retrieve",
            query=state.user_query[:50],
            ipo_id=str(state.ipo_id),
        )

        start_time = time.perf_counter()
        try:
            response = await retrieve_chunks(
                db_session,
                state.ipo_id,
                state.user_query,
                embedding_client,
                candidate_limit=settings.retrieval_candidates,
                top_k=settings.rerank_top_k,
            )
            latency = time.perf_counter() - start_time

            decision = response.decision
            hits = list(response.chunks)

            if decision.abstain:
                logger.info(
                    "abstain",
                    reason=decision.reason,
                    best_similarity=decision.best_similarity,
                    ipo_id=str(state.ipo_id),
                )
                return {
                    "retrieved_chunks": hits,
                    "abstain_decision_reason": decision.reason,
                    "outcome": "abstained",
                    "trace": {
                        **state.trace,
                        "retrieve_complete": True,
                        "hits": len(hits),
                        "best_similarity": decision.best_similarity,
                        "threshold": decision.threshold,
                        "latency_ms": round(latency * 1000, 2),
                    },
                }

            logger.info(
                "retrieval_complete",
                hits_count=len(hits),
                latency_ms=round(latency * 1000, 2),
                ipo_id=str(state.ipo_id),
            )
            return {
                "retrieved_chunks": hits,
                "trace": {
                    **state.trace,
                    "retrieve_complete": True,
                    "hits": len(hits),
                    "best_similarity": decision.best_similarity,
                    "latency_ms": round(latency * 1000, 2),
                },
            }

        except Exception as e:  # noqa: BLE001
            logger.error(
                "retrieval_error",
                error=str(e),
                ipo_id=str(state.ipo_id),
            )
            return {
                "retrieved_chunks": [],
                "outcome": "error",
                "error": str(e),
                "trace": {
                    **state.trace,
                    "retrieve_error": str(e),
                },
            }

    return _retrieve


# ---------------------------------------------------------------------------
# generate  (factory)
# ---------------------------------------------------------------------------


def make_generate_node(chat_client: ChatClient) -> NodeCallable:
    """Return a generate node with the chat client closed over."""

    system_prompt = _read_prompt("system_v1.0.txt")
    answer_prompt_template = _read_prompt("answer_v1.0.txt")

    async def _generate(state: ChatState) -> dict[str, Any]:
        # Skip generation when the graph already reached a terminal outcome.
        if state.outcome in ("abstained", "error", "refused", "blocked"):
            logger.info(
                "generate_skip",
                reason=state.outcome,
                ipo_id=str(state.ipo_id),
            )
            return {"answer": state.answer or "", "outcome": state.outcome}

        if not state.retrieved_chunks:
            logger.info("generate_skip", reason="no_chunks", ipo_id=str(state.ipo_id))
            return {
                "answer": (
                    "The prospectus does not address your question. "
                    "Please contact the issuer or regulator for more information."
                ),
                "outcome": "abstained",
            }

        logger.info(
            "generate",
            chunks_count=len(state.retrieved_chunks),
            ipo_id=str(state.ipo_id),
        )

        # Format the top-10 chunks into the answer prompt.
        passages_text = "\n\n".join(
            f"[Page {c.page_start}] {c.content}" for c in state.retrieved_chunks[:10]
        )
        human_text = answer_prompt_template.format(
            question=state.user_query,
            passages=passages_text,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_text),
        ]

        start_time = time.perf_counter()
        try:
            response: AIMessage = await chat_client.ainvoke(messages)
            latency = time.perf_counter() - start_time

            # response.content is str | list[str | dict].  Flatten to str.
            raw = response.content
            if isinstance(raw, str):
                answer = raw
            else:
                answer = " ".join(
                    block if isinstance(block, str) else str(block.get("text", ""))
                    for block in raw
                )

            logger.info(
                "generation_complete",
                latency_ms=round(latency * 1000, 2),
                answer_len=len(answer),
                ipo_id=str(state.ipo_id),
            )

            # If the LLM itself signals it couldn't find an answer, record it.
            lower = answer.lower()
            if "does not address" in lower or "i couldn't find" in lower:
                return {
                    "answer": answer,
                    "outcome": "abstained",
                    "trace": {
                        **state.trace,
                        "generate_complete": True,
                        "prompt_version": "answer_v1.0",
                        "latency_ms": round(latency * 1000, 2),
                    },
                }

            # Soft citation check — log warnings; do not block in Phase 4.
            source_text = "\n".join(c.content for c in state.retrieved_chunks)
            citation_check = check_citations(answer, source_text)
            if not citation_check.is_valid:
                logger.warning(
                    "citation_check_failed",
                    missing_numbers=citation_check.missing_numbers,
                    missing_dates=citation_check.missing_dates,
                    ipo_id=str(state.ipo_id),
                )

            usage = response.usage_metadata or {}
            return {
                "answer": answer,
                "outcome": "answered",
                "trace": {
                    **state.trace,
                    "generate_complete": True,
                    "prompt_version": "answer_v1.0",
                    "latency_ms": round(latency * 1000, 2),
                    "usage": {
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                },
            }

        except Exception as e:  # noqa: BLE001
            logger.error(
                "generation_error",
                error=str(e),
                ipo_id=str(state.ipo_id),
            )
            return {
                "answer": "",
                "outcome": "error",
                "error": str(e),
                "trace": {
                    **state.trace,
                    "generate_error": str(e),
                },
            }

    return _generate


# ---------------------------------------------------------------------------
# respond
# ---------------------------------------------------------------------------


async def respond_node(state: ChatState) -> dict[str, Any]:
    """Attach structured citations from retrieved chunks to the final state."""
    logger.info(
        "respond",
        outcome=state.outcome,
        chunks_count=len(state.retrieved_chunks),
        ipo_id=str(state.ipo_id),
    )

    # Only emit citations when we actually answered.
    citations: list[dict[str, Any]] = []
    if state.outcome == "answered":
        for chunk in state.retrieved_chunks:
            citations.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": chunk.section_path,
                    "quote": chunk.content[:200],
                }
            )

    return {
        "citations": citations,
        "trace": {
            **state.trace,
            "respond_complete": True,
        },
    }
