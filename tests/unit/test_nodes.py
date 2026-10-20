"""Unit tests for the four LangGraph nodes.

All tests are fully offline — no database, no LLM API calls.
Dependencies are provided via fakes or lightweight ``unittest.mock`` stubs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from ai.agent import nodes as nodes_module
from ai.agent.nodes import (
    make_generate_node,
    make_retrieve_node,
    respond_node,
    understand_node,
)
from ai.agent.state import ChatState
from ai.llm.fakes import FakeChatClient, FakeEmbeddingClient
from ai.retrieval.types import AbstentionDecision, RetrievalHit, RetrievalResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IPO_ID = UUID("00000000-0000-0000-0000-000000000001")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _state(**overrides) -> ChatState:
    defaults = dict(
        session_id=uuid4(),
        ipo_id=IPO_ID,
        user_id=uuid4(),
        user_query="What is the offer price?",
    )
    defaults.update(overrides)
    return ChatState(**defaults)


def _hit(index: int = 0, *, similarity: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=UUID(int=index + 1),
        ipo_id=IPO_ID,
        document_id=DOCUMENT_ID,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_path="Offer Details",
        content=f"The offer price is {index + 1}00 naira per share.",
        rrf_score=0.5,
        similarity=similarity,
    )


def _sufficient_response(hits: list[RetrievalHit] | None = None) -> RetrievalResponse:
    h = hits if hits is not None else [_hit()]
    return RetrievalResponse(
        chunks=tuple(h),
        decision=AbstentionDecision(
            abstain=False,
            reason="sufficient_similarity",
            best_similarity=h[0].similarity if h else 0.9,
            threshold=0.3,
        ),
    )


def _abstain_response(reason: str = "low_similarity") -> RetrievalResponse:
    return RetrievalResponse(
        chunks=(),
        decision=AbstentionDecision(
            abstain=True,
            reason=reason,  # type: ignore[arg-type]
            best_similarity=0.1 if reason == "low_similarity" else 0.0,
            threshold=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# understand_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_understand_node_records_trace() -> None:
    state = _state()
    result = await understand_node(state)

    assert result["trace"]["understand_complete"] is True
    # Keys from the original trace are preserved.
    for key in state.trace:
        assert key in result["trace"]


# ---------------------------------------------------------------------------
# make_retrieve_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_returns_hits_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit = _hit(similarity=0.85)
    monkeypatch.setattr(
        nodes_module,
        "retrieve_chunks",
        AsyncMock(return_value=_sufficient_response([hit])),
    )
    db = SimpleNamespace()
    embedding = FakeEmbeddingClient()
    node = make_retrieve_node(db, embedding)  # type: ignore[arg-type]

    state = _state()
    result = await node(state)

    assert result["retrieved_chunks"] == [hit]
    # Outcome must not be set to abstained/error when retrieval succeeds.
    assert result.get("outcome") is None or result.get("outcome") == "pending"
    assert result["trace"]["retrieve_complete"] is True
    assert result["trace"]["hits"] == 1


@pytest.mark.asyncio
async def test_retrieve_node_abstains_on_low_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nodes_module,
        "retrieve_chunks",
        AsyncMock(return_value=_abstain_response("low_similarity")),
    )
    node = make_retrieve_node(SimpleNamespace(), FakeEmbeddingClient())  # type: ignore[arg-type]

    result = await node(_state())

    assert result["outcome"] == "abstained"
    assert result["abstain_decision_reason"] == "low_similarity"
    assert result["trace"]["retrieve_complete"] is True


@pytest.mark.asyncio
async def test_retrieve_node_abstains_on_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nodes_module,
        "retrieve_chunks",
        AsyncMock(return_value=_abstain_response("no_results")),
    )
    node = make_retrieve_node(SimpleNamespace(), FakeEmbeddingClient())  # type: ignore[arg-type]

    result = await node(_state())

    assert result["outcome"] == "abstained"
    assert result["abstain_decision_reason"] == "no_results"


@pytest.mark.asyncio
async def test_retrieve_node_handles_exception_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nodes_module,
        "retrieve_chunks",
        AsyncMock(side_effect=RuntimeError("db down")),
    )
    node = make_retrieve_node(SimpleNamespace(), FakeEmbeddingClient())  # type: ignore[arg-type]

    result = await node(_state())

    assert result["outcome"] == "error"
    assert "db down" in result["error"]


# ---------------------------------------------------------------------------
# make_generate_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_node_produces_answer() -> None:
    answer_text = "The offer price is 100 naira per share. (Page: 5)"
    chat = FakeChatClient(response=answer_text)
    node = make_generate_node(chat)

    state = _state(retrieved_chunks=[_hit()])
    result = await node(state)

    assert result["outcome"] == "answered"
    assert result["answer"] == answer_text
    assert result["trace"]["generate_complete"] is True
    assert result["trace"]["prompt_version"] == "answer_v1.0"


@pytest.mark.asyncio
async def test_generate_node_skips_when_already_abstained() -> None:
    chat = FakeChatClient(response="should not be called")
    node = make_generate_node(chat)

    # State with outcome already set to abstained.
    state = _state(outcome="abstained", answer="Abstain message from retrieve")
    result = await node(state)

    assert result["outcome"] == "abstained"
    # The original abstain message is preserved unchanged.
    assert result["answer"] == "Abstain message from retrieve"


@pytest.mark.asyncio
async def test_generate_node_abstains_when_no_chunks() -> None:
    chat = FakeChatClient(response="irrelevant")
    node = make_generate_node(chat)

    # Outcome is not yet set but there are no chunks.
    state = _state(retrieved_chunks=[])
    result = await node(state)

    assert result["outcome"] == "abstained"
    assert "does not address" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generate_node_records_abstain_when_llm_says_unknown() -> None:
    """LLM itself signals it couldn't find the answer."""
    chat = FakeChatClient(
        response="The prospectus does not address the question about CEO compensation."
    )
    node = make_generate_node(chat)

    state = _state(retrieved_chunks=[_hit()])
    result = await node(state)

    assert result["outcome"] == "abstained"


# ---------------------------------------------------------------------------
# respond_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_respond_node_builds_citations_when_answered() -> None:
    state = _state(
        outcome="answered",
        answer="The offer price is 100 naira.",
        retrieved_chunks=[_hit(0), _hit(1)],
    )
    result = await respond_node(state)

    assert len(result["citations"]) == 2
    assert result["citations"][0]["page_start"] == 1
    assert result["citations"][1]["page_start"] == 2
    assert result["trace"]["respond_complete"] is True


@pytest.mark.asyncio
async def test_respond_node_has_no_citations_when_abstained() -> None:
    state = _state(
        outcome="abstained",
        answer="I couldn't find this in the prospectus.",
        retrieved_chunks=[_hit()],  # chunks are in state but outcome is abstained
    )
    result = await respond_node(state)

    # respond_node only emits citations for "answered" outcomes.
    assert result["citations"] == []
    assert result["trace"]["respond_complete"] is True
