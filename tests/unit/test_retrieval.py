"""Unit tests for retrieval fusion, abstention, and deterministic evaluation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from ai.retrieval import hybrid as hybrid_module
from ai.retrieval.evaluation import EvalQuestion, load_dataset, recall_at_k
from ai.retrieval.fusion import fuse_ranked_results
from ai.retrieval.gate import abstention_gate
from ai.retrieval.hybrid import expand_neighbors, hybrid_search, retrieve_chunks
from ai.retrieval.types import RetrievalHit

IPO_ID = UUID("00000000-0000-0000-0000-000000000001")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _hit(
    index: int,
    *,
    page: int | None = None,
    rrf_score: float = 0,
    similarity: float = 0,
) -> RetrievalHit:
    chunk_id = UUID(int=index + 1)
    return RetrievalHit(
        chunk_id=chunk_id,
        ipo_id=IPO_ID,
        document_id=DOCUMENT_ID,
        chunk_index=index,
        page_start=page or index + 1,
        page_end=page or index + 1,
        section_path=None,
        content=f"chunk {index}",
        rrf_score=rrf_score,
        similarity=similarity,
    )


def _chunk(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=UUID(int=index + 1),
        ipo_id=IPO_ID,
        document_id=DOCUMENT_ID,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_path="Offer",
        content=f"chunk {index}",
    )


class _EmbeddingClient:
    async def embed_query(self, text: str) -> list[float]:
        assert text == "offer"
        return [0.0] * 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        return []


def test_rrf_combines_rankings_and_deduplicates_chunks() -> None:
    vector = [_hit(0, similarity=0.8), _hit(1, similarity=0.7)]
    lexical = [_hit(1), _hit(2)]

    fused = fuse_ranked_results([vector, lexical], rank_constant=60)

    assert [hit.chunk_index for hit in fused] == [1, 0, 2]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].similarity == pytest.approx(0.7)
    assert len({hit.chunk_id for hit in fused}) == len(fused)


def test_rrf_rejects_invalid_rank_constant() -> None:
    with pytest.raises(ValueError, match="rank_constant"):
        fuse_ranked_results([[_hit(0)]], rank_constant=0)


@pytest.mark.parametrize(
    ("hits", "threshold", "abstain", "reason"),
    [
        ([], 0.3, True, "no_results"),
        ([_hit(0, similarity=0.29)], 0.3, True, "low_similarity"),
        ([_hit(0, similarity=0.3)], 0.3, False, "sufficient_similarity"),
    ],
)
def test_abstention_gate_checks_vector_similarity(
    hits: list[RetrievalHit],
    threshold: float,
    abstain: bool,
    reason: str,
) -> None:
    decision = abstention_gate(hits, threshold=threshold)

    assert decision.abstain is abstain
    assert decision.reason == reason


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_abstention_gate_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        abstention_gate([_hit(0)], threshold=threshold)


@pytest.mark.asyncio
async def test_hybrid_search_fuses_scoped_vector_and_full_text_results() -> None:
    chunk = _chunk(0)
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [(chunk, 0.2)])),
        scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [chunk])),
    )

    hits = await hybrid_search(
        session,
        IPO_ID,
        "offer",
        _EmbeddingClient(),
        candidate_limit=5,
    )

    assert len(hits) == 1
    assert hits[0].chunk_id == chunk.id
    assert hits[0].similarity == pytest.approx(0.8)
    assert hits[0].rrf_score == pytest.approx(2 / 61)
    vector_statement = session.execute.await_args.args[0]
    text_statement = session.scalars.await_args.args[0]
    vector_sql = str(vector_statement)
    text_sql = str(text_statement)
    for statement in (vector_sql, text_sql):
        assert "chunk.ipo_id" in statement
        assert "document.status" in statement
    assert "indexed" in vector_statement.compile().params.values()
    assert "indexed" in text_statement.compile().params.values()
    assert "@@" in text_sql
    assert "websearch_to_tsquery" in text_sql


@pytest.mark.asyncio
async def test_neighbor_expansion_stays_within_indexed_document() -> None:
    parent = _hit(1, rrf_score=0.04, similarity=0.8)
    session = SimpleNamespace(
        scalars=AsyncMock(
            return_value=SimpleNamespace(all=lambda: [_chunk(0), _chunk(2)])
        )
    )

    expanded = await expand_neighbors(session, IPO_ID, [parent])

    assert [hit.chunk_index for hit in expanded] == [1, 0, 2]
    assert [hit.is_neighbor for hit in expanded] == [False, True, True]
    assert all(hit.document_id == DOCUMENT_ID for hit in expanded)
    assert all(hit.rrf_score == parent.rrf_score for hit in expanded)
    statement = str(session.scalars.await_args.args[0])
    assert "chunk.ipo_id" in statement
    assert "document.status" in statement
    compiled = session.scalars.await_args.args[0].compile()
    assert "indexed" in compiled.params.values()


@pytest.mark.asyncio
async def test_retrieve_chunks_skips_neighbor_expansion_when_abstaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = AsyncMock(return_value=[_hit(0, similarity=0.1)])
    monkeypatch.setattr(hybrid_module, "hybrid_search", search)
    session = SimpleNamespace(scalars=AsyncMock())

    response = await retrieve_chunks(
        session,
        IPO_ID,
        "offer",
        _EmbeddingClient(),
        abstain_threshold=0.3,
    )

    assert response.decision.abstain
    assert response.decision.reason == "low_similarity"
    assert response.chunks == ()
    session.scalars.assert_not_awaited()


def test_recall_at_k_matches_expected_source_page() -> None:
    questions = [
        EvalQuestion(
            id="q1",
            question="offer price",
            source_pages=[34],
        ),
        EvalQuestion(
            id="q2",
            question="minimum subscription",
            source_pages=[37],
        ),
    ]

    recall = recall_at_k(
        questions,
        [[_hit(0, page=34), _hit(1, page=37)], [_hit(2, page=37)]],
        k=1,
    )

    assert recall == 1.0


def test_recall_at_k_counts_pages_inside_multi_page_chunk() -> None:
    questions = [
        EvalQuestion(id="q1", question="page range", source_pages=[35]),
    ]

    recall = recall_at_k(
        questions,
        [[_hit(0, page=34).model_copy(update={"page_end": 36})]],
        k=1,
    )

    assert recall == 1.0


def test_recall_at_k_rejects_mismatched_results_and_invalid_k() -> None:
    question = EvalQuestion(id="q1", question="offer", source_pages=[34])

    with pytest.raises(ValueError, match="one result list"):
        recall_at_k([question], [], k=1)
    with pytest.raises(ValueError, match="at least 1"):
        recall_at_k([question], [[]], k=0)


def test_dangote_eval_dataset_has_25_unique_questions() -> None:
    from pathlib import Path

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "evals"
        / "datasets"
        / "dangote_prospectus.jsonl"
    )
    questions = load_dataset(dataset_path)

    assert len(questions) == 25
    assert len({question.id for question in questions}) == 25
