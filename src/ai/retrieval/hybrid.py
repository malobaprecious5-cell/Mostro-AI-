"""Hybrid vector and full-text search with RRF and local neighbor expansion."""

from collections.abc import Sequence
from math import isfinite
from uuid import UUID

from sqlalchemy import and_, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.config import get_settings
from ai.db.models import VECTOR_DIMENSION, Chunk, Document
from ai.llm.interfaces import EmbeddingClient
from ai.retrieval.fusion import fuse_ranked_results
from ai.retrieval.gate import abstention_gate
from ai.retrieval.types import RetrievalHit, RetrievalResponse


async def hybrid_search(
    session: AsyncSession,
    ipo_id: UUID,
    query: str,
    embedding_client: EmbeddingClient,
    *,
    candidate_limit: int | None = None,
    rank_constant: int = 60,
) -> list[RetrievalHit]:
    """Search indexed chunks for one IPO using vector and full-text rankings."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if candidate_limit is None:
        candidate_limit = get_settings().retrieval_candidates
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1")

    query_vector = await embedding_client.embed_query(query)
    if len(query_vector) != VECTOR_DIMENSION:
        raise ValueError(
            f"Query embedding has {len(query_vector)} dimensions; "
            f"expected {VECTOR_DIMENSION}"
        )
    if any(not isfinite(value) for value in query_vector):
        raise ValueError("Query embedding must contain only finite values")

    distance = Chunk.embedding.cosine_distance(query_vector)
    vector_statement = (
        select(Chunk, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.ipo_id == ipo_id, Document.status == "indexed")
        .order_by(distance.asc(), Chunk.chunk_index.asc(), Chunk.id.asc())
        .limit(candidate_limit)
    )
    vector_rows = (await session.execute(vector_statement)).all()
    vector_hits = [
        _hit_from_chunk(
            chunk,
            similarity=max(0.0, min(1.0, 1.0 - float(row_distance))),
        )
        for chunk, row_distance in vector_rows
    ]

    search_query = func.websearch_to_tsquery(
        literal_column("'ai.fts_unaccent'::regconfig"),
        query,
    )
    rank = func.ts_rank_cd(Chunk.fts, search_query)
    text_statement = (
        select(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.ipo_id == ipo_id,
            Document.status == "indexed",
            Chunk.fts.op("@@")(search_query),
        )
        .order_by(rank.desc(), Chunk.chunk_index.asc(), Chunk.id.asc())
        .limit(candidate_limit)
    )
    text_chunks = (await session.scalars(text_statement)).all()
    text_hits = [_hit_from_chunk(chunk) for chunk in text_chunks]

    return fuse_ranked_results(
        [vector_hits, text_hits],
        rank_constant=rank_constant,
    )


async def expand_neighbors(
    session: AsyncSession,
    ipo_id: UUID,
    hits: Sequence[RetrievalHit],
    *,
    radius: int = 1,
) -> list[RetrievalHit]:
    """Add adjacent chunks from the same indexed documents without duplicates."""
    if radius < 0:
        raise ValueError("radius must not be negative")
    if not hits or radius == 0:
        return list(hits)

    selected_ids = {hit.chunk_id for hit in hits}
    existing_positions = {(hit.document_id, hit.chunk_index) for hit in hits}
    neighbor_parents: dict[tuple[UUID, int], RetrievalHit] = {}
    for hit in hits:
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            position = (hit.document_id, hit.chunk_index + offset)
            if position in existing_positions:
                continue
            parent = neighbor_parents.get(position)
            if parent is None or hit.rrf_score > parent.rrf_score:
                neighbor_parents[position] = hit
    if not neighbor_parents:
        return list(hits)

    conditions = [
        and_(Chunk.document_id == document_id, Chunk.chunk_index == chunk_index)
        for document_id, chunk_index in neighbor_parents
    ]
    statement = (
        select(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.ipo_id == ipo_id,
            Document.status == "indexed",
            or_(*conditions),
        )
        .order_by(Chunk.document_id, Chunk.chunk_index)
    )
    neighbors = (await session.scalars(statement)).all()

    expanded = list(hits)
    for chunk in neighbors:
        if chunk.id in selected_ids:
            continue
        parent = neighbor_parents[(chunk.document_id, chunk.chunk_index)]
        expanded.append(
            _hit_from_chunk(
                chunk,
                rrf_score=parent.rrf_score,
                is_neighbor=True,
            )
        )
        selected_ids.add(chunk.id)
    return expanded


async def retrieve_chunks(
    session: AsyncSession,
    ipo_id: UUID,
    query: str,
    embedding_client: EmbeddingClient,
    *,
    candidate_limit: int | None = None,
    top_k: int = 6,
    abstain_threshold: float | None = None,
    neighbor_radius: int = 1,
) -> RetrievalResponse:
    """Run hybrid search, abstain on weak evidence, then expand local context."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    candidates = await hybrid_search(
        session,
        ipo_id,
        query,
        embedding_client,
        candidate_limit=candidate_limit,
    )
    top_hits = candidates[:top_k]
    threshold = (
        get_settings().abstain_score_threshold
        if abstain_threshold is None
        else abstain_threshold
    )
    decision = abstention_gate(top_hits, threshold=threshold)
    if decision.abstain:
        return RetrievalResponse(chunks=(), decision=decision)

    expanded = await expand_neighbors(
        session,
        ipo_id,
        top_hits,
        radius=neighbor_radius,
    )
    return RetrievalResponse(chunks=tuple(expanded), decision=decision)


def _hit_from_chunk(
    chunk: Chunk,
    *,
    rrf_score: float = 0.0,
    similarity: float = 0.0,
    is_neighbor: bool = False,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk.id,
        ipo_id=chunk.ipo_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        content=chunk.content,
        rrf_score=rrf_score,
        similarity=similarity,
        is_neighbor=is_neighbor,
    )
