"""Reciprocal-rank fusion for independently ranked retrieval candidates."""

from collections.abc import Sequence
from uuid import UUID

from ai.retrieval.types import RetrievalHit


def fuse_ranked_results(
    ranked_lists: Sequence[Sequence[RetrievalHit]],
    *,
    rank_constant: int = 60,
) -> list[RetrievalHit]:
    """Fuse candidate lists using 1 / (rank_constant + one-based rank)."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")

    scores: dict[UUID, float] = {}
    similarities: dict[UUID, float] = {}
    hits: dict[UUID, RetrievalHit] = {}

    for ranked_list in ranked_lists:
        seen: set[UUID] = set()
        for rank, hit in enumerate(ranked_list, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (
                rank_constant + rank
            )
            similarities[hit.chunk_id] = max(
                similarities.get(hit.chunk_id, 0.0),
                hit.similarity,
            )
            hits[hit.chunk_id] = hit

    fused = [
        hit.model_copy(
            update={
                "rrf_score": scores[chunk_id],
                "similarity": similarities[chunk_id],
                "is_neighbor": False,
            }
        )
        for chunk_id, hit in hits.items()
    ]
    return sorted(
        fused,
        key=lambda hit: (
            -hit.rrf_score,
            -hit.similarity,
            hit.chunk_index,
            str(hit.chunk_id),
        ),
    )
