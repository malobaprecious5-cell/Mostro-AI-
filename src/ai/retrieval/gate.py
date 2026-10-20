"""Similarity-based retrieval abstention decision."""

from collections.abc import Sequence
from math import isfinite

from ai.retrieval.types import AbstentionDecision, RetrievalHit


def abstention_gate(
    hits: Sequence[RetrievalHit],
    *,
    threshold: float,
) -> AbstentionDecision:
    """Reject empty or low-similarity results before neighbor expansion."""
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    if not hits:
        return AbstentionDecision(
            abstain=True,
            reason="no_results",
            best_similarity=0.0,
            threshold=threshold,
        )

    best_similarity = max(hit.similarity for hit in hits)
    enough_similarity = best_similarity >= threshold
    return AbstentionDecision(
        abstain=not enough_similarity,
        reason="sufficient_similarity" if enough_similarity else "low_similarity",
        best_similarity=best_similarity,
        threshold=threshold,
    )
