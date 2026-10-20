"""Prospectus retrieval primitives."""

from ai.retrieval.fusion import fuse_ranked_results
from ai.retrieval.gate import abstention_gate
from ai.retrieval.hybrid import hybrid_search, retrieve_chunks
from ai.retrieval.types import (
    AbstentionDecision,
    RetrievalHit,
    RetrievalResponse,
)

__all__ = [
    "AbstentionDecision",
    "RetrievalHit",
    "RetrievalResponse",
    "abstention_gate",
    "fuse_ranked_results",
    "hybrid_search",
    "retrieve_chunks",
]
