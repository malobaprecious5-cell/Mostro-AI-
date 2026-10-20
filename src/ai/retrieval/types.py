"""Typed results shared by retrieval stages."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    ipo_id: UUID
    document_id: UUID
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str | None
    content: str
    rrf_score: float = Field(ge=0)
    similarity: float = Field(ge=0, le=1)
    is_neighbor: bool = False


class AbstentionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    abstain: bool
    reason: Literal["no_results", "low_similarity", "sufficient_similarity"]
    best_similarity: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: tuple[RetrievalHit, ...]
    decision: AbstentionDecision
