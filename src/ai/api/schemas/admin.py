"""Admin schemas: IPO upsert/status, documents, ingestion jobs, chunks, retrieval debug."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# IPO
# ---------------------------------------------------------------------------


class IpoUpsert(BaseModel):
    issuer_name: str
    country: str = Field(min_length=2, max_length=2)
    exchange: str | None = None


class IpoStatusUpdate(BaseModel):
    status: Literal["draft", "review", "published", "archived"]


class IpoOut(BaseModel):
    id: UUID
    issuer_name: str
    country: str
    exchange: str | None = None
    status: str
    published_at: datetime | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    id: UUID
    ipo_id: UUID
    version: int
    doc_type: str
    storage_uri: str
    file_hash: str
    page_count: int | None = None
    parser: str | None = None
    status: str
    chunk_count: int | None = None
    created_at: datetime


class DocumentAccepted(BaseModel):
    document_id: UUID
    job_id: UUID


class ReingestRequest(BaseModel):
    steps: list[Literal["parse", "chunk", "embed", "facts", "summaries"]] | None = None


class JobAccepted(BaseModel):
    job_id: UUID


class DocumentPage(BaseModel):
    items: list[DocumentOut]


# ---------------------------------------------------------------------------
# Ingestion job
# ---------------------------------------------------------------------------


class JobOut(BaseModel):
    id: UUID
    document_id: UUID
    ipo_id: UUID
    kind: str
    status: str
    step: str | None = None
    progress: int
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# Chunk inspect
# ---------------------------------------------------------------------------


class ChunkOut(BaseModel):
    id: UUID
    document_id: UUID
    ipo_id: UUID
    chunk_index: int
    chunk_type: str
    section_path: str | None = None
    page_start: int
    page_end: int
    content: str
    token_count: int | None = None


class ChunkPage(BaseModel):
    items: list[ChunkOut]
    next_cursor: str | None = None


# ---------------------------------------------------------------------------
# Retrieval debug
# ---------------------------------------------------------------------------


class RetrievalDebugRequest(BaseModel):
    ipo_id: UUID
    query: str = Field(min_length=1, max_length=1000)
    candidates: int = Field(default=30, ge=1, le=200)
    top_k: int = Field(default=6, ge=1, le=50)


class RetrievalDebugHit(BaseModel):
    chunk_id: UUID
    page_start: int
    page_end: int
    section_path: str | None = None
    rrf_score: float
    similarity: float
    is_neighbor: bool
    snippet: str


class RetrievalDebugResponse(BaseModel):
    items: list[RetrievalDebugHit]
    would_abstain: bool
    best_similarity: float
    threshold: float
    query: str


# ---------------------------------------------------------------------------
# Key facts / summaries / glossary
# ---------------------------------------------------------------------------


class KeyFactOut(BaseModel):
    id: UUID
    ipo_id: UUID
    document_id: UUID
    fact_key: str
    value: dict[str, Any]
    source_page: int
    source_quote: str | None = None
    review_status: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class KeyFactPage(BaseModel):
    items: list[KeyFactOut]


class KeyFactReview(BaseModel):
    review_status: Literal["verified", "rejected"]
    value: dict[str, Any] | None = None
    source_page: int | None = Field(default=None, ge=1)
    reviewed_by: str


class KeyFactExtractRequest(BaseModel):
    fact_keys: list[str] | None = None


class SectionSummaryOut(BaseModel):
    id: UUID
    ipo_id: UUID
    document_id: UUID
    section_key: str
    summary: str
    page_start: int | None = None
    page_end: int | None = None
    review_status: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class SectionSummaryPage(BaseModel):
    items: list[SectionSummaryOut]


class SectionSummaryReview(BaseModel):
    review_status: Literal["verified", "rejected"]
    summary: str | None = None
    reviewed_by: str


class SectionSummaryGenerateRequest(BaseModel):
    section_keys: list[str] | None = None


class GlossaryUpsert(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    language: Literal["en", "fr"]
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    definition: str = Field(min_length=1, max_length=4000)
    example: str | None = Field(default=None, max_length=2000)


class GlossaryTermOut(BaseModel):
    id: UUID
    term: str
    definition: str
    example: str | None = None
    language: str
    level: str


class GlossaryPage(BaseModel):
    items: list[GlossaryTermOut]
    next_cursor: str | None = None
