"""Public source (citation viewer) schemas."""

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field


class SourceChunk(BaseModel):
    id: UUID
    document_id: UUID
    section_path: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    chunk_type: str
    content: str


class DocumentPageView(BaseModel):
    document_id: UUID
    page: int = Field(ge=1)
    text: str
    pdf_url: str
    expires_at: AwareDatetime
