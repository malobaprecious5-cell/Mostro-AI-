"""Read-only IPO knowledge and glossary schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PublicKeyFact(BaseModel):
    fact_key: str
    value: dict[str, Any]
    source_page: int
    source_quote: str | None = None


class PublicKeyFactPage(BaseModel):
    items: list[PublicKeyFact]


class PublicSectionSummary(BaseModel):
    section_key: str
    summary: str
    page_start: int | None = None
    page_end: int | None = None


class PublicSectionSummaryPage(BaseModel):
    items: list[PublicSectionSummary]


class SuggestedQuestions(BaseModel):
    items: list[str] = Field(default_factory=list)
