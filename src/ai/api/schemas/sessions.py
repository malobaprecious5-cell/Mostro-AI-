"""Session and message schemas for the v1 API."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field

Language = Literal["en", "fr"]
Level = Literal["beginner", "intermediate", "advanced"]
MessageOutcome = Literal["answered", "abstained", "refused", "blocked", "error"]
MessageRole = Literal["user", "assistant", "system"]
FeedbackRating = Literal[-1, 0, 1]
FeedbackReason = Literal["inaccurate", "unclear", "unsafe", "other"]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    ipo_id: UUID
    language: Language = "en"
    level: Level = "beginner"


class SessionUpdate(BaseModel):
    language: Language | None = None
    level: Level | None = None


class Session(BaseModel):
    id: UUID
    user_id: UUID
    ipo_id: UUID
    language: Language
    level: Level
    created_at: AwareDatetime
    last_active: AwareDatetime


class SessionPage(BaseModel):
    items: list[Session]
    next_cursor: str | None = None


# ---------------------------------------------------------------------------
# Citation / proposal  (reused in Message)
# ---------------------------------------------------------------------------


class MessageCitationOut(BaseModel):
    index: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str | None = None
    quote: str | None = None


class Reminder(BaseModel):
    minutes_before: int = Field(ge=0)


class CalendarProposalOut(BaseModel):
    id: UUID
    key_fact_id: UUID
    title: str
    starts_at: AwareDatetime
    ends_at: AwareDatetime | None = None
    timezone: str
    reminders: list[Reminder] = Field(default_factory=list)
    status: str
    source_page: int | None = None


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class MessageOut(BaseModel):
    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    outcome: MessageOutcome | None = None
    citations: list[MessageCitationOut] = Field(default_factory=list)
    proposals: list[CalendarProposalOut] = Field(default_factory=list)
    created_at: AwareDatetime


class MessagePage(BaseModel):
    items: list[MessageOut]
    next_cursor: str | None = None


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackCreate(BaseModel):
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: Annotated[str | None, Field(max_length=500)] = None
