"""Frozen v1 chat request, response, and server-sent event schemas."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, field_validator

MessageOutcome = Literal["answered", "abstained", "refused", "blocked", "error"]


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    client_message_id: UUID
    stream: bool = True

    @field_validator("content")
    @classmethod
    def validate_content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class Citation(BaseModel):
    index: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str | None = None
    quote: str


class Reminder(BaseModel):
    minutes_before: int = Field(ge=0)


class CalendarProposal(BaseModel):
    id: UUID
    title: str
    starts_at: AwareDatetime
    ends_at: AwareDatetime | None = None
    timezone: str
    reminders: list[Reminder] = Field(default_factory=list)
    key_fact_id: UUID


class ChatMessage(BaseModel):
    id: UUID
    session_id: UUID
    role: Literal["assistant"] = "assistant"
    content: str
    outcome: MessageOutcome
    citations: list[Citation] = Field(default_factory=list)
    proposals: list[CalendarProposal] = Field(default_factory=list)
    created_at: AwareDatetime


class TokenEventData(BaseModel):
    delta: str


class CitationEventData(BaseModel):
    index: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str | None = None
    quote: str


class ProposalEventData(BaseModel):
    proposal_id: UUID
    title: str
    starts_at: AwareDatetime
    ends_at: AwareDatetime | None = None
    timezone: str
    reminders: list[Reminder] = Field(default_factory=list)
    key_fact_id: UUID


class DoneEventData(BaseModel):
    message_id: UUID
    outcome: MessageOutcome
    usage: dict[str, int | float] = Field(default_factory=dict)


class ErrorEventData(BaseModel):
    code: str
    message: str


class TokenEvent(BaseModel):
    event: Literal["token"] = "token"
    data: TokenEventData


class CitationEvent(BaseModel):
    event: Literal["citation"] = "citation"
    data: CitationEventData


class ProposalEvent(BaseModel):
    event: Literal["proposal"] = "proposal"
    data: ProposalEventData


class DoneEvent(BaseModel):
    event: Literal["done"] = "done"
    data: DoneEventData


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    data: ErrorEventData


ChatStreamEvent = Annotated[
    TokenEvent | CitationEvent | ProposalEvent | DoneEvent | ErrorEvent,
    Field(discriminator="event"),
]
