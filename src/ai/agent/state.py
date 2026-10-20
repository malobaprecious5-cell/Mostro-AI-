"""LangGraph state machine for the chat agent."""

from uuid import UUID

from pydantic import BaseModel, Field

from ai.retrieval.types import RetrievalHit


class ChatMessage(BaseModel):
    """A message in the conversation flow."""

    role: str
    content: str


class ChatState(BaseModel):
    """Shared state across all graph nodes."""

    session_id: UUID
    ipo_id: UUID
    user_id: UUID
    user_query: str
    language: str = "en"
    level: str = "beginner"

    # Retrieval stage output
    retrieved_chunks: list[RetrievalHit] = Field(default_factory=list)
    abstain_decision_reason: str | None = None

    # Generation stage output
    answer: str = ""
    outcome: str = "pending"  # answered, abstained, refused, blocked, error
    citations: list[dict] = Field(default_factory=list)

    # Trace and metadata
    trace: dict = Field(default_factory=dict)
    error: str | None = None
