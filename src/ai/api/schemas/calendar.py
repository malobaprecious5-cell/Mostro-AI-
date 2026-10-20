"""Calendar proposal request and response schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ai.api.schemas.sessions import CalendarProposalOut, Reminder

CalendarStatus = Literal["proposed", "confirmed", "rejected", "created", "failed"]


class CalendarProposeRequest(BaseModel):
    timezone: str
    fact_keys: list[str] | None = None
    reminders: list[Reminder] = Field(
        default_factory=lambda: [Reminder(minutes_before=1440)]
    )
    session_id: UUID | None = None


class CalendarProposalPage(BaseModel):
    items: list[CalendarProposalOut]


class CalendarProposalUpdate(BaseModel):
    status: CalendarStatus
    external_event_id: str | None = None
    error: str | None = None
