"""Build calendar proposals from verified key facts. Propose only; never execute."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.core.errors import AppError
from ai.db.models import CalendarProposal, ChatSession, KeyFact

FORWARD_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"confirmed", "rejected"}),
    "confirmed": frozenset({"created", "failed"}),
    "rejected": frozenset(),
    "created": frozenset(),
    "failed": frozenset(),
}


def validate_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise AppError(
            status_code=400,
            code="VALIDATION_ERROR",
            message=f"Unknown IANA time zone: {name}",
            details={"fields": [{"field": "timezone", "message": str(exc)}]},
        ) from exc


def parse_fact_datetime(value: Any) -> datetime | date | None:
    """Extract a date or datetime from a key-fact JSON value."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return _parse_datetime_string(value)
    if isinstance(value, dict):
        for key in ("datetime", "starts_at", "date", "value", "iso"):
            raw = value.get(key)
            parsed = parse_fact_datetime(raw)
            if parsed is not None:
                return parsed
    return None


def localize_start(value: datetime | date, tz: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)
    return datetime.combine(value, time(9, 0), tzinfo=tz)


def title_from_fact_key(fact_key: str) -> str:
    return fact_key.replace("_", " ").replace("-", " ").strip().capitalize()


def _parse_datetime_string(raw: str) -> datetime | date | None:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed
    except ValueError:
        pass
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


class CalendarService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def propose_from_verified_facts(
        self,
        *,
        ipo_id: UUID,
        user_id: UUID,
        timezone: str,
        fact_keys: list[str] | None,
        reminders: list[dict[str, Any]],
        session_id: UUID | None,
    ) -> list[CalendarProposal]:
        tz = validate_timezone(timezone)
        if session_id is not None:
            session = await self.db.get(ChatSession, session_id)
            if session is None or session.user_id != user_id:
                raise AppError(
                    status_code=404, code="NOT_FOUND", message="Session not found."
                )
            if session.ipo_id != ipo_id:
                raise AppError(
                    status_code=403,
                    code="FORBIDDEN",
                    message="Session is bound to a different IPO.",
                )

        stmt = select(KeyFact).where(
            KeyFact.ipo_id == ipo_id,
            KeyFact.review_status == "verified",
        )
        if fact_keys:
            stmt = stmt.where(KeyFact.fact_key.in_(fact_keys))
        facts = (await self.db.execute(stmt)).scalars().all()

        created: list[CalendarProposal] = []
        for fact in facts:
            parsed = parse_fact_datetime(fact.value)
            if parsed is None:
                continue
            starts_at = localize_start(parsed, tz)
            proposal = CalendarProposal(
                id=uuid4(),
                message_id=None,
                session_id=session_id,
                user_id=user_id,
                key_fact_id=fact.id,
                title=title_from_fact_key(fact.fact_key),
                starts_at=starts_at,
                ends_at=None,
                timezone=timezone,
                reminders=reminders,
                status="proposed",
                source_page=fact.source_page,
            )
            self.db.add(proposal)
            created.append(proposal)
        await self.db.commit()
        for proposal in created:
            await self.db.refresh(proposal)
        return created

    async def get_owned(self, proposal_id: UUID, user_id: UUID) -> CalendarProposal:
        row = await self.db.get(CalendarProposal, proposal_id)
        if row is None:
            raise AppError(
                status_code=404, code="NOT_FOUND", message="Proposal not found."
            )
        if row.user_id == user_id:
            return row
        if row.session_id is not None:
            session = await self.db.get(ChatSession, row.session_id)
            if session is not None and session.user_id == user_id:
                return row
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")

    async def update_status(
        self,
        proposal_id: UUID,
        user_id: UUID,
        status: str,
        external_event_id: str | None,
        error: str | None,
    ) -> CalendarProposal:
        row = await self.get_owned(proposal_id, user_id)
        allowed = FORWARD_TRANSITIONS.get(row.status, frozenset())
        if status not in allowed:
            raise AppError(
                status_code=409,
                code="INVALID_STATE_TRANSITION",
                message=(
                    f"Cannot move a calendar proposal from '{row.status}' to '{status}'."
                ),
                details={"from": row.status, "to": status},
            )
        row.status = status
        if external_event_id is not None:
            row.external_event_id = external_event_id
        if error is not None:
            row.error = error
        await self.db.commit()
        await self.db.refresh(row)
        return row
