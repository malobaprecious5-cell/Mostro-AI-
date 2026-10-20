"""Calendar proposal routes. The AI service proposes; it never writes a calendar."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import get_async_session, require_internal_key, require_user_id
from ai.api.schemas.calendar import (
    CalendarProposalPage,
    CalendarProposalUpdate,
    CalendarProposeRequest,
    CalendarStatus,
)
from ai.api.schemas.sessions import CalendarProposalOut, Reminder
from ai.core.errors import AppError
from ai.db.models import CalendarProposal, ChatSession, IPO
from ai.services.calendar_service import CalendarService

router = APIRouter(prefix="/v1", tags=["calendar"])

_AUTH = [Depends(require_internal_key)]


def _proposal_to_schema(row: CalendarProposal) -> CalendarProposalOut:
    return CalendarProposalOut(
        id=row.id,
        key_fact_id=row.key_fact_id,
        title=row.title,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        timezone=row.timezone,
        reminders=[Reminder.model_validate(item) for item in row.reminders],
        status=row.status,
        source_page=row.source_page,
    )


@router.post(
    "/ipos/{ipo_id}/calendar-proposals",
    response_model=CalendarProposalPage,
    status_code=201,
    dependencies=_AUTH,
)
async def create_calendar_proposals(
    ipo_id: Annotated[UUID, Path()],
    body: CalendarProposeRequest,
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> CalendarProposalPage:
    ipo = await db.get(IPO, ipo_id)
    if ipo is None or ipo.status != "published":
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")

    service = CalendarService(db)
    rows = await service.propose_from_verified_facts(
        ipo_id=ipo_id,
        user_id=user_id,
        timezone=body.timezone,
        fact_keys=body.fact_keys,
        reminders=[item.model_dump() for item in body.reminders],
        session_id=body.session_id,
    )
    return CalendarProposalPage(items=[_proposal_to_schema(row) for row in rows])


@router.get(
    "/sessions/{session_id}/calendar-proposals",
    response_model=CalendarProposalPage,
    dependencies=_AUTH,
)
async def list_session_proposals(
    session_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    status: Annotated[CalendarStatus | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> CalendarProposalPage:
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Session not found.")
    if session.user_id != user_id:
        raise AppError(status_code=403, code="FORBIDDEN", message="Access denied.")

    stmt = select(CalendarProposal).where(CalendarProposal.session_id == session_id)
    if status is not None:
        stmt = stmt.where(CalendarProposal.status == status)
    rows = (await db.execute(stmt.order_by(CalendarProposal.starts_at))).scalars().all()
    return CalendarProposalPage(items=[_proposal_to_schema(row) for row in rows])


@router.get(
    "/calendar-proposals/{proposal_id}",
    response_model=CalendarProposalOut,
    dependencies=_AUTH,
)
async def get_calendar_proposal(
    proposal_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> CalendarProposalOut:
    service = CalendarService(db)
    row = await service.get_owned(proposal_id, user_id)
    return _proposal_to_schema(row)


@router.patch(
    "/calendar-proposals/{proposal_id}",
    response_model=CalendarProposalOut,
    dependencies=_AUTH,
)
async def update_calendar_proposal(
    proposal_id: Annotated[UUID, Path()],
    body: CalendarProposalUpdate,
    user_id: Annotated[UUID, Depends(require_user_id)],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> CalendarProposalOut:
    service = CalendarService(db)
    row = await service.update_status(
        proposal_id,
        user_id,
        body.status,
        body.external_event_id,
        body.error,
    )
    return _proposal_to_schema(row)
