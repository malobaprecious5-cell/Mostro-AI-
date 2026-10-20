"""Read-only IPO knowledge: verified facts, summaries, starter questions, glossary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import get_async_session, require_internal_key, require_user_id
from ai.api.schemas.admin import GlossaryPage, GlossaryTermOut
from ai.api.schemas.knowledge import (
    PublicKeyFact,
    PublicKeyFactPage,
    PublicSectionSummary,
    PublicSectionSummaryPage,
    SuggestedQuestions,
)
from ai.api.schemas.sessions import Language, Level
from ai.core.errors import AppError
from ai.db.models import GlossaryTerm, IPO, KeyFact, SectionSummary

router = APIRouter(prefix="/v1", tags=["knowledge"])

_AUTH = [Depends(require_internal_key)]


async def _published_ipo(ipo_id: UUID, db: AsyncSession) -> IPO:
    ipo = await db.get(IPO, ipo_id)
    if ipo is None or ipo.status != "published":
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")
    return ipo


@router.get(
    "/ipos/{ipo_id}/key-facts",
    response_model=PublicKeyFactPage,
    dependencies=_AUTH,
)
async def list_verified_key_facts(
    ipo_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    keys: Annotated[str | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> PublicKeyFactPage:
    _ = user_id
    await _published_ipo(ipo_id, db)
    stmt = select(KeyFact).where(
        KeyFact.ipo_id == ipo_id,
        KeyFact.review_status == "verified",
    )
    if keys:
        fact_keys = [item.strip() for item in keys.split(",") if item.strip()]
        if fact_keys:
            stmt = stmt.where(KeyFact.fact_key.in_(fact_keys))
    rows = (await db.execute(stmt.order_by(KeyFact.fact_key))).scalars().all()
    return PublicKeyFactPage(
        items=[
            PublicKeyFact(
                fact_key=row.fact_key,
                value=row.value,
                source_page=row.source_page,
                source_quote=row.source_quote,
            )
            for row in rows
        ]
    )


@router.get(
    "/ipos/{ipo_id}/summaries",
    response_model=PublicSectionSummaryPage,
    dependencies=_AUTH,
)
async def list_verified_summaries(
    ipo_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    language: Annotated[Language, Query()] = "en",
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> PublicSectionSummaryPage:
    _ = user_id
    _ = language
    await _published_ipo(ipo_id, db)
    rows = (
        await db.execute(
            select(SectionSummary)
            .where(
                SectionSummary.ipo_id == ipo_id,
                SectionSummary.review_status == "verified",
            )
            .order_by(SectionSummary.section_key)
        )
    ).scalars().all()
    return PublicSectionSummaryPage(
        items=[
            PublicSectionSummary(
                section_key=row.section_key,
                summary=row.summary,
                page_start=row.page_start,
                page_end=row.page_end,
            )
            for row in rows
        ]
    )


@router.get(
    "/ipos/{ipo_id}/suggested-questions",
    response_model=SuggestedQuestions,
    dependencies=_AUTH,
)
async def suggested_questions(
    ipo_id: Annotated[UUID, Path()],
    user_id: Annotated[UUID, Depends(require_user_id)],
    language: Annotated[Language, Query()] = "en",
    level: Annotated[Level, Query()] = "beginner",
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> SuggestedQuestions:
    _ = user_id
    _ = level
    ipo = await _published_ipo(ipo_id, db)
    facts = (
        await db.execute(
            select(KeyFact.fact_key).where(
                KeyFact.ipo_id == ipo_id,
                KeyFact.review_status == "verified",
            )
        )
    ).scalars().all()
    return SuggestedQuestions(items=_starter_questions(language, ipo.issuer_name, facts))


@router.get("/glossary", response_model=GlossaryPage, dependencies=_AUTH)
async def list_glossary(
    user_id: Annotated[UUID, Depends(require_user_id)],
    q: Annotated[str | None, Query()] = None,
    language: Annotated[Language | None, Query()] = None,
    level: Annotated[Level | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> GlossaryPage:
    _ = user_id
    stmt = select(GlossaryTerm).order_by(GlossaryTerm.term.asc()).limit(limit + 1)
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(
            or_(
                GlossaryTerm.term.ilike(pattern),
                GlossaryTerm.definition.ilike(pattern),
            )
        )
    if language is not None:
        stmt = stmt.where(GlossaryTerm.language == language)
    if level is not None:
        stmt = stmt.where(GlossaryTerm.level == level)
    if cursor:
        stmt = stmt.where(GlossaryTerm.term > cursor)

    rows = (await db.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].term if has_more and items else None
    return GlossaryPage(
        items=[
            GlossaryTermOut(
                id=row.id,
                term=row.term,
                definition=row.definition,
                example=row.example,
                language=row.language,
                level=row.level,
            )
            for row in items
        ],
        next_cursor=next_cursor,
    )


def _starter_questions(
    language: str, issuer: str, fact_keys: Sequence[str]
) -> list[str]:
    if language == "fr":
        questions = [
            f"Quand l'offre de {issuer} ouvre-t-elle et se clôture-t-elle ?",
            "Quel est le prix de l'offre et le minimum de souscription ?",
            "Comment puis-je souscrire ?",
            "Qui sont les agents collecteurs officiels ?",
            "Quels sont les principaux risques décrits dans le prospectus ?",
        ]
    else:
        questions = [
            f"When does the {issuer} offer open and close?",
            "What is the offer price and the minimum subscription?",
            "How do I apply?",
            "Who are the official receiving agents?",
            "What are the main risks described in the prospectus?",
        ]
    extra: list[str] = []
    for key in fact_keys[:3]:
        label = key.replace("_", " ")
        extra.append(
            f"Que dit le prospectus sur {label} ?"
            if language == "fr"
            else f"What does the prospectus say about {label}?"
        )
    return questions + extra
