"""Admin routes: IPO management, documents, ingestion, review, glossary, debug."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path as FilePath
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Path,
    Query,
    UploadFile,
)
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ai.api.deps import (
    get_async_session,
    get_embedding_client_dep,
    require_admin_key,
)
from ai.api.schemas.admin import (
    ChunkOut,
    ChunkPage,
    DocumentAccepted,
    DocumentOut,
    DocumentPage,
    GlossaryTermOut,
    GlossaryUpsert,
    IpoOut,
    IpoStatusUpdate,
    IpoUpsert,
    JobAccepted,
    JobOut,
    KeyFactExtractRequest,
    KeyFactOut,
    KeyFactPage,
    KeyFactReview,
    ReingestRequest,
    RetrievalDebugHit,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    SectionSummaryGenerateRequest,
    SectionSummaryOut,
    SectionSummaryPage,
    SectionSummaryReview,
)
from ai.config import settings
from ai.core.errors import AppError
from ai.db.models import (
    Chunk,
    Document,
    GlossaryTerm,
    IngestionJob,
    IPO,
    KeyFact,
    SectionSummary,
)
from ai.db.session import AsyncSessionFactory
from ai.llm.factory import get_embedding_client
from ai.llm.interfaces import EmbeddingClient
from ai.retrieval.hybrid import retrieve_chunks

router = APIRouter(prefix="/v1/admin", tags=["admin"])

_AUTH = [Depends(require_admin_key)]
DocType = Literal["prospectus", "addendum", "supplement"]
ReviewFilter = Literal["pending", "verified", "rejected"]


def _ipo_to_schema(ipo: IPO) -> IpoOut:
    return IpoOut(
        id=ipo.id,
        issuer_name=ipo.issuer_name,
        country=ipo.country,
        exchange=ipo.exchange,
        status=ipo.status,
        published_at=ipo.published_at,
        created_at=ipo.created_at,
    )


def _doc_to_schema(doc: Document, chunk_count: int | None = None) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        ipo_id=doc.ipo_id,
        version=doc.version,
        doc_type=doc.doc_type,
        storage_uri=doc.storage_uri,
        file_hash=doc.file_hash,
        page_count=doc.page_count,
        parser=doc.parser,
        status=doc.status,
        chunk_count=chunk_count,
        created_at=doc.created_at,
    )


def _job_to_schema(row: IngestionJob) -> JobOut:
    return JobOut(
        id=row.id,
        document_id=row.document_id,
        ipo_id=row.ipo_id,
        kind=row.kind,
        status=row.status,
        step=row.step,
        progress=row.progress,
        error=row.error,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _fact_to_schema(row: KeyFact) -> KeyFactOut:
    return KeyFactOut(
        id=row.id,
        ipo_id=row.ipo_id,
        document_id=row.document_id,
        fact_key=row.fact_key,
        value=row.value,
        source_page=row.source_page,
        source_quote=row.source_quote,
        review_status=row.review_status,
        reviewed_by=row.reviewed_by,
        reviewed_at=row.reviewed_at,
    )


def _summary_to_schema(row: SectionSummary) -> SectionSummaryOut:
    return SectionSummaryOut(
        id=row.id,
        ipo_id=row.ipo_id,
        document_id=row.document_id,
        section_key=row.section_key,
        summary=row.summary,
        page_start=row.page_start,
        page_end=row.page_end,
        review_status=row.review_status,
        reviewed_by=row.reviewed_by,
        reviewed_at=row.reviewed_at,
    )


def _glossary_to_schema(row: GlossaryTerm) -> GlossaryTermOut:
    return GlossaryTermOut(
        id=row.id,
        term=row.term,
        definition=row.definition,
        example=row.example,
        language=row.language,
        level=row.level,
    )


async def _queue_index_job(
    db: AsyncSession,
    document: Document,
    kind: str,
) -> IngestionJob:
    job = IngestionJob(
        id=uuid4(),
        document_id=document.id,
        ipo_id=document.ipo_id,
        kind=kind,
        status="queued",
        step="parse",
        progress=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _run_index_job(document_id: UUID) -> None:
    from ai.ingestion.pipeline import run_ingestion

    await run_ingestion(document_id, AsyncSessionFactory, get_embedding_client())


async def _latest_indexed_document(db: AsyncSession, ipo_id: UUID) -> Document:
    document = (
        await db.execute(
            select(Document)
            .where(Document.ipo_id == ipo_id, Document.status == "indexed")
            .order_by(Document.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if document is None:
        raise AppError(
            status_code=409,
            code="PUBLISH_PRECONDITION_FAILED",
            message="No indexed document exists for this IPO.",
            details={"missing": ["indexed_document"]},
        )
    return document


@router.put("/ipos/{ipo_id}", response_model=IpoOut, dependencies=_AUTH)
async def upsert_ipo(
    ipo_id: Annotated[UUID, Path()],
    body: IpoUpsert,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> IpoOut:
    row = await db.get(IPO, ipo_id)
    if row is None:
        row = IPO(
            id=ipo_id,
            issuer_name=body.issuer_name,
            country=body.country.upper(),
            exchange=body.exchange,
            status="draft",
            created_at=datetime.now(UTC),
        )
        db.add(row)
    else:
        row.issuer_name = body.issuer_name
        row.country = body.country.upper()
        row.exchange = body.exchange
    await db.commit()
    await db.refresh(row)
    return _ipo_to_schema(row)


@router.patch("/ipos/{ipo_id}/status", response_model=IpoOut, dependencies=_AUTH)
async def update_ipo_status(
    ipo_id: Annotated[UUID, Path()],
    body: IpoStatusUpdate,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> IpoOut:
    row = await db.get(IPO, ipo_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")

    if body.status == "published":
        missing: list[str] = []
        indexed_count = (
            await db.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.ipo_id == ipo_id, Document.status == "indexed")
            )
        ).scalar_one()
        if indexed_count == 0:
            missing.append("indexed_document")
        if settings.required_key_facts:
            verified = set(
                (
                    await db.execute(
                        select(KeyFact.fact_key).where(
                            KeyFact.ipo_id == ipo_id,
                            KeyFact.review_status == "verified",
                            KeyFact.fact_key.in_(settings.required_key_facts),
                        )
                    )
                ).scalars().all()
            )
            missing.extend(
                f"key_fact:{key}"
                for key in settings.required_key_facts
                if key not in verified
            )
        if missing:
            raise AppError(
                status_code=409,
                code="PUBLISH_PRECONDITION_FAILED",
                message="IPO cannot be published: required items are missing.",
                details={"missing": missing},
            )
        row.published_at = datetime.now(UTC)

    row.status = body.status
    await db.commit()
    await db.refresh(row)
    return _ipo_to_schema(row)


@router.post(
    "/ipos/{ipo_id}/documents",
    response_model=DocumentAccepted,
    status_code=202,
    dependencies=_AUTH,
)
async def create_document(
    ipo_id: Annotated[UUID, Path()],
    background_tasks: BackgroundTasks,
    doc_type: Annotated[DocType, Form()],
    storage_uri: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> DocumentAccepted:
    ipo = await db.get(IPO, ipo_id)
    if ipo is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")
    if file is None and not storage_uri:
        raise AppError(
            status_code=400,
            code="VALIDATION_ERROR",
            message="Provide a PDF file or a storage_uri.",
        )

    document_id = uuid4()
    payload: bytes | None = None
    if file is not None:
        payload = await file.read()
        digest = sha256(payload).hexdigest()
        uri = ""
    else:
        assert storage_uri is not None
        uri = storage_uri
        source = FilePath(storage_uri.removeprefix("file://"))
        if source.is_file():
            digest = sha256(source.read_bytes()).hexdigest()
        else:
            digest = sha256(storage_uri.encode("utf-8")).hexdigest()

    existing = (
        await db.execute(
            select(Document).where(
                Document.ipo_id == ipo_id, Document.file_hash == digest
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise AppError(
            status_code=409,
            code="DUPLICATE_DOCUMENT",
            message="This file has already been ingested for this IPO.",
            details={"document_id": str(existing.id)},
        )

    if payload is not None:
        dest_dir = FilePath(settings.document_storage_dir) / str(ipo_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{document_id}.pdf"
        dest.write_bytes(payload)
        uri = dest.as_uri()

    current_version = (
        await db.execute(
            select(func.max(Document.version)).where(Document.ipo_id == ipo_id)
        )
    ).scalar_one()
    document = Document(
        id=document_id,
        ipo_id=ipo_id,
        version=(current_version or 0) + 1,
        doc_type=doc_type,
        storage_uri=uri,
        file_hash=digest,
        status="uploaded",
    )
    db.add(document)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise AppError(
            status_code=409,
            code="DUPLICATE_DOCUMENT",
            message="This file has already been ingested for this IPO.",
        ) from exc

    job = await _queue_index_job(db, document, kind="index")
    background_tasks.add_task(_run_index_job, document.id)
    return DocumentAccepted(document_id=document.id, job_id=job.id)


@router.get(
    "/ipos/{ipo_id}/documents",
    response_model=DocumentPage,
    dependencies=_AUTH,
)
async def list_documents(
    ipo_id: Annotated[UUID, Path()],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> DocumentPage:
    rows = (
        await db.execute(
            select(Document)
            .where(Document.ipo_id == ipo_id)
            .order_by(Document.version.desc())
        )
    ).scalars().all()
    return DocumentPage(items=[_doc_to_schema(r) for r in rows])


@router.get(
    "/documents/{document_id}",
    response_model=DocumentOut,
    dependencies=_AUTH,
)
async def get_document(
    document_id: Annotated[UUID, Path()],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> DocumentOut:
    row = await db.get(Document, document_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Document not found.")
    chunk_count = (
        await db.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == document_id)
        )
    ).scalar_one()
    return _doc_to_schema(row, chunk_count=int(chunk_count))


@router.post(
    "/documents/{document_id}/reingest",
    response_model=JobAccepted,
    status_code=202,
    dependencies=_AUTH,
)
async def reingest_document(
    document_id: Annotated[UUID, Path()],
    background_tasks: BackgroundTasks,
    body: ReingestRequest | None = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> JobAccepted:
    _ = body
    row = await db.get(Document, document_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Document not found.")
    job = await _queue_index_job(db, row, kind="reingest")
    background_tasks.add_task(_run_index_job, row.id)
    return JobAccepted(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=JobOut, dependencies=_AUTH)
async def get_job(
    job_id: Annotated[UUID, Path()],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> JobOut:
    row = await db.get(IngestionJob, job_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Job not found.")
    return _job_to_schema(row)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkPage,
    dependencies=_AUTH,
)
async def list_chunks(
    document_id: Annotated[UUID, Path()],
    page: Annotated[int | None, Query(ge=1)] = None,
    chunk_type: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> ChunkPage:
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit + 1)
    )
    if page is not None:
        stmt = stmt.where(Chunk.page_start <= page, Chunk.page_end >= page)
    if chunk_type is not None:
        stmt = stmt.where(Chunk.chunk_type == chunk_type)
    if q:
        stmt = stmt.where(Chunk.content.ilike(f"%{q}%"))
    if cursor:
        try:
            stmt = stmt.where(Chunk.chunk_index > int(cursor))
        except ValueError:
            pass

    rows = (await db.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = str(items[-1].chunk_index) if has_more and items else None
    return ChunkPage(
        items=[
            ChunkOut(
                id=c.id,
                document_id=c.document_id,
                ipo_id=c.ipo_id,
                chunk_index=c.chunk_index,
                chunk_type=c.chunk_type,
                section_path=c.section_path,
                page_start=c.page_start,
                page_end=c.page_end,
                content=c.content,
                token_count=c.token_count,
            )
            for c in items
        ],
        next_cursor=next_cursor,
    )


@router.get(
    "/ipos/{ipo_id}/key-facts",
    response_model=KeyFactPage,
    dependencies=_AUTH,
)
async def list_key_facts(
    ipo_id: Annotated[UUID, Path()],
    review_status: Annotated[ReviewFilter | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> KeyFactPage:
    stmt = select(KeyFact).where(KeyFact.ipo_id == ipo_id).order_by(KeyFact.fact_key)
    if review_status is not None:
        stmt = stmt.where(KeyFact.review_status == review_status)
    rows = (await db.execute(stmt)).scalars().all()
    return KeyFactPage(items=[_fact_to_schema(row) for row in rows])


@router.post(
    "/ipos/{ipo_id}/key-facts/extract",
    response_model=JobAccepted,
    status_code=202,
    dependencies=_AUTH,
)
async def extract_key_facts(
    ipo_id: Annotated[UUID, Path()],
    body: KeyFactExtractRequest | None = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> JobAccepted:
    _ = body
    ipo = await db.get(IPO, ipo_id)
    if ipo is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")
    document = await _latest_indexed_document(db, ipo_id)
    job = await _queue_index_job(db, document, kind="facts")
    return JobAccepted(job_id=job.id)


@router.patch("/key-facts/{fact_id}", response_model=KeyFactOut, dependencies=_AUTH)
async def review_key_fact(
    fact_id: Annotated[UUID, Path()],
    body: KeyFactReview,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> KeyFactOut:
    row = await db.get(KeyFact, fact_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Key fact not found.")
    row.review_status = body.review_status
    row.reviewed_by = body.reviewed_by
    row.reviewed_at = datetime.now(UTC)
    if body.value is not None:
        row.value = body.value
    if body.source_page is not None:
        row.source_page = body.source_page
    await db.commit()
    await db.refresh(row)
    return _fact_to_schema(row)


@router.get(
    "/ipos/{ipo_id}/section-summaries",
    response_model=SectionSummaryPage,
    dependencies=_AUTH,
)
async def list_section_summaries(
    ipo_id: Annotated[UUID, Path()],
    review_status: Annotated[ReviewFilter | None, Query()] = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> SectionSummaryPage:
    stmt = (
        select(SectionSummary)
        .where(SectionSummary.ipo_id == ipo_id)
        .order_by(SectionSummary.section_key)
    )
    if review_status is not None:
        stmt = stmt.where(SectionSummary.review_status == review_status)
    rows = (await db.execute(stmt)).scalars().all()
    return SectionSummaryPage(items=[_summary_to_schema(row) for row in rows])


@router.post(
    "/ipos/{ipo_id}/section-summaries/generate",
    response_model=JobAccepted,
    status_code=202,
    dependencies=_AUTH,
)
async def generate_section_summaries(
    ipo_id: Annotated[UUID, Path()],
    body: SectionSummaryGenerateRequest | None = None,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> JobAccepted:
    _ = body
    ipo = await db.get(IPO, ipo_id)
    if ipo is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="IPO not found.")
    document = await _latest_indexed_document(db, ipo_id)
    job = await _queue_index_job(db, document, kind="summaries")
    return JobAccepted(job_id=job.id)


@router.patch(
    "/section-summaries/{summary_id}",
    response_model=SectionSummaryOut,
    dependencies=_AUTH,
)
async def review_section_summary(
    summary_id: Annotated[UUID, Path()],
    body: SectionSummaryReview,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> SectionSummaryOut:
    row = await db.get(SectionSummary, summary_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Summary not found.")
    row.review_status = body.review_status
    row.reviewed_by = body.reviewed_by
    row.reviewed_at = datetime.now(UTC)
    if body.summary is not None:
        row.summary = body.summary
    await db.commit()
    await db.refresh(row)
    return _summary_to_schema(row)


@router.post(
    "/glossary",
    response_model=GlossaryTermOut,
    status_code=201,
    dependencies=_AUTH,
)
async def create_glossary_term(
    body: GlossaryUpsert,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
    embedding_client: EmbeddingClient = Depends(get_embedding_client_dep),  # noqa: B008
) -> GlossaryTermOut:
    vector = await embedding_client.embed_query(f"{body.term}\n{body.definition}")
    row = GlossaryTerm(
        id=uuid4(),
        term=body.term,
        language=body.language,
        level=body.level,
        definition=body.definition,
        example=body.example,
        embedding=vector,
    )
    db.add(row)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise AppError(
            status_code=409,
            code="VALIDATION_ERROR",
            message="A glossary term with this name and language already exists.",
        ) from exc
    await db.refresh(row)
    return _glossary_to_schema(row)


@router.put(
    "/glossary/{term_id}",
    response_model=GlossaryTermOut,
    dependencies=_AUTH,
)
async def replace_glossary_term(
    term_id: Annotated[UUID, Path()],
    body: GlossaryUpsert,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
    embedding_client: EmbeddingClient = Depends(get_embedding_client_dep),  # noqa: B008
) -> GlossaryTermOut:
    row = await db.get(GlossaryTerm, term_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Term not found.")
    row.term = body.term
    row.language = body.language
    row.level = body.level
    row.definition = body.definition
    row.example = body.example
    row.embedding = await embedding_client.embed_query(
        f"{body.term}\n{body.definition}"
    )
    await db.commit()
    await db.refresh(row)
    return _glossary_to_schema(row)


@router.delete("/glossary/{term_id}", status_code=204, dependencies=_AUTH)
async def delete_glossary_term(
    term_id: Annotated[UUID, Path()],
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Response:
    row = await db.get(GlossaryTerm, term_id)
    if row is None:
        raise AppError(status_code=404, code="NOT_FOUND", message="Term not found.")
    await db.delete(row)
    await db.commit()
    return Response(status_code=204)


@router.post(
    "/retrieval/debug",
    response_model=RetrievalDebugResponse,
    dependencies=_AUTH,
)
async def debug_retrieval(
    body: RetrievalDebugRequest,
    db: AsyncSession = Depends(get_async_session),  # noqa: B008
    embedding_client: EmbeddingClient = Depends(get_embedding_client_dep),  # noqa: B008
) -> RetrievalDebugResponse:
    if not settings.enable_retrieval_debug:
        raise AppError(
            status_code=403,
            code="FORBIDDEN",
            message="Retrieval debug is disabled in this environment.",
        )

    response = await retrieve_chunks(
        db,
        body.ipo_id,
        body.query,
        embedding_client,
        candidate_limit=body.candidates,
        top_k=body.top_k,
    )
    hits = [
        RetrievalDebugHit(
            chunk_id=h.chunk_id,
            page_start=h.page_start,
            page_end=h.page_end,
            section_path=h.section_path,
            rrf_score=h.rrf_score,
            similarity=h.similarity,
            is_neighbor=h.is_neighbor,
            snippet=h.content[:300],
        )
        for h in response.chunks
    ]
    return RetrievalDebugResponse(
        items=hits,
        would_abstain=response.decision.abstain,
        best_similarity=response.decision.best_similarity,
        threshold=response.decision.threshold,
        query=body.query,
    )
