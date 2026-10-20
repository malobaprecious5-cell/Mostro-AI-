"""Orchestrate parsing, chunking, embedding, and storage for one document."""

from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ai.core.logging import get_logger
from ai.db.models import IPO, Document, IngestionJob
from ai.ingestion.chunk import chunk_document
from ai.ingestion.embed import embed_chunks
from ai.ingestion.parse import parse_pdf
from ai.ingestion.store import store_chunks
from ai.llm.interfaces import EmbeddingClient

logger = get_logger(__name__)


async def register_document(
    ipo_id: UUID,
    pdf_path: Path,
    session_factory: async_sessionmaker[AsyncSession],
) -> UUID:
    """Register a local prospectus PDF, reusing an identical existing upload."""
    path = pdf_path.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a local PDF prospectus, got: {path}")
    with path.open("rb") as pdf_file:
        digest = file_digest(pdf_file, "sha256").hexdigest()

    async with session_factory() as session:
        ipo = await session.get(IPO, ipo_id)
        if ipo is None:
            raise ValueError(f"IPO not found: {ipo_id}")
        existing = await session.scalar(
            select(Document).where(
                Document.ipo_id == ipo_id,
                Document.file_hash == digest,
            )
        )
        if existing is not None:
            return existing.id

        current_version = await session.scalar(
            select(func.max(Document.version)).where(Document.ipo_id == ipo_id)
        )
        document = Document(
            ipo_id=ipo_id,
            version=(current_version or 0) + 1,
            doc_type="prospectus",
            storage_uri=path.as_uri(),
            file_hash=digest,
            status="uploaded",
        )
        session.add(document)
        await session.commit()
        return document.id


async def run_ingestion(
    document_id: UUID,
    session_factory: async_sessionmaker[AsyncSession],
    embedding_client: EmbeddingClient,
) -> int:
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        job = IngestionJob(
            document_id=document.id,
            ipo_id=document.ipo_id,
            kind="index",
            status="running",
            step="parse",
            progress=0,
            started_at=datetime.now(UTC),
        )
        document.status = "parsing"
        session.add(job)
        await session.commit()
        job_id = job.id

        try:
            parsed = parse_pdf(_local_pdf_path(document.storage_uri))
            job.step = "chunk"
            job.progress = 25
            await session.commit()

            chunks = chunk_document(parsed)
            if not chunks:
                raise ValueError("The prospectus did not produce any chunks")
            job.step = "embed"
            job.progress = 50
            logger.info(
                "ingestion.chunked",
                chunk_count=len(chunks)
            )
            await session.commit()

            contextualized, vectors = await embed_chunks(chunks, embedding_client)
            job.progress = 80
            await session.commit()

            await store_chunks(session, document, contextualized, vectors)
            document.page_count = parsed.page_count
            document.parser = "pymupdf"
            document.status = "indexed"
            job.status = "succeeded"
            job.step = None
            job.progress = 100
            job.finished_at = datetime.now(UTC)
            await session.commit()
            logger.info(
                "ingestion.completed",
                document_id=str(document.id),
                ipo_id=str(document.ipo_id),
                chunk_count=len(contextualized),
                page_count=parsed.page_count,
            )
            return len(contextualized)
        except Exception as error:
            await session.rollback()
            document = await session.get(Document, document_id)
            failed_job = await session.get(IngestionJob, job_id)
            if document is None or failed_job is None:
                raise RuntimeError(
                    "Could not update ingestion status after pipeline failure"
                ) from error
            failed_job.status = "failed"
            failed_job.error = str(error)[:2000]
            failed_job.finished_at = datetime.now(UTC)
            document.status = "failed"
            await session.commit()
            logger.error(
                "ingestion.failed",
                document_id=str(document.id),
                ipo_id=str(document.ipo_id),
                step=failed_job.step,
                error_type=type(error).__name__,
            )
            raise


def _local_pdf_path(storage_uri: str) -> Path:
    parsed = urlparse(storage_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("Prospectus file URI must refer to the local host")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"Unsupported document storage URI scheme: {parsed.scheme}")
    return Path(storage_uri)
