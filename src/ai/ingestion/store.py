"""Persist an indexed document's chunks atomically."""

from collections.abc import Sequence

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ai.db.models import Chunk, Document
from ai.ingestion.chunk import ChunkRecord


async def store_chunks(
    session: AsyncSession,
    document: Document,
    chunks: Sequence[ChunkRecord],
    vectors: Sequence[list[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Got {len(chunks)} chunks and {len(vectors)} embedding vectors"
        )

    await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    session.add_all(
        [
            Chunk(
                ipo_id=document.ipo_id,
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                content=chunk.content,
                embed_text=chunk.embed_text,
                token_count=chunk.token_count,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    await session.flush()
