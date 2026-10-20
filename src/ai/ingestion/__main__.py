"""CLI entry point for indexing one uploaded prospectus."""

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from ai.db.models import Chunk
from ai.db.session import AsyncSessionFactory
from ai.ingestion.pipeline import register_document, run_ingestion
from ai.llm.runtime import create_embedding_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Index one prospectus document.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser(
        "ingest", help="register and index a local PDF for an IPO"
    )
    ingest_parser.add_argument("ipo_id", type=UUID)
    ingest_parser.add_argument("pdf_path", type=Path)
    run_parser = subparsers.add_parser("run", help="parse, chunk, embed, and store")
    run_parser.add_argument("document_id", type=UUID)
    inspect_parser = subparsers.add_parser("inspect", help="print stored chunks")
    inspect_parser.add_argument("document_id", type=UUID)
    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(_ingest(args.ipo_id, args.pdf_path))
    elif args.command == "run":
        chunk_count = asyncio.run(
            run_ingestion(
                args.document_id,
                AsyncSessionFactory,
                create_embedding_client(),
            )
        )
        print(f"Indexed {chunk_count} chunks for document {args.document_id}")
    elif args.command == "inspect":
        asyncio.run(_inspect_chunks(args.document_id))


async def _inspect_chunks(document_id: UUID) -> None:
    async with AsyncSessionFactory() as session:
        result = await session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.all()
    if not chunks:
        raise ValueError(f"No stored chunks found for document {document_id}")
    for chunk in chunks:
        print(
            f"\n[{chunk.chunk_index}] {chunk.chunk_type} | "
            f"pages {chunk.page_start}-{chunk.page_end} | "
            f"{chunk.section_path or '(no section)'}\n{chunk.content}"
        )


async def _ingest(ipo_id: UUID, pdf_path: Path) -> None:
    document_id = await register_document(ipo_id, pdf_path, AsyncSessionFactory)
    chunk_count = await run_ingestion(
        document_id,
        AsyncSessionFactory,
        create_embedding_client(),
    )
    print(f"Indexed {chunk_count} chunks for document {document_id}")


if __name__ == "__main__":
    main()
