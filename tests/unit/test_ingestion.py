"""Unit coverage for the single-prospectus ingestion stages."""

from pathlib import Path

import pymupdf
import pytest

from ai.ingestion.chunk import ChunkRecord, chunk_document
from ai.ingestion.embed import embed_chunks
from ai.ingestion.parse import ParsedBlock, ParsedDocument, parse_pdf
from ai.llm.fakes import FakeEmbeddingClient


def _sample_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 45), "1 Prospectus Summary", fontsize=18)
    page.insert_text((50, 78), "1.1 Offer details", fontsize=14)

    x_positions = (50, 150, 250)
    y_positions = (105, 135, 165)
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))
    page.insert_text((55, 125), "Item", fontsize=10)
    page.insert_text((155, 125), "Value", fontsize=10)
    page.insert_text((55, 155), "Shares", fontsize=10)
    page.insert_text((155, 155), "1,000", fontsize=10)
    page.insert_text((50, 210), "The offer is described in this section.", fontsize=10)
    pdf.save(path)
    pdf.close()


def test_pdf_parser_preserves_headings_pages_and_atomic_markdown_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prospectus.pdf"
    _sample_pdf(path)

    parsed = parse_pdf(path)

    assert parsed.page_count == 1
    headings = [block for block in parsed.blocks if block.kind == "heading"]
    assert [heading.content for heading in headings] == [
        "1 Prospectus Summary",
        "1.1 Offer details",
    ]
    table = next(block for block in parsed.blocks if block.kind == "table")
    assert table.section_path == ("1 Prospectus Summary", "1.1 Offer details")
    assert "| Item | Value |" in table.content
    assert "| Shares | 1,000 |" in table.content
    assert all(block.page == 1 for block in parsed.blocks)


def test_chunking_keeps_tables_whole_and_tracks_section_paths() -> None:
    table_content = "| " + " | ".join(["header"] * 12) + " |\n"
    parsed = ParsedDocument(
        page_count=2,
        blocks=(
            ParsedBlock(
                kind="heading",
                content="Subscription",
                page=1,
                section_path=("Offer", "Subscription"),
            ),
            ParsedBlock(
                kind="text",
                content="Applicants submit an application.",
                page=1,
                section_path=("Offer", "Subscription"),
            ),
            ParsedBlock(
                kind="table",
                content=table_content,
                page=2,
                section_path=("Offer", "Subscription"),
            ),
        ),
    )

    chunks = chunk_document(parsed, max_tokens=8)

    assert [chunk.chunk_type for chunk in chunks] == ["heading", "text", "table"]
    assert chunks[-1].content == table_content
    assert chunks[-1].token_count > 8
    assert all(chunk.section_path == "Offer > Subscription" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[-1].page_start == chunks[-1].page_end == 2


@pytest.mark.asyncio
async def test_embeddings_add_neighbor_context_without_changing_source_content() -> (
    None
):
    chunks = [
        ChunkRecord(
            chunk_index=0,
            chunk_type="text",
            section_path="Offer > Eligibility",
            page_start=1,
            page_end=1,
            content="Applicants must be residents.",
            token_count=5,
        ),
        ChunkRecord(
            chunk_index=1,
            chunk_type="table",
            section_path="Offer > Eligibility",
            page_start=2,
            page_end=2,
            content="| Class | Limit |\n| --- | --- |\n| A | 10 |",
            token_count=13,
        ),
    ]

    contextualized, vectors = await embed_chunks(chunks, FakeEmbeddingClient())

    assert contextualized[1].content == chunks[1].content
    assert "Section: Offer > Eligibility" in contextualized[1].embed_text
    assert "Applicants must be residents." in contextualized[1].embed_text
    assert contextualized[1].content in contextualized[1].embed_text
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
