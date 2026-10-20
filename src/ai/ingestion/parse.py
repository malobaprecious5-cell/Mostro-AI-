"""Parse a PDF into ordered, page-aware structural blocks including images."""

import base64
import re
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field
from ai.core.logging import get_logger

logger = get_logger(__name__)


class ParsedBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["text", "table", "heading", "list", "image"]
    content: str
    image_base64: str | None = None
    page: int = Field(ge=1)
    section_path: tuple[str, ...] = ()


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    blocks: tuple[ParsedBlock, ...]
    page_count: int = Field(ge=1)


_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+\S")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•▪]|\d+[.)])\s+\S")
_ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9\s&(),./:’'-]{2,}$")


def parse_pdf(path: Path) -> ParsedDocument:
    """Read text, heading styles, tables, and images in page reading order."""
    if not path.is_file():
        raise FileNotFoundError(f"Prospectus PDF does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF prospectus, got: {path.suffix or 'no suffix'}"
        )

    blocks: list[ParsedBlock] = []
    headings: list[tuple[int, str]] = []
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            tables = page.find_tables().tables
            table_rectangles = [pymupdf.Rect(table.bbox) for table in tables]
            page_elements: list[
                tuple[
                    float,
                    int,
                    Literal["text", "table", "heading", "list", "image"],
                    str,
                    str | None,
                    int | None,
                ]
            ] = []
            
            body_sizes = [
                span["size"]
                for text_block in page.get_text("dict")["blocks"]
                if text_block.get("type") == 0
                for line in text_block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            body_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 10.0

            for table in tables:
                markdown = _table_markdown(table.extract())
                if markdown:
                    page_elements.append((table.bbox[1], 0, "table", markdown, None, None))

            for block in page.get_text("dict")["blocks"]:
                block_type = block.get("type")
                
                # Image block
                if block_type == 1:
                    image_bytes = block.get("image")
                    if image_bytes:
                        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        y_position = float(block["bbox"][1])
                        page_elements.append((
                            y_position,
                            0,
                            "image",
                            f"[Image extracted from page {page_number}]",
                            img_b64,
                            None,
                        ))
                    continue

                # Text block
                if block_type != 0:
                    continue

                block_rect = pymupdf.Rect(block["bbox"])
                if any(block_rect.intersects(rect) for rect in table_rectangles):
                    continue

                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    size = max(
                        (span.get("size", body_size) for span in spans),
                        default=body_size,
                    )
                    font_heading = size >= body_size * 1.28 and len(text) <= 180
                    numbered = _NUMBERED_HEADING.match(text)
                    uppercase_heading = (
                        _ALL_CAPS.fullmatch(text) is not None and len(text) <= 100
                    )
                    y_position = float(line["bbox"][1])
                    if font_heading or numbered or uppercase_heading:
                        level = _heading_level(text, size, body_size, numbered)
                        kind: Literal["text", "table", "heading", "list", "image"] = "heading"
                    elif _LIST_ITEM.match(text):
                        level = None
                        kind = "list"
                    else:
                        level = None
                        kind = "text"
                    page_elements.append((y_position, 1, kind, text, None, level))

            page_elements.sort(key=lambda element: (element[0], element[1]))
            for _, _, kind, content, img_b64, level in page_elements:
                if kind == "heading" and level is not None:
                    while headings and headings[-1][0] >= level:
                        headings.pop()
                    headings.append((level, content))
                blocks.append(
                    ParsedBlock(
                        kind=kind,
                        content=content,
                        image_base64=img_b64,
                        page=page_number,
                        section_path=tuple(title for _, title in headings),
                    )
                )
        page_count = len(pdf)

    if not blocks:
        raise ValueError(f"No readable content found in PDF: {path}")
    logger.info(
        "ingestion.parsed",
        blocks=len(blocks),
        page_count=page_count
    )
    return ParsedDocument(blocks=tuple(blocks), page_count=page_count)


def _heading_level(
    text: str,
    size: float,
    body_size: float,
    numbered: re.Match[str] | None,
) -> int:
    if numbered:
        return min(numbered.group(1).rstrip(".").count(".") + 1, 6)
    if size >= body_size * 1.8:
        return 1
    if size >= body_size * 1.45:
        return 2
    if _ALL_CAPS.fullmatch(text):
        return 1
    return 2


def _table_markdown(rows: list[list[object]]) -> str:
    if not rows:
        return ""
    normalized = [
        [
            str(value).replace("|", r"\|").replace("\n", " ").strip() if value else ""
            for value in row
        ]
        for row in rows
    ]
    column_count = max(map(len, normalized), default=0)
    if not column_count:
        return ""
    normalized = [row + [""] * (column_count - len(row)) for row in normalized]
    header = "| " + " | ".join(normalized[0]) + " |"
    divider = "| " + " | ".join("---" for _ in range(column_count)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, divider, *body])