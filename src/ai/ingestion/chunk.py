"""Structure-aware chunking that preserves section paths, tables, and images."""

import re
from collections.abc import Iterable
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from ai.ingestion.parse import ParsedBlock, ParsedDocument

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class ChunkRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_index: int = Field(ge=0)
    chunk_type: Literal["text", "table", "heading", "list", "image"]
    section_path: str | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    content: str
    image_base64: str | None = None
    embed_text: str = ""
    token_count: int = Field(ge=0)


def chunk_document(
    document: ParsedDocument,
    *,
    max_tokens: int = 700,
) -> list[ChunkRecord]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    result: list[ChunkRecord] = []
    pending: list[ParsedBlock] = []
    pending_type: Literal["text", "table", "heading", "list", "image"] | None = None
    pending_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal pending, pending_type
        if not pending or pending_type is None:
            return
        content = "\n\n".join(block.content for block in pending)
        _append_chunk(
            result,
            chunk_type=pending_type,
            section_path=pending_path,
            page_start=min(block.page for block in pending),
            page_end=max(block.page for block in pending),
            content=content,
        )
        pending = []
        pending_type = None

    for block in document.blocks:
        if block.kind in {"table", "heading", "image"}:
            flush()
            _append_chunk(
                result,
                chunk_type=block.kind,
                section_path=block.section_path,
                page_start=block.page,
                page_end=block.page,
                content=block.content,
                image_base64=block.image_base64,
            )
            continue

        if pending and (
            pending_path != block.section_path
            or pending_type != block.kind
            or _token_count("\n\n".join(item.content for item in pending))
            + _token_count(block.content)
            > max_tokens
        ):
            flush()
        if not pending:
            pending_path = block.section_path
            pending_type = block.kind
        for piece in _split_oversized_block(block, max_tokens):
            if (
                pending
                and _token_count(
                    "\n\n".join(item.content for item in pending) + "\n\n" + piece
                )
                > max_tokens
            ):
                flush()
                pending_path = block.section_path
                pending_type = block.kind
            pending.append(block.model_copy(update={"content": piece}))
            if (
                _token_count("\n\n".join(item.content for item in pending))
                >= max_tokens
            ):
                flush()
                pending_path = block.section_path
                pending_type = block.kind
    flush()

    return [
        chunk.model_copy(update={"chunk_index": index})
        for index, chunk in enumerate(result)
    ]


def _append_chunk(
    chunks: list[ChunkRecord],
    *,
    chunk_type: Literal["text", "table", "heading", "list", "image"],
    section_path: tuple[str, ...],
    page_start: int,
    page_end: int,
    content: str,
    image_base64: str | None = None,
) -> None:
    chunks.append(
        ChunkRecord(
            chunk_index=len(chunks),
            chunk_type=chunk_type,
            section_path=" > ".join(section_path) or None,
            page_start=page_start,
            page_end=page_end,
            content=content,
            image_base64=image_base64,
            token_count=_token_count(content),
        )
    )


def _split_oversized_block(block: ParsedBlock, max_tokens: int) -> Iterable[str]:
    if block.kind in {"table", "image"} or _token_count(block.content) <= max_tokens:
        yield block.content
        return

    pieces: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(block.content):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _token_count(sentence) > max_tokens:
            words = sentence.split()
            current: list[str] = []
            for word in words:
                candidate = " ".join((*current, word))
                if current and _token_count(candidate) > max_tokens:
                    pieces.append(" ".join(current))
                    current = []
                current.append(word)
            if current:
                pieces.append(" ".join(current))
            continue
        if pieces and _token_count(pieces[-1] + " " + sentence) <= max_tokens:
            pieces[-1] += " " + sentence
        else:
            pieces.append(sentence)
    yield from pieces


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))