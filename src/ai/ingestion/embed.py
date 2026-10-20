"""Create contextualized embedding inputs and vectorize document chunks."""

from collections.abc import Sequence

import tiktoken

from ai.db.models import VECTOR_DIMENSION
from ai.ingestion.chunk import ChunkRecord
from ai.llm.interfaces import EmbeddingClient

_CONTEXT_TOKEN_LIMIT = 120
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


async def embed_chunks(
    chunks: Sequence[ChunkRecord],
    embedding_client: EmbeddingClient,
) -> tuple[list[ChunkRecord], list[list[float]]]:
    """Embed chunks with their section and neighboring local context."""
    inputs = [
        chunk.model_copy(update={"embed_text": _embedding_text(chunks, index)})
        for index, chunk in enumerate(chunks)
    ]
    vectors = await embedding_client.embed_documents(
        [chunk.embed_text for chunk in inputs]
    )
    if len(vectors) != len(inputs):
        raise ValueError(
            f"Embedding client returned {len(vectors)} vectors for {len(inputs)} chunks"
        )
    for index, vector in enumerate(vectors):
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError(
                f"Embedding {index} has {len(vector)} dimensions; "
                f"expected {VECTOR_DIMENSION}"
            )
    return inputs, vectors


def _embedding_text(chunks: Sequence[ChunkRecord], index: int) -> str:
    chunk = chunks[index]
    parts: list[str] = []
    if chunk.section_path:
        parts.append(f"Section: {chunk.section_path}")

    neighbors: list[str] = []
    for neighbor_index in (index - 1, index + 1):
        if not 0 <= neighbor_index < len(chunks):
            continue
        neighbor = chunks[neighbor_index]
        if neighbor.section_path != chunk.section_path:
            continue
        excerpt = " ".join(neighbor.content.split())
        excerpt_tokens = _TOKENIZER.encode(excerpt)
        if len(excerpt_tokens) > _CONTEXT_TOKEN_LIMIT:
            excerpt = _TOKENIZER.decode(excerpt_tokens[:_CONTEXT_TOKEN_LIMIT])
        neighbors.append(excerpt)
    if neighbors:
        parts.append("Nearby context: " + " | ".join(neighbors))
    parts.append("Content:\n" + chunk.content)
    return "\n".join(parts)
