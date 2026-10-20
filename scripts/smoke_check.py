"""Verifies DB + pgvector, OpenAI embeddings and DeepSeek before you write any feature code.

Run:  uv run --env-file .env python scripts/smoke_check.py
"""

import asyncio

import psycopg

from ai.config import settings
from ai.llm import create_chat_client, create_embedding_client


def check_db() -> None:
    url = settings.database_url.get_secret_value().replace(
        "postgresql+psycopg://", "postgresql://"
    )
    with psycopg.connect(url) as conn:
        ext = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert ext, "pgvector extension is not installed"
        print(f"[ok] database reachable, pgvector {ext[0]}")


async def check_embeddings() -> None:
    embeddings = create_embedding_client(settings)
    vector = await embeddings.embed_query("What is the minimum subscription amount?")
    expected = settings.embedding_dim
    assert len(vector) == expected, (
        f"embedding has {len(vector)} dims, schema expects {expected}"
    )
    print(f"[ok] embeddings return {len(vector)} dims")


async def check_llm() -> None:
    llm = create_chat_client(settings, purpose="fast")
    reply = await llm.ainvoke("Reply with the single word: pong")
    print(f"[ok] DeepSeek replied: {reply.content!r}")


async def main() -> None:
    check_db()
    await check_embeddings()
    await check_llm()


if __name__ == "__main__":
    asyncio.run(main())
