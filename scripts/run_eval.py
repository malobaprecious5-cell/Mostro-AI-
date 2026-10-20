"""Measure prospectus retrieval recall@k without invoking a chat model."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ai.db.session import AsyncSessionFactory
from ai.llm.interfaces import EmbeddingClient
from ai.llm.runtime import create_embedding_client
from ai.retrieval.evaluation import EvalQuestion, load_dataset, recall_at_k
from ai.retrieval.hybrid import hybrid_search
from ai.retrieval.types import RetrievalHit

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "datasets"
    / "dangote_prospectus.jsonl"
)


async def run_evaluation(
    session: AsyncSession,
    ipo_id: UUID,
    questions: Sequence[EvalQuestion],
    embedding_client: EmbeddingClient,
    *,
    ks: Sequence[int],
) -> list[list[RetrievalHit]]:
    """Run only configured embeddings and the hybrid database retriever."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    candidate_limit = max(ks)
    results: list[list[RetrievalHit]] = []
    for question in questions:
        hits = await hybrid_search(
            session,
            ipo_id,
            question.question,
            embedding_client,
            candidate_limit=candidate_limit,
        )
        results.append(hits)
    return results


async def _run(args: argparse.Namespace) -> None:
    questions = load_dataset(args.dataset)
    ks = sorted(set(args.ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("--ks must contain positive integers")

    async with AsyncSessionFactory() as session:
        results = await run_evaluation(
            session,
            args.ipo_id,
            questions,
            create_embedding_client(),
            ks=ks,
        )

    print(
        f"Retrieval recall for {len(questions)} questions "
        f"(hybrid search; embeddings only, no LLM)"
    )
    for k in ks:
        recall = recall_at_k(questions, results, k=k)
        matched = round(recall * len(questions))
        print(f"recall@{k}: {recall:.3f} ({matched}/{len(questions)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report prospectus retrieval recall@k without calling an LLM."
    )
    parser.add_argument("--ipo-id", required=True, type=UUID)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
