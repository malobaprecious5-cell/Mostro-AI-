"""Dataset loading and page-based retrieval metrics."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt

from ai.retrieval.types import RetrievalHit


class EvalQuestion(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_pages: list[PositiveInt] = Field(min_length=1)


def load_dataset(path: Path) -> list[EvalQuestion]:
    """Load one JSON object per line, reporting malformed entries explicitly."""
    with path.open(encoding="utf-8") as dataset_file:
        questions = [
            EvalQuestion.model_validate(json.loads(line))
            for line in dataset_file
            if line.strip()
        ]
    if not questions:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    identifiers = [question.id for question in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Evaluation dataset contains duplicate IDs: {path}")
    return questions


def recall_at_k(
    questions: Sequence[EvalQuestion],
    results: Sequence[Sequence[RetrievalHit]],
    *,
    k: int,
) -> float:
    """Return the share of questions with any expected page in the top k chunks."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(questions) != len(results):
        raise ValueError("Each evaluation question must have one result list")
    if not questions:
        raise ValueError("Cannot calculate recall for an empty evaluation set")

    hits = 0
    for question, retrieved in zip(questions, results, strict=True):
        expected_pages = set(question.source_pages)
        if any(
            page in expected_pages
            for item in retrieved[:k]
            for page in range(item.page_start, item.page_end + 1)
        ):
            hits += 1
    return hits / len(questions)
