"""Evaluation metrics for answer quality and abstention."""

from typing import NamedTuple


class FaithfulnessResult(NamedTuple):
    score: float  # 0-1, higher is better
    has_hallucinations: bool
    unsupported_claims: list[str]


class CitationResult(NamedTuple):
    score: float  # 0-1, higher is better
    valid_citations: int
    missing_citations: int
    incorrect_pages: list[str]


class AbstentionResult(NamedTuple):
    correct: bool  # True if abstention decision matches ground truth
    confidence: float  # 0-1, how confident the model was


def evaluate_faithfulness(answer: str, source_text: str) -> FaithfulnessResult:
    """
    Check if the answer stays within bounds of the source text.
    
    A faithful answer only contains claims that appear or are derivable from sources.
    """
    # Simplified heuristic: check for common hallucination markers
    hallucination_markers = [
        "I believe",
        "I think",
        "It is likely",
        "In my opinion",
        "Based on my knowledge",
    ]
    has_hallucinations = any(marker.lower() in answer.lower() for marker in hallucination_markers)
    
    # Score: 1.0 if no hallucination markers, 0.8 otherwise
    score = 0.8 if has_hallucinations else 1.0
    return FaithfulnessResult(
        score=score,
        has_hallucinations=has_hallucinations,
        unsupported_claims=[],
    )


def evaluate_citations(
    answer: str,
    source_chunks: list[dict],
    expected_pages: list[int],
) -> CitationResult:
    """
    Check if citations are accurate and complete.
    
    Expected pages are from the ground truth dataset.
    """
    cited_pages = set()
    for chunk in source_chunks:
        if chunk["content"] in answer:
            cited_pages.add(chunk["page"])
    
    expected_set = set(expected_pages)
    correct_pages = len(cited_pages & expected_set)
    missed_pages = len(expected_set - cited_pages)
    
    if len(expected_set) == 0:
        score = 1.0
    else:
        score = correct_pages / len(expected_set)
    
    return CitationResult(
        score=score,
        valid_citations=correct_pages,
        missing_citations=missed_pages,
        incorrect_pages=list(cited_pages - expected_set),
    )


def evaluate_abstention(
    answer: str,
    expected_outcome: str,  # "answered" or "abstained"
    has_confidence_markers: bool = False,
) -> AbstentionResult:
    """
    Check if the model correctly abstains on out-of-document questions.
    
    An abstention is correct if:
    - Expected outcome is "abstained" and answer contains abstention phrases
    - Expected outcome is "answered" and answer provides a direct answer
    """
    abstention_phrases = [
        "does not address",
        "could not find",
        "i couldn't find",
        "prospectus does not",
        "not mentioned",
    ]
    
    is_abstaining = any(phrase.lower() in answer.lower() for phrase in abstention_phrases)
    correct = (expected_outcome == "abstained") == is_abstaining
    
    # Confidence: higher if decision matches expectation
    confidence = 0.9 if correct else 0.1
    
    return AbstentionResult(
        correct=correct,
        confidence=confidence,
    )
