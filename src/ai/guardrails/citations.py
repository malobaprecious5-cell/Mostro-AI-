"""Output validation guardrails: citation grounding, numeric/date checks."""

import re
from typing import NamedTuple


class CitationCheck(NamedTuple):
    is_valid: bool
    missing_numbers: tuple[str, ...] = ()
    missing_dates: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()  # reserved for future detailed error messages


def extract_numbers(text: str) -> list[str]:
    """Extract numeric values (including decimals, percentages, negative numbers)."""
    # Match integers, decimals, percentages, millions/billions
    pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?%?|USD\s*[\d,]+(?:\.\d+)?|N/?A\s*[\d,]+(?:\.\d+)?"
    return re.findall(pattern, text)


def extract_dates(text: str) -> list[str]:
    """Extract date-like patterns (DD/MM/YYYY, Month DD YYYY, etc.)."""
    patterns = [
        r"\d{1,2}/\d{1,2}/\d{4}",  # DD/MM/YYYY or MM/DD/YYYY
        r"\d{1,2}-\d{1,2}-\d{4}",  # DD-MM-YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\d{4}-\d{1,2}-\d{1,2}",  # YYYY-MM-DD
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, text, re.IGNORECASE))
    return results


def check_citations(answer: str, source_text: str) -> CitationCheck:
    """Verify that all numbers and dates in the answer appear in source text."""
    errors: list[str] = []
    missing_numbers: list[str] = []
    missing_dates: list[str] = []

    # Extract numbers from answer
    answer_numbers = extract_numbers(answer)
    for num in answer_numbers:
        # Normalize for comparison (remove commas, normalize spacing)
        normalized = num.replace(",", "").strip()
        if not any(
            normalized in src.replace(",", "").strip()
            or normalized.rstrip("%") in src.replace(",", "").strip()
            for src in source_text.split()
        ):
            missing_numbers.append(num)

    # Extract dates from answer
    answer_dates = extract_dates(answer)
    for date in answer_dates:
        if date not in source_text:
            missing_dates.append(date)

    is_valid = len(missing_numbers) == 0 and len(missing_dates) == 0
    return CitationCheck(
        is_valid=is_valid,
        missing_numbers=tuple(missing_numbers),
        missing_dates=tuple(missing_dates),
        errors=tuple(errors),
    )
