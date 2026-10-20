"""Input validation guardrails: injection, blocking."""

import re
from typing import NamedTuple


class InjectionCheck(NamedTuple):
    is_injection: bool
    reason: str | None


def check_injection(text: str) -> InjectionCheck:
    """Detect SQL injection and prompt injection patterns."""
    if not isinstance(text, str) or not text.strip():
        return InjectionCheck(is_injection=False, reason=None)

    # SQL injection patterns
    sql_keywords = r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC|SCRIPT)\b"
    if re.search(sql_keywords, text, re.IGNORECASE):
        # Only flag if pattern looks suspicious (multiple keywords or obvious attack syntax)
        dangerous_combos = r"(UNION.*SELECT|;.*DROP|--.*FROM)"
        if re.search(dangerous_combos, text, re.IGNORECASE):
            return InjectionCheck(is_injection=True, reason="sql_injection")

    # Prompt injection patterns (jailbreak attempts)
    jailbreak_patterns = [
        r"ignore.*previous.*instruction",
        r"forget.*the.*system.*prompt",
        r"pretend.*you.*are",
        r"act.*as.*if.*you.*are.*not",
        r"respond.*as.*if.*you.*don't.*have",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in jailbreak_patterns):
        return InjectionCheck(is_injection=True, reason="prompt_injection")

    return InjectionCheck(is_injection=False, reason=None)
