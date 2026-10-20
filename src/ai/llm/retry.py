"""Shared retry policy for transient provider failures."""

from collections.abc import Callable
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ai.core.logging import get_logger

logger = get_logger(__name__)


def _is_transient_error(error: BaseException) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 429} or (
            error.response.status_code >= 500
        )
    if isinstance(error, (APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 429} or error.status_code >= 500
    return False


def _log_retry(
    state: RetryCallState,
    *,
    provider: str,
    model: str,
    operation: str,
) -> None:
    error = state.outcome.exception() if state.outcome is not None else None
    logger.warning(
        "llm.request.retrying",
        provider=provider,
        model=model,
        operation=operation,
        attempt=state.attempt_number,
        error_type=type(error).__name__ if error else None,
    )


def retrying(
    *,
    attempts: int,
    provider: str,
    model: str,
    operation: str,
) -> AsyncRetrying:
    before_sleep: Callable[[RetryCallState], Any] = lambda state: _log_retry(
        state,
        provider=provider,
        model=model,
        operation=operation,
    )
    return AsyncRetrying(
        retry=retry_if_exception(_is_transient_error),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
        before_sleep=before_sleep,
        reraise=True,
    )
