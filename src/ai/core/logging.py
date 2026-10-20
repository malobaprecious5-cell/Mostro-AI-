"""Structured logging setup using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from ai.config import settings


def _renderer() -> Processor:
    if settings.environment == "production":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer()


def configure_logging() -> None:
    """Configure stdlib and structlog with request-context support."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.upper(),
        force=True,
    )

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_renderer(),
        foreign_pre_chain=shared_processors,
    )
    logging.root.handlers[0].setFormatter(formatter)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
