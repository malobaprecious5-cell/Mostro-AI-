"""Factories for the configured runtime providers."""

from typing import Literal

from ai.config import Settings, get_settings
from ai.llm.chat import DeepSeekChatClient
from ai.llm.embeddings import VoyageEmbeddingClient
from ai.llm.interfaces import ChatClient, EmbeddingClient


def create_chat_client(
    settings: Settings | None = None,
    *,
    purpose: Literal["fast", "main"] = "main",
) -> ChatClient:
    configured = settings or get_settings()
    model = (
        configured.llm_model_fast if purpose == "fast" else configured.llm_model_main
    )
    return DeepSeekChatClient(configured, model_name=model)


def create_embedding_client(
    settings: Settings | None = None,
) -> EmbeddingClient:
    """Always returns a VoyageEmbeddingClient (single provider for all environments)."""
    configured = settings or get_settings()
    return VoyageEmbeddingClient(configured)
