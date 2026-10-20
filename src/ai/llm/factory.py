"""LLM factory functions for dependency injection."""

from ai.config import settings
from ai.llm.chat import DeepSeekChatClient
from ai.llm.embeddings import VoyageEmbeddingClient
from ai.llm.interfaces import ChatClient, EmbeddingClient


def get_chat_client() -> ChatClient:
    """Get the configured chat client."""
    return DeepSeekChatClient(settings)


def get_embedding_client() -> EmbeddingClient:
    """Get the configured embedding client (Voyage in all environments)."""
    return VoyageEmbeddingClient(settings)


__all__ = ["get_chat_client", "get_embedding_client"]
