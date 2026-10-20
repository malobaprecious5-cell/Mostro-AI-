"""Small provider-neutral interfaces used by runtime code and test fakes."""

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage


class ChatClient(Protocol):
    async def ainvoke(self, input: str | Sequence[BaseMessage]) -> AIMessage: ...


class EmbeddingClient(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
