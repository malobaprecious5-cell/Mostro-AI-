"""Deterministic provider fakes for tests and local graph development."""

from collections.abc import Sequence
from hashlib import sha256

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage


class FakeChatClient:
    def __init__(self, response: str = "") -> None:
        self.response = response

    async def ainvoke(self, input: str | Sequence[BaseMessage]) -> AIMessage:
        _ = input
        return AIMessage(
            content=self.response,
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )


class FakeEmbeddingClient:
    def __init__(self, dimension: int = 1024) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self.dimension = dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        index = (
            int.from_bytes(sha256(text.encode()).digest()[:4], "big") % self.dimension
        )
        vector[index] = 1.0
        return vector
