"""Gemini embedding adapter (single provider for all environments)."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import perf_counter
from typing import Any

import httpx

from ai.config import Settings
from ai.core.logging import get_logger
from ai.llm.retry import retrying
from ai.llm.rate_limit import SlidingWindowRateLimiter

logger = get_logger(__name__)

_GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_BATCH_SIZE = 100
_VOYAGE_API_ROOT = "https://api.voyageai.com/v1/embeddings"
_VOYAGE_MULTIMODAL_API_ROOT = "https://api.voyageai.com/v1/multimodalembeddings"
_VOYAGE_BATCH_SIZE = 100


class _EmbeddingClient(ABC):
    provider: str

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        timeout_seconds: float,
        max_attempts: int,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._transport = transport

    async def _post(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise TypeError(f"{self.provider} embedding response must be an object")
        return body

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def _validate_vectors(
        self, vectors: list[list[float]], expected_count: int
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ValueError(
                f"{self.provider} returned {len(vectors)} embeddings for "
                f"{expected_count} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"{self.provider} returned an embedding with {len(vector)} "
                    f"dimensions; expected {self.dimension}"
                )
        return vectors

    def _log_completion(
        self,
        *,
        operation: str,
        started_at: float,
        input_tokens: int | None,
    ) -> None:
        logger.info(
            "llm.request.completed",
            provider=self.provider,
            model=self.model,
            operation=operation,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
        )

    def _log_failure(
        self, *, operation: str, started_at: float, error: Exception
    ) -> None:
        logger.error(
            "llm.request.failed",
            provider=self.provider,
            model=self.model,
            operation=operation,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            error_type=type(error).__name__,
        )

class GeminiEmbeddingClient(_EmbeddingClient):
    provider = "gemini"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.gemini_api_key.get_secret_value():
            raise ValueError("GEMINI_API_KEY must be configured")
        super().__init__(
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_attempts=settings.provider_max_attempts,
            transport=transport,
        )
        self._api_key = settings.gemini_api_key
        self._limiter = SlidingWindowRateLimiter(
            max_units=settings.embedding_requests_per_minute,
            period_seconds=60.0,
            name="gemini_embeddings",
        )
        # A batch can never exceed the per-minute quota
        self._batch_size = min(_GEMINI_BATCH_SIZE, settings.embedding_requests_per_minute)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = perf_counter()
        url = f"{_GEMINI_API_ROOT}/{self.model}:batchEmbedContents"
        vectors: list[list[float]] = []
        total_tokens = 0
        has_token_counts = True
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": text}]},
                            "outputDimensionality": self.dimension,
                        }
                        for text in batch
                    ]
                }
                async for attempt in retrying(
                    attempts=self.max_attempts,
                    provider=self.provider,
                    model=self.model,
                    operation="embed_documents",
                ):
                    with attempt:
                        await self._limiter.acquire(len(batch))
                        response = await self._post(
                            url=url,
                            payload=payload,
                            headers={
                                "x-goog-api-key": self._api_key.get_secret_value()
                            },
                        )
                raw_embeddings = response.get("embeddings")
                if not isinstance(raw_embeddings, list):
                    raise TypeError("Gemini response is missing its embeddings list")
                batch_vectors = [
                    [float(value) for value in item["values"]]
                    for item in raw_embeddings
                    if isinstance(item, dict) and isinstance(item.get("values"), list)
                ]
                vectors.extend(self._validate_vectors(batch_vectors, len(batch)))
                usage = response.get("usageMetadata")
                batch_tokens = (
                    usage.get("promptTokenCount")
                    if isinstance(usage, dict)
                    and isinstance(usage.get("promptTokenCount"), int)
                    else None
                )
                if batch_tokens is None:
                    has_token_counts = False
                else:
                    total_tokens += batch_tokens
        except Exception as error:
            self._log_failure(
                operation="embed_documents",
                started_at=started_at,
                error=error,
            )
            raise

        self._log_completion(
            operation="embed_documents",
            started_at=started_at,
            input_tokens=total_tokens if has_token_counts else None,
        )
        return vectors

# class VoyageEmbeddingClient(_EmbeddingClient):
#     provider = "voyage"

#     def __init__(
#         self,
#         settings: Settings,
#         *,
#         transport: httpx.AsyncBaseTransport | None = None,
#     ) -> None:
#         if not settings.voyage_api_key.get_secret_value():
#             raise ValueError("VOYAGE_API_KEY must be configured")
#         super().__init__(
#             model=settings.embedding_model,
#             dimension=settings.embedding_dim,
#             timeout_seconds=settings.embedding_timeout_seconds,
#             max_attempts=settings.provider_max_attempts,
#             transport=transport,
#         )
#         self._api_key = settings.voyage_api_key
#         self._limiter = SlidingWindowRateLimiter(
#             max_units=settings.embedding_requests_per_minute,
#             period_seconds=60.0,
#             name="voyage_embeddings",
#         )
#         self._batch_size = min(_VOYAGE_BATCH_SIZE, settings.embedding_requests_per_minute)

#     async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
#         if not texts:
#             return []
#         started_at = perf_counter()
#         vectors: list[list[float]] = []
#         total_tokens = 0
#         try:
#             for start in range(0, len(texts), self._batch_size):
#                 batch = texts[start : start + self._batch_size]
#                 payload = {
#                     "input": list(batch),
#                     "model": self.model,
#                     "input_type": "document",
#                 }
#                 async for attempt in retrying(
#                     attempts=self.max_attempts,
#                     provider=self.provider,
#                     model=self.model,
#                     operation="embed_documents",
#                 ):
#                     with attempt:
#                         await self._limiter.acquire(len(batch))
#                         response = await self._post(
#                             url=_VOYAGE_API_ROOT,
#                             payload=payload,
#                             headers={
#                                 "Authorization": f"Bearer {self._api_key.get_secret_value()}"
#                             },
#                         )
#                 raw_data = response.get("data")
#                 if not isinstance(raw_data, list):
#                     raise TypeError("Voyage response is missing its data list")
#                 # Sort by index to preserve input order (Voyage may return out of order)
#                 sorted_data = sorted(raw_data, key=lambda x: x.get("index", 0))
#                 batch_vectors = [
#                     [float(v) for v in item["embedding"]]
#                     for item in sorted_data
#                     if isinstance(item, dict) and isinstance(item.get("embedding"), list)
#                 ]
#                 vectors.extend(self._validate_vectors(batch_vectors, len(batch)))
#                 usage = response.get("usage")
#                 batch_tokens = (
#                     usage.get("total_tokens")
#                     if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int)
#                     else None
#                 )
#                 if batch_tokens:
#                     total_tokens += batch_tokens
#         except Exception as error:
#             self._log_failure(
#                 operation="embed_documents",
#                 started_at=started_at,
#                 error=error,
#             )
#             raise

#         self._log_completion(
#             operation="embed_documents",
#             started_at=started_at,
#             input_tokens=total_tokens if total_tokens else None,
#         )
#         return vectors

#     async def embed_query(self, text: str) -> list[float]:
#         started_at = perf_counter()
#         payload = {
#             "input": [text],
#             "model": self.model,
#             "input_type": "query",
#         }
#         try:
#             async for attempt in retrying(
#                 attempts=self.max_attempts,
#                 provider=self.provider,
#                 model=self.model,
#                 operation="embed_query",
#             ):
#                 with attempt:
#                     await self._limiter.acquire(1)
#                     response = await self._post(
#                         url=_VOYAGE_API_ROOT,
#                         payload=payload,
#                         headers={
#                             "Authorization": f"Bearer {self._api_key.get_secret_value()}"
#                         },
#                     )
#         except Exception as error:
#             self._log_failure(
#                 operation="embed_query",
#                 started_at=started_at,
#                 error=error,
#             )
#             raise

#         raw_data = response.get("data")
#         if not isinstance(raw_data, list) or len(raw_data) != 1:
#             raise TypeError("Voyage query response malformed")

#         usage = response.get("usage")
#         tokens = (
#             usage.get("total_tokens")
#             if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int)
#             else None
#         )
#         self._log_completion(
#             operation="embed_query",
#             started_at=started_at,
#             input_tokens=tokens,
#         )
#         return [float(v) for v in raw_data[0]["embedding"]]

class VoyageEmbeddingClient(_EmbeddingClient):
    provider = "voyage"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.voyage_api_key.get_secret_value():
            raise ValueError("VOYAGE_API_KEY must be configured")
        super().__init__(
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_attempts=settings.provider_max_attempts,
            transport=transport,
        )
        self._api_key = settings.voyage_api_key
        self._limiter = SlidingWindowRateLimiter(
            max_units=settings.embedding_requests_per_minute,
            period_seconds=60.0,
            name="voyage_embeddings",
        )
        self._batch_size = min(_VOYAGE_BATCH_SIZE, settings.embedding_requests_per_minute)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = perf_counter()
        vectors: list[list[float]] = []
        total_tokens = 0
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                multimodal_inputs = []
                for item in batch:
                    # Check if text string encodes an inline image specifier
                    if isinstance(item, str) and item.startswith("data:image/"):
                        multimodal_inputs.append({
                            "content": [{"type": "image_base64", "image_base64": item}]
                        })
                    else:
                        multimodal_inputs.append({
                            "content": [{"type": "text", "text": item}]
                        })

                payload = {
                    "inputs": multimodal_inputs,
                    "model": self.model,
                    "input_type": "document",
                }
                async for attempt in retrying(
                    attempts=self.max_attempts,
                    provider=self.provider,
                    model=self.model,
                    operation="embed_documents",
                ):
                    with attempt:
                        await self._limiter.acquire(len(batch))
                        response = await self._post(
                            url=_VOYAGE_MULTIMODAL_API_ROOT,
                            payload=payload,
                            headers={
                                "Authorization": f"Bearer {self._api_key.get_secret_value()}"
                            },
                        )
                raw_data = response.get("data")
                if not isinstance(raw_data, list):
                    raise TypeError("Voyage response is missing its data list")
                sorted_data = sorted(raw_data, key=lambda x: x.get("index", 0))
                batch_vectors = [
                    [float(v) for v in item["embedding"]]
                    for item in sorted_data
                    if isinstance(item, dict) and isinstance(item.get("embedding"), list)
                ]
                vectors.extend(self._validate_vectors(batch_vectors, len(batch)))
                usage = response.get("usage")
                batch_tokens = (
                    usage.get("total_tokens")
                    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int)
                    else None
                )
                if batch_tokens:
                    total_tokens += batch_tokens
        except Exception as error:
            self._log_failure(
                operation="embed_documents",
                started_at=started_at,
                error=error,
            )
            raise

        self._log_completion(
            operation="embed_documents",
            started_at=started_at,
            input_tokens=total_tokens if total_tokens else None,
        )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        started_at = perf_counter()
        payload = {
            "inputs": [{"content": [{"type": "text", "text": text}]}],
            "model": self.model,
            "input_type": "query",
        }
        try:
            async for attempt in retrying(
                attempts=self.max_attempts,
                provider=self.provider,
                model=self.model,
                operation="embed_query",
            ):
                with attempt:
                    await self._limiter.acquire(1)
                    response = await self._post(
                        url=_VOYAGE_MULTIMODAL_API_ROOT,
                        payload=payload,
                        headers={
                            "Authorization": f"Bearer {self._api_key.get_secret_value()}"
                        },
                    )
        except Exception as error:
            self._log_failure(
                operation="embed_query",
                started_at=started_at,
                error=error,
            )
            raise

        raw_data = response.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != 1:
            raise TypeError("Voyage query response malformed")

        usage = response.get("usage")
        tokens = (
            usage.get("total_tokens")
            if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int)
            else None
        )
        self._log_completion(
            operation="embed_query",
            started_at=started_at,
            input_tokens=tokens,
        )
        return [float(v) for v in raw_data[0]["embedding"]]
