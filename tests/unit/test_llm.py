import json
from typing import Any

import httpx
import pytest
import structlog
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ai.config import Settings
from ai.llm.chat import DeepSeekChatClient
from ai.llm.embeddings import VoyageEmbeddingClient
from ai.llm.fakes import FakeChatClient, FakeEmbeddingClient
from ai.llm.runtime import create_chat_client, create_embedding_client


def _settings(
    *,
    environment: str = "dev",
    voyage_api_key: str = "",
    deepseek_api_key: str = "",
    provider_max_attempts: int = 3,
) -> Settings:
    return Settings(
        _env_file=None,
        environment=environment,
        voyage_api_key=voyage_api_key,
        deepseek_api_key=deepseek_api_key,
        provider_max_attempts=provider_max_attempts,
    )


def _voyage_embedding_response(count: int, dimension: int = 1024) -> dict[str, Any]:
    """Voyage API response shape: data is a list of {embedding, index}."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.25] * dimension, "index": i}
            for i in range(count)
        ],
        "model": "voyage-3.5",
        "usage": {"total_tokens": 7},
    }


@pytest.mark.asyncio
async def test_deepseek_chat_client_returns_message_and_logs_usage() -> None:
    settings = _settings(deepseek_api_key="test-key")
    client = DeepSeekChatClient(
        settings,
        model=FakeListChatModel(responses=["grounded answer"]),
    )

    with structlog.testing.capture_logs() as logs:
        result = await client.ainvoke("question")

    assert result.content == "grounded answer"
    assert logs[-1]["event"] == "llm.request.completed"
    assert logs[-1]["provider"] == "deepseek"
    assert logs[-1]["duration_ms"] >= 0
    assert "question" not in str(logs)


def test_chat_runtime_factory_selects_configured_model() -> None:
    settings = _settings(deepseek_api_key="test-key")

    client = create_chat_client(settings, purpose="fast")

    assert isinstance(client, DeepSeekChatClient)
    assert client.model_name == settings.llm_model_fast


def test_deepseek_client_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_chat_client(_settings())


@pytest.mark.asyncio
async def test_voyage_embeddings_use_configured_dimension_and_log_tokens() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_voyage_embedding_response(2))

    settings = _settings(environment="production", voyage_api_key="test-key")
    client = VoyageEmbeddingClient(
        settings,
        transport=httpx.MockTransport(handler),
    )

    with structlog.testing.capture_logs() as logs:
        vectors = await client.embed_documents(["first", "second"])

    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "voyage-3.5"
    assert body["input_type"] == "document"
    assert len(body["input"]) == 2
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert logs[-1]["input_tokens"] == 7
    assert logs[-1]["provider"] == "voyage"
    assert logs[-1]["duration_ms"] >= 0
    assert "first" not in str(logs)


@pytest.mark.asyncio
async def test_voyage_embeddings_retry_transient_http_errors() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_voyage_embedding_response(1))

    settings = _settings(
        environment="production",
        voyage_api_key="test-key",
        provider_max_attempts=2,
    )
    client = VoyageEmbeddingClient(
        settings,
        transport=httpx.MockTransport(handler),
    )

    vector = await client.embed_query("retry this request")

    assert calls == 2
    assert len(vector) == settings.embedding_dim


@pytest.mark.asyncio
async def test_voyage_embeddings_split_large_batches() -> None:
    batch_sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        batch_size = len(body["input"])
        batch_sizes.append(batch_size)
        return httpx.Response(200, json=_voyage_embedding_response(batch_size))

    client = VoyageEmbeddingClient(
        _settings(environment="production", voyage_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )

    vectors = await client.embed_documents(["text"] * 101)

    assert batch_sizes == [100, 1]
    assert len(vectors) == 101


@pytest.mark.asyncio
async def test_embedding_clients_reject_unexpected_vector_dimensions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=_voyage_embedding_response(1, dimension=3))

    client = VoyageEmbeddingClient(
        _settings(environment="production", voyage_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="expected 1024"):
        await client.embed_query("wrong dimension")


def test_voyage_client_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        create_embedding_client(_settings(environment="production"))


@pytest.mark.asyncio
async def test_test_fakes_return_expected_shapes() -> None:
    chat = FakeChatClient(response="fake response")
    embeddings = FakeEmbeddingClient(dimension=8)

    message = await chat.ainvoke("ignored")
    vector = await embeddings.embed_query("stable")
    documents = await embeddings.embed_documents(["one", "two"])

    assert message.content == "fake response"
    assert len(vector) == 8
    assert sum(vector) == 1
    assert len(documents) == 2
    assert await embeddings.embed_documents([]) == []