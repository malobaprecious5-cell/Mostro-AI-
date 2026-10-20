import json
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic import SecretStr, TypeAdapter

from ai.api.schemas.chat import ChatStreamEvent
from ai.config import settings
from ai.main import app

client = TestClient(app)
event_adapter = TypeAdapter(ChatStreamEvent)


def _auth_headers() -> dict[str, str]:
    return {
        "X-Internal-Key": "test-internal-key",
        "X-User-Id": str(uuid4()),
    }


def test_health_is_public_and_returns_liveness_shape() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_chat_contract_is_exposed_in_openapi() -> None:
    contract = client.get("/openapi.json").json()
    operation = contract["paths"]["/v1/sessions/{session_id}/messages"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    response_content = operation["responses"]["200"]["content"]

    assert request_schema["$ref"] == "#/components/schemas/ChatRequest"
    assert (
        response_content["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ChatMessage"
    )
    assert "text/event-stream" in response_content


def test_chat_stream_matches_frozen_sse_contract(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", SecretStr("test-internal-key"))
    
    # Mock the ChatService.stream_message to return stub responses
    async def mock_stream_message(**kwargs):
        yield 'event: token\ndata: {"delta": "This is the Mostro AI chat stub. "}\n\n'
        yield 'event: token\ndata: {"delta": "I cannot answer IPO-specific questions until the prospectus service is connected."}\n\n'
        yield 'event: token\ndata: {"delta": ""}\n\n'
        yield 'event: done\ndata: {"outcome": "abstained", "message_id": "12345678-1234-5678-1234-567812345678", "created_at": "2026-10-12T03:46:00+00:00"}\n\n'
    
    session_id = uuid4()
    with patch("ai.api.routes.chat.ChatService") as mock_service_class:
        mock_service = AsyncMock()
        mock_service.stream_message = mock_stream_message
        mock_service_class.return_value = mock_service
        
        response = client.post(
            f"/v1/sessions/{session_id}/messages",
            headers=_auth_headers(),
            json={
                "content": "When does the offer close?",
                "client_message_id": str(uuid4()),
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed_events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", maxsplit=1) for line in block.splitlines())
        event = event_adapter.validate_python(
            {"event": lines["event"], "data": json.loads(lines["data"])}
        )
        parsed_events.append(event)

    assert [event.event for event in parsed_events] == [
        "token",
        "token",
        "token",
        "done",
    ]
    done = parsed_events[-1].data
    assert done.outcome == "abstained"
    assert done.message_id
    assert "".join(event.data.delta for event in parsed_events[:-1]) == (
        "This is the Mostro AI chat stub. "
        "I cannot answer IPO-specific questions until the prospectus service is connected."
    )
    assert str(session_id) in response.request.url.path


def test_chat_json_response_has_same_message_shape(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", SecretStr("test-internal-key"))
    
    # Mock the ChatService.stream_message to return stub responses
    async def mock_stream_message(**kwargs):
        yield 'event: token\ndata: {"delta": "This is the Mostro AI chat stub. "}\n\n'
        yield 'event: token\ndata: {"delta": "I cannot answer IPO-specific questions until the prospectus service is connected."}\n\n'
        yield 'event: token\ndata: {"delta": ""}\n\n'
        yield 'event: done\ndata: {"outcome": "abstained", "message_id": "12345678-1234-5678-1234-567812345678", "created_at": "2026-10-12T03:46:00+00:00"}\n\n'
    
    with patch("ai.api.routes.chat.ChatService") as mock_service_class:
        mock_service = AsyncMock()
        mock_service.stream_message = mock_stream_message
        mock_service_class.return_value = mock_service
        
        response = client.post(
            f"/v1/sessions/{uuid4()}/messages",
            headers=_auth_headers(),
            json={
                "content": "Explain the offer.",
                "client_message_id": str(uuid4()),
                "stream": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["outcome"] == "abstained"
    assert data["citations"] == []
    assert data["proposals"] == []


def test_invalid_chat_request_uses_api_error_envelope(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", SecretStr("test-internal-key"))
    response = client.post(
        f"/v1/sessions/{uuid4()}/messages",
        headers={**_auth_headers(), "X-Request-Id": "test-request"},
        json={"content": " ", "client_message_id": str(uuid4())},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["request_id"] == "test-request"
    assert response.headers["x-request-id"] == "test-request"


def test_chat_requires_internal_authentication(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", SecretStr(""))
    response = client.post(
        f"/v1/sessions/{uuid4()}/messages",
        headers={"X-User-Id": str(uuid4())},
        json={"content": "Hello", "client_message_id": str(uuid4())},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "NOT_READY"


def test_all_five_sse_event_payloads_validate() -> None:
    chunk_id = str(uuid4())
    key_fact_id = str(uuid4())
    proposal_id = str(uuid4())
    common = {
        "event": "citation",
        "data": {
            "index": 1,
            "chunk_id": chunk_id,
            "page_start": 2,
            "page_end": 3,
            "section_path": "Offer details",
            "quote": "Source quote",
        },
    }
    proposal = {
        "event": "proposal",
        "data": {
            "proposal_id": proposal_id,
            "title": "Offer closes",
            "starts_at": "2026-03-10T09:00:00+01:00",
            "timezone": "Africa/Lagos",
            "key_fact_id": key_fact_id,
        },
    }
    error = {
        "event": "error",
        "data": {"code": "UPSTREAM_LLM_ERROR", "message": "The answer failed."},
    }

    assert event_adapter.validate_python(common).event == "citation"
    assert event_adapter.validate_python(proposal).event == "proposal"
    assert event_adapter.validate_python(error).event == "error"
