from ai.db.models import VECTOR_DIMENSION, Base


def test_initial_schema_contains_core_and_api_required_tables() -> None:
    assert set(Base.metadata.tables) == {
        "ai.ipo",
        "ai.document",
        "ai.chunk",
        "ai.key_fact",
        "ai.section_summary",
        "ai.glossary_term",
        "ai.chat_session",
        "ai.message",
        "ai.message_citation",
        "ai.calendar_proposal",
        "ai.ingestion_job",
    }
    assert VECTOR_DIMENSION == 1536


def test_chat_message_supports_idempotency_and_feedback_contract() -> None:
    message_columns = Base.metadata.tables["ai.message"].columns
    assert "client_message_id" in message_columns
    assert "feedback_reason" in message_columns
    assert "feedback_comment" in message_columns
    assert any(
        constraint.name == "message_session_client_message_key"
        for constraint in Base.metadata.tables["ai.message"].constraints
    )
