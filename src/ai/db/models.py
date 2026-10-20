"""SQLAlchemy models for the AI service's Postgres schema."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

AI_SCHEMA = "ai"
VECTOR_DIMENSION = 1024


class Base(DeclarativeBase):
    metadata = MetaData(schema=AI_SCHEMA)


class IPO(Base):
    __tablename__ = "ipo"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'review', 'published', 'archived')",
            name="ipo_status_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    issuer_name: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str] = mapped_column(CHAR(2), nullable=False)
    exchange: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'draft'")
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("ipo_id", "version", name="document_ipo_version_key"),
        UniqueConstraint("ipo_id", "file_hash", name="document_ipo_file_hash_key"),
        CheckConstraint(
            "doc_type IN ('prospectus', 'addendum', 'supplement')",
            name="document_doc_type_check",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'parsing', 'indexed', 'failed', 'superseded')",
            name="document_status_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.ipo.id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    doc_type: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'prospectus'")
    )
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    parser: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'uploaded'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="chunk_document_index_key"),
        Index("chunk_ipo_idx", "ipo_id"),
        Index("chunk_doc_order_idx", "document_id", "chunk_index"),
        Index("chunk_fts_idx", "fts", postgresql_using="gin"),
        Index(
            "chunk_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        CheckConstraint(
            "chunk_type IN ('text', 'table', 'heading', 'list', 'image')",
            name="chunk_type_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.ipo.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.document.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'text'")
    )
    section_path: Mapped[str | None] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embed_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(VECTOR_DIMENSION), nullable=False
    )
    fts: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('ai.fts_unaccent', "
            "coalesce(section_path, '') || ' ' || content)",
            persisted=True,
        ),
    )


class KeyFact(Base):
    __tablename__ = "key_fact"
    __table_args__ = (
        UniqueConstraint(
            "ipo_id", "fact_key", "document_id", name="key_fact_ipo_fact_doc_key"
        ),
        CheckConstraint(
            "review_status IN ('pending', 'verified', 'rejected')",
            name="key_fact_review_status_check",
        ),
        Index(
            "key_fact_ipo_idx",
            "ipo_id",
            postgresql_where=text("review_status = 'verified'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.ipo.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(f"{AI_SCHEMA}.document.id"), nullable=False
    )
    fact_key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
    source_chunk_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.chunk.id", ondelete="SET NULL"),
    )
    source_quote: Mapped[str | None] = mapped_column(Text)
    extracted_by: Mapped[str | None] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SectionSummary(Base):
    __tablename__ = "section_summary"
    __table_args__ = (
        UniqueConstraint(
            "ipo_id",
            "section_key",
            "document_id",
            name="section_summary_ipo_key_doc_key",
        ),
        CheckConstraint(
            "review_status IN ('pending', 'verified', 'rejected')",
            name="section_summary_review_status_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.ipo.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(f"{AI_SCHEMA}.document.id"), nullable=False
    )
    section_key: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    review_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GlossaryTerm(Base):
    __tablename__ = "glossary_term"
    __table_args__ = (
        UniqueConstraint("term", "language", name="glossary_term_language_key"),
        Index(
            "glossary_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    term: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'beginner'")
    )
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    example: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(VECTOR_DIMENSION))


class ChatSession(Base):
    __tablename__ = "chat_session"
    __table_args__ = (Index("chat_session_user_idx", "user_id", "last_active"),)

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(f"{AI_SCHEMA}.ipo.id"), nullable=False
    )
    language: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'en'")
    )
    level: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'beginner'")
    )
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Message(Base):
    __tablename__ = "message"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "client_message_id",
            name="message_session_client_message_key",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')", name="message_role_check"
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN "
            "('answered', 'abstained', 'refused', 'blocked', 'error')",
            name="message_outcome_check",
        ),
        CheckConstraint(
            "feedback IS NULL OR feedback IN (-1, 0, 1)",
            name="message_feedback_check",
        ),
        Index("message_session_idx", "session_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.chat_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    client_message_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str | None] = mapped_column(Text)
    trace: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    feedback: Mapped[int | None] = mapped_column(SmallInteger)
    feedback_reason: Mapped[str | None] = mapped_column(Text)
    feedback_comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class MessageCitation(Base):
    __tablename__ = "message_citation"
    __table_args__ = (Index("message_citation_msg_idx", "message_id"),)

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.message.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(f"{AI_SCHEMA}.chunk.id"), nullable=False
    )
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str | None] = mapped_column(Text)


class CalendarProposal(Base):
    __tablename__ = "calendar_proposal"
    __table_args__ = (
        CheckConstraint(
            "status IN ('proposed', 'confirmed', 'rejected', 'created', 'failed')",
            name="calendar_proposal_status_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    message_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.message.id", ondelete="CASCADE"),
    )
    session_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.chat_session.id", ondelete="CASCADE"),
    )
    user_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    key_fact_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(f"{AI_SCHEMA}.key_fact.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str] = mapped_column(Text, nullable=False)
    reminders: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'proposed'")
    )
    external_event_id: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    source_page: Mapped[int | None] = mapped_column(Integer)


class IngestionJob(Base):
    __tablename__ = "ingestion_job"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ingestion_job_status_check",
        ),
        CheckConstraint(
            "step IS NULL OR step IN ('parse', 'chunk', 'embed', 'facts', 'summaries')",
            name="ingestion_job_step_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.document.id", ondelete="CASCADE"),
        nullable=False,
    )
    ipo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(f"{AI_SCHEMA}.ipo.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'queued'")
    )
    step: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
