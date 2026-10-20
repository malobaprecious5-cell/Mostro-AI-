"""Application settings loaded from environment variables."""

from functools import lru_cache
from math import isfinite
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Mostro AI"
    environment: Literal["dev", "test", "staging", "production"] = Field(
        default="dev",
        validation_alias=AliasChoices("ENVIRONMENT", "ENV"),
    )
    log_level: str = "INFO"
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://ai:ai@localhost:5432/ai_service"
    )
    internal_api_key: SecretStr = SecretStr("")
    admin_api_key: SecretStr = SecretStr("")
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model_fast: str = "deepseek-v4-flash"
    llm_model_main: str = "deepseek-v4-pro"
    voyage_api_key: SecretStr = SecretStr("")

    llm_timeout_seconds: float = 45.0
    embedding_model: str = "voyage-multimodal-3.5"
    embedding_timeout_seconds: float = 45.0
    embedding_requests_per_minute: int = Field(default=500, ge=1)
    provider_max_attempts: int = 4
    embedding_dim: int = 1024
    retrieval_candidates: int = 30
    rerank_top_k: int = 6
    abstain_score_threshold: float = 0.3
    enable_retrieval_debug: bool = False
    required_key_facts: list[str] = Field(default_factory=list)
    document_storage_dir: str = "data/documents"
    build: str = "dev"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "http://localhost:3000"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    @property
    def embedding_provider(self) -> str:
        # Voyage is used in all environments (dev, test, staging, production).
        return "voyage"

    @field_validator("llm_timeout_seconds", "embedding_timeout_seconds")
    @classmethod
    def validate_provider_timeout(cls, value: float) -> float:
        if not isfinite(value) or value <= 0:
            raise ValueError("Provider timeouts must be greater than zero")
        return value

    @field_validator("provider_max_attempts")
    @classmethod
    def validate_provider_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("PROVIDER_MAX_ATTEMPTS must be at least 1")
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("DATABASE_URL must not be empty")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, value: int) -> int:
        if value != 1024:
            raise ValueError(
                "EMBEDDING_DIM must remain 1024 unless the vector column and "
                "Alembic schema are migrated together"
            )
        return value

    @field_validator("abstain_score_threshold")
    @classmethod
    def validate_abstain_score_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("ABSTAIN_SCORE_THRESHOLD must be between 0 and 1")
        return value

    @property
    def sqlalchemy_database_url(self) -> str:
        url = self.database_url.get_secret_value()
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
