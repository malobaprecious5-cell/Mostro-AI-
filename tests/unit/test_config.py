import pytest

from ai.config import Settings


@pytest.mark.parametrize("environment", ["dev", "test", "staging", "production"])
def test_all_environments_use_voyage_embedding_settings(environment: str) -> None:
    settings = Settings(_env_file=None, environment=environment)

    assert settings.embedding_provider == "voyage"
    assert settings.embedding_model == "voyage-3.5"


def test_voyage_api_key_is_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")

    settings = Settings(_env_file=None, environment="production")

    assert settings.voyage_api_key.get_secret_value() == "test-voyage-key"


def test_embedding_dimension_and_deepseek_models_remain_unchanged() -> None:
    settings = Settings(_env_file=None, environment="dev")

    assert settings.embedding_dim == 1536
    assert settings.llm_model_fast == "deepseek-v4-flash"
    assert settings.llm_model_main == "deepseek-v4-pro"


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("llm_timeout_seconds", 0),
        ("embedding_timeout_seconds", -1),
        ("llm_timeout_seconds", float("nan")),
        ("provider_max_attempts", 0),
    ],
)
def test_provider_retry_settings_must_be_positive(setting: str, value: int) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{setting: value})
