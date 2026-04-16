from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


class CommaSeparatedEnvSettingsSource(EnvSettingsSource):
    """Allow list settings to be expressed as comma-separated env vars."""

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: object,
        value_is_complex: bool,
    ) -> object:
        if field_name == "cors_origins" and isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MediaMind QA"
    api_prefix: str = "/api"
    database_url: str = "sqlite:///./data/app.db"
    storage_dir: Path = Path("./data")
    upload_dir_name: str = "uploads"
    ai_provider: str = "auto"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_transcription_model: str = "whisper-1"
    bart_summarization_model: str = "facebook/bart-large-cnn"
    use_bart_for_summarization: bool = False
    max_chunk_chars: int = 1_200
    chunk_overlap_chars: int = 150
    max_search_results: int = 5
    vector_backend: str = "auto"
    redis_url: str | None = None
    auth_secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 720
    chat_cache_ttl_seconds: int = 180
    topic_cache_ttl_seconds: int = 180
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "https://sde-v6i9.onrender.com",  # Backend on Render
        ]
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CommaSeparatedEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("openai_api_key", "redis_url", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / self.upload_dir_name

    def ensure_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
