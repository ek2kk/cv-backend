from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    level: str = Field(default="INFO")
    file_path: str = Field(default="logs/app.log")
    max_bytes: int = Field(default=52_428_800)
    backup_count: int = Field(default=5, ge=1)


class OpenRouterSettings(BaseSettings):
    api_key: str = Field(default="")
    model: str = Field(default="openrouter/free")


class EmbeddingSettings(BaseSettings):
    model: str = Field(default="intfloat/multilingual-e5-small")


class RagSettings(BaseSettings):
    min_score: float = Field(default=0.83, ge=0, le=1)
    sources_count: int = Field(default=3, ge=1, le=10)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    app_name: str = Field(default="cv-rag")
    app_env: str = Field(default="dev")
    debug: bool = Field(default=False)

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rag: RagSettings = Field(default_factory=RagSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
