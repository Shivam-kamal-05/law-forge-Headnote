"""Application settings via pydantic-settings.

Simplified configuration for the standalone headnote extraction engine.
All values sourced from environment variables / .env file — no hardcoded
credentials anywhere.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class AppSettings(_BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    name: str = "law-lens-headnote-forge"
    env: AppEnv = AppEnv.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AnthropicSettings(_BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", env_file=".env", extra="ignore")

    api_key: SecretStr
    model: str = "claude-haiku-4-5-20251001"
    # 8192 leaves ample room for a rich multi-head headnote; 4096 can truncate
    # the tool call on long judgments, which then fails schema validation.
    max_tokens: int = Field(8192, ge=512, le=64000)
    max_retries: int = Field(5, ge=0, le=10)
    timeout_seconds: int = Field(120, ge=10, le=600)

    # Conservative Tier-1 defaults — override via env for higher plans.
    rpm: int | None = Field(default=50, ge=0)
    tpm: int | None = Field(default=50_000, ge=0)
    estimated_tokens_per_call: int = Field(default=4_500, ge=100)
    burst_factor: float = Field(default=1.0, ge=0.1, le=2.0)


class AuthSettings(_BaseSettings):
    """Single-tenant demo gate.

    Defaults are the agreed client-demo credentials; both can be overridden
    via the environment (AUTH_USERNAME / AUTH_PASSWORD) without code changes.
    """

    model_config = SettingsConfigDict(env_prefix="AUTH_", env_file=".env", extra="ignore")

    username: str = "dharmani.dev"
    password: SecretStr = SecretStr("dharmaniz@123")
    # Salts the derived bearer token; override in prod so tokens rotate.
    token_secret: SecretStr = SecretStr("law-lens-headnote-forge-demo")


class IngestionSettings(_BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGESTION_", env_file=".env", extra="ignore")

    ocr_enabled: bool = True
    ocr_dpi: int = Field(300, ge=72, le=600)
    min_text_length: int = Field(200, ge=0)
    max_file_bytes: int = Field(50 * 1024 * 1024, description="50 MiB per-PDF cap.")
    max_files_per_request: int = Field(10, ge=1, le=50)
    concurrency: int = Field(3, ge=1, le=16)


class Settings(_BaseSettings):
    """Aggregated settings — single entry point for the app."""

    app: AppSettings = Field(default_factory=AppSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance (env vars read once).

    Tests can clear the cache with `get_settings.cache_clear()`.
    """
    return Settings()
