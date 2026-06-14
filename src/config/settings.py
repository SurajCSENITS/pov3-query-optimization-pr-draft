"""
Application settings loaded from environment variables.

Uses Pydantic BaseSettings to:
- Read from .env file automatically
- Validate required fields at startup
- Provide typed access to configuration values

Usage:
    from src.config.settings import get_settings
    settings = get_settings()
    print(settings.snowflake_account)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the POV3 application.

    All values are read from environment variables or a .env file.
    Fields with defaults are optional; fields without defaults are
    required and will raise a validation error if missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Snowflake Connection ────────────────────────────────────
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_password: str = ""
    snowflake_warehouse: str = "COMPUTE_WH"
    snowflake_database: str = "ANALYTICS_DB"
    snowflake_schema: str = "PUBLIC"
    snowflake_role: str = "POV3_AGENT_ROLE"

    # ── FastAPI Server ──────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Feature Flags ───────────────────────────────────────────
    snowflake_enabled: bool = False

    @property
    def snowflake_configured(self) -> bool:
        """Check if minimum Snowflake credentials are provided."""
        return bool(
            self.snowflake_account
            and self.snowflake_user
            and self.snowflake_password
            and self.snowflake_enabled
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is only parsed once
    per process, regardless of how many modules import this.
    """
    return Settings()
