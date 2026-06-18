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
    bedrock_enabled: bool = False

    # ── AWS Credentials ─────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # ── Bedrock Model Configuration ─────────────────────────────
    bedrock_model_id: str = "amazon.nova-pro-v1:0"
    bedrock_screener_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_max_tokens: int = 4096

    # ── RAG / Knowledge Base ────────────────────────────────────
    bedrock_kb_id: str = ""
    bedrock_data_source_id: str = ""
    s3_bucket_name: str = "pov3-optimization-reports"
    s3_reports_prefix: str = "reports/"

    # ── Observability ───────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived properties ──────────────────────────────────────

    @property
    def snowflake_configured(self) -> bool:
        """Check if minimum Snowflake credentials are provided."""
        return bool(
            self.snowflake_account
            and self.snowflake_user
            and self.snowflake_password
            and self.snowflake_enabled
        )

    @property
    def bedrock_configured(self) -> bool:
        """Check if minimum AWS Bedrock credentials are provided."""
        return bool(
            self.aws_access_key_id
            and self.aws_secret_access_key
            and self.bedrock_model_id
            and self.bedrock_enabled
        )

    @property
    def rag_configured(self) -> bool:
        """Check if RAG Knowledge Base is configured."""
        return bool(self.bedrock_configured and self.bedrock_kb_id)


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is only parsed once
    per process, regardless of how many modules import this.
    """
    return Settings()
