"""
AWS Bedrock Connection Manager — LangChain ChatBedrock Integration.

Replaces the custom boto3 wrapper with LangChain's ChatBedrock,
providing native support for structured outputs, automatic retries,
and seamless LangSmith tracing.

Usage:
    from src.connectors.bedrock_manager import get_llm, get_screener_llm

    # For structured JSON output:
    llm = get_llm().with_structured_output(MyPydanticModel)
    result = llm.invoke("Optimize this SQL...")

    # For the fast screener (Nova Lite):
    screener = get_screener_llm().with_structured_output(SemanticCheckResult)
    check = screener.invoke("Are these two queries equivalent?")
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_aws import ChatBedrock

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def _create_chat_bedrock(
    model_id: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> ChatBedrock:
    """
    Create a configured ChatBedrock instance.

    Uses AWS credentials from application settings and returns
    a LangChain-compatible chat model that supports:
    - .invoke() for raw text responses
    - .with_structured_output(PydanticModel) for schema-validated JSON
    - Automatic LangSmith tracing when LANGSMITH_API_KEY is set
    """
    settings = get_settings()

    return ChatBedrock(
        model_id=model_id,
        region_name=settings.aws_region,
        credentials_profile_name=None,
        model_kwargs={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def get_llm(
    model_id: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> ChatBedrock:
    """
    Return a ChatBedrock instance for the primary optimization LLM.

    Args:
        model_id:    Override model (defaults to settings.bedrock_model_id).
        temperature: Sampling temperature (default 0.1 for consistency).
        max_tokens:  Max response tokens (defaults to settings.bedrock_max_tokens).

    Returns:
        A ChatBedrock instance ready for .invoke() or .with_structured_output().
    """
    settings = get_settings()
    return _create_chat_bedrock(
        model_id=model_id or settings.bedrock_model_id,
        max_tokens=max_tokens or settings.bedrock_max_tokens,
        temperature=temperature,
    )


def get_screener_llm(
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> ChatBedrock:
    """
    Return a ChatBedrock instance for the fast screener LLM (Nova Lite).

    Used for semantic equivalence checks and quick classification tasks.
    Lower temperature (0.0) for deterministic validation decisions.
    """
    settings = get_settings()
    return _create_chat_bedrock(
        model_id=settings.bedrock_screener_model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
