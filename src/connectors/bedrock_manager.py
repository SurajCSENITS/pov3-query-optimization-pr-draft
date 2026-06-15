"""
AWS Bedrock Connection Manager.

Provides a singleton client for:
- Invoking Amazon Nova Pro (primary optimization LLM)
- Invoking Amazon Nova Lite (fast screener / validation LLM)
- Structured prompt building for SQL optimization tasks

All model invocations use the Amazon Nova message format.
Claude/Anthropic format is NOT used here.

Usage:
    from src.connectors.bedrock_manager import get_bedrock_manager
    manager = get_bedrock_manager()
    response = manager.invoke(prompt="Optimize this SQL: SELECT * FROM orders")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class BedrockManager:
    """
    Singleton wrapper around boto3 bedrock-runtime client.

    Handles:
    - Client creation with credentials from settings
    - Nova Pro invocations (primary LLM)
    - Nova Lite invocations (fast screener)
    - JSON response parsing
    """

    _instance: BedrockManager | None = None
    _client: Any = None

    def __new__(cls) -> BedrockManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_client(self) -> Any:
        """Lazily create and return the boto3 bedrock-runtime client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
            logger.info(
                "Bedrock runtime client created (region=%s)", self._settings.aws_region
            )
        return self._client

    # ── Core invocation ────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "You are an expert Snowflake SQL optimization assistant.",
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Invoke a Bedrock model with a user prompt.

        Uses Amazon Nova message format (system + messages structure).
        Returns the model's text response as a string.

        Args:
            prompt: The user message content.
            system_prompt: Optional system instruction override.
            model_id: Model to use (defaults to primary bedrock_model_id).
            max_tokens: Max tokens for response (defaults to settings value).
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The model's text response string.

        Raises:
            RuntimeError: If invocation fails.
        """
        client = self._get_client()
        model = model_id or self._settings.bedrock_model_id
        tokens = max_tokens or self._settings.bedrock_max_tokens

        payload = {
            "system": [{"text": system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": tokens,
                "temperature": temperature,
            },
        }

        try:
            response = client.invoke_model(
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(response["body"].read())
            text = result["output"]["message"]["content"][0]["text"]
            input_tokens = result.get("usage", {}).get("inputTokens", 0)
            output_tokens = result.get("usage", {}).get("outputTokens", 0)
            logger.info(
                "Bedrock invoke OK model=%s tokens=in:%d/out:%d",
                model,
                input_tokens,
                output_tokens,
            )
            return text
        except Exception as e:
            logger.error("Bedrock invocation failed model=%s: %s", model, e)
            raise RuntimeError(f"Bedrock invoke failed: {e}") from e

    def invoke_screener(
        self,
        prompt: str,
        system_prompt: str = "You are a SQL semantic analysis assistant. Be concise and precise.",
        max_tokens: int = 512,
    ) -> str:
        """
        Invoke the fast screener model (Nova Lite) for quick checks.

        Used for semantic equivalence validation and quick classification.
        Cheaper and faster than Nova Pro.
        """
        return self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            model_id=self._settings.bedrock_screener_model_id,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def invoke_json(
        self,
        prompt: str,
        system_prompt: str = "You are an expert Snowflake SQL optimization assistant. Always respond with valid JSON only.",
        model_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Invoke a model and parse the response as JSON.

        The prompt should instruct the model to respond with JSON.
        Returns a parsed dict. Raises ValueError if response is not valid JSON.
        """
        raw = self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # Strip markdown fences if model wraps JSON in ```json ... ```
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            # Remove first and last fence lines
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning("Could not parse model JSON response: %s\nRaw: %s", e, raw[:500])
            raise ValueError(f"Model did not return valid JSON: {e}") from e


def get_bedrock_manager() -> BedrockManager:
    """Return the process-wide singleton BedrockManager."""
    return BedrockManager()
