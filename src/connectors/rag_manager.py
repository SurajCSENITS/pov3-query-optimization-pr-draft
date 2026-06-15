"""
RAG Manager — retrieves relevant past optimization cases from the
Amazon Bedrock Knowledge Base.

At optimization time, this manager queries the vector index for
the top-K most similar past optimization reports. These are injected
as few-shot examples into the Optimization Agent's prompt, enabling
the LLM to leverage past successes.

Usage:
    from src.connectors.rag_manager import get_rag_manager
    manager = get_rag_manager()
    context = manager.retrieve_similar_cases(
        bottleneck_types=["FULL_COLUMN_SCAN", "NON_SARGABLE_PREDICATE"],
        sql_fragment="SELECT * FROM orders WHERE YEAR(order_date) = 2025",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Maximum number of past cases to retrieve as few-shot context
_DEFAULT_TOP_K = 3


class RAGManager:
    """
    Wrapper around Amazon Bedrock Knowledge Base Retrieve API.

    Retrieves semantically similar past optimization reports
    to use as RAG context in the Optimization Agent prompt.
    """

    _instance: RAGManager | None = None
    _client: Any = None

    def __new__(cls) -> RAGManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_client(self) -> Any:
        """Lazily create and return the boto3 bedrock-agent-runtime client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-agent-runtime",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
            logger.info(
                "Bedrock agent-runtime client created (region=%s, kb=%s)",
                self._settings.aws_region,
                self._settings.bedrock_kb_id,
            )
        return self._client

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve_similar_cases(
        self,
        bottleneck_types: list[str],
        sql_fragment: str,
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-K past optimization cases similar to the current query.

        Constructs a semantic search query from the bottleneck types and SQL
        pattern, then fetches matching documents from the Bedrock KB vector
        index.

        Args:
            bottleneck_types: List of detected bottleneck type strings.
            sql_fragment:     First ~200 chars of the original SQL query.
            top_k:            Number of similar cases to retrieve.

        Returns:
            List of retrieved case dicts, each containing:
                - content: str  (text chunk from the report)
                - score:   float (relevance score 0.0–1.0)
                - source:  str  (S3 key of the source document)

        Returns [] if RAG is not configured or retrieval fails.
        """
        if not self._settings.rag_configured:
            logger.info("RAG not configured — skipping similar case retrieval")
            return []

        # ── Build the search query ─────────────────────────────────────────
        bottleneck_str = ", ".join(bottleneck_types) if bottleneck_types else "general"
        query = (
            f"Snowflake SQL optimization for bottlenecks: {bottleneck_str}. "
            f"Query pattern: {sql_fragment[:200]}"
        )

        try:
            client = self._get_client()
            response = client.retrieve(
                knowledgeBaseId=self._settings.bedrock_kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": top_k,
                    }
                },
            )

            results = []
            for item in response.get("retrievalResults", []):
                content = item.get("content", {}).get("text", "")
                score = item.get("score", 0.0)
                source = (
                    item.get("location", {})
                    .get("s3Location", {})
                    .get("uri", "unknown")
                )
                if content:
                    results.append({
                        "content": content,
                        "score": score,
                        "source": source,
                    })

            logger.info(
                "RAG retrieved %d similar cases (kb=%s)",
                len(results),
                self._settings.bedrock_kb_id,
            )
            return results

        except Exception as e:
            logger.warning("RAG retrieval failed (non-fatal): %s", e)
            return []

    def format_as_few_shot_context(
        self, cases: list[dict[str, Any]]
    ) -> str:
        """
        Format retrieved cases as a few-shot context string for the LLM prompt.

        Args:
            cases: List returned by retrieve_similar_cases().

        Returns:
            Formatted multi-line string ready to inject into an LLM prompt,
            or an empty string if no cases were retrieved.
        """
        if not cases:
            return ""

        lines = ["## Relevant Past Optimization Cases\n"]
        for i, case in enumerate(cases, 1):
            score_pct = round(case["score"] * 100, 1)
            lines.append(f"### Case {i} (Relevance: {score_pct}%)")
            lines.append(case["content"])
            lines.append("")

        return "\n".join(lines)


def get_rag_manager() -> RAGManager:
    """Return the process-wide singleton RAGManager."""
    return RAGManager()
