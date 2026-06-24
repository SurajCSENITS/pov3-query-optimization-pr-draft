"""
RAG Manager — LangChain AmazonKnowledgeBasesRetriever Integration.

Replaces the custom boto3 Knowledge Base wrapper with LangChain's
native AmazonKnowledgeBasesRetriever, which provides:
  - Standard BaseRetriever interface (compatible with LCEL chains)
  - Automatic Document object creation
  - Seamless LangSmith tracing

Usage:
    from src.connectors.rag_manager import get_retriever, retrieve_and_format

    # Get a retriever for use in LCEL chains:
    retriever = get_retriever()
    docs = retriever.invoke("FULL_COLUMN_SCAN optimization for SELECT *")

    # Or use the convenience function for few-shot context:
    context = retrieve_and_format(
        bottleneck_types=["FULL_COLUMN_SCAN"],
        sql_fragment="SELECT * FROM orders...",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Maximum number of past cases to retrieve as few-shot context
_DEFAULT_TOP_K = 3


def get_retriever(top_k: int = _DEFAULT_TOP_K) -> Any:
    """
    Return a configured AmazonKnowledgeBasesRetriever instance.

    Returns None if RAG is not configured (bedrock_kb_id not set).
    The retriever implements LangChain's BaseRetriever interface.
    """
    settings = get_settings()

    if not settings.rag_configured:
        logger.info("RAG not configured — retriever unavailable")
        return None

    from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=settings.bedrock_kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": top_k,
            }
        },
        region_name=settings.aws_region,
        credentials_profile_name=None,
    )

    logger.info(
        "AmazonKnowledgeBasesRetriever created (kb=%s, top_k=%d)",
        settings.bedrock_kb_id,
        top_k,
    )
    return retriever


def retrieve_similar_cases(
    bottleneck_types: list[str],
    sql_fragment: str,
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve similar past optimization cases from the Knowledge Base.

    Builds a semantic search query from bottleneck types and SQL pattern,
    then fetches matching documents. Returns a list of dicts compatible
    with the existing pipeline format.

    Returns [] if RAG is not configured or retrieval fails.
    """
    retriever = get_retriever(top_k=top_k)
    if retriever is None:
        return []

    # Build the search query
    bottleneck_str = ", ".join(bottleneck_types) if bottleneck_types else "general"
    query = (
        f"Snowflake SQL optimization for bottlenecks: {bottleneck_str}. "
        f"Query pattern: {sql_fragment[:200]}"
    )

    try:
        docs = retriever.invoke(query)

        results = []
        for doc in docs:
            content = doc.page_content
            score = doc.metadata.get("score", 0.0)
            source = doc.metadata.get("source", doc.metadata.get("location", {}).get("s3Location", {}).get("uri", "unknown"))
            if content:
                results.append({
                    "content": content,
                    "score": score,
                    "source": source,
                })

        logger.info("RAG retrieved %d similar cases", len(results))
        return results

    except Exception as e:
        logger.warning("RAG retrieval failed (non-fatal): %s", e)
        return []


def format_as_few_shot_context(cases: list[dict[str, Any]]) -> str:
    """
    Format retrieved cases as a few-shot context string for LLM prompts.

    Args:
        cases: List returned by retrieve_similar_cases().

    Returns:
        Formatted multi-line string, or empty string if no cases.
    """
    if not cases:
        return ""

    lines = ["## Relevant Past Optimization Cases\n"]
    for i, case in enumerate(cases, 1):
        score_pct = round(case.get("score", 0) * 100, 1)
        lines.append(f"### Case {i} (Relevance: {score_pct}%)")
        lines.append(case["content"])
        lines.append("")

    return "\n".join(lines)
