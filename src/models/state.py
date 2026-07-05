"""
LangGraph shared state definition.

Uses TypedDict so LangGraph can track granular key-level updates.
The `messages` list acts as the A2A communication ledger — every
AgentMessage exchanged during a run is appended here.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _merge_lists(existing: list, new: list) -> list:
    """Reducer: append new items to the existing list (no dedup)."""
    return existing + new


class QueryOptimizationState(TypedDict):
    """
    Shared state threaded through every node in the LangGraph.

    Each key corresponds to a pipeline stage. Agents read upstream
    keys and write to their own key. The `messages` key accumulates
    the full A2A message trail.
    """

    # ── Pipeline stage outputs ──────────────────────────────────
    input_data: dict[str, Any]
    analysis: dict[str, Any]
    optimization: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]
    pr: dict[str, Any]

    # ── Sprint 2 additions ──────────────────────────────────────
    # Raw RAG retrieval results (list of {content, score, source} dicts).
    # Written by OptimizationAgent; consumed by HTMLReportGenerator (Section 5).
    rag_results: list[dict]

    # Structured validation evidence bundle.
    # Written by ValidationAgent; consumed by HTMLReportGenerator (Section 6).
    validation_evidence: dict[str, Any]

    # ── Retry & Feedback Loop ───────────────────────────────────
    # Number of times validation has rejected the query and routed back to optimization
    retry_count: int
    # Accumulated feedback messages from validation failures
    feedback_history: list[str]

    # ── A2A message ledger ──────────────────────────────────────
    # Annotated with a reducer so each node can *append* messages
    # without overwriting previous entries.
    messages: Annotated[list[dict], _merge_lists]
