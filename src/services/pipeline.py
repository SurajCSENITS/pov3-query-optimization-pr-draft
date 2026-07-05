"""
Shared pipeline runner service.

Extracts the LangGraph pipeline invocation logic into a reusable
function that can be called from both:
  - The FastAPI HTTP endpoint (POST /alerts/ingest)
  - The NATS subscriber (inter-project messaging from POV4)

This avoids code duplication while keeping the LangGraph workflow
completely untouched.
"""

from __future__ import annotations

import logging
from typing import Any

from src.graph.workflow import build_workflow
from src.models.messages import AgentMessage

logger = logging.getLogger("pov3.services.pipeline")

# ── Build the workflow once (reused across all invocations) ─────
_workflow = build_workflow()


def run_optimization_pipeline(message: AgentMessage) -> dict[str, Any]:
    """
    Construct initial state from an AgentMessage and invoke
    the LangGraph optimization workflow.

    This is the single entry point for triggering the full
    agent chain (Analysis → Optimization → Validation → Report → PR).

    Args:
        message: An AgentMessage from POV4 (or equivalent) containing
                 the slow-query alert payload.

    Returns:
        The final LangGraph state dict with all pipeline outputs.
    """
    initial_state = {
        "input_data": message.payload,
        "analysis": {},
        "optimization": {},
        "validation": {},
        "report": {},
        "pr": {},
        "rag_results": [],
        "validation_evidence": {},
        "messages": [message.model_dump()],
    }

    logger.info(
        "Starting pipeline for query_id=%s (message_id=%s)",
        message.payload.get("query_id"),
        message.message_id,
    )

    final_state = _workflow.invoke(initial_state)

    logger.info(
        "Pipeline complete for query_id=%s — validation=%s",
        message.payload.get("query_id"),
        final_state.get("validation", {}).get("semantic_check", "N/A"),
    )

    return final_state
