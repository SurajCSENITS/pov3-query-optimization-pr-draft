"""
FastAPI route definitions for the POV3 Query Auto-Optimization Agent.

Endpoints:
    POST /alerts/ingest  — Receive a POV4 alert and trigger the optimization pipeline
    GET  /health         — Health check (Snowflake connectivity + service status)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.graph.workflow import build_workflow
from src.models.messages import AgentMessage, AgentRole

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Build the workflow once (reused across requests) ────────────
_workflow = build_workflow()


# ── Request / Response models ───────────────────────────────────

class AlertPayload(BaseModel):
    """
    Expected payload fields inside the AgentMessage.payload
    when POV4 sends a slow-query alert.
    """
    query_id: str
    warehouse: str = ""
    credits_used: float = 0.0
    execution_time_seconds: float = 0.0
    issue_type: str = ""
    query_text: str


class IngestResponse(BaseModel):
    """Response returned after a successful alert ingestion."""
    status: str = "accepted"
    message_id: str
    query_id: str
    detail: str = "Optimization pipeline triggered"


class PipelineResultResponse(BaseModel):
    """Response returned when the pipeline runs synchronously."""
    status: str
    message_id: str
    query_id: str
    pr: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    message_trail: list[dict[str, Any]] = []
    report_path: str = ""  # Sprint 2: path to the generated HTML report


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str
    timestamp: str
    snowflake_connected: bool
    snowflake_enabled: bool


# ── Helper: run pipeline ───────────────────────────────────────

def _run_pipeline(message: AgentMessage) -> dict[str, Any]:
    """
    Construct initial state from the incoming A2A message
    and invoke the LangGraph workflow.

    Returns the final state dict.
    """
    initial_state = {
        "input_data": message.payload,
        "analysis": {},
        "optimization": {},
        "validation": {},
        "report": {},
        "pr": {},
        "rag_results": [],           # Sprint 2
        "validation_evidence": {},   # Sprint 2
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


# ── Routes ──────────────────────────────────────────────────────

@router.post(
    "/alerts/ingest",
    response_model=PipelineResultResponse,
    summary="Ingest a POV4 slow-query alert",
    description=(
        "Accepts an AgentMessage from POV4 containing a slow-query alert. "
        "Validates the payload, runs the full LangGraph optimization pipeline, "
        "and returns the result including the draft PR details."
    ),
)
async def ingest_alert(message: AgentMessage) -> PipelineResultResponse:
    """
    Receive a POV4 alert and run the optimization pipeline.

    The pipeline runs synchronously for the POV demo.
    In production, consider using BackgroundTasks for async execution.
    """
    # Validate that the message is addressed to our system
    if message.receiver not in (
        AgentRole.ANALYSIS.value,
        "Orchestrator",
        "POV3",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Message addressed to '{message.receiver}', expected 'AnalysisAgent' or 'Orchestrator'",
        )

    # Validate required payload fields
    payload = message.payload
    if not payload.get("query_text"):
        raise HTTPException(
            status_code=422,
            detail="Missing required field: 'query_text' in payload",
        )
    if not payload.get("query_id"):
        raise HTTPException(
            status_code=422,
            detail="Missing required field: 'query_id' in payload",
        )

    # Run the pipeline
    try:
        final_state = _run_pipeline(message)
    except Exception as e:
        logger.exception("Pipeline failed for message_id=%s", message.message_id)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}",
        )

    return PipelineResultResponse(
        status="completed",
        message_id=message.message_id,
        query_id=payload["query_id"],
        pr=final_state.get("pr", {}),
        validation=final_state.get("validation", {}),
        message_trail=final_state.get("messages", []),
        report_path=final_state.get("report_path", ""),  # Sprint 2
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    """
    Check service health and Snowflake connectivity status.
    """
    settings = get_settings()
    snowflake_connected = False

    if settings.snowflake_configured:
        try:
            from src.connectors.snowflake_manager import get_connection_manager
            manager = get_connection_manager()
            snowflake_connected = manager.health_check()
        except Exception as e:
            logger.warning("Snowflake health check failed: %s", e)

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        snowflake_connected=snowflake_connected,
        snowflake_enabled=settings.snowflake_enabled,
    )
