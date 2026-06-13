"""
Pydantic models for Agent-to-Agent (A2A) messaging.

Each agent communicates via structured AgentMessage objects,
providing full traceability of the inter-agent data flow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Enumeration of all agents in the POV3 pipeline."""

    POV4_ALERT = "POV4AlertAgent"
    ANALYSIS = "AnalysisAgent"
    OPTIMIZATION = "OptimizationAgent"
    VALIDATION = "ValidationAgent"
    REPORT = "ReportAgent"
    PR = "PRAgent"


class AgentMessage(BaseModel):
    """
    Structured message passed between agents.

    This is the core A2A communication primitive. Every agent
    receives one message as input and emits one message as output.
    The full message chain is preserved in the shared state for
    auditability.
    """

    message_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sender: str = Field(..., description="Name of the sending agent")
    receiver: str = Field(..., description="Name of the receiving agent")
    task: str = Field(..., description="Description of the delegated task")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data passed to the next agent",
    )

    def summary(self) -> str:
        """One-line summary for console logging."""
        return (
            f"[{self.message_id}] {self.sender} → {self.receiver} | {self.task}"
        )
