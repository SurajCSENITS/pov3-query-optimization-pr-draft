"""
Base agent class that all POV3 agents inherit from.

Provides:
- Structured logging via Rich console
- AgentMessage creation helper
- State update helper
- A standard `run()` contract for subclasses
"""

from __future__ import annotations

import abc
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.models.messages import AgentMessage, AgentRole
from src.models.state import QueryOptimizationState

console = Console()


class BaseAgent(abc.ABC):
    """
    Abstract base for every agent in the pipeline.

    Subclasses implement `process()` with their domain logic.
    The `run()` method handles message construction, logging,
    and state updates — keeping agent code focused on logic.
    """

    name: str = "BaseAgent"
    role: AgentRole = AgentRole.ANALYSIS

    # ── Public entry point (called by LangGraph node) ───────────

    def run(self, state: QueryOptimizationState) -> dict[str, Any]:
        """
        Execute the agent:
        1. Log entry
        2. Delegate to `process()` for domain logic
        3. Build outgoing AgentMessage
        4. Return state patch (LangGraph merges this automatically)
        """
        self._log_entry(state)

        # Subclass does the real work
        result = self.process(state)

        # Build the outgoing A2A message
        outgoing_msg = self._build_message(result)
        self._log_message(outgoing_msg)

        # Return the state patch
        return {
            result["state_key"]: result["output"],
            "messages": [outgoing_msg.model_dump()],
        }

    # ── Subclass contract ───────────────────────────────────────

    @abc.abstractmethod
    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        """
        Perform the agent's core logic.

        Must return a dict with:
            - state_key: str   — which state key to write to
            - output: dict     — the data to store
            - next_agent: str  — receiver name for the outgoing message
            - task_desc: str   — human-readable task description
        """
        ...

    # ── Helpers ─────────────────────────────────────────────────

    def _build_message(self, result: dict) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=result["next_agent"],
            task=result["task_desc"],
            payload=result["output"],
        )

    def _log_entry(self, state: QueryOptimizationState) -> None:
        header = Text(f"⚙  {self.name}", style="bold cyan")
        console.print(Panel(header, border_style="cyan", expand=False))

    def _log_message(self, msg: AgentMessage) -> None:
        console.print(
            f"  📨 [bold green]A2A Message[/]: {msg.summary()}\n"
        )
