"""
Base agent class that all POV3 agents inherit from.

Provides:
- Structured logging via Rich console
- AgentMessage creation helper
- State update helper
- A standard `run()` contract for subclasses
- Automatic LangSmith tracing via @traceable decorator

When LANGSMITH_API_KEY is set, every agent run is automatically
traced — providing visibility into agent reasoning, chain-of-thought,
and confidence scores. When not set, @traceable is a transparent no-op.
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


def _get_traceable_decorator():
    """
    Return the @traceable decorator from langsmith if available.

    Falls back to a transparent no-op decorator if langsmith
    is not installed or not configured (no LANGSMITH_API_KEY).
    """
    try:
        from langsmith import traceable
        return traceable
    except ImportError:
        # langsmith not installed — return identity decorator
        def identity(func=None, **kwargs):
            if func is not None:
                return func
            return lambda f: f
        return identity


traceable = _get_traceable_decorator()


class BaseAgent(abc.ABC):
    """
    Abstract base for every agent in the pipeline.

    Subclasses implement `process()` with their domain logic.
    The `run()` method handles message construction, logging,
    state updates, and LangSmith tracing — keeping agent code
    focused on logic.
    """

    name: str = "BaseAgent"
    role: AgentRole = AgentRole.ANALYSIS

    # ── Public entry point (called by LangGraph node) ───────────

    @traceable(run_type="chain")
    def run(self, state: QueryOptimizationState) -> dict[str, Any]:
        """
        Execute the agent:
        1. Log entry
        2. Delegate to `process()` for domain logic
        3. Build outgoing AgentMessage
        4. Return state patch (LangGraph merges this automatically)

        Automatically traced by LangSmith when configured,
        capturing input state, output state, and timing.
        """
        self._log_entry(state)

        # Subclass does the real work
        result = self.process(state)

        # Build the outgoing A2A message
        outgoing_msg = self._build_message(result)
        self._log_message(outgoing_msg)

        # Return the state patch
        state_patch: dict = {
            result["state_key"]: result["output"],
            "messages": [outgoing_msg.model_dump()],
        }
        # Merge any extra top-level state keys (e.g. validation_evidence)
        if "extra_state" in result:
            state_patch.update(result["extra_state"])
        return state_patch

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
