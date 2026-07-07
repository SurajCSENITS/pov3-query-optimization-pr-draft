#!/usr/bin/env python3
"""
POV3 — Query Auto-Optimization Agent (MVP Runner)

Run:
    pip install -r requirements.txt
    python main.py

This script:
1. Constructs a mocked POV4 alert payload
2. Builds the LangGraph pipeline
3. Invokes the full agent chain
4. Prints the A2A message trail and final state
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.graph.workflow import build_workflow
from src.models.messages import AgentMessage, AgentRole

console = Console()

# ── Sample input (mocked POV4 alert) ───────────────────────────

SAMPLE_INPUT = {
    "query_id": "Q123",
    "warehouse": "WH_LARGE",
    "credits_used": 18,
    "execution_time_seconds": 240,
    "issue_type": "REMOTE_SPILL",
    "query_text": (
        "SELECT * FROM ORDERS o "
        "JOIN CUSTOMER c ON o.o_custkey = c.c_custkey "
        "WHERE YEAR(o.o_orderdate)=1995"
    ),
}


def print_header() -> None:
    """Print a startup banner."""
    banner = Text()
    banner.append("POV3", style="bold magenta")
    banner.append(" — Query Auto-Optimization Agent\n", style="bold white")
    banner.append("Multi-Agent Orchestration MVP using LangGraph\n", style="dim")
    banner.append(
        f"Run started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        style="dim italic",
    )
    console.print(Panel(banner, border_style="bright_magenta", padding=(1, 2)))


def print_input(data: dict) -> None:
    """Print the incoming POV4 alert payload."""
    table = Table(
        title="📥 Incoming POV4 Alert",
        box=box.ROUNDED,
        border_style="yellow",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    for key, val in data.items():
        table.add_row(key, str(val))

    console.print(table)
    console.print()


def print_message_trail(messages: list[dict]) -> None:
    """Print the complete A2A message trail."""
    console.print()
    console.print(
        Panel(
            Text("📬  Agent-to-Agent Message Trail", style="bold"),
            border_style="green",
            expand=False,
        )
    )

    table = Table(box=box.SIMPLE_HEAVY, border_style="green")
    table.add_column("#", style="dim", width=3)
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Sender", style="bold")
    table.add_column("→", style="dim", width=2)
    table.add_column("Receiver", style="bold")
    table.add_column("Task", style="white", max_width=60)

    for i, msg_dict in enumerate(messages, 1):
        msg = AgentMessage(**msg_dict)
        table.add_row(
            str(i),
            msg.message_id,
            msg.sender,
            "→",
            msg.receiver,
            msg.task,
        )

    console.print(table)


def print_final_status(state: dict) -> None:
    """Print the final pipeline status."""
    pr = state.get("pr", {})
    validation = state.get("validation", {})
    report_path = state.get("report_path", pr.get("report_path", ""))

    status_lines = [
        f"Query ID:        {pr.get('query_id', 'N/A')}",
        f"Branch:          {pr.get('branch_name', 'N/A')}",
        f"PR Title:        {pr.get('pr_title', 'N/A')}",
        f"PR State:        {pr.get('pr_state', 'N/A').upper()}",
        f"Auto-merge:      {'❌ Disabled' if not pr.get('auto_merge') else '⚠️ Enabled'}",
        f"Validation:      {validation.get('semantic_check', 'N/A')}",
        f"Labels:          {', '.join(pr.get('labels', []))}",
    ]

    if report_path:
        status_lines.append(f"HTML Report:     {report_path}")

    console.print()
    console.print(
        Panel(
            "\n".join(status_lines),
            title="[bold green]✅ Pipeline Complete[/]",
            border_style="green",
            padding=(1, 2),
        )
    )


def main() -> None:
    print_header()
    print_input(SAMPLE_INPUT)

    # ── Build the initial A2A message (POV4 → AnalysisAgent) ───
    pov4_message = AgentMessage(
        sender=AgentRole.POV4_ALERT.value,
        receiver=AgentRole.ANALYSIS.value,
        task=f"Analyze slow query {SAMPLE_INPUT['query_id']} — {SAMPLE_INPUT['issue_type']}",
        payload=SAMPLE_INPUT,
    )

    console.print(f"  📨 [bold green]A2A Message[/]: {pov4_message.summary()}\n")

    # ── Construct initial state ─────────────────────────────────
    initial_state = {
        "input_data": SAMPLE_INPUT,
        "analysis": {},
        "optimization": {},
        "validation": {},
        "report": {},
        "pr": {},
        "rag_results": [],           # Sprint 2
        "validation_evidence": {},   # Sprint 2
        "graph_location": {},        # PR Agent: codebase graph lookup result
        "messages": [pov4_message.model_dump()],
    }

    # ── Build and invoke the LangGraph workflow ─────────────────
    console.print(
        Panel(
            Text("🚀 Starting LangGraph Pipeline", style="bold"),
            border_style="bright_blue",
            expand=False,
        )
    )
    console.print()

    workflow = build_workflow()
    final_state = workflow.invoke(initial_state)

    # ── Print results ───────────────────────────────────────────
    print_message_trail(final_state["messages"])
    print_final_status(final_state)


if __name__ == "__main__":
    main()
