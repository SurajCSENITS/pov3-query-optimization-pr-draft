"""
Report Agent — assembles a human-readable optimization report.

In production this could use an LLM to generate natural-language
explanations. For the MVP it templates the report from upstream
agent outputs.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, console
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState


class ReportAgent(BaseAgent):
    name = AgentRole.REPORT.value
    role = AgentRole.REPORT

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        analysis = state["analysis"]
        optimization = state["optimization"]
        validation = state["validation"]
        metrics = validation["metrics"]

        # ── Build structured report ─────────────────────────────
        changes_table = []
        for i, change in enumerate(optimization["changes_applied"], 1):
            changes_table.append({
                "index": i,
                "change": change["action"],
                "reason": f"Resolves {change['type']} bottleneck ({change['bottleneck_id']})",
            })

        performance_table = {
            "execution_time": (
                f"{metrics['execution_time']['before_sec']}s → "
                f"{metrics['execution_time']['after_sec']}s "
                f"(-{metrics['execution_time']['improvement_pct']}%)"
            ),
            "credits_consumed": (
                f"{metrics['credits']['before']} → "
                f"{metrics['credits']['after']} "
                f"(-{metrics['credits']['improvement_pct']}%)"
            ),
            "bytes_scanned": (
                f"{metrics['bytes_scanned']['before_gb']} GB → "
                f"{metrics['bytes_scanned']['after_gb']} GB "
                f"(-{metrics['bytes_scanned']['improvement_pct']}%)"
            ),
            "partition_pruning": (
                f"{metrics['partition_pruning']['before_pct']}% → "
                f"{metrics['partition_pruning']['after_pct']}%"
            ),
        }

        summary = (
            f"Optimized query {analysis['query_id']} by applying "
            f"{optimization['change_count']} changes, reducing execution time "
            f"by {metrics['execution_time']['improvement_pct']}% and credit "
            f"consumption by {metrics['credits']['improvement_pct']}%."
        )

        report_output = {
            "query_id": analysis["query_id"],
            "title": f"🤖 AI-Generated Query Optimization — {analysis['query_id']}",
            "summary": summary,
            "original_sql": optimization["original_sql"],
            "optimized_sql": optimization["optimized_sql"],
            "changes": changes_table,
            "performance": performance_table,
            "validation_status": validation["semantic_check"],
            "row_count_match": (
                validation["row_count_original"]
                == validation["row_count_optimized"]
            ),
        }

        # Pretty-print
        console.print(f"  📄 Report: [bold]{report_output['title']}[/]")
        console.print(f"  Summary: {summary}")
        console.print(f"  Changes documented: {len(changes_table)}")

        return {
            "state_key": "report",
            "output": report_output,
            "next_agent": AgentRole.PR.value,
            "task_desc": "Create draft PR with optimization report and evidence",
        }
