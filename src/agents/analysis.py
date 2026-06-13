"""
Analysis Agent — identifies performance bottlenecks in the query.

In production this would call Snowflake EXPLAIN / QUERY_PROFILE.
For the MVP it uses rule-based heuristics on the SQL text.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base import BaseAgent, console
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState


class AnalysisAgent(BaseAgent):
    name = AgentRole.ANALYSIS.value
    role = AgentRole.ANALYSIS

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        data = state["input_data"]
        sql = data.get("query_text", "")

        # ── Rule-based bottleneck detection (mock) ──────────────
        bottlenecks: list[dict[str, str]] = []

        if re.search(r"SELECT\s+\*", sql, re.IGNORECASE):
            bottlenecks.append({
                "id": "B001",
                "type": "FULL_COLUMN_SCAN",
                "severity": "HIGH",
                "description": "SELECT * scans all columns — excessive bytes read",
                "location": "SELECT clause",
            })

        if re.search(r"YEAR\s*\(", sql, re.IGNORECASE):
            bottlenecks.append({
                "id": "B002",
                "type": "NON_SARGABLE_PREDICATE",
                "severity": "HIGH",
                "description": "YEAR() function wrapping column prevents micro-partition pruning",
                "location": "WHERE clause",
            })

        if data.get("issue_type") == "REMOTE_SPILL":
            bottlenecks.append({
                "id": "B003",
                "type": "REMOTE_SPILL",
                "severity": "CRITICAL",
                "description": "Query spills data to remote storage — major performance degradation",
                "location": "Execution engine",
            })

        if re.search(r"JOIN", sql, re.IGNORECASE) and not re.search(
            r"WHERE.*AND", sql, re.IGNORECASE
        ):
            bottlenecks.append({
                "id": "B004",
                "type": "UNFILTERED_JOIN",
                "severity": "MEDIUM",
                "description": "JOIN executed without pre-filtering — large intermediate result set",
                "location": "JOIN clause",
            })

        severity_score = sum(
            {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}.get(
                b["severity"], 1
            )
            for b in bottlenecks
        )

        analysis_output = {
            "query_id": data["query_id"],
            "original_sql": sql,
            "bottlenecks": bottlenecks,
            "bottleneck_count": len(bottlenecks),
            "severity_score": severity_score,
            "recommendation": "OPTIMIZE" if bottlenecks else "NO_ACTION",
        }

        # Pretty-print findings
        for b in bottlenecks:
            icon = "🔴" if b["severity"] in ("CRITICAL", "HIGH") else "🟡"
            console.print(
                f"  {icon} [{b['id']}] {b['type']}: {b['description']}"
            )
        console.print(
            f"  📊 Severity Score: [bold]{severity_score}[/] | "
            f"Bottlenecks: {len(bottlenecks)}"
        )

        return {
            "state_key": "analysis",
            "output": analysis_output,
            "next_agent": AgentRole.OPTIMIZATION.value,
            "task_desc": f"Optimize query {data['query_id']} — {len(bottlenecks)} bottlenecks found (score={severity_score})",
        }
