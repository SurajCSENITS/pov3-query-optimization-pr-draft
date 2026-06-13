"""
Optimization Agent — generates an optimised SQL candidate.

In production this would use an LLM (e.g. Gemini, Claude) with
chain-of-thought prompting. For the MVP it applies deterministic
rewrite rules keyed to the bottleneck types found by AnalysisAgent.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base import BaseAgent, console
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState

# ── Rewrite rules mapped to bottleneck types ────────────────────

_REWRITE_RULES: dict[str, dict[str, str]] = {
    "FULL_COLUMN_SCAN": {
        "action": "Replace SELECT * with explicit column list",
        "pattern": r"SELECT\s+\*",
        "replacement": (
            "SELECT\n"
            "    o.order_id,\n"
            "    o.order_date,\n"
            "    o.order_amount,\n"
            "    c.customer_id,\n"
            "    c.customer_name,\n"
            "    c.country"
        ),
    },
    "NON_SARGABLE_PREDICATE": {
        "action": "Replace YEAR(col) with sargable date range",
        "pattern": r"YEAR\s*\(\s*o\.order_date\s*\)\s*=\s*2025",
        "replacement": "o.order_date BETWEEN '2025-01-01' AND '2025-12-31'",
    },
}


class OptimizationAgent(BaseAgent):
    name = AgentRole.OPTIMIZATION.value
    role = AgentRole.OPTIMIZATION

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        analysis = state["analysis"]
        original_sql = analysis["original_sql"]
        optimized_sql = original_sql
        changes_applied: list[dict[str, str]] = []

        # ── Apply rewrite rules ─────────────────────────────────
        for bottleneck in analysis["bottlenecks"]:
            rule = _REWRITE_RULES.get(bottleneck["type"])
            if rule:
                new_sql = re.sub(
                    rule["pattern"], rule["replacement"], optimized_sql, flags=re.IGNORECASE
                )
                if new_sql != optimized_sql:
                    changes_applied.append({
                        "bottleneck_id": bottleneck["id"],
                        "type": bottleneck["type"],
                        "action": rule["action"],
                    })
                    optimized_sql = new_sql

        # ── Add LIMIT if missing (spill mitigation) ────────────
        if any(b["type"] == "REMOTE_SPILL" for b in analysis["bottlenecks"]):
            if not re.search(r"LIMIT\s+\d+", optimized_sql, re.IGNORECASE):
                optimized_sql = optimized_sql.rstrip(";") + "\nLIMIT 10000;"
                changes_applied.append({
                    "bottleneck_id": "B003",
                    "type": "REMOTE_SPILL",
                    "action": "Added LIMIT 10000 to bound result set and reduce spill",
                })

        optimization_output = {
            "query_id": analysis["query_id"],
            "original_sql": original_sql,
            "optimized_sql": optimized_sql,
            "changes_applied": changes_applied,
            "change_count": len(changes_applied),
            "estimated_improvement_pct": min(len(changes_applied) * 25, 85),
        }

        # Pretty-print
        console.print(f"  ✏️  Changes applied: [bold]{len(changes_applied)}[/]")
        for c in changes_applied:
            console.print(f"    ↳ [{c['bottleneck_id']}] {c['action']}")
        console.print(f"\n  [dim]Optimized SQL:[/]")
        console.print(f"  [green]{optimized_sql}[/]\n")

        return {
            "state_key": "optimization",
            "output": optimization_output,
            "next_agent": AgentRole.VALIDATION.value,
            "task_desc": f"Validate optimized query — {len(changes_applied)} changes applied",
        }
