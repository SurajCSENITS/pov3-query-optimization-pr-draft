"""
Validation Agent — verifies semantic equivalence and simulates
performance comparison between original and optimised queries.

In production this would run both queries against Snowflake (with
LIMIT) and compare row counts / sample rows. For the MVP it
uses mock metrics.
"""

from __future__ import annotations

import random
from typing import Any

from src.agents.base import BaseAgent, console
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState


class ValidationAgent(BaseAgent):
    name = AgentRole.VALIDATION.value
    role = AgentRole.VALIDATION

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        optimization = state["optimization"]
        input_data = state["input_data"]
        change_count = optimization["change_count"]

        # ── Mock: simulate performance metrics ──────────────────
        original_time = input_data["execution_time_seconds"]
        original_credits = input_data["credits_used"]
        improvement_factor = max(0.15, 1 - (change_count * 0.22))

        optimized_time = round(original_time * improvement_factor, 1)
        optimized_credits = round(original_credits * improvement_factor, 2)

        original_bytes = 480_000_000_000  # 480 GB mock
        optimized_bytes = int(original_bytes * improvement_factor)

        # ── Mock: semantic equivalence check ────────────────────
        original_row_count = 2_847_391
        optimized_row_count = original_row_count  # simulated match
        rows_match = original_row_count == optimized_row_count

        original_pruning = 3
        optimized_pruning = min(91, 3 + change_count * 28)

        validation_output = {
            "query_id": optimization["query_id"],
            "is_valid": rows_match,
            "semantic_check": "PASS" if rows_match else "FAIL",
            "row_count_original": original_row_count,
            "row_count_optimized": optimized_row_count,
            "metrics": {
                "execution_time": {
                    "before_sec": original_time,
                    "after_sec": optimized_time,
                    "improvement_pct": round(
                        (1 - optimized_time / original_time) * 100, 1
                    ),
                },
                "credits": {
                    "before": original_credits,
                    "after": optimized_credits,
                    "improvement_pct": round(
                        (1 - optimized_credits / original_credits) * 100, 1
                    ),
                },
                "bytes_scanned": {
                    "before_gb": round(original_bytes / 1e9, 1),
                    "after_gb": round(optimized_bytes / 1e9, 1),
                    "improvement_pct": round(
                        (1 - optimized_bytes / original_bytes) * 100, 1
                    ),
                },
                "partition_pruning": {
                    "before_pct": original_pruning,
                    "after_pct": optimized_pruning,
                },
            },
        }

        # Pretty-print
        m = validation_output["metrics"]
        status = "✅ PASS" if rows_match else "❌ FAIL"
        console.print(f"  Semantic check: [bold]{status}[/]")
        console.print(
            f"  Row counts: {original_row_count:,} → {optimized_row_count:,}"
        )
        console.print(
            f"  ⏱  Execution: {m['execution_time']['before_sec']}s → "
            f"{m['execution_time']['after_sec']}s "
            f"([green]-{m['execution_time']['improvement_pct']}%[/])"
        )
        console.print(
            f"  💰 Credits: {m['credits']['before']} → "
            f"{m['credits']['after']} "
            f"([green]-{m['credits']['improvement_pct']}%[/])"
        )
        console.print(
            f"  📦 Bytes: {m['bytes_scanned']['before_gb']} GB → "
            f"{m['bytes_scanned']['after_gb']} GB "
            f"([green]-{m['bytes_scanned']['improvement_pct']}%[/])"
        )
        console.print(
            f"  🌿 Partition pruning: {m['partition_pruning']['before_pct']}% → "
            f"{m['partition_pruning']['after_pct']}%"
        )

        return {
            "state_key": "validation",
            "output": validation_output,
            "next_agent": AgentRole.REPORT.value,
            "task_desc": f"Generate optimization report — validation {validation_output['semantic_check']}",
        }
