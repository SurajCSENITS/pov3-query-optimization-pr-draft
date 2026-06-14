"""
Analysis Agent — identifies performance bottlenecks in the query.

Operates in two modes:
  1. SNOWFLAKE MODE: Executes EXPLAIN and queries QUERY_HISTORY for real
     performance metadata. Activated when Snowflake is configured and enabled.
  2. MOCK MODE: Uses rule-based regex heuristics on the SQL text.
     This is the fallback when Snowflake is unavailable.

Both modes produce the same output schema, so downstream agents
(OptimizationAgent, etc.) work identically regardless of mode.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    name = AgentRole.ANALYSIS.value
    role = AgentRole.ANALYSIS

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        data = state["input_data"]
        sql = data.get("query_text", "")
        settings = get_settings()

        # ── Choose execution mode ───────────────────────────────
        if settings.snowflake_configured:
            console.print("  🔗 [bold blue]Mode: SNOWFLAKE[/] — using real metadata")
            analysis_output = self._analyze_with_snowflake(data, sql)
        else:
            console.print("  🧪 [bold yellow]Mode: MOCK[/] — using rule-based heuristics")
            analysis_output = self._analyze_with_mock(data, sql)

        # ── Pretty-print findings ───────────────────────────────
        for b in analysis_output["bottlenecks"]:
            icon = "🔴" if b["severity"] in ("CRITICAL", "HIGH") else "🟡"
            console.print(
                f"  {icon} [{b['id']}] {b['type']}: {b['description']}"
            )
        console.print(
            f"  📊 Severity Score: [bold]{analysis_output['severity_score']}[/] | "
            f"Bottlenecks: {analysis_output['bottleneck_count']}"
        )

        return {
            "state_key": "analysis",
            "output": analysis_output,
            "next_agent": AgentRole.OPTIMIZATION.value,
            "task_desc": (
                f"Optimize query {data.get('query_id', 'unknown')} — "
                f"{analysis_output['bottleneck_count']} bottlenecks found "
                f"(score={analysis_output['severity_score']})"
            ),
        }

    # ── Snowflake-backed analysis ───────────────────────────────

    def _analyze_with_snowflake(
        self, data: dict[str, Any], sql: str
    ) -> dict[str, Any]:
        """
        Fetch real execution metadata from Snowflake:
        1. EXPLAIN USING TEXT for the query plan
        2. QUERY_HISTORY for execution stats
        3. Combine with rule-based heuristics
        """
        from src.connectors.snowflake_manager import get_connection_manager

        manager = get_connection_manager()
        bottlenecks: list[dict[str, str]] = []
        query_plan_text = ""
        snowflake_metadata: dict[str, Any] = {}

        # ── Step 1: Run EXPLAIN ─────────────────────────────────
        try:
            explain_results = manager.explain_query(sql)
            if explain_results:
                # EXPLAIN returns rows with plan details
                plan_lines = []
                for row in explain_results:
                    # Snowflake EXPLAIN USING TEXT returns a single-column result
                    for col_name, col_val in row.items():
                        if col_val:
                            plan_lines.append(str(col_val))
                query_plan_text = "\n".join(plan_lines)
                console.print(f"  📋 EXPLAIN plan retrieved ({len(explain_results)} rows)")

                # Detect issues from the EXPLAIN plan
                plan_lower = query_plan_text.lower()
                if "tableScan" in query_plan_text or "table scan" in plan_lower:
                    bottlenecks.append({
                        "id": "B001",
                        "type": "FULL_TABLE_SCAN",
                        "severity": "HIGH",
                        "description": "Full table scan detected in query plan",
                        "location": "EXPLAIN plan",
                    })

        except Exception as e:
            logger.warning("EXPLAIN failed, falling back to heuristics: %s", e)
            console.print(f"  ⚠️  EXPLAIN failed: {e}")

        # ── Step 2: Fetch QUERY_HISTORY ─────────────────────────
        query_id = data.get("query_id", "")
        try:
            history = manager.get_query_history(
                query_id=query_id if len(query_id) > 10 else None,
                query_text_fragment=sql[:80] if not query_id or len(query_id) <= 10 else None,
                limit=1,
            )
            if history:
                record = history[0]
                snowflake_metadata = {
                    "snowflake_query_id": record.get("QUERY_ID"),
                    "execution_time_seconds": record.get("EXECUTION_TIME_SECONDS"),
                    "bytes_scanned": record.get("BYTES_SCANNED"),
                    "rows_produced": record.get("ROWS_PRODUCED"),
                    "partitions_scanned": record.get("PARTITIONS_SCANNED"),
                    "partitions_total": record.get("PARTITIONS_TOTAL"),
                    "bytes_spilled_local": record.get("BYTES_SPILLED_TO_LOCAL_STORAGE"),
                    "bytes_spilled_remote": record.get("BYTES_SPILLED_TO_REMOTE_STORAGE"),
                    "credits_used": record.get("CREDITS_USED_CLOUD_SERVICES"),
                }
                console.print(
                    f"  📊 QUERY_HISTORY: exec={snowflake_metadata.get('execution_time_seconds')}s, "
                    f"bytes={snowflake_metadata.get('bytes_scanned')}"
                )

                # Detect spilling from real metadata
                spill_remote = record.get("BYTES_SPILLED_TO_REMOTE_STORAGE", 0) or 0
                spill_local = record.get("BYTES_SPILLED_TO_LOCAL_STORAGE", 0) or 0
                if spill_remote > 0:
                    bottlenecks.append({
                        "id": "B003",
                        "type": "REMOTE_SPILL",
                        "severity": "CRITICAL",
                        "description": f"Query spills {spill_remote:,} bytes to remote storage",
                        "location": "Execution engine",
                    })
                elif spill_local > 0:
                    bottlenecks.append({
                        "id": "B003",
                        "type": "LOCAL_SPILL",
                        "severity": "HIGH",
                        "description": f"Query spills {spill_local:,} bytes to local storage",
                        "location": "Execution engine",
                    })

                # Detect poor partition pruning
                parts_scanned = record.get("PARTITIONS_SCANNED", 0) or 0
                parts_total = record.get("PARTITIONS_TOTAL", 1) or 1
                if parts_total > 0:
                    prune_pct = (1 - parts_scanned / parts_total) * 100
                    if prune_pct < 20:
                        bottlenecks.append({
                            "id": "B005",
                            "type": "POOR_PARTITION_PRUNING",
                            "severity": "HIGH",
                            "description": (
                                f"Only {prune_pct:.0f}% of partitions pruned "
                                f"({parts_scanned}/{parts_total} scanned)"
                            ),
                            "location": "Storage layer",
                        })

            else:
                console.print("  ℹ️  No QUERY_HISTORY records found (may be within 45-min latency)")

        except Exception as e:
            logger.warning("QUERY_HISTORY fetch failed: %s", e)
            console.print(f"  ⚠️  QUERY_HISTORY unavailable: {e}")

        # ── Step 3: Always apply SQL heuristics too ─────────────
        bottlenecks.extend(self._detect_sql_patterns(data, sql))

        # Deduplicate by bottleneck type
        seen_types = set()
        unique_bottlenecks = []
        for b in bottlenecks:
            if b["type"] not in seen_types:
                seen_types.add(b["type"])
                unique_bottlenecks.append(b)

        severity_score = self._compute_severity(unique_bottlenecks)

        return {
            "query_id": data.get("query_id", "unknown"),
            "original_sql": sql,
            "bottlenecks": unique_bottlenecks,
            "bottleneck_count": len(unique_bottlenecks),
            "severity_score": severity_score,
            "recommendation": "OPTIMIZE" if unique_bottlenecks else "NO_ACTION",
            "query_plan": query_plan_text,
            "snowflake_metadata": snowflake_metadata,
            "analysis_mode": "snowflake",
        }

    # ── Mock/regex-based analysis (original MVP logic) ──────────

    def _analyze_with_mock(
        self, data: dict[str, Any], sql: str
    ) -> dict[str, Any]:
        """Original MVP analysis using regex pattern matching."""
        bottlenecks = self._detect_sql_patterns(data, sql)

        # Check metadata-based issues
        if data.get("issue_type") == "REMOTE_SPILL":
            bottlenecks.append({
                "id": "B003",
                "type": "REMOTE_SPILL",
                "severity": "CRITICAL",
                "description": "Query spills data to remote storage — major performance degradation",
                "location": "Execution engine",
            })

        severity_score = self._compute_severity(bottlenecks)

        return {
            "query_id": data.get("query_id", "unknown"),
            "original_sql": sql,
            "bottlenecks": bottlenecks,
            "bottleneck_count": len(bottlenecks),
            "severity_score": severity_score,
            "recommendation": "OPTIMIZE" if bottlenecks else "NO_ACTION",
            "query_plan": "",
            "snowflake_metadata": {},
            "analysis_mode": "mock",
        }

    # ── Shared helpers ──────────────────────────────────────────

    @staticmethod
    def _detect_sql_patterns(
        data: dict[str, Any], sql: str
    ) -> list[dict[str, str]]:
        """
        Regex-based SQL pattern detection.
        Used in both Snowflake and mock modes.
        """
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

        return bottlenecks

    @staticmethod
    def _compute_severity(bottlenecks: list[dict[str, str]]) -> int:
        """Compute a weighted severity score from bottleneck list."""
        return sum(
            {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}.get(
                b["severity"], 1
            )
            for b in bottlenecks
        )
