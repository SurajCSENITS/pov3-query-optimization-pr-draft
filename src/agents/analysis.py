"""
Analysis Agent — LLM-powered bottleneck identification.

Replaces the MVP's regex-based bottleneck detection with LLM-guided
analysis. The agent identifies performance issues using the LLM's
understanding of SQL anti-patterns, EXPLAIN plans, and execution history.

Operates in two modes:
  1. SNOWFLAKE MODE: Fetches real EXPLAIN and QUERY_HISTORY metadata,
     passes them as context to the LLM for informed analysis.
  2. STANDALONE MODE: Passes only the raw SQL to the LLM. The LLM
     identifies bottlenecks from SQL text patterns alone.

Both modes use the same LLM-backed analysis — no regex rules.
Falls back to a minimal static response only when Bedrock is
completely unavailable.
"""

from __future__ import annotations

import logging
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

        # ── Collect Snowflake context (if available) ────────────
        query_plan_text = ""
        snowflake_metadata: dict[str, Any] = {}

        if settings.snowflake_configured:
            console.print("  🔗 [bold blue]Mode: SNOWFLAKE[/] — fetching real metadata")
            query_plan_text, snowflake_metadata = self._fetch_snowflake_context(data, sql)
        else:
            console.print("  🧪 [bold yellow]Mode: STANDALONE[/] — SQL text analysis only")

        # ── LLM-backed analysis ─────────────────────────────────
        if settings.bedrock_configured:
            console.print(
                f"  🤖 Analyzing bottlenecks via [bold cyan]"
                f"{settings.bedrock_screener_model_id}[/]..."
            )
            analysis_output = self._analyze_with_llm(
                data, sql, query_plan_text, snowflake_metadata
            )
        else:
            console.print(
                "  ⚠️  [yellow]Bedrock not configured — returning minimal analysis[/]"
            )
            analysis_output = self._minimal_fallback(data, sql)

        # ── Attach Snowflake context to output ──────────────────
        analysis_output["query_plan"] = query_plan_text
        analysis_output["snowflake_metadata"] = snowflake_metadata
        analysis_output["analysis_mode"] = (
            "snowflake" if settings.snowflake_configured else "standalone"
        )

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

    # ── LLM-backed analysis ─────────────────────────────────────

    def _analyze_with_llm(
        self,
        data: dict[str, Any],
        sql: str,
        explain_plan: str,
        snowflake_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Use the LLM to identify bottlenecks in the SQL query.

        Passes the SQL, EXPLAIN plan, and QUERY_HISTORY metadata
        as context to the LLM. Returns structured bottleneck data.
        """
        from src.connectors.bedrock_manager import get_llm
        from src.models.llm_outputs import BottleneckAnalysis
        from src.prompts.analysis_prompt import (
            ANALYSIS_PROMPT,
            format_explain_plan_section,
            format_query_history_section,
        )

        try:
            llm = get_llm(temperature=0.1).with_structured_output(BottleneckAnalysis)

            result: BottleneckAnalysis = (ANALYSIS_PROMPT | llm).invoke({
                "sql": sql,
                "explain_plan_section": format_explain_plan_section(explain_plan),
                "query_history_section": format_query_history_section(snowflake_metadata),
            })

            # Convert Pydantic models to dicts for pipeline compatibility
            bottlenecks = [b.model_dump() for b in result.bottlenecks]

            console.print(
                f"  ✅ LLM analysis complete — {len(bottlenecks)} bottleneck(s), "
                f"reasoning: {result.reasoning[:80]}..."
                if result.reasoning else
                f"  ✅ LLM analysis complete — {len(bottlenecks)} bottleneck(s)"
            )

            return {
                "query_id": data.get("query_id", "unknown"),
                "original_sql": sql,
                "bottlenecks": bottlenecks,
                "bottleneck_count": len(bottlenecks),
                "severity_score": result.severity_score,
                "recommendation": result.recommendation,
                "llm_reasoning": result.reasoning,
            }

        except Exception as e:
            logger.warning("LLM analysis failed, using minimal fallback: %s", e)
            console.print(
                f"  ⚠️  [yellow]LLM analysis failed ({e}) — using minimal fallback[/]"
            )
            return self._minimal_fallback(data, sql)

    # ── Minimal fallback (no LLM available) ─────────────────────

    def _minimal_fallback(
        self, data: dict[str, Any], sql: str
    ) -> dict[str, Any]:
        """
        Minimal static response when both Snowflake and Bedrock
        are unavailable. Does NOT apply regex rules — simply reports
        that analysis could not be performed and recommends manual review.
        """
        return {
            "query_id": data.get("query_id", "unknown"),
            "original_sql": sql,
            "bottlenecks": [],
            "bottleneck_count": 0,
            "severity_score": 0,
            "recommendation": "MANUAL_REVIEW",
            "llm_reasoning": (
                "Automated analysis unavailable — Bedrock LLM is not configured. "
                "Manual review recommended."
            ),
            "query_plan": "",
            "snowflake_metadata": {},
            "analysis_mode": "unavailable",
        }

    # ── Snowflake context fetching ──────────────────────────────

    def _fetch_snowflake_context(
        self, data: dict[str, Any], sql: str
    ) -> tuple[str, dict[str, Any]]:
        """
        Fetch EXPLAIN plan and QUERY_HISTORY from Snowflake.

        Returns (plan_text, metadata_dict). Non-fatal — returns
        empty values on any failure.
        """
        from src.connectors.snowflake_manager import get_connection_manager

        manager = get_connection_manager()
        query_plan_text = ""
        snowflake_metadata: dict[str, Any] = {}

        # ── EXPLAIN plan ────────────────────────────────────────
        try:
            explain_results = manager.explain_query(sql)
            if explain_results:
                plan_lines = []
                for row in explain_results:
                    for col_val in row.values():
                        if col_val:
                            plan_lines.append(str(col_val))
                query_plan_text = "\n".join(plan_lines)
                console.print(
                    f"  📋 EXPLAIN plan retrieved ({len(explain_results)} rows)"
                )
        except Exception as e:
            logger.warning("EXPLAIN failed: %s", e)
            console.print(f"  ⚠️  EXPLAIN failed: {e}")

        # ── QUERY_HISTORY ───────────────────────────────────────
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
            else:
                console.print(
                    "  ℹ️  No QUERY_HISTORY records found (may be within 45-min latency)"
                )
        except Exception as e:
            logger.warning("QUERY_HISTORY fetch failed: %s", e)
            console.print(f"  ⚠️  QUERY_HISTORY unavailable: {e}")

        return query_plan_text, snowflake_metadata
