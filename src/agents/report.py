"""
Report Agent — assembles, persists, and prints the optimization report.

Enhancements over MVP:
  1. Builds an OptimizationReport Pydantic model with all pipeline outputs
  2. Uploads the report to S3 (feeds the RAG Knowledge Base)
  3. Includes EXPLAIN plan diff insights in the report
  4. Includes validation decision (APPROVED / REVIEW / REJECTED) in summary
  5. Falls back gracefully when S3 is not configured
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.models.messages import AgentRole
from src.models.optimization_report import OptimizationReport
from src.models.state import QueryOptimizationState

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    name = AgentRole.REPORT.value
    role = AgentRole.REPORT

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        analysis = state["analysis"]
        optimization = state["optimization"]
        validation = state["validation"]
        metrics = validation["metrics"]

        # ── Build structured report ──────────────────────────────────────────
        report = OptimizationReport.from_pipeline_state(
            analysis=analysis,
            optimization=optimization,
            validation=validation,
            llm_model=optimization.get("llm_model", ""),
            rag_cases_used=optimization.get("rag_cases_used", 0),
        )

        # ── Upload to S3 (feeds RAG KB) ──────────────────────────────────────
        decision = validation.get("decision", "APPROVED")
        s3_key = ""
        if decision == "APPROVED":
            s3_key = self._try_s3_upload(report)
        else:
            logger.info("Validation decision is %s — skipping S3 upload", decision)

        # ── Build the legacy changes_table for backward compatibility ─────────
        changes_table = []
        for i, change in enumerate(optimization.get("changes_applied", []), 1):
            changes_table.append({
                "index": i,
                "change": change.get("action", ""),
                "reason": change.get("reason", f"Resolves {change.get('type', '')} bottleneck"),
            })

        # ── Helper: format a metric delta with correct sign-aware wording ──────
        def _fmt_metric(pct: float, unit: str = "%") -> str:
            """Return a human-readable delta string, correctly handling regressions."""
            if pct > 0:
                return f"-{pct}{unit} ✅"
            elif pct < 0:
                return f"+{abs(pct)}{unit} 🔴 REGRESSION"
            else:
                return f"0{unit} (no change)"

        time_pct    = metrics["execution_time"]["improvement_pct"]
        credits_pct = metrics["credits"]["improvement_pct"]
        bytes_pct   = metrics["bytes_scanned"]["improvement_pct"]

        performance_table = {
            "execution_time": (
                f"{metrics['execution_time']['before_sec']}s → "
                f"{metrics['execution_time']['after_sec']}s "
                f"({_fmt_metric(time_pct)})"
            ),
            "credits_consumed": (
                f"{metrics['credits']['before']} → "
                f"{metrics['credits']['after']} "
                f"({_fmt_metric(credits_pct)})"
            ),
            "bytes_scanned": (
                f"{metrics['bytes_scanned']['before_gb']} GB → "
                f"{metrics['bytes_scanned']['after_gb']} GB "
                f"({_fmt_metric(bytes_pct)})"
            ),
            "partition_pruning": (
                f"{metrics['partition_pruning']['before_pct']}% → "
                f"{metrics['partition_pruning']['after_pct']}%"
            ),
        }

        decision = validation.get("decision", "APPROVED")
        decision_icon = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "?")

        # ── Build an accurate summary — never claim improvement if regressing ──
        change_count = optimization.get("change_count", 0)

        def _describe_metric(label: str, pct: float) -> str:
            if pct > 0:
                return f"reducing {label} by {pct}%"
            elif pct < 0:
                return f"INCREASING {label} by {abs(pct)}% (REGRESSION)"
            else:
                return f"{label} unchanged"

        if decision == "REJECTED":
            summary = (
                f"Optimization REJECTED for query {analysis['query_id']} — "
                f"{_describe_metric('execution time', time_pct)}, "
                f"{_describe_metric('credit consumption', credits_pct)}. "
                f"The optimized query performs WORSE than the original and was not approved."
            )
        else:
            summary = (
                f"Optimized query {analysis['query_id']} by applying "
                f"{change_count} change(s), "
                f"{_describe_metric('execution time', time_pct)} and "
                f"{_describe_metric('credit consumption', credits_pct)}. "
                f"Validation: {decision}."
            )

        # ── Explain diff insights ────────────────────────────────────────────
        explain_diff = validation.get("explain_diff", {})
        diff_insights = explain_diff.get("insights", [])

        report_output = {
            "report_id": report.report_id,
            "query_id": analysis["query_id"],
            "title": f"🤖 AI-Generated Query Optimization — {analysis['query_id']}",
            "summary": summary,
            "original_sql": optimization["original_sql"],
            "optimized_sql": optimization["optimized_sql"],
            "changes": changes_table,
            "performance": performance_table,
            "validation_status": validation.get("semantic_check", "PASS"),
            "validation_decision": decision,
            "row_count_match": (
                validation.get("row_count_original", 0)
                == validation.get("row_count_optimized", 0)
            ),
            "explain_diff_insights": diff_insights,
            "optimization_mode": optimization.get("optimization_mode", "unavailable"),
            "llm_model": optimization.get("llm_model", ""),
            "rag_cases_used": optimization.get("rag_cases_used", 0),
            "confidence_score": validation.get("confidence_score"),
            "s3_key": s3_key,
        }

        # ── Pretty-print ─────────────────────────────────────────────────────
        console.print(f"  📄 Report: [bold]{report_output['title']}[/]")
        console.print(f"  {decision_icon} Validation: [bold]{decision}[/]")
        console.print(f"  Summary: {summary}")
        console.print(f"  Changes documented: {len(changes_table)}")
        if diff_insights:
            console.print("  📊 EXPLAIN Plan Insights:")
            for insight in diff_insights[:3]:
                console.print(f"    → {insight}")
        if optimization.get("optimization_mode") == "llm":
            console.print(
                f"  🤖 LLM: [cyan]{optimization.get('llm_model', '')}[/] | "
                f"RAG cases: {optimization.get('rag_cases_used', 0)} | "
                f"Confidence: {validation.get('confidence_score', 0):.0%}"
            )
        if s3_key:
            console.print(f"  ☁️  Report stored: [dim]s3://{self._settings.s3_bucket_name}/{s3_key}[/]")

        return {
            "state_key": "report",
            "output": report_output,
            "next_agent": AgentRole.PR.value,
            "task_desc": f"Create draft PR with optimization report — decision: {decision}",
        }

    # ── S3 upload ─────────────────────────────────────────────────────────────

    def _try_s3_upload(self, report: OptimizationReport) -> str:
        """
        Attempt to upload the report to S3.

        Non-fatal — if AWS credentials or bucket are missing, logs a warning
        and returns an empty string so the pipeline continues.
        """
        if not self._settings.s3_bucket_name:
            logger.info("S3 not configured — skipping report upload")
            return ""
        try:
            from src.connectors.s3_manager import get_s3_manager

            s3 = get_s3_manager()
            key = s3.upload_report(
                report_id=report.report_id,
                data=report.to_s3_dict(),
            )
            return key
        except Exception as e:
            logger.warning("S3 upload failed (non-fatal): %s", e)
            return ""
