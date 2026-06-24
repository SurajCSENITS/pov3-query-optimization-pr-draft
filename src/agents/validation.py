"""
Validation Agent — LLM-powered semantic check + safety rules.

Three-stage validation:

  Stage 1 — Safety Rules (SQLSafetyEngine)
    Deterministic checks: no DDL/DML, WHERE preserved, DISTINCT preserved, etc.
    Any CRITICAL failure → REJECTED immediately.

  Stage 2 — Explain Plan Diff (ExplainPlanDiffEngine)
    Compares EXPLAIN plans (when Snowflake available) or mocks the diff.
    Extracts bytes/partition/operation deltas and human-readable insights.

  Stage 3 — LLM Semantic Equivalence (ChatBedrock + SemanticCheckResult)
    Uses structured output to confirm the two queries are semantically identical.
    Returns typed SemanticCheckResult — no manual JSON parsing.

Decision logic:
  - Any CRITICAL safety failure → REJECTED
  - LLM says NOT equivalent with high confidence → REVIEW
  - All checks pass → APPROVED

The decision is stored in state and drives PR body wording.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.engines.explain_plan_diff import ExplainPlanDiffEngine
from src.engines.performance_comparison import PerformanceComparisonEngine
from src.engines.sql_safety import SQLSafetyEngine
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState
from src.models.validation_evidence import CheckResult, ValidationEvidence

logger = logging.getLogger(__name__)

# Snowflake warehouse sizes → credits consumed per hour.
# Source: Snowflake documentation (standard credit table).
_WAREHOUSE_CREDITS_PER_HOUR: dict[str, float] = {
    "X-SMALL":  1.0,
    "SMALL":     2.0,
    "MEDIUM":    4.0,
    "LARGE":     8.0,
    "X-LARGE":  16.0,
    "2X-LARGE": 32.0,
    "3X-LARGE": 64.0,
    "4X-LARGE": 128.0,
}


# ── Module-level helper ────────────────────────────────────────────────────────

def _diff_to_summary(diff: Any):
    """
    Convert an ExplainPlanDiff object (or dict) to an ExplainDiffSummary.

    Used when the diff object doesn't expose a .to_summary() method.
    Falls back to an empty ExplainDiffSummary if extraction fails.
    """
    from src.models.optimization_report import ExplainDiffSummary
    try:
        diff_dict = diff.to_dict() if hasattr(diff, "to_dict") else (diff if isinstance(diff, dict) else {})
        return ExplainDiffSummary(
            removed_operations=diff_dict.get("removed_operations", []),
            added_operations=diff_dict.get("added_operations", []),
            insights=diff_dict.get("insights", []),
            rows_reduced_pct=diff_dict.get("metrics", {}).get("bytes_scanned_reduction_pct", 0.0),
            bytes_reduced_pct=diff_dict.get("metrics", {}).get("bytes_scanned_reduction_pct", 0.0),
            overall_improvement_score=diff_dict.get("overall_improvement_score", 0.0),
        )
    except Exception:
        from src.models.optimization_report import ExplainDiffSummary
        return ExplainDiffSummary()


class ValidationAgent(BaseAgent):
    name = AgentRole.VALIDATION.value
    role = AgentRole.VALIDATION

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()
        self._safety_engine = SQLSafetyEngine()
        self._diff_engine = ExplainPlanDiffEngine()
        self._perf_engine = PerformanceComparisonEngine()

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        optimization = state["optimization"]
        input_data = state["input_data"]
        analysis = state["analysis"]

        original_sql = optimization["original_sql"]
        optimized_sql = optimization["optimized_sql"]

        # ── Stage 1: Safety checks ───────────────────────────────────────────
        console.print("  🛡️  Running safety checks...")
        safety_report = self._safety_engine.run_checks(original_sql, optimized_sql)

        if safety_report.critical_failures:
            console.print(
                f"  ❌ [red]CRITICAL safety failures[/]: "
                f"{', '.join(safety_report.critical_failures)}"
            )
        else:
            console.print(
                f"  ✅ Safety checks passed "
                f"({len(safety_report.passed_checks)} checks, "
                f"{len(safety_report.warnings)} warning(s))"
            )
        for w in safety_report.warnings:
            console.print(f"    ⚠️  [yellow]{w}[/]")

        # ── Stage 2: Explain Plan Diff ───────────────────────────────────────
        console.print("  📊 Running EXPLAIN plan diff...")
        # Use query_plan from AnalysisAgent, falling back to original_explain if needed
        original_explain = analysis.get("query_plan", "") or analysis.get("original_explain", "")

        # Fetch optimized explain plan from Snowflake if configured
        optimized_explain = ""
        if self._settings.snowflake_configured:
            try:
                from src.connectors.snowflake_manager import get_connection_manager
                manager = get_connection_manager()
                explain_results = manager.explain_query(optimized_sql)
                if explain_results:
                    plan_lines = []
                    for row in explain_results:
                        for col_val in row.values():
                            if col_val:
                                plan_lines.append(str(col_val))
                    optimized_explain = "\n".join(plan_lines)
            except Exception as e:
                logger.warning("Failed to fetch EXPLAIN for optimized query: %s", e)
        else:
            # Fallback if there's any mock optimized_explain injected in analysis
            optimized_explain = analysis.get("optimized_explain", "")

        diff = self._diff_engine.compare(original_explain, optimized_explain)

        console.print(
            f"  📈 EXPLAIN diff: score={diff.overall_improvement_score:.2f}, "
            f"removed_ops={len(diff.removed_operations)}, "
            f"insights={len(diff.insights)}"
        )
        for insight in diff.insights[:3]:
            console.print(f"    → {insight}")

        # ── Stage 3: LLM Semantic Check ──────────────────────────────────────
        semantic_equivalent = True
        llm_confidence = 0.0
        llm_concerns: list[str] = []
        llm_reasoning = ""
        llm_used = False

        if self._settings.bedrock_configured:
            console.print(
                f"  🤖 Semantic check via [bold cyan]"
                f"{self._settings.bedrock_screener_model_id}[/]..."
            )
            semantic_equivalent, llm_confidence, llm_concerns, llm_reasoning, llm_used = (
                self._llm_semantic_check(original_sql, optimized_sql)
            )
        else:
            console.print(
                "  ⚠️  [yellow]Bedrock not configured — skipping LLM semantic check[/]"
            )

        # ── Stage 3½: Live Snowflake execution (optional) ───────────────────
        real_optimized_time: float | None = None
        real_optimized_credits: float | None = None
        real_bytes: int | None = None
        real_parts_scanned: int | None = None
        real_parts_total: int | None = None

        if self._settings.snowflake_configured:
            console.print("  ⏱️  Executing optimized query on Snowflake for real telemetry...")
            real_optimized_time, real_optimized_credits, real_bytes, real_parts_scanned, real_parts_total = (
                self._execute_optimized_query(optimized_sql)
            )
            if real_optimized_time is not None:
                console.print(
                    f"  ✅ Live execution: "
                    f"{real_optimized_time:.2f}s, "
                    f"{real_optimized_credits:.4f} credits"
                )
            else:
                console.print(
                    "  ⚠️  [yellow]Live execution failed — falling back to heuristic metrics[/]"
                )
        else:
            console.print(
                "  ⚠️  [yellow]Snowflake not configured — using simulated performance metrics[/]"
            )

        # ── Performance metrics via PerformanceComparisonEngine ───────────────
        snowflake_metadata = analysis.get("snowflake_metadata", {})
        
        perf_diff = self._perf_engine.from_pipeline_data(
            input_data=input_data,
            optimization=optimization,
            explain_diff=diff,
            real_optimized_time=real_optimized_time,
            real_optimized_credits=real_optimized_credits,
            real_optimized_bytes_scanned=real_bytes,
            real_optimized_partitions_scanned=real_parts_scanned,
            real_optimized_partitions_total=real_parts_total,
            real_original_bytes_scanned=snowflake_metadata.get("bytes_scanned"),
            real_original_partitions_scanned=snowflake_metadata.get("partitions_scanned"),
            real_original_partitions_total=snowflake_metadata.get("partitions_total"),
        )
        metrics = self._perf_engine.to_metrics_dict(perf_diff)

        # ── Decision logic (AFTER performance, so regressions are detected) ──
        decision = self._make_decision(
            safety_report=safety_report,
            semantic_equivalent=semantic_equivalent,
            llm_confidence=llm_confidence,
            perf_diff=perf_diff,
        )

        # ── Build ValidationEvidence bundle ──────────────────────────────────
        stage1_results: list[CheckResult] = []
        for check in safety_report.passed_checks:
            stage1_results.append(CheckResult(
                check_name=check,
                passed=True,
                severity="INFO",
                detail="Check passed",
                evidence="passed",
            ))
        for check in safety_report.critical_failures:
            stage1_results.append(CheckResult(
                check_name=check,
                passed=False,
                severity="CRITICAL",
                detail="Critical safety failure — optimization rejected",
                evidence="failed",
            ))
        for check in safety_report.warnings:
            stage1_results.append(CheckResult(
                check_name=check,
                passed=True,
                severity="WARNING",
                detail="Check passed with warning",
                evidence="warning",
            ))

        stage3_check = CheckResult(
            check_name="LLM Semantic Equivalence",
            passed=semantic_equivalent,
            severity="WARNING" if not semantic_equivalent else "INFO",
            detail=(
                f"LLM confirmed {'equivalent' if semantic_equivalent else 'NOT equivalent'} "
                f"(confidence={llm_confidence:.0%}). {llm_reasoning}"
            ) if llm_used else "LLM check not run (Bedrock not configured)",
            evidence=(
                f"confidence={llm_confidence:.2f}; concerns={'; '.join(llm_concerns)}"
                if llm_concerns else f"confidence={llm_confidence:.2f}"
            ),
        )

        evidence = ValidationEvidence(
            stage1_safety=stage1_results,
            stage2_diff=diff.to_summary() if hasattr(diff, "to_summary") else _diff_to_summary(diff),
            stage3_semantic=stage3_check,
            overall_decision=decision,
            confidence_score=llm_confidence if llm_used else None,
        )

        # ── Assemble output ──────────────────────────────────────────────────
        validation_output = {
            "query_id": optimization["query_id"],
            "decision": decision,
            "is_valid": decision == "APPROVED",
            "semantic_check": "PASS" if semantic_equivalent else "FAIL",
            "semantic_equivalent": semantic_equivalent,
            "llm_confidence": llm_confidence,
            "llm_concerns": llm_concerns,
            "llm_reasoning": llm_reasoning,
            "llm_used": llm_used,

            # Safety
            "safety_checks_passed": safety_report.passed_checks,
            "safety_checks_failed": safety_report.critical_failures,
            "safety_warnings": safety_report.warnings,

            # Explain diff
            "explain_diff": diff.to_dict(),

            # Performance
            "metrics": metrics,
            "perf_diff": perf_diff.model_dump(),

            # Convenience fields used by ReportAgent
            "confidence_score": llm_confidence if llm_used else None,

            # Structured evidence bundle
            "validation_evidence": evidence.model_dump(),
        }

        # ── Final console summary ─────────────────────────────────────────────
        status_icon = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "?")
        console.print(
            f"\n  {status_icon} Validation decision: [bold]{decision}[/] "
            f"(safety={'PASS' if not safety_report.critical_failures else 'FAIL'}, "
            f"semantic={'PASS' if semantic_equivalent else 'FAIL'}, "
            f"confidence={llm_confidence:.0%})"
        )

        if metrics.get("execution_time", {}).get("before_sec") is not None:
            time_pct = metrics['execution_time']['improvement_pct']
            credits_pct = metrics['credits']['improvement_pct']
            bytes_pct = metrics['bytes_scanned']['improvement_pct']

            def _fmt_pct(pct: float) -> str:
                """Format percentage with color: green for improvement, red for regression."""
                if pct > 0:
                    return f"[green]↓{pct}%[/]"
                elif pct < 0:
                    return f"[red]↑{abs(pct)}% REGRESSION[/]"
                else:
                    return f"[dim]0.0% (no change)[/]"

            console.print(
                f"  ⏱  Execution: {metrics['execution_time']['before_sec']}s → "
                f"{metrics['execution_time']['after_sec']}s "
                f"({_fmt_pct(time_pct)})"
            )
            console.print(
                f"  💰 Credits: {metrics['credits']['before']} → "
                f"{metrics['credits']['after']} "
                f"({_fmt_pct(credits_pct)})"
            )
            console.print(
                f"  📦 Bytes: {metrics['bytes_scanned']['before_gb']} GB → "
                f"{metrics['bytes_scanned']['after_gb']} GB "
                f"({_fmt_pct(bytes_pct)})"
            )
        else:
            console.print(
                "  ℹ️  Performance metrics: N/A (no real telemetry available)"
            )

        return {
            "state_key": "validation",
            "output": validation_output,
            "next_agent": AgentRole.REPORT.value,
            "task_desc": (
                f"Generate optimization report — validation {decision}"
            ),
            "extra_state": {"validation_evidence": evidence.model_dump()},
        }

    # ── Live Snowflake execution ────────────────────────────────────────────

    def _execute_optimized_query(
        self,
        optimized_sql: str,
    ) -> tuple[float | None, float | None, int | None, int | None, int | None]:
        """
        Execute the optimized SQL inside a transaction that is always
        rolled back so no data is modified (DML/DDL safety).

        Returns:
            (execution_time_sec, credits_used, bytes_scanned, partitions_scanned, partitions_total) on success, or
            (None, None, None, None, None) on any connection / execution failure.
        """
        from src.connectors.snowflake_manager import get_connection_manager

        try:
            manager = get_connection_manager()

            # Determine actual warehouse size from Snowflake
            warehouse_name = self._settings.snowflake_warehouse
            warehouse_size = "X-SMALL"
            try:
                wh_results = manager.execute_query(f"SHOW WAREHOUSES LIKE '{warehouse_name}'")
                if wh_results:
                    row = wh_results[0]
                    warehouse_size = str(row.get("SIZE", row.get("size", "X-SMALL"))).upper()
            except Exception as e:
                logger.warning("Failed to fetch warehouse size, defaulting to X-SMALL: %s", e)

            credits_per_hour = _WAREHOUSE_CREDITS_PER_HOUR.get(warehouse_size, 1.0)
            if warehouse_size not in _WAREHOUSE_CREDITS_PER_HOUR:
                logger.warning(
                    "Unknown warehouse size '%s'; defaulting to X-SMALL (1 credit/hour).",
                    warehouse_size,
                )

            # Open an explicit transaction — the ROLLBACK guarantees no
            # persistent writes even if the query performs DML.
            manager.execute_query("ALTER SESSION SET USE_CACHED_RESULT = FALSE")
            manager.execute_query("BEGIN")

            start = time.perf_counter()
            sfqid = None
            try:
                _, sfqid = manager.execute_query(
                    optimized_sql, fetch_results=False, return_sfqid=True
                )
            finally:
                # Always roll back — even if the query raised an exception.
                elapsed_wall = time.perf_counter() - start
                manager.execute_query("ROLLBACK")
                try:
                    manager.execute_query("ALTER SESSION SET USE_CACHED_RESULT = TRUE")
                except Exception:
                    pass

            # ── Retrieve Exact Telemetry from Session History ───────────────
            compute_time_sec = elapsed_wall  # Fallback
            credits_used = (elapsed_wall / 3600.0) * credits_per_hour
            real_bytes = None
            real_parts_scanned = None
            real_parts_total = None

            if sfqid:
                try:
                    for _ in range(6):  # Retry up to 3 seconds for metadata to flush
                        history = manager.execute_query(
                            f"""
                            SELECT EXECUTION_TIME, BYTES_SCANNED
                            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION())
                            WHERE QUERY_ID = '{sfqid}'
                            """
                        )
                        if history:
                            row = history[0]
                            if row.get("EXECUTION_TIME") is not None:
                                # EXECUTION_TIME is in milliseconds and represents pure compute
                                compute_time_sec = row["EXECUTION_TIME"] / 1000.0
                                credits_used = (compute_time_sec / 3600.0) * credits_per_hour
                            real_bytes = row.get("BYTES_SCANNED")
                            real_parts_scanned = None
                            real_parts_total = None
                            
                            # If BYTES_SCANNED is still None, it might still be flushing metadata, continue waiting
                            if real_bytes is None:
                                time.sleep(0.5)
                                continue
                                
                            logger.info(
                                "Exact Snowflake telemetry retrieved for query %s: "
                                "compute_time=%.3fs vs wall_time=%.3fs",
                                sfqid, compute_time_sec, elapsed_wall
                            )
                            break
                        time.sleep(0.5)
                except Exception as e:
                    logger.warning("Failed to retrieve query history for %s: %s", sfqid, e)

            logger.info(
                "Optimized query executed (warehouse=%s, rate=%.0f cr/h, "
                "compute=%.3fs, credits=%.6f).",
                warehouse_size,
                credits_per_hour,
                compute_time_sec,
                credits_used,
            )
            return compute_time_sec, credits_used, real_bytes, real_parts_scanned, real_parts_total

        except Exception as exc:
            logger.warning(
                "Live Snowflake execution failed (non-fatal); "
                "falling back to heuristic metrics. Error: %s",
                exc,
            )
            return None, None, None, None, None

    # ── LLM semantic check ────────────────────────────────────────────────────

    def _llm_semantic_check(
        self, original_sql: str, optimized_sql: str
    ) -> tuple[bool, float, list[str], str, bool]:
        """
        Ask the LLM if the two queries are semantically equivalent.

        Uses ChatBedrock with structured output (SemanticCheckResult)
        for schema-validated responses.

        Returns:
            (is_equivalent, confidence, concerns, reasoning, llm_was_used)
        """
        from src.connectors.bedrock_manager import get_screener_llm
        from src.models.llm_outputs import SemanticCheckResult
        from src.prompts.optimization_prompt import SCREENER_PROMPT

        try:
            llm = get_screener_llm().with_structured_output(SemanticCheckResult)
            result: SemanticCheckResult = (SCREENER_PROMPT | llm).invoke({
                "original_sql": original_sql.strip(),
                "optimized_sql": optimized_sql.strip(),
            })

            if not result.semantically_equivalent:
                console.print(
                    f"    ⚠️  LLM semantic check: [red]NOT equivalent[/] "
                    f"(confidence={result.confidence:.0%})"
                )
                for c in result.concerns:
                    console.print(f"       → {c}")
            else:
                console.print(
                    f"    ✅ LLM semantic check: [green]equivalent[/] "
                    f"(confidence={result.confidence:.0%})"
                )
            return (
                result.semantically_equivalent,
                result.confidence,
                result.concerns,
                result.reasoning,
                True,
            )
        except Exception as e:
            logger.warning("LLM semantic check failed (non-fatal): %s", e)
            console.print(
                f"    ⚠️  [yellow]LLM screener failed ({e}) — assuming equivalent[/]"
            )
            return True, 0.0, [], "", False

    # ── Decision logic ────────────────────────────────────────────────────────

    def _make_decision(
        self,
        safety_report: Any,
        semantic_equivalent: bool,
        llm_confidence: float,
        perf_diff: Any = None,
    ) -> str:
        """
        Map check results to APPROVED / REVIEW / REJECTED.

        Decision factors (in priority order):
          1. CRITICAL safety failures → REJECTED
          2. Significant performance regression → REJECTED
          3. Minor performance regression → REVIEW
          4. Semantic non-equivalence with high confidence → REVIEW
          5. All checks pass and no regression → APPROVED
        """
        threshold = self._settings.validation_confidence_threshold

        # ── Safety failures are always fatal ──────────────────────
        if safety_report.critical_failures:
            return "REJECTED"

        # ── Performance regression detection ──────────────────────
        if perf_diff is not None:
            time_pct = getattr(perf_diff, "execution_time_improvement_pct", 0.0)
            credits_pct = getattr(perf_diff, "credits_improvement_pct", 0.0)

            # Significant regression: execution time or credits got >5% worse
            if time_pct < -5.0 or credits_pct < -5.0:
                console.print(
                    f"  ❌ [red]PERFORMANCE REGRESSION detected — "
                    f"time: {time_pct:+.1f}%, credits: {credits_pct:+.1f}%[/]"
                )
                return "REJECTED"

            # Minor regression: any negative improvement
            if time_pct < 0.0 or credits_pct < 0.0:
                console.print(
                    f"  ⚠️  [yellow]Minor performance regression — "
                    f"time: {time_pct:+.1f}%, credits: {credits_pct:+.1f}%[/]"
                )
                return "REVIEW"

        # ── Semantic equivalence ──────────────────────────────────
        if not semantic_equivalent and llm_confidence >= threshold:
            return "REVIEW"

        return "APPROVED"
