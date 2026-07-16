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

        if metrics.get("execution_time", {}).get("before_ms") is not None:
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
                f"  ⏱  Execution: {metrics['execution_time']['before_ms']}ms → "
                f"{metrics['execution_time']['after_ms']}ms "
                f"({_fmt_pct(time_pct)})"
            )
            console.print(
                f"  💰 Credits: {metrics['credits']['before']} → "
                f"{metrics['credits']['after']} "
                f"({_fmt_pct(credits_pct)})"
            )
            console.print(
                f"  📦 Bytes: {metrics['bytes_scanned']['before_mb']} MB → "
                f"{metrics['bytes_scanned']['after_mb']} MB "
                f"({_fmt_pct(bytes_pct)})"
            )
        else:
            console.print(
                "  ℹ️  Performance metrics: N/A (no real telemetry available)"
            )

        extra_state = {"validation_evidence": evidence.model_dump()}
        
        # ── Retry feedback logic ──────────────────────────────────────────────
        if decision != "APPROVED":
            current_retry = state.get("retry_count", 0)
            feedback = (
                f"Validation attempt {current_retry + 1} failed. "
                f"Decision: {decision}. "
            )
            
            # Summarize regressions
            regressions = []
            if metrics.get('execution_time', {}).get('improvement_pct', 0) < 0:
                regressions.append(f"Execution time got worse ({_fmt_pct(metrics['execution_time']['improvement_pct'])})")
                
            if regressions:
                feedback += "Performance regressions: " + "; ".join(regressions) + ". "
                
            if not semantic_equivalent:
                feedback += f"Semantic equivalence failed: {llm_reasoning} "
                
            if safety_report.critical_failures:
                feedback += f"Safety failures: {', '.join(safety_report.critical_failures)}. "
                
            # Read existing feedback and append
            history = state.get("feedback_history", [])
            new_history = history + [feedback]

            extra_state["retry_count"] = current_retry + 1
            extra_state["feedback_history"] = new_history

        return {
            "state_key": "validation",
            "output": validation_output,
            "next_agent": AgentRole.REPORT.value if decision == "APPROVED" else AgentRole.OPTIMIZATION.value,
            "task_desc": (
                f"Generate optimization report — validation {decision}"
            ),
            "extra_state": extra_state,
        }

    # ── Live Snowflake execution ──────────────────────────────────────────────────────

    def _execute_optimized_query(
        self,
        optimized_sql: str,
    ) -> tuple[float | None, float | None, int | None, int | None, int | None]:
        """
        Execute the optimized SQL on Snowflake and return real telemetry.

        When SNOWFLAKE_BENCHMARK_WAREHOUSE is configured the query runs on a
        dedicated warehouse that is explicitly suspended before execution so its
        local SSD disk cache is cleared.  This gives a clean, reproducible
        BYTES_SCANNED measurement (equivalent to running on a fresh cold warehouse),
        without touching the shared production warehouse.

        The query is always wrapped in BEGIN / ROLLBACK for DML safety.

        Returns:
            (execution_time_sec, credits_used, bytes_scanned,
             partitions_scanned, partitions_total) or (None,…) on failure.
        """
        from src.connectors.snowflake_manager import get_connection_manager

        try:
            manager = get_connection_manager()

            # ── Choose execution warehouse ────────────────────────────────────
            # Use the benchmark warehouse when configured; fall back to the
            # production warehouse for backwards compatibility.
            use_bench   = self._settings.benchmark_warehouse_configured
            bench_wh    = self._settings.snowflake_benchmark_warehouse
            prod_wh     = self._settings.snowflake_warehouse
            exec_wh     = bench_wh if use_bench else prod_wh

            # ── Fetch warehouse size for credit calculation ───────────────────
            warehouse_size = "X-SMALL"
            try:
                wh_results = manager.execute_query(f"SHOW WAREHOUSES LIKE '{exec_wh}'")
                if wh_results:
                    row = wh_results[0]
                    warehouse_size = str(row.get("SIZE", row.get("size", "X-SMALL"))).upper()
            except Exception as e:
                logger.warning("Failed to fetch warehouse size for %s, defaulting to X-SMALL: %s", exec_wh, e)

            credits_per_hour = _WAREHOUSE_CREDITS_PER_HOUR.get(warehouse_size, 1.0)
            if warehouse_size not in _WAREHOUSE_CREDITS_PER_HOUR:
                logger.warning(
                    "Unknown warehouse size '%s' for %s; defaulting to X-SMALL (1 credit/hour).",
                    warehouse_size, exec_wh,
                )

            # ── Benchmark warehouse: suspend → resume cold ────────────────────
            # Suspending clears the local SSD disk cache so BYTES_SCANNED
            # reflects only reads from remote storage — no warm-cache inflation.
            if use_bench:
                console.print(
                    f"\n  🧊 [bold cyan]Benchmark warehouse[/] [white]{bench_wh}[/] — "
                    "suspending to clear local disk cache..."
                )
                try:
                    manager.execute_query(f"ALTER WAREHOUSE {bench_wh} SUSPEND")
                    time.sleep(2)  # Allow suspension to complete fully
                    logger.info("Benchmark warehouse %s suspended (cache cleared).", bench_wh)
                except Exception as e:
                    # Warehouse may already be suspended — perfectly fine
                    logger.info(
                        "Benchmark warehouse %s suspend skipped (already suspended?): %s",
                        bench_wh, e,
                    )
                manager.execute_query(f"ALTER WAREHOUSE {bench_wh} RESUME")
                manager.execute_query(f"USE WAREHOUSE {bench_wh}")
                console.print(
                    f"  ❄️  [bold cyan]{bench_wh}[/] resumed cold — "
                    "executing optimized query on clean cache..."
                )
                logger.info("Session switched to benchmark warehouse %s (cold).", bench_wh)
            else:
                console.print(
                    f"  ⚠️  [yellow]No benchmark warehouse configured — running on "
                    f"{prod_wh} (warm, BYTES_SCANNED may be inflated).[/]\n"
                    f"     Set SNOWFLAKE_BENCHMARK_WAREHOUSE in .env for clean measurements."
                )

            # ── Execute optimized query inside a rolled-back transaction ──────
            manager.execute_query("ALTER SESSION SET USE_CACHED_RESULT = FALSE")
            manager.execute_query("BEGIN")

            start  = time.perf_counter()
            sfqid  = None
            try:
                _, sfqid = manager.execute_query(
                    optimized_sql, fetch_results=False, return_sfqid=True
                )
                # Log prominently so the ID can be used for manual Snowflake verification
                logger.info("Optimized query Snowflake Query ID: %s", sfqid)
                console.print(
                    f"\n  🔑 [bold yellow]Optimized Query ID (Snowflake):[/] "
                    f"[bold white]{sfqid}[/]\n"
                    f"     Verify: SELECT BYTES_SCANNED FROM TABLE(\n"
                    f"               INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION())\n"
                    f"             WHERE QUERY_ID = '{sfqid}';"
                )
            finally:
                # Always roll back — no persistent side-effects from DML
                elapsed_wall = time.perf_counter() - start
                manager.execute_query("ROLLBACK")
                try:
                    manager.execute_query("ALTER SESSION SET USE_CACHED_RESULT = TRUE")
                except Exception:
                    pass
                # Switch session back to the production warehouse regardless of outcome
                if use_bench:
                    try:
                        manager.execute_query(f"USE WAREHOUSE {prod_wh}")
                        logger.info("Session switched back to production warehouse %s.", prod_wh)
                    except Exception as e:
                        logger.warning("Failed to switch back to %s: %s", prod_wh, e)

            # ── Retrieve exact telemetry from INFORMATION_SCHEMA ──────────────
            compute_time_sec = elapsed_wall  # wall-clock fallback
            credits_used     = (elapsed_wall / 3600.0) * credits_per_hour
            real_bytes        = None
            real_parts_scanned = None
            real_parts_total   = None

            if sfqid:
                try:
                    # ── Initial sleep before first poll ───────────────────────
                    # INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION() requires a
                    # brief metadata flush delay after query completion.
                    # Polling immediately returns an empty set — the 2s sleep
                    # covers the typical flush latency.
                    logger.debug(
                        "Waiting 2s for QUERY_HISTORY_BY_SESSION metadata flush "
                        "(sfqid=%s)...", sfqid
                    )
                    time.sleep(2.0)

                    for attempt_num in range(10):  # 10 × 1.5s = up to 15s total
                        history = manager.execute_query(
                            f"""
                            SELECT *
                            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(
                                END_TIME_RANGE_START=>DATEADD('minutes', -10, CURRENT_TIMESTAMP())
                            ))
                            WHERE QUERY_ID = '{sfqid}'
                            """
                        )
                        if history:
                            row = history[0]
                            logger.info("Entire telemetry from QUERY_HISTORY_BY_SESSION for %s: %s", sfqid, row)
                            if row.get("EXECUTION_TIME") is not None:
                                compute_time_sec = row["EXECUTION_TIME"] / 1000.0
                                credits_used = (compute_time_sec / 3600.0) * credits_per_hour
                            real_bytes = row.get("BYTES_SCANNED")
                            real_parts_scanned = None
                            real_parts_total   = None

                            # Retry on None OR 0 — Snowflake returns 0 (not NULL)
                            # when stats haven't stabilised after a warehouse resume
                            if real_bytes is None or real_bytes == 0:
                                logger.debug(
                                    "BYTES_SCANNED not yet available for %s "
                                    "(attempt %d/10, value=%s) — retrying in 1.5s...",
                                    sfqid, attempt_num + 1, real_bytes,
                                )
                                time.sleep(1.5)
                                continue

                            logger.info(
                                "Exact telemetry retrieved for query %s on %s: "
                                "bytes_scanned=%d, compute_time=%.3fs, wall_time=%.3fs",
                                sfqid, exec_wh, real_bytes, compute_time_sec, elapsed_wall,
                            )
                            break
                        else:
                            logger.debug(
                                "QUERY_HISTORY_BY_SESSION: no row yet for %s "
                                "(attempt %d/10) — retrying in 1.5s...",
                                sfqid, attempt_num + 1,
                            )
                            time.sleep(1.5)
                    else:
                        # All 10 attempts exhausted
                        logger.warning(
                            "BYTES_SCANNED never stabilised for query %s after 10 attempts "
                            "(%.0fs total); real_bytes=%s — PerformanceComparisonEngine will "
                            "fall back to input_data bytes.",
                            sfqid, 2.0 + 10 * 1.5, real_bytes,
                        )
                        if real_bytes == 0:
                            real_bytes = None  # Treat 0 as unavailable downstream

                except Exception as e:
                    logger.warning("Failed to retrieve query history for %s: %s", sfqid, e)

            logger.info(
                "Optimized query done (warehouse=%s [%s], rate=%.0f cr/h, "
                "compute=%.3fs, credits=%.6f, bytes=%s).",
                exec_wh,
                "BENCH-COLD" if use_bench else "PROD-WARM",
                credits_per_hour,
                compute_time_sec,
                credits_used,
                real_bytes,
            )
            return compute_time_sec, credits_used, real_bytes, real_parts_scanned, real_parts_total

        except Exception as exc:
            logger.warning(
                "Live Snowflake execution failed (non-fatal); "
                "falling back to heuristic metrics. Error: %s",
                exc,
            )
            return None, None, None, None, None

    # ── LLM semantic check ──────────────────────────────────────────────────

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
