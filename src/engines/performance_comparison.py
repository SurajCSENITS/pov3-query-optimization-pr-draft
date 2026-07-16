"""
Performance Comparison Engine.

Computes a structured, reusable before/after performance diff for
a single optimization run. Decouples the performance calculation
logic from ValidationAgent so it can be consumed independently
by both ValidationAgent and the HTML Report Generator.

Usage:
    engine = PerformanceComparisonEngine()
    diff = engine.from_pipeline_data(
        input_data=state["input_data"],
        optimization=state["optimization"],
        explain_diff=diff_result,
        # Optional — pass real Snowflake telemetry captured during validation:
        real_optimized_time=12.4,       # seconds (perf_counter)
        real_optimized_credits=0.0035,  # computed from warehouse size
    )

    When real_optimized_time / real_optimized_credits are provided the engine
    uses them verbatim. When they are None (offline / mock mode), the engine
    reports None for metrics that cannot be determined — it does NOT fabricate
    improvement numbers with hardcoded heuristics.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.optimization_report import PerformanceDiff, PerformanceSnapshot

logger = logging.getLogger(__name__)

# ── Verdict thresholds ───────────────────────────────────────────────────────
_VERDICT_THRESHOLDS = [
    (60.0,  "EXCELLENT"),
    (30.0,  "GOOD"),
    (10.0,  "MARGINAL"),
    (0.0,   "NO_IMPROVEMENT"),
    # Negative scores indicate regression — query got WORSE
]

# ── Composite score weights ──────────────────────────────────────────────────
_W_TIME    = 0.50   # execution time is the primary user-visible metric
_W_CREDITS = 0.30   # cost reduction is the business metric
_W_BYTES   = 0.20   # bytes scanned proxy for scan efficiency


class PerformanceComparisonEngine:
    """
    Stateless engine that computes before/after performance deltas.

    Uses ONLY real telemetry data. When real data is unavailable,
    affected metrics are set to 0.0 with a clear indication that
    they are not available — no synthetic improvement numbers.
    """

    # ── Public factory ────────────────────────────────────────────────────────

    def from_pipeline_data(
        self,
        input_data: dict[str, Any],
        optimization: dict[str, Any],
        explain_diff: Any,                      # ExplainPlanDiff object or dict
        *,
        real_optimized_time: float | None = None,
        real_optimized_credits: float | None = None,
        real_optimized_bytes_scanned: int | None = None,
        real_optimized_partitions_scanned: int | None = None,
        real_optimized_partitions_total: int | None = None,
        real_original_bytes_scanned: int | None = None,
        real_original_partitions_scanned: int | None = None,
        real_original_partitions_total: int | None = None,
    ) -> PerformanceDiff:
        """
        Build a PerformanceDiff from the raw pipeline stage outputs.

        Args:
            input_data:            state["input_data"]   — original POV4 alert payload
            optimization:          state["optimization"] — OptimizationAgent output
            explain_diff:          ExplainPlanDiff object (has .metrics attribute)
            real_optimized_time:   Actual execution duration (seconds) captured by
                                   ValidationAgent via perf_counter.
            real_optimized_credits: Actual credit cost derived from warehouse size and
                                   measured wall-clock time.
        """
        original_time = float(input_data.get("execution_time_seconds", 0.0))
        
        # ── Fix apples-to-oranges credit comparison ─────────────────────────
        # POV4 provides CREDITS_USED_CLOUD_SERVICES, which is massively smaller 
        # than warehouse compute. To ensure a 1:1 comparison against the optimized 
        # query, we must calculate original credits using the exact same pure 
        # compute formula.
        warehouse_size = input_data.get("warehouse_name", "X-SMALL").upper().replace("WH", "").strip("_- ")
        if not warehouse_size or warehouse_size == "":
            warehouse_size = "X-SMALL"
            
        rate_map = {
            "X-SMALL": 1.0, "SMALL": 2.0, "MEDIUM": 4.0, "LARGE": 8.0,
            "X-LARGE": 16.0, "X1": 16.0, "XX-LARGE": 32.0, "X2": 32.0,
            "X3": 64.0, "X4": 128.0, "X5": 256.0, "X6": 512.0
        }
        # Attempt to map standard sizes, fallback to X-SMALL
        rate = rate_map.get(warehouse_size, 1.0)
        original_credits = (original_time / 3600.0) * rate

        # ── Time / credits: use real telemetry when available ─────────────────
        use_real = real_optimized_time is not None and real_optimized_credits is not None

        if use_real:
            optimized_time    = round(real_optimized_time, 1)
            optimized_credits = round(real_optimized_credits, 4)
            logger.info(
                "PerformanceComparisonEngine: using real Snowflake telemetry "
                "(time=%.2fs, credits=%.4f)",
                optimized_time,
                optimized_credits,
            )
        else:
            # No real telemetry — report original values as-is with no fake improvement
            optimized_time    = original_time
            optimized_credits = original_credits
            logger.info(
                "PerformanceComparisonEngine: no real telemetry available — "
                "reporting original values (no fabricated improvement)"
            )

        # ── Bytes scanned ─────────────────────────────────────────────────────
        #
        # EXPLAIN USING TEXT embeds `bytesAssigned` — a static PRE-EXECUTION estimate
        # of bytes allocated to each scan node, computed BEFORE micro-partition pruning.
        # For a well-pruned query this can be 3–4× larger than the actual bytes read.
        # It must NEVER be used as the primary source of truth for bytes scanned.
        #
        # Priority:
        #   bytes_before → real_original_bytes_scanned > input_data["BYTES_SCANNED"]
        #                   > EXPLAIN bytesAssigned (last resort, logged as WARNING)
        #   bytes_after  → real_optimized_bytes_scanned
        #                   > EXPLAIN bytesAssigned (last resort, logged as WARNING)

        # Read EXPLAIN estimates — used as a last-resort fallback only
        try:
            diff_metrics      = explain_diff.metrics
            explain_bytes_before = diff_metrics.bytes_scanned_before
            explain_bytes_after  = diff_metrics.bytes_scanned_after
            explain_bytes_pct    = diff_metrics.bytes_scanned_reduction_pct
        except AttributeError:
            _em = explain_diff.get("metrics", {}) if isinstance(explain_diff, dict) else {}
            explain_bytes_before = _em.get("bytes_scanned_before", 0)
            explain_bytes_after  = _em.get("bytes_scanned_after", 0)
            explain_bytes_pct    = _em.get("bytes_scanned_reduction_pct", 0.0)

        # ── bytes_before: real runtime data always wins over EXPLAIN estimate ─
        bytes_before = 0
        if real_original_bytes_scanned is not None and real_original_bytes_scanned > 0:
            bytes_before = int(real_original_bytes_scanned)
            logger.info(
                "PerformanceComparisonEngine: bytes_before = %d "
                "(source: real_original_bytes_scanned)", bytes_before
            )
        else:
            # POV4 alert payload carries the original query's actual BYTES_SCANNED
            # from ACCOUNT_USAGE — use it before falling back to EXPLAIN estimates.
            original_bytes = input_data.get("BYTES_SCANNED", input_data.get("bytes_scanned", 0))
            if original_bytes:
                bytes_before = int(original_bytes)
                logger.info(
                    "PerformanceComparisonEngine: bytes_before = %d "
                    "(source: input_data BYTES_SCANNED)", bytes_before
                )
            elif explain_bytes_before > 0:
                bytes_before = explain_bytes_before
                logger.warning(
                    "PerformanceComparisonEngine: bytes_before = %d "
                    "(source: EXPLAIN bytesAssigned — pre-pruning estimate, not actual scan bytes)",
                    bytes_before,
                )

        # ── bytes_after: real QUERY_HISTORY telemetry; EXPLAIN is a last resort ─
        bytes_after = 0
        bytes_pct   = 0.0
        if real_optimized_bytes_scanned is not None and real_optimized_bytes_scanned > 0:
            bytes_after = real_optimized_bytes_scanned
            logger.info(
                "PerformanceComparisonEngine: bytes_after = %d "
                "(source: real_optimized_bytes_scanned from INFORMATION_SCHEMA)", bytes_after
            )
        elif explain_bytes_after > 0:
            bytes_after = explain_bytes_after
            logger.warning(
                "PerformanceComparisonEngine: bytes_after = %d "
                "(source: EXPLAIN bytesAssigned — pre-pruning estimate; "
                "real telemetry unavailable. Result may be inflated.)",
                bytes_after,
            )
            bytes_pct = explain_bytes_pct

        # Recompute reduction pct from the final before/after values
        if bytes_before > 0 and bytes_after > 0:
            bytes_pct = (bytes_before - bytes_after) / bytes_before * 100.0

        if bytes_before > 0:
            bytes_before_mb = round(bytes_before / 1e6, 2)
            bytes_after_mb  = round(bytes_after  / 1e6, 2)
            bytes_pct       = round(bytes_pct, 1)
        else:
            # No real bytes data — report 0.0 instead of fabricating numbers
            bytes_before_mb = 0.0
            bytes_after_mb  = 0.0
            bytes_pct       = 0.0
            logger.info(
                "PerformanceComparisonEngine: no bytes data available — "
                "reporting 0.0 (no fabricated bytes)"
            )

        # ── Partition pruning ────────────────────────────────────────────────
        try:
            _p_before = explain_diff.metrics.partitions_before
            _p_after  = explain_diff.metrics.partitions_after
            _p_total_before = explain_diff.metrics.partitions_total_before
            _p_total_after  = explain_diff.metrics.partitions_total_after
        except AttributeError:
            metrics = explain_diff.get("metrics", {}) if isinstance(explain_diff, dict) else {}
            _p_before = metrics.get("partitions_before", 0)
            _p_after = metrics.get("partitions_after", 0)
            _p_total_before = metrics.get("partitions_total_before", 0)
            _p_total_after = metrics.get("partitions_total_after", 0)
            
        if real_optimized_partitions_scanned is not None and real_optimized_partitions_total is not None:
            _p_after = real_optimized_partitions_scanned
            _p_total_after = real_optimized_partitions_total
            
            if real_original_partitions_scanned is not None:
                _p_before = int(real_original_partitions_scanned)
            else:
                original_p = input_data.get("PARTITIONS_SCANNED", input_data.get("partitions_scanned", 0))
                if original_p:
                    _p_before = int(original_p)
                    
            if real_original_partitions_total is not None:
                _p_total_before = int(real_original_partitions_total)
            else:
                original_pt = input_data.get("PARTITIONS_TOTAL", input_data.get("partitions_total", 0))
                if original_pt:
                    _p_total_before = int(original_pt)

        if _p_total_before > 0:
            baseline_total = max(_p_total_before, _p_total_after)
            pruning_before = round((1 - _p_before / baseline_total) * 100, 1)
            pruning_after  = round((1 - _p_after / baseline_total) * 100, 1)
            logger.info(
                "PerformanceComparisonEngine: real pruning data "
                "(before: %d/%d assigned, after: %d/%d assigned; pruning_pct: %.1f%% -> %.1f%%)",
                _p_before, baseline_total, _p_after, baseline_total,
                pruning_before, pruning_after
            )
        else:
            # No partition data — report 0.0 instead of fabricating
            pruning_before = 0.0
            pruning_after  = 0.0
            logger.info(
                "PerformanceComparisonEngine: no EXPLAIN partition data — "
                "reporting 0.0 (no fabricated pruning)"
            )

        # ── Improvement percentages ──────────────────────────────────────────
        time_pct    = round((1 - optimized_time    / original_time)    * 100, 1) if original_time > 0 else 0.0
        credits_pct = round((1 - optimized_credits / original_credits) * 100, 1) if original_credits > 0 else 0.0

        # Log regressions clearly
        if time_pct < 0:
            logger.warning(
                "PerformanceComparisonEngine: REGRESSION detected — "
                "execution time increased by %.1f%% (%.2fs → %.2fs)",
                abs(time_pct), original_time, optimized_time,
            )
        if credits_pct < 0:
            logger.warning(
                "PerformanceComparisonEngine: REGRESSION detected — "
                "credits increased by %.1f%% (%.4f → %.4f)",
                abs(credits_pct), original_credits, optimized_credits,
            )

        before = PerformanceSnapshot(
            execution_time_ms=round(original_time * 1000.0, 1),
            credits=original_credits,
            bytes_scanned_mb=bytes_before_mb,
            partition_pruning_pct=pruning_before,
        )
        after = PerformanceSnapshot(
            execution_time_ms=round(optimized_time * 1000.0, 1),
            credits=optimized_credits,
            bytes_scanned_mb=bytes_after_mb,
            partition_pruning_pct=pruning_after,
        )

        overall_score = self._compute_score(time_pct, credits_pct, bytes_pct)
        verdict       = self._score_to_verdict(overall_score)

        return PerformanceDiff(
            before=before,
            after=after,
            execution_time_improvement_pct=time_pct,
            credits_improvement_pct=credits_pct,
            bytes_scanned_improvement_pct=bytes_pct,
            pruning_improvement_pct=round(pruning_after - pruning_before, 1),
            overall_score=overall_score,
            verdict=verdict,
        )

    def to_metrics_dict(self, diff: PerformanceDiff) -> dict[str, Any]:
        """
        Convert a PerformanceDiff into the legacy metrics dict shape
        expected by ReportAgent, PRAgent, and the rest of the pipeline.
        """
        def _format_bytes(gb: float) -> str:
            if gb == 0.0:
                return "0.0 GB"
            bytes_val = gb * 1e9
            if bytes_val >= 1e9: return f"{bytes_val/1e9:.1f} GB"
            if bytes_val >= 1e6: return f"{bytes_val/1e6:.1f} MB"
            if bytes_val >= 1e3: return f"{bytes_val/1e3:.1f} KB"
            return f"{int(bytes_val)} B"

        return {
            "execution_time": {
                "before_ms": diff.before.execution_time_ms,
                "after_ms": diff.after.execution_time_ms,
                "improvement_pct": diff.execution_time_improvement_pct,
            },
            "credits": {
                "before": round(diff.before.credits, 4),
                "after": round(diff.after.credits, 4),
                "improvement_pct": diff.credits_improvement_pct,
            },
            "bytes_scanned": {
                "before_mb": diff.before.bytes_scanned_mb,
                "after_mb": diff.after.bytes_scanned_mb,
                "improvement_pct": diff.bytes_scanned_improvement_pct,
            },
            "partition_pruning": {
                "before_pct": diff.before.partition_pruning_pct,
                "after_pct": diff.after.partition_pruning_pct,
                "improvement_pct": diff.pruning_improvement_pct,
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_score(
        time_pct: float, credits_pct: float, bytes_pct: float
    ) -> float:
        """Weighted composite score: -100 to +100. Negative = regression."""
        score = (
            time_pct    * _W_TIME
            + credits_pct * _W_CREDITS
            + bytes_pct   * _W_BYTES
        )
        return round(max(-100.0, min(score, 100.0)), 1)

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        """Map score to verdict. Negative scores → REGRESSION."""
        if score < 0:
            return "REGRESSION"
        for threshold, label in _VERDICT_THRESHOLDS:
            if score >= threshold:
                return label
        return "NO_IMPROVEMENT"
