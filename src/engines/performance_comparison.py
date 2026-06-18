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
    # diff.verdict  →  "EXCELLENT" | "GOOD" | "MARGINAL" | "NO_IMPROVEMENT"
    # diff.overall_score  →  0.0 - 100.0

    When real_optimized_time / real_optimized_credits are provided the engine
    uses them verbatim instead of the simulated 22%-per-change heuristic.
    When they are None (offline / mock mode) the heuristic is kept as the
    fallback so the pipeline degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.optimization_report import PerformanceDiff, PerformanceSnapshot

logger = logging.getLogger(__name__)

# ── Verdict thresholds ───────────────────────────────────────────────────────
_VERDICT_THRESHOLDS = [
    (60.0, "EXCELLENT"),
    (30.0, "GOOD"),
    (10.0, "MARGINAL"),
    (0.0,  "NO_IMPROVEMENT"),
]

# ── Composite score weights ──────────────────────────────────────────────────
_W_TIME    = 0.50   # execution time is the primary user-visible metric
_W_CREDITS = 0.30   # cost reduction is the business metric
_W_BYTES   = 0.20   # bytes scanned proxy for scan efficiency


class PerformanceComparisonEngine:
    """
    Stateless engine that computes before/after performance deltas.

    The engine accepts raw pipeline dictionaries so it requires no
    coupling to specific agent classes — it can be called from any
    context that has access to the pipeline state.
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
    ) -> PerformanceDiff:
        """
        Build a PerformanceDiff from the raw pipeline stage outputs.

        This replaces the inline `_build_metrics()` logic in ValidationAgent.
        The output PerformanceDiff contains the same numbers — they are just
        structured and independently verifiable.

        Args:
            input_data:            state["input_data"]   — original POV4 alert payload
            optimization:          state["optimization"] — OptimizationAgent output
            explain_diff:          ExplainPlanDiff object (has .metrics attribute)
            real_optimized_time:   Actual execution duration (seconds) captured by
                                   ValidationAgent via perf_counter.  When provided,
                                   replaces the simulated improvement-factor estimate.
            real_optimized_credits: Actual credit cost derived from warehouse size and
                                   measured wall-clock time.  When provided, replaces
                                   the heuristic credit estimate.
        """
        change_count     = optimization.get("change_count", 0)
        original_time    = float(input_data.get("execution_time_seconds", 120.0))
        original_credits = float(input_data.get("credits_used", 4.5))

        # ── Time / credits: real telemetry wins; heuristic is the fallback ──────
        use_real = real_optimized_time is not None and real_optimized_credits is not None

        if use_real:
            # Live Snowflake execution — use measured values verbatim
            optimized_time    = round(real_optimized_time, 1)
            optimized_credits = round(real_optimized_credits, 4)
            logger.info(
                "PerformanceComparisonEngine: using real Snowflake telemetry "
                "(time=%.2fs, credits=%.4f)",
                optimized_time,
                optimized_credits,
            )
        else:
            # Offline / mock mode — keep the 22%-per-change heuristic
            improvement_factor = max(0.15, 1.0 - (change_count * 0.22))
            optimized_time    = round(original_time    * improvement_factor, 1)
            optimized_credits = round(original_credits * improvement_factor, 2)
            logger.debug(
                "PerformanceComparisonEngine: using heuristic improvement_factor=%.2f",
                improvement_factor,
            )

        # ── Bytes scanned: use real EXPLAIN diff if available ─────────────────
        try:
            diff_metrics = explain_diff.metrics
            bytes_before  = diff_metrics.bytes_scanned_before
            bytes_after   = diff_metrics.bytes_scanned_after
            bytes_pct     = diff_metrics.bytes_scanned_reduction_pct
        except AttributeError:
            # explain_diff may be a plain dict in fallback paths
            bytes_before = explain_diff.get("metrics", {}).get("bytes_scanned_before", 0) if isinstance(explain_diff, dict) else 0
            bytes_after  = explain_diff.get("metrics", {}).get("bytes_scanned_after", 0)  if isinstance(explain_diff, dict) else 0
            bytes_pct    = 0.0

        if bytes_before <= 0:
            # Fall back to synthetic bytes.
            # When real telemetry is in use, derive a proxy ratio from the
            # measured time improvement so the bytes estimate stays coherent.
            # In mock mode `improvement_factor` is already defined above.
            _bytes_factor = (
                (optimized_time / original_time) if use_real else improvement_factor
            )
            _original_bytes  = 480_000_000_000          # 480 GB sentinel
            _optimized_bytes = int(_original_bytes * _bytes_factor)
            bytes_before_gb  = round(_original_bytes  / 1e9, 1)
            bytes_after_gb   = round(_optimized_bytes / 1e9, 1)
            bytes_pct        = round((1 - _optimized_bytes / _original_bytes) * 100, 1)
        else:
            bytes_before_gb  = round(bytes_before / 1e9, 1)
            bytes_after_gb   = round(bytes_after  / 1e9, 1)
            bytes_pct        = round(bytes_pct, 1)

        # ── Partition pruning ────────────────────────────────────────────────
        # Use real partitionsAssigned and partitionsTotal from the EXPLAIN diff
        # when available. Pruning % = (1 - assigned / total) * 100.
        try:
            _p_before = explain_diff.metrics.partitions_before
            _p_after  = explain_diff.metrics.partitions_after
            _p_total_before = explain_diff.metrics.partitions_total_before
            _p_total_after  = explain_diff.metrics.partitions_total_after
        except AttributeError:
            _p_before = 0
            _p_after = 0
            _p_total_before = 0
            _p_total_after = 0

        if _p_total_before > 0 and _p_total_after > 0:
            pruning_before = round((1 - _p_before / _p_total_before) * 100, 1)
            pruning_after  = round((1 - _p_after / _p_total_after) * 100, 1)
            logger.info(
                "PerformanceComparisonEngine: real pruning data "
                "(before: %d/%d assigned, after: %d/%d assigned; pruning_pct: %.1f%% -> %.1f%%)",
                _p_before, _p_total_before, _p_after, _p_total_after,
                pruning_before, pruning_after
            )
        else:
            # Heuristic fallback when no EXPLAIN partition data is present
            pruning_before = 3.0
            pruning_after  = float(min(91, 3 + change_count * 28))
            logger.debug(
                "PerformanceComparisonEngine: no EXPLAIN partition data — "
                "using heuristic pruning estimate"
            )

        # ── Improvement percentages ──────────────────────────────────────────
        time_pct    = round((1 - optimized_time    / original_time)    * 100, 1)
        credits_pct = round((1 - optimized_credits / original_credits) * 100, 1)

        before = PerformanceSnapshot(
            execution_time_sec=original_time,
            credits=original_credits,
            bytes_scanned_gb=bytes_before_gb,
            partition_pruning_pct=pruning_before,
        )
        after = PerformanceSnapshot(
            execution_time_sec=optimized_time,
            credits=optimized_credits,
            bytes_scanned_gb=bytes_after_gb,
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

        This ensures full backward compatibility while the internals
        are now cleanly computed by this engine.
        """
        return {
            "execution_time": {
                "before_sec":       diff.before.execution_time_sec,
                "after_sec":        diff.after.execution_time_sec,
                "improvement_pct":  diff.execution_time_improvement_pct,
            },
            "credits": {
                "before":           diff.before.credits,
                "after":            diff.after.credits,
                "improvement_pct":  diff.credits_improvement_pct,
            },
            "bytes_scanned": {
                "before_gb":        diff.before.bytes_scanned_gb,
                "after_gb":         diff.after.bytes_scanned_gb,
                "improvement_pct":  diff.bytes_scanned_improvement_pct,
            },
            "partition_pruning": {
                "before_pct":       diff.before.partition_pruning_pct,
                "after_pct":        diff.after.partition_pruning_pct,
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_score(
        time_pct: float, credits_pct: float, bytes_pct: float
    ) -> float:
        """Weighted composite score: 0-100."""
        score = (
            max(0.0, time_pct)    * _W_TIME
            + max(0.0, credits_pct) * _W_CREDITS
            + max(0.0, bytes_pct)   * _W_BYTES
        )
        return round(min(score, 100.0), 1)

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        for threshold, label in _VERDICT_THRESHOLDS:
            if score >= threshold:
                return label
        return "NO_IMPROVEMENT"
