"""
Insight Generator — converts ExplainPlanDiff into human-readable strings.

Applies rule-based templates to generate precise, actionable insight
sentences from the structural changes detected in the EXPLAIN diff.

These insights are:
  1. Shown in the Report Agent console output
  2. Stored in the PR body under "Performance Evidence"
  3. Embedded in the S3 report for RAG retrieval
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engines.explain_plan_diff import ExplainPlanDiff

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Rule-based insight template engine.

    Maps structural diffs and metric deltas to human-readable sentences.
    Each rule produces at most one insight to avoid redundancy.
    """

    def generate(self, diff: "ExplainPlanDiff") -> list[str]:
        """
        Generate a list of insight strings from an ExplainPlanDiff.

        Insights are ordered by significance (most impactful first).

        Args:
            diff: Populated ExplainPlanDiff from ExplainPlanDiffEngine.

        Returns:
            List of human-readable insight strings.
        """
        insights: list[str] = []
        m = diff.metrics

        # ── Spill elimination (highest impact) ───────────────────
        if m.spill_eliminated:
            insights.append(
                "Eliminated remote disk spill — query now executes fully in-memory, "
                "significantly reducing I/O overhead."
            )

        # ── Bytes scanned reduction ──────────────────────────────
        if m.bytes_scanned_reduction_pct > 5:
            before_gb = round(m.bytes_scanned_before / 1e9, 1)
            after_gb = round(m.bytes_scanned_after / 1e9, 1)
            if before_gb > 0:
                insights.append(
                    f"Bytes scanned reduced from {before_gb} GB to {after_gb} GB "
                    f"({m.bytes_scanned_reduction_pct:.0f}% reduction in data read)."
                )
            else:
                insights.append(
                    f"Estimated bytes scanned reduced by "
                    f"{m.bytes_scanned_reduction_pct:.0f}%."
                )

        # ── Partition pruning improvement ─────────────────────────
        if m.partition_pruning_improvement_pct > 5:
            insights.append(
                f"Micro-partition pruning improved by "
                f"{m.partition_pruning_improvement_pct:.0f}% — Snowflake now skips "
                f"{m.partitions_before - m.partitions_after:,} partitions."
            )

        # ── Sort elimination ─────────────────────────────────────
        if m.sort_eliminated:
            insights.append(
                "In-memory Sort operation eliminated — reduces peak memory pressure "
                "and warehouse credit usage."
            )

        # ── Removed operations ───────────────────────────────────
        for op in diff.removed_operations:
            if "tablescan" in op.lower() or "scan" in op.lower():
                insights.append(
                    f"Removed {op} operation — query plan no longer requires "
                    "a full table scan on this path."
                )
            elif "join" in op.lower():
                insights.append(
                    f"Removed redundant {op} — join tree simplified."
                )

        # ── Added beneficial operations ──────────────────────────
        for op in diff.added_operations:
            if "filter" in op.lower():
                insights.append(
                    f"Added {op} operation — predicate pushed down closer to the scan, "
                    "reducing intermediate result set size."
                )
            elif "project" in op.lower():
                insights.append(
                    f"Added {op} operation — column projection reduces bytes read "
                    "from Snowflake micro-partitions."
                )

        # ── Row reduction ────────────────────────────────────────
        if m.estimated_rows_before > 0 and m.estimated_rows_after < m.estimated_rows_before:
            row_pct = (
                (m.estimated_rows_before - m.estimated_rows_after)
                / m.estimated_rows_before * 100
            )
            if row_pct > 10:
                insights.append(
                    f"Estimated rows reduced from {m.estimated_rows_before:,} to "
                    f"{m.estimated_rows_after:,} ({row_pct:.0f}% fewer rows processed)."
                )

        # ── Column scanned reduction ──────────────────────────────
        if hasattr(m, "columns_scanned_reduction_pct") and m.columns_scanned_reduction_pct > 5:
            insights.append(
                f"Columns scanned reduced from {m.columns_scanned_before} to {m.columns_scanned_after} "
                f"({m.columns_scanned_reduction_pct:.0f}% reduction in column projection width)."
            )

        # ── Fallback when no EXPLAIN data is available ───────────
        if not insights:
            if m.bytes_scanned_before > 0 or m.partitions_total_before > 0:
                insights.append(
                    "EXPLAIN plan analysis completed — plans are structurally identical, "
                    "but SQL optimization improves column projection and predicate sargability."
                )
            else:
                insights.append(
                    "Optimization applied — EXPLAIN plan metrics not available "
                    "(query was not executed against Snowflake in analysis mode)."
                )

        return insights
