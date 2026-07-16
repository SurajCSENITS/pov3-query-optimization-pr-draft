"""
Explain Plan Diff Engine.

Compares two Snowflake EXPLAIN plans (original vs. optimized) and
produces a structured ExplainPlanDiff with:
  - Removed and added operations
  - Quantitative metrics (bytes, partitions, rows)
  - Human-readable insights via InsightGenerator

The diff is stored in the LangGraph state and included in the
optimization report that is persisted to S3/RAG.

Usage:
    from src.engines.explain_plan_diff import ExplainPlanDiffEngine

    engine = ExplainPlanDiffEngine()
    diff = engine.compare(original_plan_text, optimized_plan_text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.engines.insight_generator import InsightGenerator

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class DiffMetrics:
    """Quantitative before/after metrics extracted from EXPLAIN output.

    IMPORTANT — bytes_scanned_before / bytes_scanned_after:
        These values come from `bytesAssigned` in `EXPLAIN USING TEXT`, which is a
        PRE-EXECUTION estimate of bytes allocated to each scan node, calculated BEFORE
        micro-partition pruning takes effect.  For well-pruned queries they can be
        3–4× larger than the actual bytes read at runtime.

        Do NOT use them as ground-truth BYTES_SCANNED.  Actual runtime bytes come from
        INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION() and are handled exclusively by
        PerformanceComparisonEngine.
    """
    # EXPLAIN bytesAssigned estimates (pre-pruning, NOT actual runtime scan bytes)
    bytes_scanned_before: int = 0
    bytes_scanned_after: int = 0
    bytes_scanned_reduction_pct: float = 0.0

    partitions_before: int = 0
    partitions_after: int = 0
    partitions_total_before: int = 0
    partitions_total_after: int = 0
    partition_pruning_improvement_pct: float = 0.0

    estimated_rows_before: int = 0
    estimated_rows_after: int = 0

    spill_eliminated: bool = False
    sort_eliminated: bool = False

    # Column scanned metrics
    columns_scanned_before: int = 0
    columns_scanned_after: int = 0
    columns_scanned_reduction_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes_scanned_before": self.bytes_scanned_before,
            "bytes_scanned_after": self.bytes_scanned_after,
            "bytes_scanned_reduction_pct": round(self.bytes_scanned_reduction_pct, 1),
            "partitions_before": self.partitions_before,
            "partitions_after": self.partitions_after,
            "partitions_total_before": self.partitions_total_before,
            "partitions_total_after": self.partitions_total_after,
            "partition_pruning_improvement_pct": round(self.partition_pruning_improvement_pct, 1),
            "estimated_rows_before": self.estimated_rows_before,
            "estimated_rows_after": self.estimated_rows_after,
            "spill_eliminated": self.spill_eliminated,
            "sort_eliminated": self.sort_eliminated,
            "columns_scanned_before": self.columns_scanned_before,
            "columns_scanned_after": self.columns_scanned_after,
            "columns_scanned_reduction_pct": round(self.columns_scanned_reduction_pct, 1),
        }


@dataclass
class ExplainPlanDiff:
    """Complete diff result from comparing two EXPLAIN plans."""
    removed_operations: list[str] = field(default_factory=list)
    added_operations: list[str] = field(default_factory=list)
    modified_operations: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    metrics: DiffMetrics = field(default_factory=DiffMetrics)
    overall_improvement_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed_operations": self.removed_operations,
            "added_operations": self.added_operations,
            "modified_operations": self.modified_operations,
            "insights": self.insights,
            "metrics": self.metrics.to_dict(),
            "overall_improvement_score": round(self.overall_improvement_score, 3),
        }


# ── Engine ───────────────────────────────────────────────────────────────────

class ExplainPlanDiffEngine:
    """
    Compares original and optimized Snowflake EXPLAIN plans.

    Supports EXPLAIN USING TEXT output (most commonly available).
    Falls back to regex-based text analysis when structured JSON
    is not available (which is common with Snowflake free-tier access).
    """

    def __init__(self) -> None:
        self._insight_gen = InsightGenerator()

    def compare(
        self,
        original_plan: str,
        optimized_plan: str,
    ) -> ExplainPlanDiff:
        """
        Compare two EXPLAIN plan text outputs and produce a diff.

        Args:
            original_plan:  EXPLAIN output for the original SQL.
            optimized_plan: EXPLAIN output for the optimized SQL.

        Returns:
            ExplainPlanDiff with insights, metrics, and operation lists.
        """
        diff = ExplainPlanDiff()

        if not original_plan and not optimized_plan:
            diff.insights.append("No EXPLAIN data available — diff skipped.")
            return diff

        # ── Extract operations from both plans ───────────────────
        orig_ops = self._extract_operations(original_plan)
        opt_ops = self._extract_operations(optimized_plan)

        orig_set = set(orig_ops)
        opt_set = set(opt_ops)

        diff.removed_operations = sorted(orig_set - opt_set)
        diff.added_operations = sorted(opt_set - orig_set)

        # ── Extract metrics ─────────────────────────────────────
        diff.metrics = self._extract_metrics(original_plan, optimized_plan)

        # ── Detect sort elimination ─────────────────────────────
        orig_has_sort = self._has_operation(original_plan, "Sort")
        opt_has_sort = self._has_operation(optimized_plan, "Sort")
        if orig_has_sort and not opt_has_sort:
            diff.metrics.sort_eliminated = True

        # ── Detect spill elimination ────────────────────────────
        orig_has_spill = self._has_spill(original_plan)
        opt_has_spill = self._has_spill(optimized_plan)
        if orig_has_spill and not opt_has_spill:
            diff.metrics.spill_eliminated = True

        # ── Generate human-readable insights ─────────────────────
        diff.insights = self._insight_gen.generate(diff)

        # ── Compute overall improvement score ────────────────────
        diff.overall_improvement_score = self._compute_score(diff)

        return diff

    # ── Parsing helpers ──────────────────────────────────────────────────────

    def _extract_operations(self, plan_text: str) -> list[str]:
        """
        Extract operation names from EXPLAIN USING TEXT output.

        Snowflake EXPLAIN text uses patterns like:
          TableScan [ORDERS] ...
          Filter [YEAR(ORDER_DATE) = 2025]
          Sort [ORDER_DATE ASC]
        """
        if not plan_text:
            return []
        ops = []
        # Match leading operation keywords
        pattern = re.compile(
            r"\b(TableScan|Filter|Sort|Join|Aggregate|Project|Limit|"
            r"HashJoin|MergeJoin|NestedLoop|Scan|Expand|Union|Except|"
            r"Intersect|Window|WithReference|WithClause)\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(plan_text):
            ops.append(match.group(0))
        return ops

    def _has_operation(self, plan_text: str, op_name: str) -> bool:
        if not plan_text:
            return False
        return bool(re.search(re.escape(op_name), plan_text, re.IGNORECASE))

    def _has_spill(self, plan_text: str) -> bool:
        if not plan_text:
            return False
        return bool(re.search(r"spill|remote.storage|disk", plan_text, re.IGNORECASE))

    def _extract_scanned_columns_count(self, plan_text: str) -> int:
        """Helper to extract and count unique scanned columns from TableScans in the plan."""
        if not plan_text:
            return 0
        cols = set()
        pattern = re.compile(
            r"TableScan\s+\S+(?:\s+as\s+\S+)?\s+([A-Za-z0-9_, ]+?)(?:\s*\{|\s*$)",
            re.IGNORECASE
        )
        for match in pattern.finditer(plan_text):
            for col in match.group(1).split(","):
                col_name = col.strip()
                if col_name:
                    cols.add(col_name.upper())
        return len(cols)

    def _extract_metrics(self, original: str, optimized: str) -> DiffMetrics:
        """
        Extract numeric metrics from EXPLAIN text via regex.

        Snowflake EXPLAIN USING TEXT embeds stats like:
          bytesAssigned=12884901888
          partitionsAssigned=1024
          partitionsTotal=1024
          estimatedRows=50000000
        """
        m = DiffMetrics()

        def extract_int(text: str, key: str) -> int:
            pattern = rf'["\']?{key}["\']?\s*[=:]\s*(\d+)'
            match = re.search(pattern, text, re.IGNORECASE)
            return int(match.group(1)) if match else 0

        # Extract EXPLAIN bytesAssigned — a pre-pruning estimate of bytes allocated to
        # each scan node. This is NOT the same as the actual BYTES_SCANNED at runtime.
        # The real runtime bytes are retrieved by ValidationAgent from
        # INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION() and override these values
        # in PerformanceComparisonEngine.
        m.bytes_scanned_before = extract_int(original, "bytesAssigned")
        m.bytes_scanned_after = extract_int(optimized, "bytesAssigned")

        m.partitions_before = extract_int(original, "partitionsAssigned")
        m.partitions_after = extract_int(optimized, "partitionsAssigned")

        m.partitions_total_before = extract_int(original, "partitionsTotal")
        m.partitions_total_after = extract_int(optimized, "partitionsTotal")

        m.estimated_rows_before = extract_int(original, "estimatedRows")
        m.estimated_rows_after = extract_int(optimized, "estimatedRows")

        # ── Column Scanned Metrics ──────────────────────────────
        m.columns_scanned_before = self._extract_scanned_columns_count(original)
        m.columns_scanned_after = self._extract_scanned_columns_count(optimized)

        # ── Compute reduction percentages ─────────────────────────
        if m.bytes_scanned_before > 0:
            m.bytes_scanned_reduction_pct = (
                (m.bytes_scanned_before - m.bytes_scanned_after)
                / m.bytes_scanned_before * 100
            )

        if m.partitions_total_before > 0 and m.partitions_total_after > 0:
            pruning_before = (m.partitions_total_before - m.partitions_before) / m.partitions_total_before * 100
            pruning_after = (m.partitions_total_after - m.partitions_after) / m.partitions_total_after * 100
            if pruning_after > pruning_before:
                m.partition_pruning_improvement_pct = pruning_after - pruning_before

        if m.columns_scanned_before > 0:
            m.columns_scanned_reduction_pct = (
                (m.columns_scanned_before - m.columns_scanned_after)
                / m.columns_scanned_before * 100
            )

        return m

    def _compute_score(self, diff: ExplainPlanDiff) -> float:
        """
        Compute an overall improvement score 0.0-1.0.

        Weights:
          - bytes reduction:         30%
          - partition pruning:       20%
          - columns scanned red:     20%
          - removed operations:      15%
          - spill elimination:       7.5%
          - sort elimination:        7.5%
        """
        score = 0.0

        # bytes reduction (cap at 100%)
        bytes_pct = min(diff.metrics.bytes_scanned_reduction_pct, 100.0)
        score += 0.30 * (bytes_pct / 100.0)

        # partition pruning
        prune_pct = min(diff.metrics.partition_pruning_improvement_pct, 100.0)
        score += 0.20 * (prune_pct / 100.0)

        # columns scanned reduction
        cols_pct = min(diff.metrics.columns_scanned_reduction_pct, 100.0)
        score += 0.20 * (cols_pct / 100.0)

        # removed operations (each removal = 7.5%, cap at 15%)
        removals = min(len(diff.removed_operations), 2)
        score += 0.15 * (removals / 2.0)

        # spill elimination
        if diff.metrics.spill_eliminated:
            score += 0.075

        # sort elimination
        if diff.metrics.sort_eliminated:
            score += 0.075

        return round(min(score, 1.0), 3)
