"""
Optimization Report schema.

This Pydantic model defines the structure of an optimization report
that is persisted to S3 and ingested into the Bedrock Knowledge Base.

Every successful optimization run generates one of these reports.
Over time, the collection of reports forms the RAG knowledge base
that makes future optimizations more accurate.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Before/after performance metrics for a single optimization."""
    execution_time_before_sec: float = 0.0
    execution_time_after_sec: float = 0.0
    execution_time_improvement_pct: float = 0.0

    credits_before: float = 0.0
    credits_after: float = 0.0
    credits_improvement_pct: float = 0.0

    bytes_scanned_before_gb: float = 0.0
    bytes_scanned_after_gb: float = 0.0
    bytes_scanned_improvement_pct: float = 0.0

    partition_pruning_before_pct: float = 0.0
    partition_pruning_after_pct: float = 0.0

    spill_eliminated: bool = False


class PerformanceSnapshot(BaseModel):
    """A point-in-time snapshot of query performance numbers."""

    execution_time_sec: float = 0.0
    credits: float = 0.0
    bytes_scanned_gb: float = 0.0
    partition_pruning_pct: float = 0.0


class PerformanceDiff(BaseModel):
    """
    Computed before/after performance delta for a single optimization run.

    `overall_score` is a weighted composite (0–100):
      - 50% execution time improvement
      - 30% credit improvement
      - 20% bytes-scanned improvement

    `verdict` maps the score to a human-readable label:
      ≥ 60  → EXCELLENT
      ≥ 30  → GOOD
      ≥ 10  → MARGINAL
      <  10 → NO_IMPROVEMENT
    """

    before: PerformanceSnapshot = PerformanceSnapshot()
    after: PerformanceSnapshot = PerformanceSnapshot()
    execution_time_improvement_pct: float = 0.0
    credits_improvement_pct: float = 0.0
    bytes_scanned_improvement_pct: float = 0.0
    pruning_improvement_pct: float = 0.0
    overall_score: float = 0.0
    verdict: str = "NO_IMPROVEMENT"  # EXCELLENT | GOOD | MARGINAL | NO_IMPROVEMENT


class ExplainDiffSummary(BaseModel):
    """High-level summary of the Explain Plan diff."""
    removed_operations: list[str] = Field(default_factory=list)
    added_operations: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    rows_reduced_pct: float = 0.0
    bytes_reduced_pct: float = 0.0
    overall_improvement_score: float = 0.0


class OptimizationReport(BaseModel):
    """
    Full optimization report stored in S3 and indexed in Bedrock KB.

    This is the canonical output of a complete optimization pipeline run.
    Fields are designed for both human readability and RAG retrieval.
    """

    # ── Identity ────────────────────────────────────────────────
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── SQL ─────────────────────────────────────────────────────
    original_sql: str = ""
    optimized_sql: str = ""
    original_sql_hash: str = ""
    optimized_sql_hash: str = ""

    # ── Analysis ────────────────────────────────────────────────
    bottleneck_types: list[str] = Field(default_factory=list)
    bottleneck_count: int = 0
    severity_score: int = 0

    # ── Optimization ────────────────────────────────────────────
    optimizations_applied: list[str] = Field(default_factory=list)
    root_cause: str = ""
    optimization_rationale: str = ""

    # ── Explain Plan Diff ────────────────────────────────────────
    explain_diff: ExplainDiffSummary = Field(default_factory=ExplainDiffSummary)

    # ── Performance ─────────────────────────────────────────────
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)

    # ── Validation ──────────────────────────────────────────────
    validation_decision: str = "UNKNOWN"   # APPROVED | REVIEW | REJECTED
    confidence_score: float = 0.0
    semantic_equivalent: bool = True
    safety_checks_passed: list[str] = Field(default_factory=list)
    safety_checks_failed: list[str] = Field(default_factory=list)
    validated: bool = False

    # ── Context ─────────────────────────────────────────────────
    analysis_mode: str = "mock"            # mock | snowflake
    llm_model_used: str = ""
    rag_cases_used: int = 0
    snowflake_context: dict[str, Any] = Field(default_factory=dict)

    # ── Lifecycle helpers ────────────────────────────────────────

    def compute_hashes(self) -> None:
        """Compute and set SQL hash fields."""
        self.original_sql_hash = hashlib.sha256(
            self.original_sql.encode()
        ).hexdigest()[:16]
        self.optimized_sql_hash = hashlib.sha256(
            self.optimized_sql.encode()
        ).hexdigest()[:16]

    @classmethod
    def from_pipeline_state(
        cls,
        analysis: dict[str, Any],
        optimization: dict[str, Any],
        validation: dict[str, Any],
        llm_model: str = "",
        rag_cases_used: int = 0,
    ) -> OptimizationReport:
        """
        Construct an OptimizationReport from the three upstream state outputs.

        This is the factory method called by the ReportAgent after all
        agents have run.
        """
        metrics_raw = validation.get("metrics", {})

        # ── Performance metrics ──────────────────────────────────
        perf = PerformanceMetrics(
            execution_time_before_sec=metrics_raw.get("execution_time", {}).get("before_sec", 0.0),
            execution_time_after_sec=metrics_raw.get("execution_time", {}).get("after_sec", 0.0),
            execution_time_improvement_pct=metrics_raw.get("execution_time", {}).get("improvement_pct", 0.0),
            credits_before=metrics_raw.get("credits", {}).get("before", 0.0),
            credits_after=metrics_raw.get("credits", {}).get("after", 0.0),
            credits_improvement_pct=metrics_raw.get("credits", {}).get("improvement_pct", 0.0),
            bytes_scanned_before_gb=metrics_raw.get("bytes_scanned", {}).get("before_gb", 0.0),
            bytes_scanned_after_gb=metrics_raw.get("bytes_scanned", {}).get("after_gb", 0.0),
            bytes_scanned_improvement_pct=metrics_raw.get("bytes_scanned", {}).get("improvement_pct", 0.0),
            partition_pruning_before_pct=metrics_raw.get("partition_pruning", {}).get("before_pct", 0.0),
            partition_pruning_after_pct=metrics_raw.get("partition_pruning", {}).get("after_pct", 0.0),
        )

        # ── Explain diff ─────────────────────────────────────────
        diff_raw = validation.get("explain_diff", {})
        explain_diff = ExplainDiffSummary(
            removed_operations=diff_raw.get("removed_operations", []),
            added_operations=diff_raw.get("added_operations", []),
            insights=diff_raw.get("insights", []),
            rows_reduced_pct=diff_raw.get("metrics", {}).get("bytes_scanned_reduction_pct", 0.0),
            bytes_reduced_pct=diff_raw.get("metrics", {}).get("bytes_scanned_reduction_pct", 0.0),
            overall_improvement_score=diff_raw.get("overall_improvement_score", 0.0),
        )

        # ── Build optimizations_applied list ─────────────────────
        optimizations_applied = [
            c.get("action", c.get("type", ""))
            for c in optimization.get("changes_applied", [])
        ]

        report = cls(
            query_id=analysis.get("query_id", "unknown"),
            original_sql=optimization.get("original_sql", ""),
            optimized_sql=optimization.get("optimized_sql", ""),
            bottleneck_types=[b.get("type", "") for b in analysis.get("bottlenecks", [])],
            bottleneck_count=analysis.get("bottleneck_count", 0),
            severity_score=analysis.get("severity_score", 0),
            optimizations_applied=optimizations_applied,
            root_cause=analysis.get("root_cause", ""),
            optimization_rationale=optimization.get("rationale", ""),
            explain_diff=explain_diff,
            performance=perf,
            validation_decision=validation.get("decision", "UNKNOWN"),
            confidence_score=validation.get("confidence_score", 0.0),
            semantic_equivalent=validation.get("semantic_equivalent", True),
            safety_checks_passed=validation.get("safety_checks_passed", []),
            safety_checks_failed=validation.get("safety_checks_failed", []),
            validated=validation.get("decision") == "APPROVED",
            analysis_mode=analysis.get("analysis_mode", "mock"),
            llm_model_used=llm_model,
            rag_cases_used=rag_cases_used,
            snowflake_context=analysis.get("snowflake_metadata", {}),
        )
        report.compute_hashes()
        return report

    def to_s3_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for S3 JSON storage."""
        return self.model_dump()
