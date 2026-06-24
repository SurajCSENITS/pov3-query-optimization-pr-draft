"""
Validation Evidence models.

Provides structured, proof-carrying representations of every validation
check performed by the ValidationAgent. These objects are stored in
pipeline state and consumed by the HTML Report Generator (Section 6).
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.optimization_report import ExplainDiffSummary


class CheckResult(BaseModel):
    """
    Result of a single validation check.

    `evidence` carries the raw value or excerpt that justifies the
    pass/fail verdict — e.g. the failed check name, the LLM confidence
    score, or the EXPLAIN diff score.
    """

    check_name: str
    passed: bool
    severity: str = "INFO"       # CRITICAL | WARNING | INFO
    detail: str = ""
    evidence: str = ""           # machine-readable proof value


class ValidationEvidence(BaseModel):
    """
    Aggregated evidence bundle for one complete validation run.

    Fields map 1-to-1 with the three stages in ValidationAgent:
      stage1_safety  — SQLSafetyEngine check results
      stage2_diff    — ExplainPlanDiff summary
      stage3_semantic — LLM semantic equivalence check

    `evidence_bundle_id` is a UUID that correlates this evidence
    record with the matching OptimizationReport / HTML report.
    """

    # ── Stage 1: Safety ────────────────────────────────────────────
    stage1_safety: list[CheckResult] = Field(default_factory=list)

    # ── Stage 2: Explain Plan Diff ─────────────────────────────────
    stage2_diff: ExplainDiffSummary = Field(default_factory=ExplainDiffSummary)

    # ── Stage 3: LLM Semantic Check ───────────────────────────────
    stage3_semantic: CheckResult = Field(
        default_factory=lambda: CheckResult(
            check_name="LLM Semantic Equivalence",
            passed=True,
            severity="INFO",
            detail="LLM check not run (Bedrock not configured)",
            evidence="skipped",
        )
    )

    # ── Aggregate ──────────────────────────────────────────────────
    overall_decision: str = "UNKNOWN"   # APPROVED | REVIEW | REJECTED
    confidence_score: Optional[float] = None  # None when LLM not used
    evidence_bundle_id: str = Field(default_factory=lambda: str(uuid4())[:12])
