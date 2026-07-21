"""
Pydantic models for the offline prompt evaluation framework.

These models represent:
  - EvalTestCase      — a single row from the golden JSONL dataset
  - EvalCaseResult    — per-case scoring output after running the evaluator
  - EvalSuiteReport   — aggregate report across all test cases in a run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# ── Golden dataset models ─────────────────────────────────────────────────────

class EvalTestCase(BaseModel):
    """
    A single golden test case loaded from the JSONL dataset.

    Required fields:
        id          — evaluation identifier (can also be a real Snowflake query ID)
        original_sql — the SQL to optimize (maps to `query_text` in the real pipeline)

    Optional POV4-payload fields (used to construct the real pipeline input_data):
        warehouse, credits_used, execution_time_seconds, bytes_scanned, issue_type
        If omitted, sensible defaults are used so the pipeline still runs.
    """

    id: str = Field(description="Unique identifier — can be a real Snowflake query ID or tc_XXX")
    description: str = Field(description="Human-readable description of what is being tested")
    original_sql: str = Field(description="The SQL query to optimize — maps to query_text in the pipeline")
    expected_bottleneck_types: list[str] = Field(
        default_factory=list,
        description="Bottleneck types we expect the optimizer to address (used by AST scorer)",
    )
    expected_min_changes: int = Field(
        default=0,
        description="Minimum number of verified changes we expect the pipeline to produce",
    )
    expected_semantic_equivalence: bool = Field(
        default=True,
        description="Whether the optimized SQL should be semantically equivalent to the original",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for grouping and filtering, e.g. ['select_star', 'critical']",
    )

    # ── Optional POV4 payload overrides ──────────────────────────────────────
    # These let you embed real Snowflake metadata from your QUERY_HISTORY.
    # If absent, the evaluator fills in neutral defaults so the pipeline
    # does not crash (analysis still runs in STANDALONE mode).
    warehouse: str = Field(default="EVAL_WH", description="Snowflake warehouse name")
    credits_used: float = Field(default=0.0, description="Credits consumed by this query")
    execution_time_seconds: float = Field(default=0.0, description="Wall-clock execution time")
    bytes_scanned: int = Field(default=0, description="Bytes scanned by Snowflake")
    issue_type: str = Field(
        default="GENERAL",
        description="Hint to the pipeline about what bottleneck triggered the alert",
    )

    def to_pipeline_payload(self) -> dict:
        """
        Convert this test case into the input_data dict the real pipeline expects.
        Mirrors the POV4 alert payload format shown in main.py:SAMPLE_INPUT.
        """
        return {
            "query_id": self.id,
            "warehouse": self.warehouse,
            "credits_used": self.credits_used,
            "execution_time_seconds": self.execution_time_seconds,
            "bytes_scanned": self.bytes_scanned,
            "issue_type": self.issue_type,
            "query_text": self.original_sql,
        }



# ── Per-case scoring result ───────────────────────────────────────────────────

@dataclass
class PromptScores:
    """
    Individual dimension scores for a single test case.

    All scores are in the range [0.0, 1.0] unless documented otherwise.
    """

    # Did the LLM produce a schema-valid response? (0 = invalid, 1 = valid)
    format_score: float = 0.0

    # Fraction of claimed changes that survived ChangeVerifier (1.0 = no hallucinations)
    hallucination_score: float = 1.0

    # Did the screener report semantic equivalence? (0 or 1)
    semantic_equivalence_score: float = 0.0

    # LLM self-reported confidence from OptimizationResult
    confidence_score: float = 0.0

    # Were the expected bottleneck types (from the golden dataset) actually addressed?
    # Fraction of expected types that appeared in changes_applied.
    bottleneck_coverage_score: float = 0.0

    # Composite score: weighted average of the above dimensions
    composite_score: float = 0.0


@dataclass
class EvalCaseResult:
    """Full evaluation result for a single golden test case."""

    test_case: EvalTestCase
    scores: PromptScores = field(default_factory=PromptScores)

    # Raw LLM outputs for inspection
    optimized_sql: str = ""
    changes_applied: list[dict] = field(default_factory=list)
    verified_changes: list[dict] = field(default_factory=list)
    hallucinated_change_count: int = 0
    claimed_change_count: int = 0
    semantic_equivalent: bool = False
    semantic_confidence: float = 0.0

    # Whether the LLM call itself succeeded
    llm_call_succeeded: bool = True
    error_message: str = ""

    # Extra detail for diagnostics
    raw_llm_result: Any = None


# ── Aggregate suite report ────────────────────────────────────────────────────

@dataclass
class EvalSuiteReport:
    """
    Aggregate evaluation report across all test cases in a run.

    Produced by PromptEvaluator after scoring every case.
    """

    prompt_version: str = ""
    total_cases: int = 0
    passed_cases: int = 0   # cases where composite_score >= pass_threshold
    failed_cases: int = 0
    error_cases: int = 0    # cases where the LLM call itself failed

    # Macro-averages across all cases
    avg_format_score: float = 0.0
    avg_hallucination_score: float = 0.0
    avg_semantic_equivalence_score: float = 0.0
    avg_confidence_score: float = 0.0
    avg_bottleneck_coverage_score: float = 0.0
    avg_composite_score: float = 0.0

    # Worst-performing cases for easier debugging
    worst_cases: list[EvalCaseResult] = field(default_factory=list)

    # All individual results (sorted by composite score ascending)
    results: list[EvalCaseResult] = field(default_factory=list)
