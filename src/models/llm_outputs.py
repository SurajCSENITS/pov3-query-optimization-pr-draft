"""
Pydantic models for LLM structured outputs.

These models are used with ChatBedrock.with_structured_output() to
guarantee schema-validated responses from the LLM — replacing all
manual JSON parsing and markdown fence stripping.

Each model maps to a specific agent's LLM call:
  - BottleneckAnalysis  → AnalysisAgent (LLM-backed bottleneck detection)
  - OptimizationResult  → OptimizationAgent (SQL rewrite + change log)
  - SemanticCheckResult → ValidationAgent (semantic equivalence check)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── AnalysisAgent structured output ──────────────────────────────────────────

class Bottleneck(BaseModel):
    """A single performance bottleneck identified by the LLM."""

    id: str = Field(description="Bottleneck identifier, e.g. B001, B002")
    type: str = Field(description="Bottleneck category, e.g. FULL_COLUMN_SCAN, NON_SARGABLE_PREDICATE, REMOTE_SPILL")
    severity: str = Field(description="One of: CRITICAL, HIGH, MEDIUM, LOW")
    description: str = Field(description="Human-readable description of the bottleneck")
    location: str = Field(description="Where in the query the bottleneck occurs, e.g. SELECT clause, WHERE clause")


class BottleneckAnalysis(BaseModel):
    """LLM-generated bottleneck analysis for a SQL query."""

    bottlenecks: list[Bottleneck] = Field(
        default_factory=list,
        description="List of detected performance bottlenecks",
    )
    severity_score: int = Field(
        default=0,
        description="Overall severity score (0-100) based on bottleneck weights",
    )
    recommendation: str = Field(
        default="NO_ACTION",
        description="OPTIMIZE if bottlenecks found, NO_ACTION otherwise",
    )
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought explanation of the analysis",
    )


# ── OptimizationAgent structured output ──────────────────────────────────────

class ChangeApplied(BaseModel):
    """A single optimization change made to the SQL query."""

    type: str = Field(description="Bottleneck type this change addresses, e.g. FULL_COLUMN_SCAN")
    action: str = Field(description="One-sentence description of the change")
    reason: str = Field(description="Why this change improves performance")
    bottleneck_id: str = Field(default="", description="ID of the bottleneck this resolves, e.g. B001")


class OptimizationResult(BaseModel):
    """LLM-generated SQL optimization result."""

    optimized_sql: str = Field(description="Complete optimized SQL query")
    changes_applied: list[ChangeApplied] = Field(
        default_factory=list,
        description="List of optimization changes made",
    )
    rationale: str = Field(
        default="",
        description="2-3 sentence overall optimization rationale",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM confidence in the optimization (0.0-1.0)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any caveats or assumptions made during optimization",
    )


# ── ValidationAgent structured output ────────────────────────────────────────

class SemanticCheckResult(BaseModel):
    """LLM-generated semantic equivalence check result."""

    semantically_equivalent: bool = Field(
        description="Whether the two queries produce identical result sets",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="LLM confidence in the equivalence judgment (0.0-1.0)",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Specific concerns about semantic differences, if any",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the equivalence assessment",
    )
