"""
Scoring metrics for the offline prompt evaluation framework.

Each function scores exactly one dimension of the prompt's output:

  - format_score             — Is the LLM output schema-valid?
  - hallucination_score      — What fraction of claimed changes are real?
  - semantic_equivalence_score — Did the screener accept the optimized SQL?
  - confidence_score         — Self-reported LLM confidence (pass-through)
  - bottleneck_coverage_score — Did changes address the expected bottleneck types?
  - composite_score          — Weighted average of all dimensions

All scores return a float in [0.0, 1.0].
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Scoring weights for composite ────────────────────────────────────────────

SCORE_WEIGHTS: dict[str, float] = {
    "format_score":               0.20,
    "hallucination_score":        0.30,
    "semantic_equivalence_score": 0.25,
    "confidence_score":           0.10,
    "bottleneck_coverage_score":  0.15,
}


# ── Individual dimension metrics ─────────────────────────────────────────────

def format_score(llm_call_succeeded: bool, optimized_sql: str) -> float:
    """
    Format Adherence Score.

    Measures whether the LLM produced a schema-valid, parseable response.

    Returns:
        1.0 — LLM call succeeded and returned non-empty SQL.
        0.0 — LLM call failed (exception, JSON parse error, empty SQL).
    """
    if not llm_call_succeeded:
        return 0.0
    if not optimized_sql or not optimized_sql.strip():
        return 0.0
    return 1.0


def hallucination_score(total_claimed: int, total_removed: int) -> float:
    """
    Hallucination Score.

    Measures what fraction of the LLM's claimed changes were real (i.e.,
    survived the ChangeVerifier). A higher score means fewer hallucinations.

    Formula: verified / total_claimed
    Edge case: if the LLM claimed zero changes, it cannot hallucinate → 1.0.

    Returns:
        float in [0.0, 1.0], where 1.0 = no hallucinations.
    """
    if total_claimed == 0:
        return 1.0  # no claims → no hallucinations
    verified = total_claimed - total_removed
    return max(0.0, verified / total_claimed)


def semantic_equivalence_score(
    semantically_equivalent: bool,
    screener_confidence: float,
    expected_equivalence: bool = True,
) -> float:
    """
    Semantic Equivalence Score.

    Checks whether the screener's verdict matches the expected equivalence
    defined in the golden dataset.

    Scoring:
        - Correct verdict (matches expected) + screener_confidence → score
        - Correct verdict but low screener confidence → scaled score
        - Wrong verdict → 0.0

    Returns:
        float in [0.0, 1.0]
    """
    verdict_correct = (semantically_equivalent == expected_equivalence)
    if not verdict_correct:
        return 0.0
    # Scale by screener confidence (min 0.5 so a correct answer is never zero)
    return max(0.5, screener_confidence)


def confidence_score(llm_confidence: float) -> float:
    """
    Confidence Score.

    Pass-through of the LLM's self-reported optimization confidence.
    Already in [0.0, 1.0] by the OptimizationResult schema.

    Returns:
        float in [0.0, 1.0]
    """
    return float(max(0.0, min(1.0, llm_confidence)))


def bottleneck_coverage_score(
    expected_bottleneck_types: list[str],
    original_sql: str,
    optimized_sql: str,
    changes_applied: list[dict],
    expected_min_changes: int,
) -> float:
    """
    Bottleneck Coverage Score — sqlglot AST-based structural diff.

    Instead of matching the `type` string the LLM wrote (which is non-deterministic),
    we use sqlglot to parse both SQLs into ASTs and check whether the optimized SQL
    made a *structural change in the area that corresponds to each expected bottleneck*.

    This is insensitive to how the LLM named the bottleneck type — we verify the
    actual SQL structure changed where we expected, not what label the model used.

    Bottleneck → AST check mapping:
        FULL_COLUMN_SCAN    — SELECT clause no longer contains '*'
        NON_SARGABLE_PREDICATE / FUNCTION_ON_FILTER_COLUMN
                            — WHERE clause AST changed (function wrapping removed)
        CORRELATED_SUBQUERY / NOT_IN_SUBQUERY
                            — Subquery count in WHERE/SELECT reduced
        REPEATED_SUBQUERY / REDUNDANT_SUBQUERY / NESTED_SUBQUERY
                            — CTE (WITH clause) added, or nested subquery depth reduced
        REDUNDANT_DISTINCT / UNNECESSARY_DISTINCT
                            — DISTINCT removed from outer SELECT
        MISSING_LIMIT       — LIMIT clause added
        EXPENSIVE_SORT      — ORDER BY clause removed or reduced
        IMPLICIT_JOIN / UNFILTERED_JOIN / CARTESIAN_PRODUCT
                            — FROM clause changed from comma-join to explicit JOIN ON
        LEADING_WILDCARD    — LIKE pattern changed

    Falls back to change-count check when sqlglot cannot parse either query.

    Returns:
        float in [0.0, 1.0]
    """
    # Case: No optimizations expected
    if not expected_bottleneck_types and expected_min_changes == 0:
        if not changes_applied:
            return 1.0  # Correctly identified no-op
        else:
            penalty = min(1.0, len(changes_applied) * 0.2)
            return max(0.0, 1.0 - penalty)

    # Case: Optimizations expected but none produced
    if not changes_applied:
        return 0.0

    # Try AST-based structural check if expected types are known
    if expected_bottleneck_types:
        try:
            import sqlglot
            import sqlglot.expressions as exp

            orig_ast = sqlglot.parse_one(original_sql, dialect="snowflake")
            opt_ast  = sqlglot.parse_one(optimized_sql,  dialect="snowflake")

            matched = sum(
                1 for et in expected_bottleneck_types
                if _ast_change_detected(et.upper(), orig_ast, opt_ast)
            )
            return matched / len(expected_bottleneck_types)

        except Exception:
            # sqlglot parse failure → fall back to change-count heuristic
            pass

    # Fallback: at least the minimum number of changes were produced
    if len(changes_applied) >= expected_min_changes:
        return 1.0
    return len(changes_applied) / expected_min_changes


# ── AST-level structural change detectors ────────────────────────────────────

def _ast_change_detected(bottleneck_type: str, orig_ast, opt_ast) -> bool:
    """
    Return True if the optimized AST shows a structural change that resolves
    the given bottleneck type. Uses sqlglot expression types for detection.
    """
    import sqlglot.expressions as exp

    _FULL_COLUMN_SCAN_TYPES = {
        "FULL_COLUMN_SCAN", "SELECT_STAR", "FULL_STAR",
    }
    _NON_SARGABLE_TYPES = {
        "NON_SARGABLE_PREDICATE", "NON_SARGABLE",
        "FUNCTION_ON_FILTER_COLUMN", "LEADING_WILDCARD",
    }
    _SUBQUERY_TYPES = {
        "CORRELATED_SUBQUERY", "NOT_IN_SUBQUERY",
    }
    _REPEATED_SUBQUERY_TYPES = {
        "REPEATED_SUBQUERY", "REDUNDANT_SUBQUERY", "NESTED_SUBQUERY",
    }
    _DISTINCT_TYPES = {
        "REDUNDANT_DISTINCT", "UNNECESSARY_DISTINCT",
    }
    _IMPLICIT_JOIN_TYPES = {
        "IMPLICIT_JOIN", "UNFILTERED_JOIN", "CARTESIAN_PRODUCT", "CROSS_JOIN",
    }

    if bottleneck_type in _FULL_COLUMN_SCAN_TYPES:
        # Star removed from SELECT
        orig_stars = len(list(orig_ast.find_all(exp.Star)))
        opt_stars  = len(list(opt_ast.find_all(exp.Star)))
        return opt_stars < orig_stars

    if bottleneck_type in _NON_SARGABLE_TYPES:
        # Any function call in the WHERE clause was removed or reduced
        def _where_func_count(ast):
            where = ast.find(exp.Where)
            if not where:
                return 0
            return len(list(where.find_all(exp.Anonymous))) + len(list(where.find_all(exp.Year))) + len(list(where.find_all(exp.Month)))
        return _where_func_count(opt_ast) < _where_func_count(orig_ast)

    if bottleneck_type in _SUBQUERY_TYPES:
        # Correlated subqueries (Subquery nodes inside WHERE/SELECT) reduced
        def _inner_subquery_count(ast):
            where  = ast.find(exp.Where)
            select = ast.find(exp.Select)
            count  = 0
            for node in [where, select]:
                if node:
                    count += len(list(node.find_all(exp.Subquery)))
            return count
        return _inner_subquery_count(opt_ast) < _inner_subquery_count(orig_ast)

    if bottleneck_type in _REPEATED_SUBQUERY_TYPES:
        # CTE added OR overall subquery depth reduced
        orig_ctes = len(list(orig_ast.find_all(exp.With)))
        opt_ctes  = len(list(opt_ast.find_all(exp.With)))
        if opt_ctes > orig_ctes:
            return True
        # Otherwise check if subquery count dropped
        orig_sq = len(list(orig_ast.find_all(exp.Subquery)))
        opt_sq  = len(list(opt_ast.find_all(exp.Subquery)))
        return opt_sq < orig_sq

    if bottleneck_type in _DISTINCT_TYPES:
        # DISTINCT removed from top-level SELECT
        orig_has_distinct = orig_ast.find(exp.Distinct) is not None
        opt_has_distinct  = opt_ast.find(exp.Distinct)  is not None
        return orig_has_distinct and not opt_has_distinct

    if bottleneck_type == "MISSING_LIMIT":
        orig_has_limit = orig_ast.find(exp.Limit) is not None
        opt_has_limit  = opt_ast.find(exp.Limit)  is not None
        return not orig_has_limit and opt_has_limit

    if bottleneck_type == "EXPENSIVE_SORT":
        orig_has_order = orig_ast.find(exp.Order) is not None
        opt_has_order  = opt_ast.find(exp.Order)  is not None
        return orig_has_order and not opt_has_order

    if bottleneck_type in _IMPLICIT_JOIN_TYPES:
        # Comma-join → explicit JOIN: count Cross/implicit joins in FROM
        orig_cross = len(list(orig_ast.find_all(exp.CrossJoin)))
        opt_cross  = len(list(opt_ast.find_all(exp.CrossJoin)))
        # Also detect comma-style (From with multiple tables)
        orig_from = orig_ast.find(exp.From)
        opt_from  = opt_ast.find(exp.From)
        orig_tables_in_from = len(list(orig_from.find_all(exp.Table))) if orig_from else 0
        opt_tables_in_from  = len(list(opt_from.find_all(exp.Table)))  if opt_from  else 0
        orig_joins = len(list(orig_ast.find_all(exp.Join)))
        opt_joins  = len(list(opt_ast.find_all(exp.Join)))
        # If the original had multiple tables in FROM (implicit join) and the
        # optimized moved them to explicit JOIN ON, that's a detected change.
        return (opt_tables_in_from < orig_tables_in_from and opt_joins > orig_joins) \
            or (opt_cross < orig_cross)

    # Unknown bottleneck type: cannot verify structurally → assume covered
    # if the LLM produced at least one change
    return len(list(opt_ast.find_all(exp.Expression))) != \
           len(list(orig_ast.find_all(exp.Expression)))



def compute_composite_score(
    fmt: float,
    hallucination: float,
    semantic: float,
    confidence: float,
    coverage: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the weighted composite score from all dimension scores.

    Args:
        fmt:           format_score
        hallucination: hallucination_score
        semantic:      semantic_equivalence_score
        confidence:    confidence_score
        coverage:      bottleneck_coverage_score
        weights:       Optional override for SCORE_WEIGHTS.

    Returns:
        float in [0.0, 1.0]
    """
    w = weights or SCORE_WEIGHTS
    composite = (
        fmt        * w["format_score"]
        + hallucination * w["hallucination_score"]
        + semantic  * w["semantic_equivalence_score"]
        + confidence * w["confidence_score"]
        + coverage  * w["bottleneck_coverage_score"]
    )
    return round(min(1.0, max(0.0, composite)), 4)
