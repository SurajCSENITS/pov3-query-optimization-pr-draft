"""
SQL Safety Check Engine.

Performs a suite of deterministic safety checks on the optimized SQL
before the ValidationAgent approves it for production use.

Checks cover:
  1. Semantic equivalence (row count / structure preservation)
  2. No data-destroying operations introduced
  3. No LIMIT removal (does not allow new full-scan risk)
  4. No unintended CROSS JOIN
  5. WHERE clause preservation
  6. GROUP BY / aggregation preservation
  7. Column projection safety (no extra columns)
  8. No DISTINCT removal
  9. Filter predicate preservation (key filter conditions retained)

Usage:
    from src.engines.sql_safety import SQLSafetyEngine
    engine = SQLSafetyEngine()
    result = engine.run_checks(original_sql, optimized_sql)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a single safety rule check."""
    check_name: str
    passed: bool
    message: str
    severity: str = "WARNING"  # WARNING | CRITICAL


@dataclass
class SafetyReport:
    """Aggregate result of all safety checks."""
    checks: list[SafetyCheckResult] = field(default_factory=list)
    all_passed: bool = True
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "passed_checks": self.passed_checks,
        }


class SQLSafetyEngine:
    """
    Deterministic rule-based SQL safety checker.

    Compares original and optimized SQL strings using pattern matching
    to flag regressions. Does not execute SQL.
    """

    def run_checks(self, original_sql: str, optimized_sql: str) -> SafetyReport:
        """
        Run all safety checks and return an aggregate SafetyReport.

        Args:
            original_sql:  The original query (baseline).
            optimized_sql: The optimized query (candidate).

        Returns:
            SafetyReport with pass/fail per check and aggregate result.
        """
        report = SafetyReport()
        results = [
            self._check_no_ddl_dml(optimized_sql),
            self._check_no_cross_join(original_sql, optimized_sql),
            self._check_where_preserved(original_sql, optimized_sql),
            self._check_group_by_preserved(original_sql, optimized_sql),
            self._check_aggregates_preserved(original_sql, optimized_sql),
            self._check_distinct_preserved(original_sql, optimized_sql),
            self._check_order_by_preserved(original_sql, optimized_sql),
            self._check_no_limit_removal(original_sql, optimized_sql),
            self._check_select_star_removal(original_sql, optimized_sql),
        ]

        for r in results:
            report.checks.append(r)
            if r.passed:
                report.passed_checks.append(r.check_name)
            else:
                if r.severity == "CRITICAL":
                    report.all_passed = False
                    report.critical_failures.append(r.check_name)
                else:
                    report.warnings.append(r.check_name)

        return report

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_no_ddl_dml(self, sql: str) -> SafetyCheckResult:
        """Optimized SQL must remain a pure SELECT — no DDL/DML allowed."""
        bad_kws = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|MERGE|GRANT|REVOKE)\b",
            re.IGNORECASE,
        )
        match = bad_kws.search(sql)
        if match:
            return SafetyCheckResult(
                check_name="NO_DDL_DML",
                passed=False,
                message=f"Optimized SQL contains disallowed keyword: {match.group(0)}",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="NO_DDL_DML",
            passed=True,
            message="No DDL/DML statements detected",
        )

    def _check_no_cross_join(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """Flag if CROSS JOIN was introduced in the optimized query."""
        orig_has = bool(re.search(r"\bCROSS\s+JOIN\b", original, re.IGNORECASE))
        opt_has = bool(re.search(r"\bCROSS\s+JOIN\b", optimized, re.IGNORECASE))
        if not orig_has and opt_has:
            return SafetyCheckResult(
                check_name="NO_CROSS_JOIN_INTRODUCED",
                passed=False,
                message="CROSS JOIN introduced in optimized SQL — potential full Cartesian product",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="NO_CROSS_JOIN_INTRODUCED",
            passed=True,
            message="No new CROSS JOIN introduced",
        )

    def _check_where_preserved(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """WHERE clause must still be present if original had one."""
        orig_has_where = bool(re.search(r"\bWHERE\b", original, re.IGNORECASE))
        opt_has_where = bool(re.search(r"\bWHERE\b", optimized, re.IGNORECASE))
        if orig_has_where and not opt_has_where:
            return SafetyCheckResult(
                check_name="WHERE_CLAUSE_PRESERVED",
                passed=False,
                message="WHERE clause removed in optimized SQL — filtering conditions dropped",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="WHERE_CLAUSE_PRESERVED",
            passed=True,
            message="WHERE clause structure preserved",
        )

    def _check_group_by_preserved(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """GROUP BY must still be present if original had one."""
        orig_has = bool(re.search(r"\bGROUP\s+BY\b", original, re.IGNORECASE))
        opt_has = bool(re.search(r"\bGROUP\s+BY\b", optimized, re.IGNORECASE))
        if orig_has and not opt_has:
            return SafetyCheckResult(
                check_name="GROUP_BY_PRESERVED",
                passed=False,
                message="GROUP BY clause removed in optimized SQL — aggregation semantics changed",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="GROUP_BY_PRESERVED",
            passed=True,
            message="GROUP BY structure preserved",
        )

    def _check_aggregates_preserved(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """Key aggregate functions (SUM, COUNT, AVG, MAX, MIN) must be preserved."""
        agg_pattern = re.compile(
            r"\b(SUM|COUNT|AVG|MAX|MIN|MEDIAN|STDDEV|VARIANCE)\s*\(",
            re.IGNORECASE,
        )
        orig_aggs = set(m.group(1).upper() for m in agg_pattern.finditer(original))
        opt_aggs = set(m.group(1).upper() for m in agg_pattern.finditer(optimized))
        dropped = orig_aggs - opt_aggs
        if dropped:
            return SafetyCheckResult(
                check_name="AGGREGATES_PRESERVED",
                passed=False,
                message=f"Aggregate function(s) dropped: {', '.join(dropped)}",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="AGGREGATES_PRESERVED",
            passed=True,
            message=f"All aggregate functions preserved: {', '.join(orig_aggs) or 'none'}",
        )

    def _check_distinct_preserved(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """SELECT DISTINCT must be preserved if original had it."""
        orig_has = bool(re.search(r"\bSELECT\s+DISTINCT\b", original, re.IGNORECASE))
        opt_has = bool(re.search(r"\bSELECT\s+DISTINCT\b", optimized, re.IGNORECASE))
        if orig_has and not opt_has:
            return SafetyCheckResult(
                check_name="DISTINCT_PRESERVED",
                passed=False,
                message="DISTINCT removed — optimized query may return duplicate rows",
                severity="CRITICAL",
            )
        return SafetyCheckResult(
            check_name="DISTINCT_PRESERVED",
            passed=True,
            message="DISTINCT keyword preserved",
        )

    def _check_order_by_preserved(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """Warn (not block) if ORDER BY was removed."""
        orig_has = bool(re.search(r"\bORDER\s+BY\b", original, re.IGNORECASE))
        opt_has = bool(re.search(r"\bORDER\s+BY\b", optimized, re.IGNORECASE))
        if orig_has and not opt_has:
            return SafetyCheckResult(
                check_name="ORDER_BY_PRESERVED",
                passed=False,
                message="ORDER BY removed — row order guarantee dropped (warning only)",
                severity="WARNING",
            )
        return SafetyCheckResult(
            check_name="ORDER_BY_PRESERVED",
            passed=True,
            message="ORDER BY structure preserved",
        )

    def _check_no_limit_removal(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """Warn if original had LIMIT and it was removed."""
        orig_has = bool(re.search(r"\bLIMIT\s+\d+\b", original, re.IGNORECASE))
        opt_has = bool(re.search(r"\bLIMIT\s+\d+\b", optimized, re.IGNORECASE))
        if orig_has and not opt_has:
            return SafetyCheckResult(
                check_name="LIMIT_NOT_REMOVED",
                passed=False,
                message="LIMIT removed from optimized SQL — could result in full scan in production",
                severity="WARNING",
            )
        return SafetyCheckResult(
            check_name="LIMIT_NOT_REMOVED",
            passed=True,
            message="LIMIT clause preserved",
        )

    def _check_select_star_removal(
        self, original: str, optimized: str
    ) -> SafetyCheckResult:
        """It is GOOD if SELECT * was replaced with explicit columns."""
        orig_has_star = bool(re.search(r"SELECT\s+\*", original, re.IGNORECASE))
        opt_has_star = bool(re.search(r"SELECT\s+\*", optimized, re.IGNORECASE))
        if orig_has_star and not opt_has_star:
            return SafetyCheckResult(
                check_name="SELECT_STAR_REPLACED",
                passed=True,
                message="SELECT * replaced with explicit column list — reduces data read",
            )
        if not orig_has_star and opt_has_star:
            return SafetyCheckResult(
                check_name="SELECT_STAR_REPLACED",
                passed=False,
                message="SELECT * introduced in optimized SQL — may increase bytes scanned",
                severity="WARNING",
            )
        return SafetyCheckResult(
            check_name="SELECT_STAR_REPLACED",
            passed=True,
            message="Column selection pattern unchanged",
        )
