"""
Change Verifier — post-LLM hallucination filter.

After the LLM generates an optimized SQL and a list of "changes_applied",
this engine verifies each claimed change against the actual diff between
the original and optimized SQL.

Changes that cannot be verified (i.e., the LLM claimed a change that
doesn't exist in the SQL diff) are removed from the list and logged as
hallucinations.

This is a GENERIC verifier — it works for any SQL query, not just
specific patterns. It operates by:
  1. Normalising both SQL strings
  2. Extracting clause-level segments (SELECT, FROM, WHERE, JOIN, ORDER BY, etc.)
  3. For each claimed change, checking whether the relevant clause actually differs
  4. Filtering out non-changes (e.g., "kept X unchanged") and hallucinated changes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single claimed change."""

    original_action: str
    verified: bool
    reason: str = ""


@dataclass
class ChangeVerificationReport:
    """Full report of change verification for one optimization run."""

    verified_changes: list[dict] = field(default_factory=list)
    removed_changes: list[VerificationResult] = field(default_factory=list)
    total_claimed: int = 0
    total_verified: int = 0
    total_removed: int = 0


class ChangeVerifier:
    """
    Verifies LLM-claimed optimization changes against actual SQL diffs.

    Usage:
        verifier = ChangeVerifier()
        report = verifier.verify(
            original_sql="SELECT * FROM ...",
            optimized_sql="SELECT col1, col2 FROM ...",
            changes_applied=[{"type": "...", "action": "...", "reason": "..."}],
        )
        # report.verified_changes contains only real changes
        # report.removed_changes contains hallucinated ones
    """

    # ── Keywords that indicate a non-change (the LLM describing inaction) ────
    _NON_CHANGE_PATTERNS = [
        r"\bkept\b.*\b(unchanged|same|as[- ]is|intact)\b",
        r"\bmaintain(s|ed)?\b.*\b(existing|original|current)\b",
        r"\bpreserv(e[ds]?|ing)\b",
        r"\bretain(s|ed)?\b",
        r"\bno\s+change\b",
        r"\bunchanged\b",
    ]

    # ── Clause extraction patterns ───────────────────────────────────────────
    # These extract major SQL clause segments for comparison.
    _CLAUSE_PATTERNS = {
        "select": r"(?i)\bSELECT\b(.*?)(?=\bFROM\b)",
        "from": r"(?i)\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)",
        "where": r"(?i)\bWHERE\b(.*?)(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)",
        "join": r"(?i)\b(?:(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)\s+)*JOIN\b(.*?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|\b(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)\s+JOIN\b|\bJOIN\b|$)",
        "order_by": r"(?i)\bORDER\s+BY\b(.*?)(?=\bLIMIT\b|$)",
        "group_by": r"(?i)\bGROUP\s+BY\b(.*?)(?=\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        "having": r"(?i)\bHAVING\b(.*?)(?=\bORDER\s+BY\b|\bLIMIT\b|$)",
        "limit": r"(?i)\bLIMIT\b(.*?)$",
    }

    # ── Action text → clause mapping keywords ────────────────────────────────
    _ACTION_TO_CLAUSE = {
        "select *": "select",
        "select clause": "select",
        "column": "select",
        "replaced select": "select",
        "specific columns": "select",
        "where clause": "where",
        "where predicate": "where",
        "predicate": "where",
        "filter": "where",
        "sargable": "where",
        "join": "join",
        "join condition": "join",
        "order by": "order_by",
        "sort": "order_by",
        "group by": "group_by",
        "limit": "limit",
        "having": "having",
    }

    def verify(
        self,
        original_sql: str,
        optimized_sql: str,
        changes_applied: list[dict],
    ) -> ChangeVerificationReport:
        """
        Verify each claimed change against the actual SQL diff.

        Args:
            original_sql:    The original SQL query.
            optimized_sql:   The LLM-generated optimized SQL query.
            changes_applied: List of change dicts from the LLM
                             (each has 'type', 'action', 'reason').

        Returns:
            ChangeVerificationReport with verified and removed changes.
        """
        report = ChangeVerificationReport(total_claimed=len(changes_applied))

        # Quick check: if SQL is identical, all changes are hallucinated
        norm_orig = self._normalise(original_sql)
        norm_opt = self._normalise(optimized_sql)

        if norm_orig == norm_opt:
            for change in changes_applied:
                action = change.get("action", "")
                result = VerificationResult(
                    original_action=action,
                    verified=False,
                    reason="SQL is identical after normalisation — no changes were made",
                )
                report.removed_changes.append(result)
                logger.warning(
                    "Hallucinated change removed (SQL identical): %s", action
                )
            report.total_removed = len(changes_applied)
            return report

        # Extract clauses from both SQLs
        orig_clauses = self._extract_clauses(original_sql)
        opt_clauses = self._extract_clauses(optimized_sql)

        for change in changes_applied:
            action = change.get("action", "")
            action_lower = action.lower()

            # ── Check 1: Is this a non-change? ───────────────────
            if self._is_non_change(action_lower):
                result = VerificationResult(
                    original_action=action,
                    verified=False,
                    reason="Describes keeping something unchanged — not an optimization",
                )
                report.removed_changes.append(result)
                logger.warning("Non-change removed: %s", action)
                continue

            # ── Check 2: Does the claimed change match an actual diff? ──
            relevant_clause = self._identify_clause(action_lower)

            if relevant_clause:
                orig_clause_text = orig_clauses.get(relevant_clause, "")
                opt_clause_text = opt_clauses.get(relevant_clause, "")

                if self._normalise(orig_clause_text) == self._normalise(opt_clause_text):
                    # The clause the LLM claims to have changed is identical
                    result = VerificationResult(
                        original_action=action,
                        verified=False,
                        reason=(
                            f"Claimed change in {relevant_clause.upper()} clause, "
                            f"but that clause is identical in both queries"
                        ),
                    )
                    report.removed_changes.append(result)
                    logger.warning(
                        "Hallucinated change removed (clause identical): %s "
                        "(clause: %s)",
                        action,
                        relevant_clause,
                    )
                    continue

            # ── Check 3: Specific pattern checks ────────────────
            specific_result = self._check_specific_patterns(
                action_lower, original_sql, optimized_sql
            )
            if specific_result is not None and not specific_result.verified:
                report.removed_changes.append(specific_result)
                logger.warning(
                    "Hallucinated change removed (specific check): %s — %s",
                    action,
                    specific_result.reason,
                )
                continue

            # ── Change passed verification ───────────────────────
            report.verified_changes.append(change)

        report.total_verified = len(report.verified_changes)
        report.total_removed = len(report.removed_changes)

        if report.total_removed > 0:
            logger.info(
                "Change verification: %d/%d claimed changes verified, "
                "%d hallucinated changes removed",
                report.total_verified,
                report.total_claimed,
                report.total_removed,
            )

        return report

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(sql: str) -> str:
        """Normalise SQL for comparison: lowercase, collapse whitespace, strip."""
        if not sql:
            return ""
        text = sql.lower().strip()
        text = re.sub(r"\s+", " ", text)
        # Remove trailing semicolons
        text = text.rstrip(";").strip()
        return text

    def _extract_clauses(self, sql: str) -> dict[str, str]:
        """Extract major SQL clauses as raw text segments."""
        clauses: dict[str, str] = {}
        for name, pattern in self._CLAUSE_PATTERNS.items():
            match = re.search(pattern, sql, re.DOTALL | re.IGNORECASE)
            if match:
                clauses[name] = match.group(1).strip() if match.group(1) else ""
        return clauses

    def _is_non_change(self, action_lower: str) -> bool:
        """Check if the action text describes a non-change."""
        for pattern in self._NON_CHANGE_PATTERNS:
            if re.search(pattern, action_lower):
                return True
        return False

    def _identify_clause(self, action_lower: str) -> str | None:
        """Identify which SQL clause an action description refers to."""
        for keyword, clause in self._ACTION_TO_CLAUSE.items():
            if keyword in action_lower:
                return clause
        return None

    def _check_specific_patterns(
        self, action_lower: str, original_sql: str, optimized_sql: str
    ) -> VerificationResult | None:
        """
        Check specific well-known patterns that the LLM commonly hallucinates.

        Returns None if no specific check applies (pass through to general check).
        Returns a VerificationResult if a specific check was performed.
        """
        orig_lower = original_sql.lower()
        opt_lower = optimized_sql.lower()

        # ── "Replaced SELECT * with specific columns" ────────────
        if "select *" in action_lower and ("replaced" in action_lower or "column" in action_lower):
            if "select *" not in orig_lower:
                return VerificationResult(
                    original_action=action_lower,
                    verified=False,
                    reason="Claims to replace SELECT * but original query does not use SELECT *",
                )
            if "select *" in opt_lower:
                return VerificationResult(
                    original_action=action_lower,
                    verified=False,
                    reason="Claims to replace SELECT * but optimized query still contains SELECT *",
                )

        # ── "Added filter conditions to the join" ────────────────
        if "added" in action_lower and ("filter" in action_lower or "condition" in action_lower) and "join" in action_lower:
            # Extract JOIN clauses from both and compare
            orig_joins = re.findall(
                r"(?i)\bJOIN\b.*?\bON\b.*?(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
                original_sql,
                re.DOTALL,
            )
            opt_joins = re.findall(
                r"(?i)\bJOIN\b.*?\bON\b.*?(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
                optimized_sql,
                re.DOTALL,
            )
            orig_join_norm = " ".join(self._normalise(j) for j in orig_joins)
            opt_join_norm = " ".join(self._normalise(j) for j in opt_joins)
            if orig_join_norm == opt_join_norm:
                return VerificationResult(
                    original_action=action_lower,
                    verified=False,
                    reason="Claims to add filter conditions to JOIN but JOIN clauses are identical",
                )

        return None  # No specific check applies — pass through
