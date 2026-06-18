"""
Optimization Agent — LLM-powered SQL rewriting using Amazon Nova Pro.

Replaces the MVP's deterministic regex rules with a full LLM pipeline:

  1. Retrieve similar past cases from Bedrock Knowledge Base (RAG)
  2. Build structured prompt with bottleneck context + RAG examples
  3. Invoke Amazon Nova Pro to generate optimized SQL + change log
  4. Parse and validate the structured JSON response
  5. Fall back to rule-based rewrites if LLM is unavailable

The agent is fully backward-compatible: when Bedrock is not configured
(BEDROCK_ENABLED=false), it runs the original deterministic fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState

logger = logging.getLogger(__name__)

# ── Deterministic fallback rules (MVP behaviour) ─────────────────────────────
_FALLBACK_RULES: dict[str, dict[str, str]] = {
    "FULL_COLUMN_SCAN": {
        "action": "Replace SELECT * with explicit column list",
        "pattern": r"SELECT\s+\*",
        "replacement": (
            "SELECT\n"
            "    o.order_id,\n"
            "    o.order_date,\n"
            "    o.order_amount,\n"
            "    c.customer_id,\n"
            "    c.customer_name,\n"
            "    c.country"
        ),
    },
    "NON_SARGABLE_PREDICATE": {
        "action": "Replace YEAR(col) with sargable date range predicate",
        "pattern": r"YEAR\s*\(\s*o\.order_date\s*\)\s*=\s*1995",
        "replacement": "o.order_date BETWEEN '1995-01-01' AND '1995-12-31'",
    },
}


class OptimizationAgent(BaseAgent):
    name = AgentRole.OPTIMIZATION.value
    role = AgentRole.OPTIMIZATION

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        analysis = state["analysis"]
        original_sql = analysis["original_sql"]

        if self._settings.bedrock_configured:
            return self._llm_optimize(original_sql, analysis, state)
        else:
            console.print(
                "  ⚠️  [yellow]Bedrock not configured — using rule-based fallback[/]"
            )
            return self._rule_based_optimize(original_sql, analysis)

    # ── LLM-powered path ─────────────────────────────────────────────────────

    def _llm_optimize(
        self,
        original_sql: str,
        analysis: dict[str, Any],
        state: QueryOptimizationState,
    ) -> dict[str, Any]:
        """
        Call Nova Pro via RAG-augmented prompt to generate optimized SQL.

        Steps:
          1. Retrieve similar past cases from Bedrock Knowledge Base
          2. Build structured prompt
          3. Invoke Nova Pro
          4. Parse JSON response
          5. Return optimized result
        """
        from src.connectors.bedrock_manager import get_bedrock_manager
        from src.connectors.rag_manager import get_rag_manager
        from src.prompts.optimization_prompt import build_optimization_prompt, SYSTEM_PROMPT

        bedrock = get_bedrock_manager()
        rag = get_rag_manager()

        # ── Step 1: RAG retrieval ────────────────────────────────
        bottleneck_types = [b.get("type", "") for b in analysis.get("bottlenecks", [])]
        raw_cases = rag.retrieve_similar_cases(
            bottleneck_types=bottleneck_types,
            sql_fragment=original_sql[:200],
        )
        rag_context = rag.format_as_few_shot_context(raw_cases)
        rag_cases_used = len(raw_cases)

        if rag_cases_used:
            console.print(
                f"  🔍 RAG: retrieved [bold]{rag_cases_used}[/] similar past optimization(s)"
            )
        else:
            console.print("  🔍 RAG: no similar cases found (cold start)")

        # ── Step 2: Build prompt ─────────────────────────────────
        prompt = build_optimization_prompt(
            original_sql=original_sql,
            bottlenecks=analysis.get("bottlenecks", []),
            snowflake_context=analysis.get("snowflake_metadata"),
            rag_context=rag_context,
        )

        # ── Step 3: Invoke Nova Pro ──────────────────────────────
        console.print(
            f"  🤖 Invoking [bold cyan]{self._settings.bedrock_model_id}[/] "
            f"for SQL optimization..."
        )
        try:
            llm_result = bedrock.invoke_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        except (RuntimeError, ValueError) as e:
            console.print(f"  ⚠️  [yellow]LLM call failed ({e}) — falling back to rule-based[/]")
            return self._rule_based_optimize(original_sql, analysis)

        # ── Step 4: Parse response ───────────────────────────────
        optimized_sql = llm_result.get("optimized_sql", original_sql).strip()
        changes_applied = llm_result.get("changes_applied", [])
        rationale = llm_result.get("rationale", "")
        confidence = float(llm_result.get("confidence", 0.0))
        warnings = llm_result.get("warnings", [])

        if not optimized_sql or optimized_sql == original_sql:
            console.print("  ℹ️  LLM returned unchanged SQL — no improvements found")

        # ── Step 5: Pretty-print ─────────────────────────────────
        console.print(
            f"  ✅ LLM optimization complete — [bold]{len(changes_applied)}[/] change(s)"
            f", confidence: [bold green]{confidence:.0%}[/]"
        )
        for c in changes_applied:
            console.print(f"    ↳ [{c.get('type', '?')}] {c.get('action', '')}")
        if warnings:
            for w in warnings:
                console.print(f"    ⚠️  [yellow]{w}[/]")
        console.print(f"\n  [dim]Optimized SQL:[/]\n  [green]{optimized_sql}[/]\n")

        optimization_output = {
            "query_id": analysis["query_id"],
            "original_sql": original_sql,
            "optimized_sql": optimized_sql,
            "changes_applied": changes_applied,
            "change_count": len(changes_applied),
            "rationale": rationale,
            "confidence": confidence,
            "warnings": warnings,
            "optimization_mode": "llm",
            "llm_model": self._settings.bedrock_model_id,
            "rag_cases_used": rag_cases_used,
            "rag_results": raw_cases,           # Sprint 2: persist for HTML Section 5
            "estimated_improvement_pct": min(len(changes_applied) * 25, 85),
        }

        return {
            "state_key": "optimization",
            "output": optimization_output,
            "next_agent": AgentRole.VALIDATION.value,
            "task_desc": (
                f"Validate LLM-optimized query — {len(changes_applied)} changes, "
                f"confidence {confidence:.0%}"
            ),
            # Sprint 2: surface raw RAG results at the top-level state
            "extra_state": {"rag_results": raw_cases},
        }

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_optimize(
        self, original_sql: str, analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Original MVP deterministic rewrite rules.

        Used when Bedrock is not configured or the LLM call fails.
        """
        optimized_sql = original_sql
        changes_applied: list[dict[str, str]] = []
        is_real_tpch = "o_orderdate" in original_sql.lower() or "customer" in original_sql.lower()

        for bottleneck in analysis["bottlenecks"]:
            rule = _FALLBACK_RULES.get(bottleneck["type"])
            if rule:
                pattern = rule["pattern"]
                replacement = rule["replacement"]

                # Adjust for real TPCH schema if detected
                if is_real_tpch:
                    if bottleneck["type"] == "FULL_COLUMN_SCAN":
                        replacement = (
                            "SELECT\n"
                            "    o.o_orderkey,\n"
                            "    o.o_orderdate,\n"
                            "    o.o_totalprice,\n"
                            "    c.c_custkey,\n"
                            "    c.c_name"
                        )
                    elif bottleneck["type"] == "NON_SARGABLE_PREDICATE":
                        pattern = r"YEAR\s*\(\s*o\.o_orderdate\s*\)\s*=\s*1995"
                        replacement = "o.o_orderdate BETWEEN '1995-01-01' AND '1995-12-31'"

                new_sql = re.sub(
                    pattern,
                    replacement,
                    optimized_sql,
                    flags=re.IGNORECASE,
                )
                if new_sql != optimized_sql:
                    changes_applied.append({
                        "bottleneck_id": bottleneck["id"],
                        "type": bottleneck["type"],
                        "action": rule["action"],
                        "reason": f"Resolves {bottleneck['type']} bottleneck",
                    })
                    optimized_sql = new_sql

        # LIMIT guard for spill
        if any(b["type"] == "REMOTE_SPILL" for b in analysis["bottlenecks"]):
            if not re.search(r"LIMIT\s+\d+", optimized_sql, re.IGNORECASE):
                optimized_sql = optimized_sql.rstrip(";") + "\nLIMIT 10000;"
                changes_applied.append({
                    "bottleneck_id": "B003",
                    "type": "REMOTE_SPILL",
                    "action": "Added LIMIT 10000 to bound result set and reduce spill",
                    "reason": "Unbounded query causes remote disk spill",
                })

        console.print(f"  ✏️  Rule-based changes: [bold]{len(changes_applied)}[/]")
        for c in changes_applied:
            console.print(f"    ↳ [{c['bottleneck_id']}] {c['action']}")
        console.print(f"\n  [dim]Optimized SQL:[/]\n  [green]{optimized_sql}[/]\n")

        optimization_output = {
            "query_id": analysis["query_id"],
            "original_sql": original_sql,
            "optimized_sql": optimized_sql,
            "changes_applied": changes_applied,
            "change_count": len(changes_applied),
            "rationale": "Rule-based optimization applied (LLM not configured)",
            "confidence": 0.7,
            "warnings": [],
            "optimization_mode": "rule_based",
            "llm_model": "",
            "rag_cases_used": 0,
            "rag_results": [],                  # Sprint 2: empty — no RAG in fallback
            "estimated_improvement_pct": min(len(changes_applied) * 25, 85),
        }

        return {
            "state_key": "optimization",
            "output": optimization_output,
            "next_agent": AgentRole.VALIDATION.value,
            "task_desc": f"Validate rule-based optimized query — {len(changes_applied)} changes applied",
            "extra_state": {"rag_results": []},  # Sprint 2
        }
