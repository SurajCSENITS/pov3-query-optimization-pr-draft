"""
Optimization Agent — LLM-powered SQL rewriting using ChatBedrock.

All optimization is performed by the LLM. There are NO hardcoded regex
rewrite rules — this was the core feedback from Barandeep Singh:
"The model must determine the necessary steps based on guidelines
and examples, not hard-coded Python scripts."

Pipeline:
  1. Retrieve similar past cases from Bedrock Knowledge Base (RAG)
  2. Build structured prompt with bottleneck context + RAG examples
  3. Invoke ChatBedrock with structured output (OptimizationResult)
  4. Return validated, schema-guaranteed result

When Bedrock is not configured, the agent emits a no-change result
with a clear recommendation for manual review — it does NOT fall back
to regex substitution rules.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState

logger = logging.getLogger(__name__)


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
                "  ⚠️  [yellow]Bedrock not configured — optimization unavailable[/]"
            )
            return self._no_change_fallback(original_sql, analysis)

    # ── LLM-powered optimization ─────────────────────────────────────────────

    def _llm_optimize(
        self,
        original_sql: str,
        analysis: dict[str, Any],
        state: QueryOptimizationState,
    ) -> dict[str, Any]:
        """
        Call ChatBedrock via RAG-augmented prompt to generate optimized SQL.

        Uses LangChain's structured output to guarantee schema-validated
        responses — no manual JSON parsing needed.
        """
        from src.connectors.bedrock_manager import get_llm
        from src.connectors.rag_manager import (
            retrieve_similar_cases,
            format_as_few_shot_context,
        )
        from src.models.llm_outputs import OptimizationResult
        from src.prompts.optimization_prompt import (
            OPTIMIZATION_PROMPT,
            format_bottleneck_section,
            format_snowflake_context,
        )

        # ── Step 1: RAG retrieval ────────────────────────────────
        bottleneck_types = [b.get("type", "") for b in analysis.get("bottlenecks", [])]
        raw_cases = retrieve_similar_cases(
            bottleneck_types=bottleneck_types,
            sql_fragment=original_sql[:200],
        )
        rag_context = format_as_few_shot_context(raw_cases)
        rag_cases_used = len(raw_cases)

        if rag_cases_used:
            console.print(
                f"  🔍 RAG: retrieved [bold]{rag_cases_used}[/] similar past optimization(s)"
            )
        else:
            console.print("  🔍 RAG: no similar cases found (cold start)")

        # ── Step 2: Build prompt inputs ──────────────────────────
        # Inject feedback from previous validation failures if this is a retry
        feedback_history = state.get("feedback_history", [])
        feedback_section = ""
        if feedback_history:
            feedback_section = "## Validation Feedback\n" + "\n".join(
                f"- {msg}" for msg in feedback_history
            )
            console.print(f"  🔄 Retrying: injecting {len(feedback_history)} previous failure feedback(s)")

        prompt_inputs = {
            "original_sql": original_sql.strip(),
            "bottleneck_section": format_bottleneck_section(
                analysis.get("bottlenecks", [])
            ),
            "snowflake_context": format_snowflake_context(
                analysis.get("snowflake_metadata")
            ),
            "rag_context": rag_context.strip() if rag_context else "",
            "feedback_section": feedback_section,
        }

        # ── Step 3: Invoke ChatBedrock with structured output ────
        console.print(
            f"  🤖 Invoking [bold cyan]{self._settings.bedrock_model_id}[/] "
            f"for SQL optimization..."
        )
        try:
            llm = get_llm().with_structured_output(OptimizationResult)
            result: OptimizationResult = (OPTIMIZATION_PROMPT | llm).invoke(prompt_inputs)
        except Exception as e:
            console.print(
                f"  ⚠️  [yellow]LLM call failed ({e}) — returning no-change result[/]"
            )
            return self._no_change_fallback(original_sql, analysis)

        # ── Step 4: Process result ───────────────────────────────
        optimized_sql = result.optimized_sql.strip()
        changes_applied = [c.model_dump() for c in result.changes_applied]

        if not optimized_sql or optimized_sql == original_sql:
            console.print("  ℹ️  LLM returned unchanged SQL — no improvements found")

        # ── Step 4½: Verify claimed changes against actual diff ──
        from src.engines.change_verifier import ChangeVerifier

        verifier = ChangeVerifier()
        verification = verifier.verify(
            original_sql=original_sql,
            optimized_sql=optimized_sql,
            changes_applied=changes_applied,
        )

        if verification.total_removed > 0:
            console.print(
                f"  🔍 Change verification: [yellow]{verification.total_removed}[/] "
                f"hallucinated change(s) removed out of {verification.total_claimed}"
            )
            for removed in verification.removed_changes:
                console.print(
                    f"    ✂️  [dim]Removed:[/] {removed.original_action} "
                    f"— [yellow]{removed.reason}[/]"
                )

        # Use only verified changes going forward
        changes_applied = verification.verified_changes

        # ── Pretty-print ─────────────────────────────────────────
        console.print(
            f"  ✅ LLM optimization complete — [bold]{len(changes_applied)}[/] verified change(s)"
            f", confidence: [bold green]{result.confidence:.0%}[/]"
        )
        for c in changes_applied:
            console.print(f"    ↳ [{c.get('type', '?')}] {c.get('action', '')}")
        if result.warnings:
            for w in result.warnings:
                console.print(f"    ⚠️  [yellow]{w}[/]")
        console.print(f"\n  [dim]Optimized SQL:[/]\n  [green]{optimized_sql}[/]\n")

        optimization_output = {
            "query_id": analysis["query_id"],
            "original_sql": original_sql,
            "optimized_sql": optimized_sql,
            "changes_applied": changes_applied,
            "change_count": len(changes_applied),
            "rationale": result.rationale,
            "confidence": result.confidence,
            "warnings": result.warnings,
            "optimization_mode": "llm",
            "llm_model": self._settings.bedrock_model_id,
            "rag_cases_used": rag_cases_used,
            "rag_results": raw_cases,
        }

        return {
            "state_key": "optimization",
            "output": optimization_output,
            "next_agent": AgentRole.VALIDATION.value,
            "task_desc": (
                f"Validate LLM-optimized query — {len(changes_applied)} changes, "
                f"confidence {result.confidence:.0%}"
            ),
            "extra_state": {"rag_results": raw_cases},
        }

    # ── No-change fallback (Bedrock unavailable) ─────────────────────────────

    def _no_change_fallback(
        self, original_sql: str, analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Emit a no-change result when the LLM is unavailable.

        Instead of silently applying regex rewrite rules, this clearly
        reports that optimization could not be performed and recommends
        manual review — matching Barandeep's described fallback procedure:
        "Provide a recommendation, add comments, and tag the responsible party."
        """
        console.print(
            "  📋 [yellow]No optimization performed — manual review recommended[/]"
        )
        console.print(
            "    → Tag the query owner for manual optimization review."
        )

        optimization_output = {
            "query_id": analysis["query_id"],
            "original_sql": original_sql,
            "optimized_sql": original_sql,  # unchanged
            "changes_applied": [],
            "change_count": 0,
            "rationale": (
                "Automated optimization unavailable — Bedrock LLM is not configured. "
                "Manual review recommended. Please tag the responsible engineer."
            ),
            "confidence": 0.0,
            "warnings": [
                "No optimization was applied — LLM service is unavailable.",
                "This query should be reviewed manually by the responsible engineer.",
            ],
            "optimization_mode": "unavailable",
            "llm_model": "",
            "rag_cases_used": 0,
            "rag_results": [],
        }

        return {
            "state_key": "optimization",
            "output": optimization_output,
            "next_agent": AgentRole.VALIDATION.value,
            "task_desc": "Validate query — no automated optimization performed (manual review needed)",
            "extra_state": {"rag_results": []},
        }
