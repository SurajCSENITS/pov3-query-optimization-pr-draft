"""
Prompt Evaluator — Offline LLM-as-a-Judge scoring for system prompts.

Evaluates the OPTIMIZATION_PROMPT (and SCREENER_PROMPT) by running each
golden test case through the *real* Analysis → Optimization pipeline and
scoring the output across multiple dimensions:

  1. Format Adherence  — did the pipeline return a schema-valid optimization?
  2. Hallucination     — what fraction of claimed changes are real (ChangeVerifier)?
  3. Semantic Equivalence — does the SCREENER_PROMPT accept the optimized SQL?
  4. Confidence        — LLM self-reported confidence pass-through
  5. Bottleneck Coverage — sqlglot AST diff: did the SQL change where expected?
  6. Composite         — weighted aggregate of the above

Pipeline invocation:
    Each test case is converted to the same input_data payload the real pipeline
    expects (matching main.py:SAMPLE_INPUT / the POV4 alert format).
    The evaluator runs only the Analysis and Optimization agents — skipping
    Validation, Report, and PR since we only care about prompt quality.

Usage:
    from src.eval.prompt_evaluator import PromptEvaluator

    evaluator = PromptEvaluator()
    report = evaluator.evaluate(dataset_path="eval/golden_dataset.jsonl")
    print(report.avg_composite_score)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.models import (
    EvalCaseResult,
    EvalSuiteReport,
    EvalTestCase,
    PromptScores,
)
from src.eval.metrics import (
    SCORE_WEIGHTS,
    bottleneck_coverage_score,
    compute_composite_score,
    confidence_score,
    format_score,
    hallucination_score,
    semantic_equivalence_score,
)

logger = logging.getLogger(__name__)

# Composite score threshold for pass/fail classification
PASS_THRESHOLD = 0.60


class PromptEvaluator:
    """
    Offline evaluator for the SQL optimization system prompts.

    Runs each golden test case through the real Analysis + Optimization agents,
    then scores the output using the metrics module.

    Each test case is converted to a valid pipeline input_data dict via
    EvalTestCase.to_pipeline_payload(), which matches the POV4 alert format.
    """

    def __init__(self, pass_threshold: float = PASS_THRESHOLD) -> None:
        self.pass_threshold = pass_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataset_path: str | Path = "eval/golden_dataset.jsonl",
        prompt_version: str | None = None,
        max_cases: int | None = None,
        tags_filter: list[str] | None = None,
    ) -> EvalSuiteReport:
        """
        Run the full evaluation suite against the golden dataset.

        Args:
            dataset_path:   Path to the JSONL golden dataset.
            prompt_version: Optional override; defaults to OPTIMIZATION_PROMPT_VERSION.
            max_cases:      Limit evaluation to the first N cases (useful for quick checks).
            tags_filter:    Only evaluate cases whose tags include at least one of these.

        Returns:
            EvalSuiteReport with per-case results and macro-averages.
        """
        from src.prompts.optimization_prompt import OPTIMIZATION_PROMPT_VERSION

        version = prompt_version or OPTIMIZATION_PROMPT_VERSION
        test_cases = self._load_dataset(dataset_path, tags_filter)

        if max_cases:
            test_cases = test_cases[:max_cases]

        logger.info(
            "Starting prompt evaluation — version=%s, cases=%d",
            version,
            len(test_cases),
        )

        results: list[EvalCaseResult] = []
        for tc in test_cases:
            logger.info("Evaluating case %s: %s", tc.id, tc.description)
            result = self._evaluate_single(tc)
            results.append(result)

        report = self._aggregate(results, version)
        logger.info(
            "Evaluation complete — composite=%.2f%%, pass=%d/%d",
            report.avg_composite_score * 100,
            report.passed_cases,
            report.total_cases,
        )
        return report

    # ── Single case evaluation ─────────────────────────────────────────────

    def _evaluate_single(self, tc: EvalTestCase) -> EvalCaseResult:
        """
        Run one golden test case through the real Analysis → Optimization pipeline
        and return a fully scored EvalCaseResult.

        Steps:
          1. Convert test case to pipeline input_data (POV4 payload format)
          2. Run AnalysisAgent to detect bottlenecks from the SQL
          3. Run OptimizationAgent with the analysis output
          4. Run ChangeVerifier on the optimization output
          5. Run SCREENER_PROMPT on (original, optimized) pair
          6. Score all five dimensions
        """
        result = EvalCaseResult(test_case=tc)

        # ── Step 1: Build pipeline input_data ────────────────────────────────
        input_data = tc.to_pipeline_payload()

        # ── Step 2: Run AnalysisAgent ─────────────────────────────────────────
        analysis_output, analysis_ok, analysis_err = self._run_analysis(input_data)
        if not analysis_ok:
            result.llm_call_succeeded = False
            result.error_message = f"AnalysisAgent failed: {analysis_err}"
            result.scores = self._zero_scores()
            return result

        # ── Step 3: Run OptimizationAgent ─────────────────────────────────────
        opt_output, opt_ok, opt_err = self._run_optimization(analysis_output)
        if not opt_ok:
            result.llm_call_succeeded = False
            result.error_message = f"OptimizationAgent failed: {opt_err}"
            result.scores = self._zero_scores(fmt=0.0)
            return result

        optimized_sql: str = opt_output.get("optimized_sql", "").strip()
        claimed_changes: list[dict] = opt_output.get("changes_applied", [])
        llm_confidence: float = opt_output.get("confidence", 0.0)

        result.optimized_sql = optimized_sql
        result.changes_applied = claimed_changes
        result.claimed_change_count = opt_output.get("change_count", len(claimed_changes))

        # ── Step 4: Re-run ChangeVerifier (OptimizationAgent already ran it,
        #            but we want the raw counts for scoring) ───────────────────
        from src.engines.change_verifier import ChangeVerifier

        verifier = ChangeVerifier()
        verification = verifier.verify(
            original_sql=tc.original_sql,
            optimized_sql=optimized_sql,
            changes_applied=claimed_changes,
        )
        result.verified_changes = verification.verified_changes
        result.hallucinated_change_count = verification.total_removed

        # ── Step 5: Run SCREENER_PROMPT ────────────────────────────────────────
        sem_equiv, sem_confidence = self._run_screener(tc.original_sql, optimized_sql)
        result.semantic_equivalent = sem_equiv
        result.semantic_confidence = sem_confidence

        # ── Step 6: Score all dimensions ───────────────────────────────────────
        fmt = format_score(
            llm_call_succeeded=True,
            optimized_sql=optimized_sql,
        )
        hall = hallucination_score(
            total_claimed=verification.total_claimed,
            total_removed=verification.total_removed,
        )
        sem = semantic_equivalence_score(
            semantically_equivalent=sem_equiv,
            screener_confidence=sem_confidence,
            expected_equivalence=tc.expected_semantic_equivalence,
        )
        conf = confidence_score(llm_confidence)
        cov = bottleneck_coverage_score(
            expected_bottleneck_types=tc.expected_bottleneck_types,
            original_sql=tc.original_sql,
            optimized_sql=optimized_sql,
            changes_applied=verification.verified_changes,
            expected_min_changes=tc.expected_min_changes,
        )
        comp = compute_composite_score(fmt, hall, sem, conf, cov)

        result.scores = PromptScores(
            format_score=fmt,
            hallucination_score=hall,
            semantic_equivalence_score=sem,
            confidence_score=conf,
            bottleneck_coverage_score=cov,
            composite_score=comp,
        )
        return result

    # ── Real pipeline agent calls ──────────────────────────────────────────

    def _run_analysis(
        self, input_data: dict
    ) -> tuple[dict[str, Any], bool, str]:
        """
        Run the real AnalysisAgent on the input_data payload.

        Constructs a minimal QueryOptimizationState and calls the agent
        directly — no LangGraph graph invocation needed for eval.

        Returns: (analysis_output dict, succeeded: bool, error_message: str)
        """
        try:
            from src.agents.analysis import AnalysisAgent
            from src.models.messages import AgentMessage, AgentRole

            state: dict[str, Any] = {
                "input_data": input_data,
                "analysis": {},
                "optimization": {},
                "validation": {},
                "report": {},
                "pr": {},
                "rag_results": [],
                "validation_evidence": {},
                "graph_location": {},
                "retry_count": 0,
                "feedback_history": [],
                "messages": [],
            }

            agent = AnalysisAgent()
            patch = agent.run(state)

            # agent.run() returns a state patch; analysis is under 'analysis'
            analysis_output = patch.get("analysis", state.get("analysis", {}))
            if not analysis_output:
                return {}, False, "AnalysisAgent returned empty analysis output"
            return analysis_output, True, ""

        except Exception as exc:
            logger.error("AnalysisAgent call failed: %s", exc)
            return {}, False, str(exc)

    def _run_optimization(
        self, analysis_output: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, str]:
        """
        Run the real OptimizationAgent using the analysis output.

        Constructs the state with the analysis already populated, then calls
        the OptimizationAgent directly.

        Returns: (optimization_output dict, succeeded: bool, error_message: str)
        """
        try:
            from src.agents.optimization import OptimizationAgent

            state: dict[str, Any] = {
                "input_data": {
                    "query_id": analysis_output.get("query_id", "eval"),
                    "query_text": analysis_output.get("original_sql", ""),
                },
                "analysis": analysis_output,
                "optimization": {},
                "validation": {},
                "report": {},
                "pr": {},
                "rag_results": [],
                "validation_evidence": {},
                "graph_location": {},
                "retry_count": 0,
                "feedback_history": [],
                "messages": [],
            }

            agent = OptimizationAgent()
            patch = agent.run(state)

            opt_output = patch.get("optimization", {})
            if not opt_output:
                return {}, False, "OptimizationAgent returned empty output"
            return opt_output, True, ""

        except Exception as exc:
            logger.error("OptimizationAgent call failed: %s", exc)
            return {}, False, str(exc)

    def _run_screener(
        self, original_sql: str, optimized_sql: str
    ) -> tuple[bool, float]:
        """
        Invoke SCREENER_PROMPT via ChatBedrock with structured output.

        This is the only direct prompt call in the evaluator — the screener
        is a lightweight semantic check, not part of the optimization pipeline.

        Returns: (semantically_equivalent: bool, confidence: float)
        """
        from src.connectors.bedrock_manager import get_llm
        from src.models.llm_outputs import SemanticCheckResult
        from src.prompts.optimization_prompt import SCREENER_PROMPT

        try:
            llm = get_llm().with_structured_output(SemanticCheckResult)
            result: SemanticCheckResult = (SCREENER_PROMPT | llm).invoke(
                {"original_sql": original_sql, "optimized_sql": optimized_sql}
            )
            return result.semantically_equivalent, result.confidence
        except Exception as exc:
            logger.warning("SCREENER_PROMPT call failed: %s", exc)
            # On screener failure: assume non-equivalent (conservative), confidence=0
            return False, 0.0

    # ── Aggregation ────────────────────────────────────────────────────────

    def _aggregate(
        self, results: list[EvalCaseResult], prompt_version: str
    ) -> EvalSuiteReport:
        """Aggregate per-case results into a suite-level report."""
        total = len(results)
        if total == 0:
            return EvalSuiteReport(prompt_version=prompt_version)

        passed = sum(
            1 for r in results
            if r.scores.composite_score >= self.pass_threshold
        )
        errors = sum(1 for r in results if not r.llm_call_succeeded)
        failed = total - passed - errors

        def _mean(attr: str) -> float:
            vals = [getattr(r.scores, attr) for r in results]
            return round(sum(vals) / len(vals), 4)

        sorted_results = sorted(results, key=lambda r: r.scores.composite_score)
        worst = sorted_results[:3]

        return EvalSuiteReport(
            prompt_version=prompt_version,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            avg_format_score=_mean("format_score"),
            avg_hallucination_score=_mean("hallucination_score"),
            avg_semantic_equivalence_score=_mean("semantic_equivalence_score"),
            avg_confidence_score=_mean("confidence_score"),
            avg_bottleneck_coverage_score=_mean("bottleneck_coverage_score"),
            avg_composite_score=_mean("composite_score"),
            worst_cases=worst,
            results=sorted_results,
        )

    def _zero_scores(self, fmt: float = 0.0) -> PromptScores:
        """Return a zero-score PromptScores for failed pipeline calls."""
        return PromptScores(
            format_score=fmt,
            hallucination_score=0.0,
            semantic_equivalence_score=0.0,
            confidence_score=0.0,
            bottleneck_coverage_score=0.0,
            composite_score=0.0,
        )

    # ── Dataset loading ────────────────────────────────────────────────────

    @staticmethod
    def _load_dataset(
        path: str | Path,
        tags_filter: list[str] | None,
    ) -> list[EvalTestCase]:
        """Load and optionally filter golden test cases from a JSONL file."""
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found at '{dataset_path}'. "
                f"Expected a JSONL file at eval/golden_dataset.jsonl."
            )

        cases: list[EvalTestCase] = []
        with dataset_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    raw = json.loads(line)
                    tc = EvalTestCase(**raw)
                    if tags_filter:
                        if not any(tag in tc.tags for tag in tags_filter):
                            continue
                    cases.append(tc)
                except Exception as exc:
                    logger.warning("Skipping malformed line %d in dataset: %s", lineno, exc)

        logger.info("Loaded %d test case(s) from '%s'", len(cases), dataset_path)
        return cases
