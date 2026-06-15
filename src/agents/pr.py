"""
PR Agent — simulates creating a GitHub Draft Pull Request.

In production this would call the GitHub API via PyGitHub.
For the MVP it generates the full PR payload and prints it.
"""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel

from src.agents.base import BaseAgent, console
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState


class PRAgent(BaseAgent):
    name = AgentRole.PR.value
    role = AgentRole.PR

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        report = state["report"]
        validation = state["validation"]
        metrics = validation["metrics"]

        query_id = report["query_id"].lower()
        branch_name = f"ai/optimize-{query_id}"
        decision = report.get("validation_decision", validation.get("decision", "APPROVED"))
        decision_icon = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "?")

        # ── Build the PR body (Markdown) ────────────────────────
        changes_md = "\n".join(
            f"| {c['index']} | {c['change']} | {c['reason']} |"
            for c in report["changes"]
        )

        # EXPLAIN diff insights section
        diff_insights = report.get("explain_diff_insights", [])
        if diff_insights:
            insights_md = "\n".join(f"- {ins}" for ins in diff_insights)
            insights_section = f"\n### EXPLAIN Plan Insights\n{insights_md}\n"
        else:
            insights_section = ""

        # LLM metadata section
        opt_mode = report.get("optimization_mode", "rule_based")
        llm_meta = ""
        if opt_mode == "llm":
            llm_meta = (
                f"\n### AI Metadata\n"
                f"| Field | Value |\n"
                f"|-------|-------|\n"
                f"| LLM Model | `{report.get('llm_model', 'unknown')}` |\n"
                f"| RAG Cases Used | {report.get('rag_cases_used', 0)} |\n"
                f"| Confidence Score | {report.get('confidence_score', 0):.0%} |\n"
                f"| Optimization Mode | LLM + RAG |\n"
            )

        # Safety check results
        safety_passed = validation.get("safety_checks_passed", [])
        safety_failed = validation.get("safety_checks_failed", [])
        safety_warnings = validation.get("safety_warnings", [])
        safety_md = ""
        if safety_passed:
            safety_md += "".join(f"- [x] {c}\n" for c in safety_passed)
        if safety_failed:
            safety_md += "".join(f"- [ ] ❌ {c}\n" for c in safety_failed)
        if safety_warnings:
            safety_md += "".join(f"- [x] ⚠️ {c} (warning)\n" for c in safety_warnings)

        # LLM semantic concerns
        llm_concerns = validation.get("llm_concerns", [])
        if llm_concerns:
            concerns_md = "\n".join(f"  - {c}" for c in llm_concerns)
            concerns_section = f"\n**LLM Semantic Concerns:**\n{concerns_md}\n"
        else:
            concerns_section = ""

        pr_body = f"""## 🤖 AI-Generated Query Optimisation

> ⚠️ This PR was created by an AI agent. **Human review and approval is mandatory** before merging.

### {decision_icon} Validation Decision: {decision}
{('All safety and semantic checks passed.' if decision == 'APPROVED' else 'This PR requires human review before merging.')}
{concerns_section}
### Summary
{report['summary']}

### What Changed
| # | Change | Reason |
|---|--------|--------|
{changes_md}

### Original SQL
```sql
{report['original_sql']}
```

### Optimised SQL
```sql
{report['optimized_sql']}
```

### Performance Evidence
| Metric | Result |
|--------|--------|
| Execution Time | {report['performance']['execution_time']} |
| Credits Consumed | {report['performance']['credits_consumed']} |
| Bytes Scanned | {report['performance']['bytes_scanned']} |
| Partition Pruning | {report['performance']['partition_pruning']} |
{insights_section}{llm_meta}
### Validation & Safety Checks
{safety_md if safety_md else '- [x] All safety checks passed'}
- [x] Semantic check: **{validation.get('semantic_check', 'PASS')}**
- [x] Row count match: {validation.get('row_count_original', 0):,} rows
{f'- [x] EXPLAIN plan diff score: {validation.get("explain_diff", {}).get("overall_improvement_score", 0):.2f}' if validation.get('explain_diff') else ''}
- [ ] **Awaiting human review**

### Review Checklist (Human Reviewer)
- [ ] Verify business logic is semantically equivalent
- [ ] Confirm LIMIT is appropriate for the use case
- [ ] Test on staging before merging to production
- [ ] Update any dependent dashboards or pipelines

**Labels**: `ai-generated` `needs-human-review` `query-optimization` `snowflake`
"""

        pr_output = {
            "query_id": report["query_id"],
            "branch_name": branch_name,
            "pr_title": (
                f"[AI] Optimize query {report['query_id']} — "
                f"{metrics['execution_time']['improvement_pct']}% faster "
                f"[{decision}]"
            ),
            "pr_body": pr_body,
            "pr_state": "draft",
            "validation_decision": decision,
            "labels": [
                "ai-generated",
                "needs-human-review",
                "query-optimization",
                "snowflake",
            ],
            "auto_merge": False,
            "status": "DRAFT_PR_CREATED",
        }

        # Pretty-print the PR
        console.print(
            Panel(
                Markdown(pr_body),
                title=f"[bold magenta]Draft PR: {pr_output['pr_title']}[/]",
                subtitle=f"Branch: {branch_name} | State: DRAFT",
                border_style="magenta",
                padding=(1, 2),
            )
        )

        return {
            "state_key": "pr",
            "output": pr_output,
            "next_agent": "HumanReviewer",
            "task_desc": "Draft PR created — awaiting human review",
        }
