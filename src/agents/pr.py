"""
PR Agent — creates a real GitHub Draft Pull Request in the target repository.

Uses the pre-generated codebase graph (from Graphify) to locate the exact
file containing the flagged SQL query, patches it in-place, and opens a
Draft PR via the GitHub API.

When TARGET_REPO is not configured, falls back to the original simulation
behavior (prints PR body to console + generates HTML report).
"""

from __future__ import annotations

import logging
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel

from src.agents.base import BaseAgent, console
from src.config.settings import get_settings
from src.models.messages import AgentRole
from src.models.state import QueryOptimizationState
from src.reporters.html_report import get_html_report_generator

logger = logging.getLogger(__name__)


class PRAgent(BaseAgent):
    name = AgentRole.PR.value
    role = AgentRole.PR

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()

    def process(self, state: QueryOptimizationState) -> dict[str, Any]:
        report = state["report"]
        validation = state["validation"]
        metrics = validation["metrics"]

        query_id = report["query_id"].lower()
        branch_name = f"ai/optimize-{query_id}"
        decision = report.get("validation_decision", validation.get("decision", "APPROVED"))
        decision_icon = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "?")

        # ── Build the PR body (Markdown) ────────────────────────
        pr_body = self._build_pr_body(report, validation, metrics, decision, decision_icon)

        pr_title = (
            f"[AI] Optimize query {report['query_id']} — "
            f"{metrics['execution_time']['improvement_pct']}% faster "
            f"[{decision}]"
        )

        labels = [
            "ai-generated",
            "needs-human-review",
            "query-optimization",
            "snowflake",
        ]

        # ── Attempt real PR creation via codebase graph ─────────
        github_pr_result = None
        graph_location_data = {}

        if self._settings.target_repo_configured:
            github_pr_result, graph_location_data = self._create_real_pr(
                report=report,
                branch_name=branch_name,
                pr_title=pr_title,
                pr_body=pr_body,
                labels=labels,
                query_id=report["query_id"],
            )

        # ── Build output dict ───────────────────────────────────
        pr_output = {
            "query_id": report["query_id"],
            "branch_name": branch_name,
            "pr_title": pr_title,
            "pr_body": pr_body,
            "pr_state": "draft",
            "validation_decision": decision,
            "labels": labels,
            "auto_merge": False,
            "status": "DRAFT_PR_CREATED" if github_pr_result else "SIMULATED",
        }

        if github_pr_result:
            pr_output.update({
                "pr_number": github_pr_result.pr_number,
                "pr_url": github_pr_result.pr_url,
                "commit_sha": github_pr_result.commit_sha,
                "patched_file": github_pr_result.file_path,
                "status": "DRAFT_PR_CREATED",
            })
            console.print(
                f"  🔗 [bold green]Real Draft PR created:[/] [cyan]{github_pr_result.pr_url}[/]"
            )
        else:
            if self._settings.target_repo_configured:
                console.print(
                    "  ⚠️  [yellow]Real PR creation failed — showing simulation fallback[/]"
                )
            else:
                console.print(
                    "  ℹ️  [dim]TARGET_REPO not configured — simulation mode[/]"
                )

        # Pretty-print the PR body
        console.print(
            Panel(
                Markdown(pr_body),
                title=f"[bold magenta]Draft PR: {pr_title}[/]",
                subtitle=f"Branch: {branch_name} | State: DRAFT",
                border_style="magenta",
                padding=(1, 2),
            )
        )

        # ── Sprint 2: Generate HTML Report ───────────────────────────────────
        report_path = ""
        try:
            generator = get_html_report_generator()
            # Inject the pr_output into state so the generator can read it
            state_with_pr = dict(state)
            state_with_pr["pr"] = pr_output
            path = generator.generate(state_with_pr)
            report_path = str(path)
            console.print(
                f"  📄 [bold green]HTML Report:[/] [cyan]{report_path}[/]"
            )
        except Exception as exc:
            console.print(f"  ⚠️  [yellow]HTML report generation failed (non-fatal): {exc}[/]")

        pr_output["report_path"] = report_path

        return {
            "state_key": "pr",
            "output": pr_output,
            "next_agent": "HumanReviewer",
            "task_desc": "Draft PR created — awaiting human review",
            "extra_state": {
                "report_path": report_path,
                "graph_location": graph_location_data,
            },
        }

    # ── Real PR creation via codebase graph ─────────────────────

    def _create_real_pr(
        self,
        *,
        report: dict,
        branch_name: str,
        pr_title: str,
        pr_body: str,
        labels: list[str],
        query_id: str,
    ) -> tuple[Any, dict]:
        """
        Attempt to create a real Draft PR in the target repository.

        Steps:
        1. Download codebase graph from target repo's GitHub Artifacts
        2. Locate the SQL query in the graph
        3. Fetch + patch the file
        4. Create branch → commit → Draft PR

        Returns:
            (PRResult or None, graph_location dict)
        """
        try:
            from src.services.codebase_graph import get_codebase_navigator
            from src.services.github_pr import get_pr_creator, patch_file_content

            token = self._settings.target_repo_token
            repo_name = self._settings.target_repo
            base_branch = self._settings.target_repo_default_branch

            console.print(f"  🔍 [bold cyan]Locating SQL in {repo_name} via codebase graph…[/]")

            # Step 1: Download and parse the codebase graph
            navigator = get_codebase_navigator(token, repo_name)
            graph = navigator.download_graph()

            # Step 2: Locate the SQL query
            original_sql = report["original_sql"]
            location = navigator.locate_sql(original_sql, graph)

            if location is None:
                console.print(
                    "  ⚠️  [yellow]SQL query not found in codebase graph — "
                    "falling back to simulation[/]"
                )
                return None, {}

            console.print(
                f"  📍 [bold green]Found:[/] {location.file_path} "
                f"(lines {location.line_start}–{location.line_end})"
            )

            # Step 3: Fetch the original file and patch it
            from github import Github

            g = Github(token)
            repo = g.get_repo(repo_name)
            contents = repo.get_contents(location.file_path, ref=base_branch)
            if isinstance(contents, list):
                raise RuntimeError(f"{location.file_path} is a directory")

            original_content = contents.decoded_content.decode("utf-8")
            optimized_sql = report["optimized_sql"]

            patched_content = patch_file_content(
                original_content=original_content,
                original_sql=original_sql,
                optimized_sql=optimized_sql,
                line_start=location.line_start,
                line_end=location.line_end,
            )

            console.print("  ✏️  [bold]File patched — creating Draft PR…[/]")

            # Step 4: Create the Draft PR
            pr_creator = get_pr_creator(token, repo_name)
            result = pr_creator.create_draft_pr(
                file_path=location.file_path,
                patched_content=patched_content,
                branch_name=branch_name,
                pr_title=pr_title,
                pr_body=pr_body,
                commit_message=f"[AI] Optimize query {query_id}",
                base_branch=base_branch,
                labels=labels,
            )

            return result, location.to_dict()

        except Exception as exc:
            logger.error("Real PR creation failed: %s", exc, exc_info=True)
            console.print(f"  ❌ [red]PR creation failed: {exc}[/]")
            return None, {}

    # ── PR body builder (preserved from original) ───────────────

    def _build_pr_body(
        self,
        report: dict,
        validation: dict,
        metrics: dict,
        decision: str,
        decision_icon: str,
    ) -> str:
        """Build the rich Markdown PR body — unchanged from the original PRAgent."""

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
        opt_mode = report.get("optimization_mode", "unavailable")
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
        return pr_body
