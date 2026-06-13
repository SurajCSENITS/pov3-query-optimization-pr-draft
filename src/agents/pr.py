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

        # ── Build the PR body (Markdown) ────────────────────────
        changes_md = "\n".join(
            f"| {c['index']} | {c['change']} | {c['reason']} |"
            for c in report["changes"]
        )

        pr_body = f"""## 🤖 AI-Generated Query Optimisation

> ⚠️ This PR was created by an AI agent. **Human review and approval is mandatory** before merging.

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

### Validation
- [x] Semantic check: **{validation['semantic_check']}**
- [x] Row count match: {validation['row_count_original']:,} rows
- [x] EXPLAIN plan verified
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
            "pr_title": f"[AI] Optimize query {report['query_id']} — "
            f"{metrics['execution_time']['improvement_pct']}% faster",
            "pr_body": pr_body,
            "pr_state": "draft",
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
