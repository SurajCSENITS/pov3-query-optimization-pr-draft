"""
HTML Report Generator — Sprint 2.

Generates a professional, self-contained HTML report for every
POV3 optimization workflow execution.

Output location:
    reports/execution_<query_id>.html

The report contains 8 sections:
    1. Execution Summary
    2. Original Query
    3. Optimized Query
    4. Root Cause Analysis
    5. Retrieved RAG Knowledge
    6. Validation Results
    7. Performance Metrics
    8. Draft PR Summary

All CSS and JS is embedded inline — the file is fully self-contained
and opens correctly in any browser without a server.

Zero new pip dependencies — uses Python stdlib only.
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Output directory ─────────────────────────────────────────────────────────
_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


# ── Severity badge colours ───────────────────────────────────────────────────
_SEVERITY_CLASSES = {
    "CRITICAL": "badge-critical",
    "HIGH":     "badge-high",
    "MEDIUM":   "badge-medium",
    "LOW":      "badge-low",
    "INFO":     "badge-info",
    "WARNING":  "badge-warning",
}

_DECISION_CLASSES = {
    "APPROVED": "decision-approved",
    "REVIEW":   "decision-review",
    "REJECTED": "decision-rejected",
}

_VERDICT_ICONS = {
    "EXCELLENT":      "🚀",
    "GOOD":           "✅",
    "MARGINAL":       "⚠️",
    "NO_IMPROVEMENT": "➡️",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTMLReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class HTMLReportGenerator:
    """
    Builds a self-contained HTML report from the LangGraph final state.

    Usage:
        gen = HTMLReportGenerator()
        path = gen.generate(final_state)
        # path → PosixPath("reports/execution_Q123.html")
    """

    def __init__(self) -> None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def generate(self, state: dict[str, Any]) -> Path:
        """
        Generate and write the HTML report from the complete pipeline state.

        Args:
            state:  The final LangGraph state dict after all agents have run.

        Returns:
            Path to the written HTML file.
        """
        input_data  = state.get("input_data", {})
        analysis    = state.get("analysis",   {})
        optimization = state.get("optimization", {})
        validation  = state.get("validation", {})
        report      = state.get("report",     {})
        pr          = state.get("pr",         {})
        rag_results = state.get("rag_results", [])
        val_evidence = state.get("validation_evidence", {})

        query_id    = input_data.get("query_id", report.get("query_id", "unknown"))
        timestamp   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        run_id      = report.get("report_id", "N/A")
        warehouse   = input_data.get("warehouse", "N/A")

        sections = [
            self._section_execution_summary(query_id, timestamp, warehouse, run_id, input_data),
            self._section_original_query(optimization),
            self._section_optimized_query(optimization),
            self._section_root_cause(analysis),
            self._section_rag_knowledge(rag_results, optimization),
            self._section_validation(validation, val_evidence),
            self._section_performance(validation),
            self._section_pr_summary(pr, report),
        ]

        body = "\n".join(sections)
        full_html = _HTML_TEMPLATE.format(
            query_id=html.escape(str(query_id)),
            timestamp=html.escape(timestamp),
            body=body,
        )

        # Sanitise query_id for use in a filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(query_id))
        out_path = _REPORTS_DIR / f"execution_{safe_id}.html"
        out_path.write_text(full_html, encoding="utf-8")

        logger.info("HTML report written → %s", out_path)
        return out_path

    # ── Section 1 — Execution Summary ─────────────────────────────────────────

    def _section_execution_summary(
        self,
        query_id: str,
        timestamp: str,
        warehouse: str,
        run_id: str,
        input_data: dict,
    ) -> str:
        exec_time = input_data.get("execution_time_seconds", "N/A")
        credits   = input_data.get("credits_used", "N/A")
        issue     = input_data.get("issue_type", "N/A")
        return f"""
        <section class="card" id="s1-execution-summary">
          <div class="section-header">
            <span class="section-num">01</span>
            <h2>Execution Summary</h2>
          </div>
          <div class="summary-grid">
            <div class="summary-item">
              <div class="summary-label">Query ID</div>
              <div class="summary-value mono">{html.escape(str(query_id))}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Timestamp</div>
              <div class="summary-value">{html.escape(timestamp)}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Warehouse</div>
              <div class="summary-value mono">{html.escape(str(warehouse))}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Agent Run ID</div>
              <div class="summary-value mono">{html.escape(str(run_id))}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Execution Time (Before)</div>
              <div class="summary-value">{html.escape(str(exec_time))} s</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Credits Used (Before)</div>
              <div class="summary-value">{html.escape(str(credits))}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Issue Type</div>
              <div class="summary-value"><span class="badge badge-high">{html.escape(str(issue))}</span></div>
            </div>
          </div>
        </section>"""

    # ── Section 2 — Original Query ─────────────────────────────────────────────

    def _section_original_query(self, optimization: dict) -> str:
        sql = optimization.get("original_sql", "— not available —")
        return f"""
        <section class="card" id="s2-original-query">
          <div class="section-header">
            <span class="section-num">02</span>
            <h2>Original Query</h2>
          </div>
          <div class="code-block-wrapper">
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
            <pre class="code-block sql-block"><code>{html.escape(sql)}</code></pre>
          </div>
        </section>"""

    # ── Section 3 — Optimized Query ────────────────────────────────────────────

    def _section_optimized_query(self, optimization: dict) -> str:
        sql      = optimization.get("optimized_sql", "— not available —")
        mode     = optimization.get("optimization_mode", "rule_based")
        model    = optimization.get("llm_model", "")
        changes  = optimization.get("changes_applied", [])
        conf     = optimization.get("confidence", 0.0)

        mode_badge = (
            f'<span class="badge badge-info">LLM — {html.escape(model)}</span>'
            if mode == "llm" else
            '<span class="badge badge-medium">Rule-Based</span>'
        )

        change_rows = "".join(
            f"""<tr>
                  <td>{i}</td>
                  <td><code>{html.escape(c.get('type', c.get('bottleneck_id', '?')))}</code></td>
                  <td>{html.escape(c.get('action', ''))}</td>
                  <td>{html.escape(c.get('reason', ''))}</td>
                </tr>"""
            for i, c in enumerate(changes, 1)
        ) or "<tr><td colspan='4' class='empty'>No changes recorded</td></tr>"

        return f"""
        <section class="card" id="s3-optimized-query">
          <div class="section-header">
            <span class="section-num">03</span>
            <h2>Optimized Query</h2>
            <div class="section-meta">{mode_badge}
              {f'<span class="badge badge-info">Confidence: {conf:.0%}</span>' if mode == "llm" else ""}
            </div>
          </div>
          <div class="code-block-wrapper">
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
            <pre class="code-block sql-block"><code>{html.escape(sql)}</code></pre>
          </div>
          <h3 class="sub-heading">Changes Applied</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead><tr><th>#</th><th>Type</th><th>Action</th><th>Reason</th></tr></thead>
              <tbody>{change_rows}</tbody>
            </table>
          </div>
        </section>"""

    # ── Section 4 — Root Cause Analysis ───────────────────────────────────────

    def _section_root_cause(self, analysis: dict) -> str:
        bottlenecks  = analysis.get("bottlenecks", [])
        root_cause   = analysis.get("root_cause", "")
        severity_score = analysis.get("severity_score", 0)
        mode         = analysis.get("analysis_mode", "mock")

        mode_badge = (
            '<span class="badge badge-info">Snowflake Mode</span>'
            if mode == "snowflake" else
            '<span class="badge badge-medium">Mock / Heuristic Mode</span>'
        )

        cards = ""
        for b in bottlenecks:
            sev   = b.get("severity", "LOW")
            btype = b.get("type", "UNKNOWN")
            desc  = b.get("description", "")
            loc   = b.get("location", "")
            bid   = b.get("id", "")
            badge_cls = _SEVERITY_CLASSES.get(sev, "badge-info")

            icon_map = {
                "FULL_TABLE_SCAN":       "🔍",
                "FULL_COLUMN_SCAN":      "📋",
                "REMOTE_SPILL":          "💾",
                "LOCAL_SPILL":           "💿",
                "NON_SARGABLE_PREDICATE":"🔒",
                "UNFILTERED_JOIN":       "🔗",
                "POOR_PARTITION_PRUNING":"🗂️",
                "WINDOW_FUNCTION_BOTTLENECK": "🪟",
            }
            icon = icon_map.get(btype, "⚠️")

            cards += f"""
            <div class="bottleneck-card sev-{sev.lower()}">
              <div class="bottleneck-header">
                <span class="bottleneck-icon">{icon}</span>
                <span class="bottleneck-type">{html.escape(btype)}</span>
                <span class="badge {badge_cls}">{html.escape(sev)}</span>
                <span class="bottleneck-id dim">{html.escape(bid)}</span>
              </div>
              <div class="bottleneck-desc">{html.escape(desc)}</div>
              {f'<div class="bottleneck-loc dim">📍 {html.escape(loc)}</div>' if loc else ""}
            </div>"""

        if not cards:
            cards = '<div class="empty-state">No bottlenecks detected</div>'

        root_section = (
            f'<div class="root-cause-box"><strong>Root Cause:</strong> {html.escape(root_cause)}</div>'
            if root_cause else ""
        )

        return f"""
        <section class="card" id="s4-root-cause">
          <div class="section-header">
            <span class="section-num">04</span>
            <h2>Root Cause Analysis</h2>
            <div class="section-meta">
              {mode_badge}
              <span class="badge badge-critical">Severity Score: {severity_score}</span>
            </div>
          </div>
          {root_section}
          <div class="bottleneck-grid">{cards}</div>
        </section>"""

    # ── Section 5 — RAG Knowledge ──────────────────────────────────────────────

    def _section_rag_knowledge(self, rag_results: list, optimization: dict) -> str:
        rag_cases_used = optimization.get("rag_cases_used", 0)
        pattern        = optimization.get("rationale", "")

        if not rag_results:
            # Show placeholder regardless of why (disabled or cold-start)
            placeholder_msg = (
                "RAG retrieval returned no similar past optimization reports. "
                "This may be because Bedrock Knowledge Base is not configured "
                "(BEDROCK_ENABLED=false), the knowledge base is empty (cold start), "
                "or no sufficiently similar cases were found."
            )
            return f"""
        <section class="card" id="s5-rag-knowledge">
          <div class="section-header">
            <span class="section-num">05</span>
            <h2>Retrieved RAG Knowledge</h2>
          </div>
          <div class="rag-placeholder">
            <span class="rag-placeholder-icon">🔍</span>
            <div class="rag-placeholder-title">RAG not configured / No results</div>
            <div class="rag-placeholder-desc">{html.escape(placeholder_msg)}</div>
          </div>
        </section>"""

        case_cards = ""
        for i, case in enumerate(rag_results, 1):
            score    = case.get("score", 0.0)
            content  = case.get("content", "")
            source   = case.get("source", "unknown")
            score_pct = round(score * 100, 1)
            score_cls = "score-high" if score_pct >= 70 else ("score-mid" if score_pct >= 40 else "score-low")

            case_cards += f"""
            <div class="rag-card">
              <div class="rag-card-header">
                <span class="rag-case-num">Case {i}</span>
                <span class="rag-score {score_cls}">{score_pct}% match</span>
                <span class="rag-source dim" title="{html.escape(source)}">
                  📄 {html.escape(source.split("/")[-1] if "/" in source else source)}
                </span>
              </div>
              <details class="rag-content-details">
                <summary>View retrieved content</summary>
                <pre class="rag-content">{html.escape(content[:2000])}{'…' if len(content) > 2000 else ''}</pre>
              </details>
            </div>"""

        pattern_section = (
            f'<div class="pattern-box"><strong>Optimization Pattern Used:</strong> {html.escape(pattern)}</div>'
            if pattern else ""
        )

        return f"""
        <section class="card" id="s5-rag-knowledge">
          <div class="section-header">
            <span class="section-num">05</span>
            <h2>Retrieved RAG Knowledge</h2>
            <div class="section-meta">
              <span class="badge badge-info">{rag_cases_used} case(s) retrieved</span>
            </div>
          </div>
          {pattern_section}
          <div class="rag-cards">{case_cards}</div>
        </section>"""

    # ── Section 6 — Validation Results ────────────────────────────────────────

    def _section_validation(self, validation: dict, val_evidence: dict) -> str:
        decision    = validation.get("decision", "UNKNOWN")
        dec_cls     = _DECISION_CLASSES.get(decision, "decision-unknown")
        dec_icon    = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "❓")

        sem_check   = validation.get("semantic_check", "N/A")
        sem_pass    = sem_check == "PASS"
        conf_score  = validation.get("confidence_score", 0.0)
        llm_used    = validation.get("llm_used", False)
        llm_concerns = validation.get("llm_concerns", [])

        safety_passed  = validation.get("safety_checks_passed", [])
        safety_failed  = validation.get("safety_checks_failed", [])
        safety_warnings = validation.get("safety_warnings", [])

        # Explain diff
        diff = validation.get("explain_diff", {})
        diff_score  = diff.get("overall_improvement_score", 0.0)
        diff_insights = diff.get("insights", [])

        def check_row(name: str, passed: bool, detail: str = "", sev: str = "INFO") -> str:
            icon = "✅" if passed else ("❌" if sev == "CRITICAL" else "⚠️")
            row_cls = "check-pass" if passed else ("check-fail" if sev == "CRITICAL" else "check-warn")
            return f"""<tr class="{row_cls}">
              <td>{icon}</td>
              <td>{html.escape(name)}</td>
              <td>{html.escape(detail)}</td>
            </tr>"""

        safety_rows = ""
        for c in safety_passed:
            safety_rows += check_row(c, True, "Passed")
        for c in safety_failed:
            safety_rows += check_row(c, False, "Critical failure", "CRITICAL")
        for c in safety_warnings:
            safety_rows += check_row(c, True, "Warning — review recommended", "WARNING")

        if not safety_rows:
            safety_rows = "<tr><td colspan='3' class='empty'>No safety checks recorded</td></tr>"

        sem_row = check_row(
            "LLM Semantic Equivalence",
            sem_pass,
            (f"Confidence: {conf_score:.0%}" + (f" | Concerns: {', '.join(llm_concerns)}" if llm_concerns else ""))
            if llm_used else "LLM check skipped (Bedrock not configured)",
            "WARNING" if not sem_pass else "INFO",
        )

        concerns_html = ""
        if llm_concerns:
            concerns_html = "<ul class='concerns-list'>" + "".join(f"<li>{html.escape(c)}</li>" for c in llm_concerns) + "</ul>"

        insights_html = ""
        if diff_insights:
            insights_html = "<ul class='insights-list'>" + "".join(f"<li>{html.escape(ins)}</li>" for ins in diff_insights) + "</ul>"

        return f"""
        <section class="card" id="s6-validation">
          <div class="section-header">
            <span class="section-num">06</span>
            <h2>Validation Results</h2>
            <div class="section-meta">
              <span class="decision-badge {dec_cls}">{dec_icon} {decision}</span>
            </div>
          </div>

          <div class="val-columns">
            <div class="val-col">
              <h3 class="sub-heading">Stage 1 — SQL Safety Checks</h3>
              <div class="table-wrapper">
                <table class="data-table">
                  <thead><tr><th>Status</th><th>Check</th><th>Detail</th></tr></thead>
                  <tbody>{safety_rows}</tbody>
                </table>
              </div>
            </div>

            <div class="val-col">
              <h3 class="sub-heading">Stage 3 — LLM Semantic Check</h3>
              <div class="table-wrapper">
                <table class="data-table">
                  <thead><tr><th>Status</th><th>Check</th><th>Detail</th></tr></thead>
                  <tbody>{sem_row}</tbody>
                </table>
              </div>
              {concerns_html}
            </div>
          </div>

          <h3 class="sub-heading">Stage 2 — EXPLAIN Plan Diff (Score: {diff_score:.2f})</h3>
          {insights_html if insights_html else '<p class="dim">No EXPLAIN plan insights available.</p>'}
        </section>"""

    # ── Section 7 — Performance Metrics ───────────────────────────────────────

    def _section_performance(self, validation: dict) -> str:
        metrics = validation.get("metrics", {})
        perf_diff_raw = validation.get("perf_diff", {})
        verdict  = perf_diff_raw.get("verdict", "")
        score    = perf_diff_raw.get("overall_score", 0.0)
        verdict_icon = _VERDICT_ICONS.get(verdict, "")

        et   = metrics.get("execution_time", {})
        cr   = metrics.get("credits", {})
        bs   = metrics.get("bytes_scanned", {})
        pp   = metrics.get("partition_pruning", {})

        def pct_bar(pct: float) -> str:
            clamped = max(0.0, min(pct, 100.0))
            bar_cls = "bar-excellent" if clamped >= 60 else ("bar-good" if clamped >= 30 else "bar-marginal")
            return f"""<div class="pct-bar-wrap">
              <div class="pct-bar {bar_cls}" style="width:{clamped}%"></div>
              <span class="pct-label">{pct}%</span>
            </div>"""

        def metric_row(label: str, before: Any, after: Any, unit: str, pct: float) -> str:
            return f"""<tr>
              <td>{html.escape(label)}</td>
              <td class="before-val">{html.escape(str(before))} {html.escape(unit)}</td>
              <td class="after-val">{html.escape(str(after))} {html.escape(unit)}</td>
              <td>{pct_bar(pct)}</td>
            </tr>"""

        rows = (
            metric_row("Execution Time", et.get("before_sec", "N/A"), et.get("after_sec", "N/A"), "s",  et.get("improvement_pct", 0.0))
            + metric_row("Credits Consumed", cr.get("before", "N/A"), cr.get("after", "N/A"), "", cr.get("improvement_pct", 0.0))
            + metric_row("Bytes Scanned", bs.get("before_gb", "N/A"), bs.get("after_gb", "N/A"), "GB", bs.get("improvement_pct", 0.0))
            + metric_row("Partition Pruning", f"{pp.get('before_pct', 0)}%", f"{pp.get('after_pct', 0)}%", "", 0.0)
        )

        return f"""
        <section class="card" id="s7-performance">
          <div class="section-header">
            <span class="section-num">07</span>
            <h2>Performance Metrics</h2>
            {f'<div class="section-meta"><span class="badge badge-info">{verdict_icon} {verdict} (score: {score})</span></div>' if verdict else ""}
          </div>
          <div class="table-wrapper">
            <table class="data-table perf-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Before</th>
                  <th>After</th>
                  <th>Improvement</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </section>"""

    # ── Section 8 — Draft PR Summary ──────────────────────────────────────────

    def _section_pr_summary(self, pr: dict, report: dict) -> str:
        pr_title   = pr.get("pr_title", "N/A")
        branch     = pr.get("branch_name", "N/A")
        decision   = pr.get("validation_decision", "N/A")
        labels     = pr.get("labels", [])
        summary    = report.get("summary", "")
        pr_body    = pr.get("pr_body", "")
        dec_cls    = _DECISION_CLASSES.get(decision, "decision-unknown")
        dec_icon   = {"APPROVED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}.get(decision, "❓")

        label_tags = "".join(f'<span class="pr-label">{html.escape(l)}</span>' for l in labels)

        # Show first ~1500 chars of PR body as evidence
        body_excerpt = pr_body[:1500] + ("…" if len(pr_body) > 1500 else "")

        return f"""
        <section class="card" id="s8-pr-summary">
          <div class="section-header">
            <span class="section-num">08</span>
            <h2>Draft PR Summary</h2>
            <div class="section-meta">
              <span class="decision-badge {dec_cls}">{dec_icon} {decision}</span>
            </div>
          </div>
          <div class="pr-meta-grid">
            <div class="pr-meta-item">
              <div class="summary-label">PR Title</div>
              <div class="summary-value">{html.escape(pr_title)}</div>
            </div>
            <div class="pr-meta-item">
              <div class="summary-label">Branch</div>
              <div class="summary-value mono">{html.escape(branch)}</div>
            </div>
            <div class="pr-meta-item">
              <div class="summary-label">Labels</div>
              <div class="summary-value">{label_tags}</div>
            </div>
          </div>
          {f'<div class="pr-summary-box"><strong>Optimization Summary:</strong><br>{html.escape(summary)}</div>' if summary else ""}
          <h3 class="sub-heading">Generated Evidence (PR Body Excerpt)</h3>
          <details class="pr-body-details" open>
            <summary>View PR body</summary>
            <pre class="code-block">{html.escape(body_excerpt)}</pre>
          </details>
        </section>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template (embedded CSS + JS, dark-theme, responsive)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>POV3 Optimization Report — {query_id}</title>
  <meta name="description" content="AI-Generated Query Optimization Report for {query_id} — {timestamp}">
  <style>
    /* ── Reset & base ─────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:         #0d1117;
      --surface:    #161b22;
      --surface2:   #21262d;
      --border:     #30363d;
      --text:       #c9d1d9;
      --text-dim:   #8b949e;
      --accent:     #58a6ff;
      --green:      #3fb950;
      --yellow:     #d29922;
      --red:        #f85149;
      --purple:     #bc8cff;
      --radius:     10px;
      --font:       'Segoe UI', system-ui, sans-serif;
      --font-mono:  'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
    }}
    body {{
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      font-size: 14px;
    }}
    a {{ color: var(--accent); }}

    /* ── Layout ───────────────────────────────────────────── */
    .header-bar {{
      background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
      border-bottom: 1px solid var(--border);
      padding: 28px 40px;
      display: flex;
      align-items: center;
      gap: 20px;
    }}
    .header-logo {{
      font-size: 28px;
      font-weight: 800;
      background: linear-gradient(90deg, #58a6ff, #bc8cff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .header-sub {{
      color: var(--text-dim);
      font-size: 13px;
    }}
    .header-meta {{
      margin-left: auto;
      text-align: right;
    }}
    .header-qid {{
      font-family: var(--font-mono);
      color: var(--accent);
      font-size: 18px;
      font-weight: 700;
    }}

    .sidebar-nav {{
      position: fixed;
      top: 80px;
      left: 0;
      width: 220px;
      height: calc(100vh - 80px);
      overflow-y: auto;
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 20px 0;
      z-index: 100;
    }}
    .sidebar-nav a {{
      display: block;
      padding: 8px 20px;
      color: var(--text-dim);
      text-decoration: none;
      font-size: 12px;
      border-left: 3px solid transparent;
      transition: all 0.15s;
    }}
    .sidebar-nav a:hover, .sidebar-nav a.active {{
      color: var(--accent);
      border-left-color: var(--accent);
      background: rgba(88, 166, 255, 0.05);
    }}
    .sidebar-num {{
      display: inline-block;
      width: 24px;
      font-weight: 700;
      color: var(--accent);
    }}

    .main-content {{
      margin-left: 220px;
      padding: 32px 40px;
      max-width: 1200px;
    }}

    /* ── Cards / Sections ─────────────────────────────────── */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
      margin-bottom: 24px;
      scroll-margin-top: 20px;
      transition: box-shadow 0.2s;
    }}
    .card:hover {{ box-shadow: 0 0 0 1px var(--border), 0 4px 24px rgba(0,0,0,0.4); }}

    .section-header {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .section-num {{
      font-size: 11px;
      font-weight: 800;
      font-family: var(--font-mono);
      color: var(--accent);
      background: rgba(88,166,255,0.1);
      border: 1px solid rgba(88,166,255,0.3);
      border-radius: 4px;
      padding: 2px 8px;
    }}
    .section-header h2 {{
      font-size: 18px;
      font-weight: 700;
      color: #f0f6fc;
    }}
    .section-meta {{
      display: flex;
      gap: 8px;
      margin-left: auto;
      flex-wrap: wrap;
    }}
    .sub-heading {{
      font-size: 14px;
      font-weight: 600;
      color: var(--text-dim);
      margin: 20px 0 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    /* ── Summary grid ─────────────────────────────────────── */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 16px;
    }}
    .summary-item {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .summary-label {{
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .summary-value {{
      font-size: 15px;
      font-weight: 600;
      color: #f0f6fc;
      word-break: break-all;
    }}

    /* ── Code blocks ──────────────────────────────────────── */
    .code-block-wrapper {{ position: relative; }}
    .copy-btn {{
      position: absolute;
      top: 10px;
      right: 12px;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text-dim);
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.15s;
      z-index: 2;
    }}
    .copy-btn:hover {{ background: var(--border); color: var(--text); }}
    .code-block {{
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.7;
      color: #a9d49b;
      white-space: pre;
    }}
    .sql-block {{ color: #79c0ff; }}

    /* ── Tables ───────────────────────────────────────────── */
    .table-wrapper {{ overflow-x: auto; border-radius: 8px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .data-table th {{
      background: var(--surface2);
      color: var(--text-dim);
      font-weight: 600;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}
    .data-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid rgba(48,54,61,0.5);
      vertical-align: top;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tr:hover td {{ background: rgba(88,166,255,0.03); }}
    .data-table code {{ font-family: var(--font-mono); font-size: 12px; }}

    .check-pass td:first-child {{ color: var(--green); }}
    .check-fail td:first-child {{ color: var(--red); }}
    .check-warn td:first-child {{ color: var(--yellow); }}
    .empty {{ color: var(--text-dim); text-align: center; padding: 20px; }}

    /* ── Badges ───────────────────────────────────────────── */
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge-critical {{ background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.3); }}
    .badge-high     {{ background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }}
    .badge-medium   {{ background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }}
    .badge-low      {{ background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }}
    .badge-info     {{ background: rgba(188,140,255,0.15);color: #bc8cff; border: 1px solid rgba(188,140,255,0.3); }}
    .badge-warning  {{ background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }}

    .decision-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
    }}
    .decision-approved {{ background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.4); }}
    .decision-review   {{ background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.4); }}
    .decision-rejected {{ background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.4); }}
    .decision-unknown  {{ background: var(--surface2); color: var(--text-dim); border: 1px solid var(--border); }}

    /* ── Bottleneck cards ─────────────────────────────────── */
    .bottleneck-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 8px;
    }}
    .bottleneck-card {{
      background: var(--surface2);
      border-radius: 8px;
      padding: 16px;
      border-left: 4px solid var(--border);
      transition: transform 0.15s;
    }}
    .bottleneck-card:hover {{ transform: translateY(-2px); }}
    .sev-critical {{ border-left-color: var(--red); }}
    .sev-high     {{ border-left-color: var(--yellow); }}
    .sev-medium   {{ border-left-color: var(--accent); }}
    .sev-low      {{ border-left-color: var(--green); }}
    .bottleneck-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .bottleneck-icon {{ font-size: 18px; }}
    .bottleneck-type {{ font-weight: 700; font-size: 13px; font-family: var(--font-mono); color: #f0f6fc; }}
    .bottleneck-id  {{ font-size: 11px; }}
    .bottleneck-desc {{ font-size: 13px; color: var(--text); }}
    .bottleneck-loc  {{ font-size: 11px; margin-top: 6px; }}
    .root-cause-box {{
      background: rgba(88,166,255,0.07);
      border: 1px solid rgba(88,166,255,0.2);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
      font-size: 14px;
    }}

    /* ── RAG section ──────────────────────────────────────── */
    .rag-placeholder {{
      text-align: center;
      padding: 48px 20px;
      background: var(--surface2);
      border: 1px dashed var(--border);
      border-radius: 8px;
    }}
    .rag-placeholder-icon {{ font-size: 40px; display: block; margin-bottom: 12px; }}
    .rag-placeholder-title {{ font-size: 16px; font-weight: 700; color: var(--text-dim); margin-bottom: 8px; }}
    .rag-placeholder-desc  {{ font-size: 13px; color: var(--text-dim); max-width: 480px; margin: 0 auto; }}
    .rag-cards {{ display: flex; flex-direction: column; gap: 14px; }}
    .rag-card {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .rag-card-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: rgba(88,166,255,0.04);
      border-bottom: 1px solid var(--border);
    }}
    .rag-case-num {{ font-weight: 700; color: var(--accent); font-size: 13px; }}
    .rag-score {{
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
    }}
    .score-high {{ background: rgba(63,185,80,0.15); color: #3fb950; }}
    .score-mid  {{ background: rgba(210,153,34,0.15); color: #d29922; }}
    .score-low  {{ background: rgba(248,81,73,0.15);  color: #f85149; }}
    .rag-source {{ font-size: 11px; margin-left: auto; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .rag-content-details {{ padding: 14px 16px; }}
    .rag-content-details summary {{ cursor: pointer; color: var(--accent); font-size: 13px; margin-bottom: 10px; }}
    .rag-content {{
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--bg);
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      max-height: 300px;
      overflow-y: auto;
    }}
    .pattern-box {{
      background: rgba(188,140,255,0.07);
      border: 1px solid rgba(188,140,255,0.2);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
      font-size: 14px;
    }}

    /* ── Validation ───────────────────────────────────────── */
    .val-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .val-columns {{ grid-template-columns: 1fr; }} }}
    .concerns-list, .insights-list {{
      margin-top: 10px;
      padding-left: 20px;
      font-size: 13px;
      color: var(--yellow);
    }}
    .insights-list {{ color: var(--accent); }}

    /* ── Performance bars ─────────────────────────────────── */
    .perf-table .before-val {{ color: var(--red); }}
    .perf-table .after-val  {{ color: var(--green); }}
    .pct-bar-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 160px;
    }}
    .pct-bar {{
      height: 8px;
      border-radius: 4px;
      transition: width 1s ease;
      min-width: 4px;
    }}
    .bar-excellent {{ background: linear-gradient(90deg, #3fb950, #58a6ff); }}
    .bar-good      {{ background: linear-gradient(90deg, #58a6ff, #bc8cff); }}
    .bar-marginal  {{ background: var(--yellow); }}
    .pct-label {{ font-size: 12px; font-weight: 700; white-space: nowrap; color: var(--text); }}

    /* ── PR section ───────────────────────────────────────── */
    .pr-meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: 14px; margin-bottom: 16px; }}
    .pr-meta-item {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }}
    .pr-label {{ display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 20px; font-size: 11px; background: rgba(88,166,255,0.1); color: var(--accent); border: 1px solid rgba(88,166,255,0.2); margin-right: 4px; }}
    .pr-summary-box {{ background: rgba(63,185,80,0.07); border: 1px solid rgba(63,185,80,0.2); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 14px; }}
    .pr-body-details summary {{ cursor: pointer; color: var(--accent); font-size: 13px; margin-bottom: 12px; }}
    .pr-body-details .code-block {{ font-size: 12px; color: var(--text); }}

    /* ── Utilities ────────────────────────────────────────── */
    .mono {{ font-family: var(--font-mono); }}
    .dim  {{ color: var(--text-dim); }}
    .empty-state {{ color: var(--text-dim); padding: 24px; text-align: center; font-size: 13px; }}

    /* ── Footer ───────────────────────────────────────────── */
    .footer {{
      text-align: center;
      padding: 32px 40px;
      color: var(--text-dim);
      font-size: 12px;
      border-top: 1px solid var(--border);
      margin-top: 24px;
    }}

    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  </style>
</head>
<body>

<!-- Header -->
<div class="header-bar">
  <div>
    <div class="header-logo">POV3</div>
    <div class="header-sub">Query Auto-Optimization Agent</div>
  </div>
  <div class="header-meta">
    <div class="header-qid">{query_id}</div>
    <div class="header-sub">{timestamp}</div>
  </div>
</div>

<!-- Sidebar navigation -->
<nav class="sidebar-nav" id="sidebar">
  <a href="#s1-execution-summary"><span class="sidebar-num">01</span> Execution Summary</a>
  <a href="#s2-original-query"><span class="sidebar-num">02</span> Original Query</a>
  <a href="#s3-optimized-query"><span class="sidebar-num">03</span> Optimized Query</a>
  <a href="#s4-root-cause"><span class="sidebar-num">04</span> Root Cause Analysis</a>
  <a href="#s5-rag-knowledge"><span class="sidebar-num">05</span> RAG Knowledge</a>
  <a href="#s6-validation"><span class="sidebar-num">06</span> Validation Results</a>
  <a href="#s7-performance"><span class="sidebar-num">07</span> Performance Metrics</a>
  <a href="#s8-pr-summary"><span class="sidebar-num">08</span> Draft PR Summary</a>
</nav>

<!-- Main content -->
<main class="main-content">
{body}
</main>

<footer class="footer">
  Generated by POV3 Query Auto-Optimization Agent &middot; {timestamp} &middot; AI-generated &mdash; requires human review before merging
</footer>

<script>
  // Copy button
  function copyCode(btn) {{
    const pre = btn.parentElement.querySelector('pre');
    navigator.clipboard.writeText(pre.innerText).then(() => {{
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    }});
  }}

  // Sidebar active link on scroll
  const sections = document.querySelectorAll('section[id]');
  const navLinks  = document.querySelectorAll('.sidebar-nav a');
  const observer  = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        navLinks.forEach(l => l.classList.remove('active'));
        const active = document.querySelector('.sidebar-nav a[href="#' + entry.target.id + '"]');
        if (active) active.classList.add('active');
      }}
    }});
  }}, {{ rootMargin: '-20% 0px -70% 0px' }});
  sections.forEach(s => observer.observe(s));
</script>
</body>
</html>
"""


# ── Module-level singleton accessor ─────────────────────────────────────────

def get_html_report_generator() -> HTMLReportGenerator:
    """Return a fresh HTMLReportGenerator (lightweight, stateless)."""
    return HTMLReportGenerator()
