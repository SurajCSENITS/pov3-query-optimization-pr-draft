"""
HTML Report Generator — Sprint 2 (Visual Redesign).

Generates a premium, self-contained HTML report for every
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
import re
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
    "REGRESSION":     "🔴",
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

    # ── SQL Syntax Highlighting Helper ────────────────────────────────────────

    @staticmethod
    def _highlight_sql(escaped_sql: str) -> str:
        """Add CSS-class spans around SQL keywords in *already HTML-escaped* text."""
        kw = (
            r'SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AND|OR|NOT|IN|'
            r'AS|ORDER|BY|GROUP|HAVING|LIMIT|UNION|ALL|DISTINCT|CASE|WHEN|'
            r'THEN|ELSE|END|WITH|OVER|PARTITION|BETWEEN|LIKE|IS|NULL|EXISTS|'
            r'INTO|VALUES|SET|ASC|DESC|USING|CROSS|FULL|NATURAL|LATERAL|'
            r'FLATTEN|QUALIFY|WINDOW|ROWS|RANGE|CURRENT|ROW|OFFSET|TOP|'
            r'COUNT|SUM|AVG|MIN|MAX|YEAR|MONTH|DAY|DATE|CAST|COALESCE|'
            r'TRIM|UPPER|LOWER|REPLACE|LENGTH|ROUND|EXTRACT|CONCAT|'
            r'IFF|IFNULL|NVL|DECODE|GREATEST|LEAST|DATEADD|DATEDIFF|'
            r'TO_DATE|TO_TIMESTAMP|TO_CHAR|TO_NUMBER|TRY_CAST|'
            r'ARRAY_AGG|PARSE_JSON|GET_PATH|OBJECT_CONSTRUCT|'
            r'DENSE_RANK|ROW_NUMBER|RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE'
        )
        result = re.sub(
            rf'\b({kw})\b',
            r'<span class="sql-kw">\1</span>',
            escaped_sql,
            flags=re.IGNORECASE,
        )
        # Highlight single-quoted string literals
        result = re.sub(
            r"'([^']*)'",
            r"<span class=\"sql-str\">'\1'</span>",
            result,
        )
        return result

    @staticmethod
    def _format_credits(val: Any) -> str:
        if val is None or val == "N/A":
            return "N/A"
        try:
            val_f = float(val)
            # Format to 8 decimal places and strip trailing zeros, leaving at least 1 decimal
            formatted = f"{val_f:.8f}".rstrip("0")
            if formatted.endswith("."):
                formatted += "0"
            return formatted
        except (ValueError, TypeError):
            return str(val)

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
        credits   = self._format_credits(input_data.get("credits_used", "N/A"))
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
              <div class="summary-label">Credits Used (CREDITS_USED_CLOUD_SERVICES) (Before)</div>
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
        highlighted = self._highlight_sql(html.escape(sql))
        return f"""
        <section class="card" id="s2-original-query">
          <div class="section-header">
            <span class="section-num">02</span>
            <h2>Original Query</h2>
          </div>
          <div class="code-block-wrapper">
            <button class="copy-btn" onclick="copyCode(this)">📋 Copy</button>
            <pre class="code-block sql-block"><code>{highlighted}</code></pre>
          </div>
        </section>"""

    # ── Section 3 — Optimized Query ────────────────────────────────────────────

    def _section_optimized_query(self, optimization: dict) -> str:
        sql      = optimization.get("optimized_sql", "— not available —")
        mode     = optimization.get("optimization_mode", "unavailable")
        model    = optimization.get("llm_model", "")
        changes  = optimization.get("changes_applied", [])
        conf     = optimization.get("confidence", 0.0)

        highlighted = self._highlight_sql(html.escape(sql))

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
            <button class="copy-btn" onclick="copyCode(this)">📋 Copy</button>
            <pre class="code-block sql-block"><code>{highlighted}</code></pre>
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

    # ── Section 7 — Performance Metrics (Radial Gauges) ───────────────────────

    def _section_performance(self, validation: dict) -> str:
        metrics = validation.get("metrics", {})
        perf_diff_raw = validation.get("perf_diff", {})
        verdict  = perf_diff_raw.get("verdict", "")
        score    = perf_diff_raw.get("overall_score", 0.0)
        verdict_icon = _VERDICT_ICONS.get(verdict, "")
        is_regression = verdict == "REGRESSION"

        et   = metrics.get("execution_time", {})
        cr   = metrics.get("credits", {})
        bs   = metrics.get("bytes_scanned", {})
        pp   = metrics.get("partition_pruning", {})

        CIRC = 339.29  # 2 × π × 54

        def gauge(label: str, before: Any, after: Any, unit: str, pct: float) -> str:
            try:
                pct_f = float(pct)
            except (TypeError, ValueError):
                pct_f = 0.0

            is_reg = pct_f < 0
            # For regressions: show 0% fill (ring stays empty) + red class
            # For improvements: clamp to 0-100 and pick colour tier
            if is_reg:
                clamped = 0.0
                fill_cls = "g-regression"
                pct_display = f"+{abs(pct_f):.0f}"
                after_cls = "gv-after gv-regression"
                badge = '<span class="regression-badge">REGRESSION</span>'
            else:
                clamped = min(pct_f, 100.0)
                fill_cls = "g-excellent" if clamped >= 60 else ("g-good" if clamped >= 30 else "g-low")
                pct_display = f"{pct_f:.0f}"
                after_cls = "gv-after"
                badge = ""

            offset = CIRC * (1.0 - clamped / 100.0)

            return f"""
              <div class="gauge-card{'  gauge-card-regression' if is_reg else ''}">
                <div class="gauge-ring-wrap">
                  <svg class="gauge-svg" viewBox="0 0 120 120">
                    <circle class="gauge-track" cx="60" cy="60" r="54"/>
                    <circle class="gauge-fill {fill_cls}" cx="60" cy="60" r="54"
                      stroke-dasharray="{CIRC}" stroke-dashoffset="{CIRC}"
                      data-target="{offset:.2f}"
                      transform="rotate(-90 60 60)"/>
                  </svg>
                  <span class="gauge-pct{'  gauge-pct-reg' if is_reg else ''}">{pct_display}<small>%</small></span>
                </div>
                <div class="gauge-meta">
                  <div class="gauge-label">{html.escape(label)}</div>
                  <div class="gauge-vals">
                    <span class="gv-before">{html.escape(str(before))}{' ' + html.escape(unit) if unit else ''}</span>
                    <span class="gv-arrow">→</span>
                    <span class="{after_cls}">{html.escape(str(after))}{' ' + html.escape(unit) if unit else ''}</span>
                  </div>
                  {badge}
                </div>
              </div>"""

        cards = (
            gauge("Execution Time",
                  et.get("before_ms", "N/A"), et.get("after_ms", "N/A"),
                  "ms", et.get("improvement_pct", 0.0))
            + gauge("Credits Used",
                    self._format_credits(cr.get("before", "N/A")), self._format_credits(cr.get("after", "N/A")),
                    "", cr.get("improvement_pct", 0.0))
            + gauge("Bytes Scanned",
                    bs.get("before_mb", "N/A"), bs.get("after_mb", "N/A"),
                    "MB", bs.get("improvement_pct", 0.0))
            + gauge("Partition Pruning",
                    f'{pp.get("before_pct", 0)}%', f'{pp.get("after_pct", 0)}%',
                    "", pp.get("improvement_pct", 0.0))
        )

        verdict_badge_cls = "badge-critical" if is_regression else "badge-info"
        verdict_badge = (
            f'<div class="section-meta"><span class="badge {verdict_badge_cls}">'
            f'{verdict_icon} {verdict} (score: {score})'
            f'</span></div>'
        ) if verdict else ""

        return f"""
        <section class="card" id="s7-performance">
          <div class="section-header">
            <span class="section-num">07</span>
            <h2>Performance Metrics</h2>
            {verdict_badge}
          </div>
          <div class="gauge-grid">{cards}</div>
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
# HTML Template — Premium dark theme with glassmorphism, animations, gauges
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="#fcfbf2">
  <title>POV3 Optimization Report — {query_id}</title>
  <meta name="description" content="AI-Generated Query Optimization Report for {query_id} — {timestamp}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    /* ── Reset & Design Tokens ──────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:          #fcfbf2;
      --bg-surface:  #faf7e1;
      --bg-card:     rgba(255, 255, 255, 0.7);
      --border:      rgba(180, 83, 9, 0.12);
      --border-hover:rgba(180, 83, 9, 0.28);
      --text:        #2c2a21;
      --text-dim:    #5c5747;
      --text-muted:  #8a826b;
      --indigo:      #b45309;
      --violet:      #9a3412;
      --cyan:        #0369a1;
      --emerald:     #047857;
      --amber:       #b45309;
      --rose:        #b91c1c;
      --radius:      12px;
      --font:        'Inter', system-ui, -apple-system, sans-serif;
      --mono:        'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
      --ease:        cubic-bezier(0.4, 0, 0.2, 1);
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
      font-size: 14px;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }}

    a {{ color: var(--indigo); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* ── Keyframes ──────────────────────────────────────────── */
    @keyframes gradientShift {{
      0%, 100% {{ background-position: 0% 50%; }}
      50%      {{ background-position: 100% 50%; }}
    }}

    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50%      {{ opacity: 0.55; }}
    }}

    @keyframes float {{
      0%, 100% {{ transform: translateY(0px); }}
      50%      {{ transform: translateY(-6px); }}
    }}

    /* ── Scroll Progress Bar ───────────────────────────────── */
    .scroll-progress {{
      position: fixed;
      top: 0; left: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--indigo), var(--violet), var(--cyan));
      z-index: 9999;
      transition: width 80ms linear;
      border-radius: 0 2px 2px 0;
    }}

    /* ── Header ────────────────────────────────────────────── */
    .header {{
      position: relative;
      overflow: hidden;
      border-bottom: 1px solid var(--border);
    }}

    .header-bg {{
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, #fdf8e2 0%, #f6e6b4 25%, #e9ce82 50%, #d9a05b 75%, #fdf8e2 100%);
      background-size: 300% 300%;
      animation: gradientShift 12s ease infinite;
    }}

    .header-bg::before {{
      content: '';
      position: absolute;
      inset: 0;
      background:
        radial-gradient(ellipse at 20% 50%, rgba(217,119,6,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(245,158,11,0.12) 0%, transparent 40%),
        radial-gradient(ellipse at 50% 90%, rgba(251,191,36,0.1) 0%, transparent 45%);
    }}

    .header-bg::after {{
      content: '';
      position: absolute;
      inset: 0;
      background-image: radial-gradient(circle, rgba(0,0,0,0.03) 1px, transparent 1px);
      background-size: 28px 28px;
    }}

    .header-content {{
      position: relative;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 38px 48px;
      z-index: 1;
    }}

    .header-logo {{
      font-size: 34px;
      font-weight: 900;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    .header-tagline {{
      font-size: 13px;
      color: rgba(69, 26, 3, 0.75);
      font-weight: 400;
      margin-top: 4px;
      letter-spacing: 0.03em;
    }}

    .header-info {{ text-align: right; }}

    .header-qid {{
      font-family: var(--mono);
      font-size: 18px;
      font-weight: 700;
      color: #78350f;
      letter-spacing: -0.01em;
    }}

    .header-time {{
      font-size: 12px;
      color: rgba(69, 26, 3, 0.65);
      margin-top: 4px;
    }}

    /* ── Sidebar ───────────────────────────────────────────── */
    .sidebar-nav {{
      position: fixed;
      top: 0; left: 0;
      width: 240px;
      height: 100vh;
      background: rgba(246, 243, 222, 0.95);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border-right: 1px solid var(--border);
      padding: 114px 0 32px;
      z-index: 50;
      overflow-y: auto;
      transition: transform 0.35s var(--ease);
    }}

    .sidebar-nav a {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 24px;
      color: var(--text-dim);
      text-decoration: none;
      font-size: 12px;
      font-weight: 500;
      border-left: 3px solid transparent;
      transition: all 0.2s var(--ease);
      margin: 1px 0;
    }}

    .sidebar-nav a:hover {{
      color: var(--text);
      background: rgba(180, 83, 9, 0.05);
      text-decoration: none;
    }}

    .sidebar-nav a.active {{
      color: #b45309;
      border-left-color: var(--indigo);
      background: rgba(180, 83, 9, 0.08);
    }}

    .sidebar-num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px; height: 22px;
      font-size: 10px;
      font-weight: 700;
      font-family: var(--mono);
      color: #b45309;
      background: rgba(180, 83, 9, 0.08);
      border-radius: 6px;
      flex-shrink: 0;
      transition: all 0.2s var(--ease);
    }}

    .sidebar-nav a.active .sidebar-num {{
      background: var(--indigo);
      color: #fff;
      box-shadow: 0 0 12px rgba(180, 83, 9, 0.3);
    }}

    /* ── Main Content ──────────────────────────────────────── */
    .main-content {{
      margin-left: 240px;
      padding: 36px 48px;
      max-width: 1200px;
    }}

    /* ── Cards / Sections ──────────────────────────────────── */
    .card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 32px;
      margin-bottom: 28px;
      scroll-margin-top: 24px;
      opacity: 0;
      transform: translateY(28px);
      transition: opacity 0.6s var(--ease),
                  transform 0.6s var(--ease),
                  border-color 0.3s var(--ease),
                  box-shadow 0.3s var(--ease);
    }}

    .card.visible {{
      opacity: 1;
      transform: translateY(0);
    }}

    .card:hover {{
      border-color: var(--border-hover);
      box-shadow: 0 0 40px rgba(180, 83, 9, 0.04),
                  0 8px 32px rgba(0, 0, 0, 0.08);
    }}

    /* ── Section Headers ───────────────────────────────────── */
    .section-header {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 24px;
      padding-bottom: 18px;
      border-bottom: 1px solid rgba(180, 83, 9, 0.08);
      flex-wrap: wrap;
    }}

    .section-num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px; height: 34px;
      font-size: 12px;
      font-weight: 800;
      font-family: var(--mono);
      color: #fff;
      background: linear-gradient(135deg, var(--indigo), var(--violet));
      border-radius: 9px;
      flex-shrink: 0;
      box-shadow: 0 2px 12px rgba(180, 83, 9, 0.25);
    }}

    .section-header h2 {{
      font-size: 19px;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.01em;
    }}

    .section-meta {{
      display: flex;
      gap: 8px;
      margin-left: auto;
      flex-wrap: wrap;
    }}

    .sub-heading {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-dim);
      margin: 24px 0 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    /* ── Summary Grid ──────────────────────────────────────── */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 14px;
    }}

    .summary-item {{
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
      transition: all 0.25s var(--ease);
    }}

    .summary-item:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }}

    .summary-label {{
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
      margin-bottom: 8px;
    }}

    .summary-value {{
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      word-break: break-all;
    }}

    /* ── Code Blocks ───────────────────────────────────────── */
    .code-block-wrapper {{ position: relative; margin-top: 4px; }}

    .copy-btn {{
      position: absolute;
      top: 12px; right: 14px;
      background: rgba(180, 83, 9, 0.08);
      border: 1px solid rgba(180, 83, 9, 0.18);
      color: #b45309;
      padding: 6px 14px;
      border-radius: 7px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s var(--ease);
      z-index: 2;
      font-family: var(--font);
    }}

    .copy-btn:hover {{
      background: rgba(180, 83, 9, 0.18);
      border-color: rgba(180, 83, 9, 0.3);
      color: #78350f;
      transform: translateY(-1px);
    }}

    .code-block {{
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 22px 24px;
      overflow-x: auto;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.8;
      color: #4b4537;
      white-space: pre;
    }}

    .sql-block {{ color: #4b4537; }}

    /* SQL syntax highlighting */
    .sql-kw  {{ color: #b45309; font-weight: 600; }}
    .sql-str {{ color: #047857; }}

    /* ── Tables ─────────────────────────────────────────────── */
    .table-wrapper {{
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    .data-table th {{
      background: rgba(180, 83, 9, 0.05);
      color: var(--text-dim);
      font-weight: 600;
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 10px;
      letter-spacing: 0.06em;
    }}

    .data-table td {{
      padding: 12px 16px;
      border-bottom: 1px solid rgba(180, 83, 9, 0.05);
      vertical-align: top;
    }}

    .data-table tr:last-child td {{ border-bottom: none; }}

    .data-table tr:hover td {{ background: rgba(180, 83, 9, 0.02); }}

    .data-table code {{
      font-family: var(--mono);
      font-size: 12px;
      color: #b45309;
      background: rgba(180, 83, 9, 0.06);
      padding: 2px 6px;
      border-radius: 4px;
    }}

    .check-pass td:first-child {{ color: var(--emerald); }}
    .check-fail td:first-child {{ color: var(--rose); }}
    .check-warn td:first-child {{ color: var(--amber); }}
    .empty {{ color: var(--text-muted); text-align: center; padding: 20px; }}

    /* ── Badges ─────────────────────────────────────────────── */
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 11px;
      border-radius: 20px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      transition: all 0.2s var(--ease);
    }}

    .badge:hover {{ transform: scale(1.04); }}

    .badge-critical {{ background: rgba(185,28,28,0.08); color: #b91c1c; border: 1px solid rgba(185,28,28,0.18); }}
    .badge-high     {{ background: rgba(180,83,9,0.08); color: #b45309; border: 1px solid rgba(180,83,9,0.18); }}
    .badge-medium   {{ background: rgba(67,56,202,0.08); color: #4338ca; border: 1px solid rgba(67,56,202,0.18); }}
    .badge-low      {{ background: rgba(4,120,87,0.08); color: #047857; border: 1px solid rgba(4,120,87,0.18); }}
    .badge-info     {{ background: rgba(109,40,217,0.08); color: #6d28d9; border: 1px solid rgba(109,40,217,0.18); }}
    .badge-warning  {{ background: rgba(180,83,9,0.08); color: #b45309; border: 1px solid rgba(180,83,9,0.18); }}

    .decision-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.02em;
    }}

    .decision-approved {{ background: rgba(4,120,87,0.08); color: #047857; border: 1px solid rgba(4,120,87,0.25); }}
    .decision-review   {{ background: rgba(180,83,9,0.08); color: #b45309; border: 1px solid rgba(180,83,9,0.25); }}
    .decision-rejected {{ background: rgba(185,28,28,0.08); color: #b91c1c; border: 1px solid rgba(185,28,28,0.25); }}
    .decision-unknown  {{ background: rgba(92,87,71,0.08); color: var(--text-dim); border: 1px solid rgba(92,87,71,0.15); }}

    /* ── Bottleneck Cards ──────────────────────────────────── */
    .bottleneck-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 8px;
    }}

    .bottleneck-card {{
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
      border-left: 4px solid var(--border);
      transition: all 0.25s var(--ease);
    }}

    .bottleneck-card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15);
      border-color: var(--border-hover);
    }}

    .sev-critical {{ border-left-color: var(--rose) !important; }}
    .sev-high     {{ border-left-color: var(--amber) !important; }}
    .sev-medium   {{ border-left-color: var(--indigo) !important; }}
    .sev-low      {{ border-left-color: var(--emerald) !important; }}

    .sev-critical.bottleneck-card {{ animation: pulse 2.5s ease-in-out infinite; }}

    .bottleneck-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}

    .bottleneck-icon {{
      font-size: 18px;
      width: 36px; height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(180, 83, 9, 0.08);
      border-radius: 8px;
      flex-shrink: 0;
    }}

    .bottleneck-type {{ font-weight: 700; font-size: 12px; font-family: var(--mono); color: var(--text); }}
    .bottleneck-id   {{ font-size: 11px; }}
    .bottleneck-desc {{ font-size: 13px; color: var(--text); line-height: 1.55; }}
    .bottleneck-loc  {{ font-size: 11px; margin-top: 8px; }}

    .root-cause-box {{
      background: rgba(180, 83, 9, 0.04);
      border: 1px solid rgba(180, 83, 9, 0.12);
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 18px;
      font-size: 14px;
      line-height: 1.65;
    }}

    /* ── RAG Section ───────────────────────────────────────── */
    .rag-placeholder {{
      text-align: center;
      padding: 54px 24px;
      background: rgba(255, 255, 255, 0.3);
      border: 1px dashed rgba(180, 83, 9, 0.15);
      border-radius: 12px;
    }}

    .rag-placeholder-icon  {{ font-size: 44px; display: block; margin-bottom: 14px; animation: float 3s ease-in-out infinite; }}
    .rag-placeholder-title {{ font-size: 16px; font-weight: 700; color: var(--text-dim); margin-bottom: 8px; }}
    .rag-placeholder-desc  {{ font-size: 13px; color: var(--text-muted); max-width: 480px; margin: 0 auto; line-height: 1.65; }}

    .rag-cards {{ display: flex; flex-direction: column; gap: 14px; }}

    .rag-card {{
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      transition: all 0.2s var(--ease);
    }}

    .rag-card:hover {{ border-color: var(--border-hover); }}

    .rag-card-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 18px;
      background: rgba(180, 83, 9, 0.03);
      border-bottom: 1px solid var(--border);
    }}

    .rag-case-num {{ font-weight: 700; color: #b45309; font-size: 13px; }}

    .rag-score {{
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
    }}

    .score-high {{ background: rgba(16,185,129,0.1); color: #047857; }}
    .score-mid  {{ background: rgba(245,158,11,0.1); color: #b45309; }}
    .score-low  {{ background: rgba(239,68,68,0.1);  color: #b91c1c; }}

    .rag-source {{ font-size: 11px; margin-left: auto; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

    .rag-content-details {{ padding: 16px 18px; }}
    .rag-content-details summary {{ cursor: pointer; color: var(--indigo); font-size: 13px; font-weight: 500; margin-bottom: 10px; }}

    .rag-content {{
      font-family: var(--mono);
      font-size: 12px;
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(255, 255, 255, 0.7);
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      max-height: 300px;
      overflow-y: auto;
    }}

    .pattern-box {{
      background: rgba(109, 40, 217, 0.04);
      border: 1px solid rgba(109, 40, 217, 0.12);
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 18px;
      font-size: 14px;
    }}

    /* ── Validation ─────────────────────────────────────────── */
    .val-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}

    .concerns-list, .insights-list {{
      margin-top: 12px;
      padding-left: 20px;
      font-size: 13px;
      line-height: 1.7;
    }}

    .concerns-list {{ color: #b45309; }}
    .insights-list {{ color: #4338ca; }}

    /* ── Performance Gauges ─────────────────────────────────── */
    .gauge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 18px;
    }}

    .gauge-card {{
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px 20px;
      text-align: center;
      transition: all 0.3s var(--ease);
    }}

    .gauge-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-4px);
      box-shadow: 0 0 30px rgba(180, 83, 9, 0.06),
                  0 8px 24px rgba(0, 0, 0, 0.08);
    }}

    .gauge-ring-wrap {{
      position: relative;
      width: 110px; height: 110px;
      margin: 0 auto 18px;
    }}

    .gauge-svg {{ width: 100%; height: 100%; }}

    .gauge-track {{
      fill: none;
      stroke: rgba(180, 83, 9, 0.08);
      stroke-width: 8;
    }}

    .gauge-fill {{
      fill: none;
      stroke-width: 8;
      stroke-linecap: round;
      transition: stroke-dashoffset 1.6s cubic-bezier(0.22, 1, 0.36, 1);
    }}

    .g-excellent  {{ stroke: var(--emerald); }}
    .g-good       {{ stroke: var(--indigo); }}
    .g-low        {{ stroke: var(--amber); }}
    .g-regression {{ stroke: var(--rose); }}

    .gauge-card-regression {{
      border: 1px solid rgba(239, 68, 68, 0.25);
      background: rgba(239, 68, 68, 0.03);
    }}

    .gauge-pct-reg {{
      color: var(--rose);
    }}

    .gv-regression {{
      color: var(--rose);
      font-weight: 600;
    }}

    .regression-badge {{
      display: inline-block;
      margin-top: 4px;
      padding: 2px 8px;
      background: rgba(239,68,68,0.08);
      color: var(--rose);
      border: 1px solid rgba(239,68,68,0.22);
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}

    .gauge-pct {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
      font-family: var(--mono);
    }}

    .gauge-pct small {{ font-size: 13px; color: var(--text-dim); font-weight: 500; }}

    .gauge-label {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 8px; }}

    .gauge-vals  {{ font-size: 12px; color: var(--text-dim); }}
    .gv-before   {{ color: #b91c1c; font-weight: 500; font-family: var(--mono); font-size: 11px; }}
    .gv-arrow    {{ margin: 0 6px; color: var(--text-muted); }}
    .gv-after    {{ color: #047857; font-weight: 500; font-family: var(--mono); font-size: 11px; }}

    /* ── PR Section ────────────────────────────────────────── */
    .pr-meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}

    .pr-meta-item {{
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      transition: all 0.2s var(--ease);
    }}

    .pr-meta-item:hover {{ border-color: var(--border-hover); transform: translateY(-1px); }}

    .pr-label {{
      display: inline-flex;
      align-items: center;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 500;
      background: rgba(180, 83, 9, 0.08);
      color: #b45309;
      border: 1px solid rgba(180, 83, 9, 0.15);
      margin: 0 4px 4px 0;
    }}

    .pr-summary-box {{
      background: rgba(4, 120, 87, 0.04);
      border: 1px solid rgba(4, 120, 87, 0.12);
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 18px;
      font-size: 14px;
      line-height: 1.65;
    }}

    .pr-body-details summary {{ cursor: pointer; color: var(--indigo); font-size: 13px; font-weight: 500; margin-bottom: 12px; }}
    .pr-body-details .code-block {{ font-size: 12px; color: var(--text); }}

    /* ── Utilities ──────────────────────────────────────────── */
    .mono {{ font-family: var(--mono); }}
    .dim  {{ color: var(--text-dim); }}
    .empty-state {{ color: var(--text-muted); padding: 28px; text-align: center; font-size: 13px; }}

    /* ── Footer ────────────────────────────────────────────── */
    .footer {{
      text-align: center;
      padding: 36px 48px;
      color: var(--text-muted);
      font-size: 12px;
      border-top: 1px solid var(--border);
      margin-top: 28px;
      margin-left: 240px;
      background: rgba(255, 255, 255, 0.25);
      letter-spacing: 0.01em;
    }}

    .footer-brand {{
      display: inline;
      background: linear-gradient(90deg, var(--indigo), var(--violet));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      font-weight: 700;
    }}

    /* ── Back to Top ───────────────────────────────────────── */
    .back-to-top {{
      position: fixed;
      bottom: 32px; right: 32px;
      width: 44px; height: 44px;
      border-radius: 12px;
      background: rgba(180, 83, 9, 0.12);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(180, 83, 9, 0.22);
      color: #b45309;
      font-size: 18px;
      cursor: pointer;
      opacity: 0;
      transform: translateY(20px);
      transition: all 0.3s var(--ease);
      z-index: 200;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .back-to-top.visible {{
      opacity: 1;
      transform: translateY(0);
    }}

    .back-to-top:hover {{
      background: rgba(180, 83, 9, 0.25);
      transform: translateY(-2px);
      box-shadow: 0 4px 16px rgba(180, 83, 9, 0.2);
    }}

    /* ── Mobile Menu Button ────────────────────────────────── */
    .mobile-menu-btn {{
      display: none;
      position: fixed;
      top: 16px; left: 16px;
      z-index: 200;
      width: 42px; height: 42px;
      border-radius: 10px;
      background: rgba(180, 83, 9, 0.12);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(180, 83, 9, 0.22);
      color: #b45309;
      font-size: 20px;
      cursor: pointer;
      align-items: center;
      justify-content: center;
      transition: all 0.2s var(--ease);
    }}

    .mobile-menu-btn:hover {{
      background: rgba(180, 83, 9, 0.25);
    }}

    /* ── Scrollbar ──────────────────────────────────────────── */
    ::-webkit-scrollbar       {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(180, 83, 9, 0.18); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(180, 83, 9, 0.35); }}

    /* ── Responsive ────────────────────────────────────────── */
    @media (max-width: 900px) {{
      .val-columns {{ grid-template-columns: 1fr; }}
    }}

    @media (max-width: 768px) {{
      .sidebar-nav {{
        transform: translateX(-100%);
        width: 280px;
        z-index: 500;
      }}

      .sidebar-nav.open {{ transform: translateX(0); }}

      .mobile-menu-btn {{ display: flex; }}

      .main-content, .footer {{ margin-left: 0; }}

      .main-content {{ padding: 24px 20px; }}

      .header-content {{
        padding: 28px 20px;
        flex-direction: column;
        align-items: flex-start;
        gap: 14px;
      }}

      .header-info {{ text-align: left; }}

      .summary-grid {{ grid-template-columns: 1fr 1fr; }}
      .gauge-grid   {{ grid-template-columns: 1fr 1fr; }}
      .bottleneck-grid {{ grid-template-columns: 1fr; }}
      .pr-meta-grid {{ grid-template-columns: 1fr; }}
    }}

    @media (max-width: 480px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
      .gauge-grid   {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<!-- Scroll progress bar -->
<div class="scroll-progress" id="scrollProgress"></div>

<!-- Mobile menu button -->
<button class="mobile-menu-btn" id="mobileMenuBtn" aria-label="Toggle navigation">☰</button>

<!-- Header -->
<header class="header">
  <div class="header-bg"></div>
  <div class="header-content">
    <div>
      <div class="header-logo">⚡ POV3</div>
    </div>
    <div class="header-info">
      <div class="header-qid">{query_id}</div>
      <div class="header-time">{timestamp}</div>
    </div>
  </div>
</header>

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
  <span class="footer-brand">⚡ POV3</span> Query Auto-Optimization Agent &middot; {timestamp} &middot; AI-generated &mdash; requires human review before merging
</footer>

<!-- Back to top -->
<button class="back-to-top" id="backToTop" aria-label="Back to top">↑</button>

<script>
  /* ── Copy button ─────────────────────────────────────── */
  function copyCode(btn) {{
    const pre = btn.parentElement.querySelector('pre');
    navigator.clipboard.writeText(pre.innerText).then(() => {{
      btn.textContent = '✓ Copied!';
      setTimeout(() => btn.textContent = '📋 Copy', 2000);
    }});
  }}

  /* ── Scroll progress bar ─────────────────────────────── */
  const progressBar = document.getElementById('scrollProgress');
  function updateProgress() {{
    const winH = document.documentElement.scrollHeight - window.innerHeight;
    const pct = winH > 0 ? (window.scrollY / winH) * 100 : 0;
    progressBar.style.width = pct + '%';
  }}
  window.addEventListener('scroll', updateProgress, {{ passive: true }});

  /* ── Sidebar active link on scroll ───────────────────── */
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.sidebar-nav a');
  const sectionObs = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        navLinks.forEach(l => l.classList.remove('active'));
        const active = document.querySelector('.sidebar-nav a[href="#' + entry.target.id + '"]');
        if (active) active.classList.add('active');
      }}
    }});
  }}, {{ rootMargin: '-20% 0px -70% 0px' }});
  sections.forEach(s => sectionObs.observe(s));

  /* ── Card entrance animations ────────────────────────── */
  const cards = document.querySelectorAll('.card');
  const cardObs = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        entry.target.classList.add('visible');
        cardObs.unobserve(entry.target);
      }}
    }});
  }}, {{ threshold: 0.06, rootMargin: '0px 0px -40px 0px' }});
  cards.forEach((c, i) => {{
    c.style.transitionDelay = `${{i * 0.07}}s`;
    cardObs.observe(c);
  }});

  /* ── Gauge animations (fill on scroll) ───────────────── */
  const gaugeObs = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        entry.target.querySelectorAll('.gauge-fill').forEach(f => {{
          f.style.strokeDashoffset = f.getAttribute('data-target');
        }});
        gaugeObs.unobserve(entry.target);
      }}
    }});
  }}, {{ threshold: 0.25 }});
  document.querySelectorAll('.gauge-card').forEach(c => gaugeObs.observe(c));

  /* ── Back to top button ──────────────────────────────── */
  const backToTop = document.getElementById('backToTop');
  window.addEventListener('scroll', () => {{
    backToTop.classList.toggle('visible', window.scrollY > 400);
  }}, {{ passive: true }});
  backToTop.addEventListener('click', () => {{
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }});

  /* ── Mobile menu toggle ──────────────────────────────── */
  const menuBtn = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('sidebar');
  menuBtn.addEventListener('click', () => {{
    sidebar.classList.toggle('open');
    menuBtn.textContent = sidebar.classList.contains('open') ? '✕' : '☰';
  }});

  /* Close sidebar on nav click (mobile) */
  navLinks.forEach(link => {{
    link.addEventListener('click', () => {{
      if (window.innerWidth <= 768) {{
        sidebar.classList.remove('open');
        menuBtn.textContent = '☰';
      }}
    }});
  }});
</script>
</body>
</html>
"""


# ── Module-level singleton accessor ─────────────────────────────────────────

def get_html_report_generator() -> HTMLReportGenerator:
    """Return a fresh HTMLReportGenerator (lightweight, stateless)."""
    return HTMLReportGenerator()
