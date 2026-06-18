"""
Optimization Prompt Builder.

Constructs the structured system + user prompts sent to Amazon Nova Pro
for SQL optimization. The prompt is designed to:

  1. Ground the model with a Snowflake-specific system persona
  2. Inject detected bottlenecks (from AnalysisAgent)
  3. Inject Snowflake metadata (columns, table stats) if available
  4. Inject RAG few-shot examples (similar past optimizations)
  5. Request structured JSON output for deterministic parsing

Keeping prompts in a separate module makes them easy to tune and
version-control without touching agent logic.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert Snowflake SQL optimization engineer with deep knowledge of:
- Snowflake micro-partition pruning and clustering
- Predicate pushdown and filter placement
- Column pruning (eliminating SELECT *)
- Join type selection (HASH JOIN vs. MERGE JOIN)
- Avoiding non-sargable predicates (e.g., YEAR(date_col) vs date_col >= '...')
- WITH clause / CTE factoring to eliminate repeated subqueries
- Window function optimization
- LIMIT pushdown for exploratory queries
- Eliminating unnecessary DISTINCT and ORDER BY
- Snowflake-specific features: QUALIFY, LATERAL FLATTEN, SAMPLE

You always:
1. Preserve the semantic equivalence of the query (same result set), EXCEPT for exploratory queries (e.g. unbounded SELECT * without an aggregation) where adding a LIMIT clause or pruning unused columns is highly encouraged and considered a valid optimization.
2. Return valid Snowflake SQL — no PostgreSQL/MySQL-specific syntax
3. Explain every change you make
4. Respond ONLY with valid JSON — no markdown, no prose outside the JSON
"""


def build_optimization_prompt(
    original_sql: str,
    bottlenecks: list[dict],
    snowflake_context: dict | None = None,
    rag_context: str = "",
) -> str:
    """
    Build the user-turn prompt for Nova Pro to optimize a SQL query.

    Args:
        original_sql:       The original query to optimize.
        bottlenecks:        List of bottleneck dicts from AnalysisAgent.
        snowflake_context:  Optional dict with table/column metadata.
        rag_context:        Pre-formatted few-shot context from RAGManager.

    Returns:
        A complete user-turn prompt string.
    """
    # ── Bottleneck summary ───────────────────────────────────────
    if bottlenecks:
        bottleneck_lines = "\n".join(
            f"  - [{b.get('type', 'UNKNOWN')}] {b.get('description', '')} "
            f"(severity: {b.get('severity', '?')}/10)"
            for b in bottlenecks
        )
        bottleneck_section = f"## Detected Bottlenecks\n{bottleneck_lines}"
    else:
        bottleneck_section = "## Detected Bottlenecks\n  - No specific bottlenecks detected — general optimization requested"

    # ── Snowflake context ────────────────────────────────────────
    if snowflake_context and snowflake_context.get("tables"):
        ctx_lines = []
        for table_info in snowflake_context["tables"]:
            tname = table_info.get("table_name", "unknown")
            cols = ", ".join(
                f"{c['column_name']} ({c['data_type']})"
                for c in table_info.get("columns", [])[:15]  # cap at 15 columns
            )
            ctx_lines.append(f"  - {tname}: {cols}")
        ctx_str = "\n".join(ctx_lines)
        context_section = f"## Snowflake Table Schema\n{ctx_str}"
    else:
        context_section = ""

    # ── RAG few-shot section ─────────────────────────────────────
    rag_section = rag_context.strip() if rag_context else ""

    # ── Assemble prompt ──────────────────────────────────────────
    prompt_parts = [
        "## Task",
        "Optimize the following Snowflake SQL query.",
        "",
        "## Original SQL",
        "```sql",
        original_sql.strip(),
        "```",
        "",
        bottleneck_section,
    ]

    if context_section:
        prompt_parts += ["", context_section]

    if rag_section:
        prompt_parts += ["", rag_section]

    prompt_parts += [
        "",
        "## Response Format",
        "Respond with ONLY a JSON object matching this exact schema — no prose:",
        "",
        """```json
{
  "optimized_sql": "<complete optimized SQL query>",
  "changes_applied": [
    {
      "type": "<bottleneck type, e.g. FULL_COLUMN_SCAN>",
      "action": "<one-sentence description of the change>",
      "reason": "<why this change improves performance>",
      "bottleneck_id": "<b1, b2, etc.>"
    }
  ],
  "rationale": "<2-3 sentence overall optimization rationale>",
  "confidence": <0.0-1.0 float>,
  "warnings": ["<any caveats or assumptions made>"]
}
```""",
        "",
        "IMPORTANT: Return ONLY the JSON. Do not wrap it in markdown fences in your response.",
    ]

    return "\n".join(prompt_parts)


def build_screener_prompt(original_sql: str, optimized_sql: str) -> str:
    """
    Build a prompt for Nova Lite to check semantic equivalence.

    Used by ValidationAgent for a fast LLM-based sanity check.

    Args:
        original_sql:  The original query.
        optimized_sql: The optimized candidate.

    Returns:
        A concise user-turn prompt string.
    """
    return f"""\
## Task
Determine whether the following two SQL queries are semantically equivalent.
"Semantically equivalent" means they would produce identical result sets
(same columns, same rows, same order if ORDER BY exists) on the same data.

EXCEPTION: If the original query is an unbounded exploratory query (e.g. lacks a LIMIT), and the optimized query adds a LIMIT clause or drops unnecessary columns to improve performance, you MUST treat them as semantically equivalent and return true.

## Original SQL
```sql
{original_sql.strip()}
```

## Optimized SQL
```sql
{optimized_sql.strip()}
```

## Response Format
Respond with ONLY a JSON object — no prose:
{{
  "semantically_equivalent": true | false,
  "confidence": <0.0-1.0>,
  "concerns": ["<concern 1>", "<concern 2>"]
}}
"""
