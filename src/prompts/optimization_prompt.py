"""
Optimization Prompt Templates — LangChain ChatPromptTemplate.

Replaces manual f-string prompt builders with typed, testable,
composable LangChain prompt templates.

Templates:
  - OPTIMIZATION_PROMPT: Main SQL optimization prompt (system + human)
  - SCREENER_PROMPT: Semantic equivalence check prompt (system + human)

The ≤3 column constraint has been removed per Barandeep's feedback.
Agents now select necessary columns based on query purpose, not an
arbitrary limit.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ── System prompt for SQL optimization ───────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Snowflake SQL optimization engineer with deep knowledge of:
- Snowflake micro-partition pruning and clustering
- Predicate pushdown and filter placement
- Column pruning (eliminating SELECT *)
- Join type selection (HASH JOIN vs. MERGE JOIN)
- Avoiding non-sargable predicates (e.g., YEAR(date_col) vs date_col >= '...')
- Rewriting correlated subqueries as JOINs with GROUP BY for better performance
- WITH clause / CTE factoring to eliminate repeated subqueries
- Window function optimization
- LIMIT pushdown for exploratory queries
- Eliminating unnecessary DISTINCT and ORDER BY
- Snowflake-specific features: QUALIFY, LATERAL FLATTEN, SAMPLE

CRITICAL ACCURACY RULES — you MUST follow these:
1. Preserve the semantic equivalence of the query (same result set).
2. In "changes_applied", ONLY describe changes that are ACTUALLY reflected in \
your optimized SQL. Do NOT claim changes you did not make. For example:
   - Do NOT say "Replaced SELECT * with specific columns" unless the original \
     query actually uses SELECT * and your optimized SQL replaces it.
   - Do NOT say "Added filter conditions to the join" unless you actually \
     added new conditions to a JOIN clause.
   - Do NOT describe keeping something unchanged as a "change". If you kept \
     the ORDER BY clause as-is, that is NOT a change — do not list it.
   - Each change must be verifiable: someone comparing the original and \
     optimized SQL side-by-side must be able to see the exact difference.
3. When a query uses SELECT *, identify the columns actually needed to satisfy \
the query's apparent purpose (filters, aggregations, joins, output requirements). \
Replace SELECT * with only those necessary columns. If the necessary columns \
cannot be determined with confidence, keep SELECT * and flag it as a warning. \
Do not arbitrarily limit column count.
4. Pay special attention to CORRELATED SUBQUERIES in SELECT or WHERE clauses. \
These are a major performance bottleneck because they execute once per row of \
the outer query. When you detect a correlated subquery, consider rewriting it \
as a JOIN with GROUP BY, a CTE, or a window function.
5. Return valid Snowflake SQL — no PostgreSQL/MySQL-specific syntax.
6. Explain every change you make — but ONLY changes you actually make.
7. Respond ONLY with valid JSON — no markdown, no prose outside the JSON.
8. If you cannot find any meaningful optimizations, return the original SQL \
unchanged with an empty changes_applied list. Do NOT invent fake changes."""


# ── Optimization prompt template ─────────────────────────────────────────────

OPTIMIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """\
## Task
Optimize the following Snowflake SQL query.

## Original SQL
```sql
{original_sql}
```

## Detected Bottlenecks
{bottleneck_section}

{snowflake_context}

{rag_context}

## Response Format
Respond with ONLY a JSON object matching this exact schema — no prose:

```json
{{
  "optimized_sql": "<complete optimized SQL query>",
  "changes_applied": [
    {{
      "type": "<bottleneck type, e.g. FULL_COLUMN_SCAN>",
      "action": "<one-sentence description of the change>",
      "reason": "<why this change improves performance>",
      "bottleneck_id": "<b1, b2, etc.>"
    }}
  ],
  "rationale": "<2-3 sentence overall optimization rationale>",
  "confidence": <0.0-1.0 float>,
  "warnings": ["<any caveats or assumptions made>"]
}}
```

IMPORTANT: Return ONLY the JSON. Do not wrap it in markdown fences in your response."""),
])


# ── Screener prompt template (semantic equivalence check) ────────────────────

SCREENER_SYSTEM_PROMPT = """\
You are a SQL semantic analysis assistant. Be concise and respond only with JSON."""


SCREENER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCREENER_SYSTEM_PROMPT),
    ("human", """\
## Task
Determine whether the following two SQL queries are semantically equivalent.
"Semantically equivalent" means they would produce identical result sets
(same columns, same rows, same order if ORDER BY exists) on the same data.

Note: If the optimized query replaces SELECT * with specific columns that
cover the query's functional needs, this is an acceptable optimization and
should be treated as semantically equivalent as long as no required columns
are dropped.

## Original SQL
```sql
{original_sql}
```

## Optimized SQL
```sql
{optimized_sql}
```

## Response Format
Respond with ONLY a JSON object — no prose:
{{
  "semantically_equivalent": true | false,
  "confidence": <0.0-1.0>,
  "concerns": ["<concern 1>", "<concern 2>"],
  "reasoning": "<brief explanation>"
}}"""),
])


# ── Helper functions for backward compatibility ──────────────────────────────

def format_bottleneck_section(bottlenecks: list[dict]) -> str:
    """Format bottleneck list into a prompt section string."""
    if bottlenecks:
        lines = "\n".join(
            f"  - [{b.get('type', 'UNKNOWN')}] {b.get('description', '')} "
            f"(severity: {b.get('severity', '?')})"
            for b in bottlenecks
        )
        return f"## Detected Bottlenecks\n{lines}"
    return "## Detected Bottlenecks\n  - No specific bottlenecks detected — general optimization requested"


def format_snowflake_context(snowflake_context: dict | None) -> str:
    """Format Snowflake metadata into a prompt section string."""
    if snowflake_context and snowflake_context.get("tables"):
        ctx_lines = []
        for table_info in snowflake_context["tables"]:
            tname = table_info.get("table_name", "unknown")
            cols = ", ".join(
                f"{c['column_name']} ({c['data_type']})"
                for c in table_info.get("columns", [])[:15]
            )
            ctx_lines.append(f"  - {tname}: {cols}")
        ctx_str = "\n".join(ctx_lines)
        return f"## Snowflake Table Schema\n{ctx_str}"
    return ""
