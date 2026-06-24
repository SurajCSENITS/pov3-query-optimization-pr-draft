"""
Analysis Prompt Templates — LangChain ChatPromptTemplate.

Used by the AnalysisAgent to ask the LLM to identify performance
bottlenecks in a SQL query. Replaces the hardcoded regex pattern
matching (_detect_sql_patterns) with LLM-guided analysis.

The LLM receives:
  - The raw SQL query
  - Optional EXPLAIN plan text (from Snowflake)
  - Optional QUERY_HISTORY metadata (from Snowflake)

And returns structured bottleneck analysis as a BottleneckAnalysis
Pydantic model.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ── System prompt for SQL analysis ───────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert Snowflake SQL performance analyst. Your task is to identify \
performance bottlenecks in SQL queries by analyzing:

1. The SQL text itself (anti-patterns like SELECT *, non-sargable predicates, \
   unfiltered JOINs, missing LIMIT on exploratory queries, correlated subqueries)
2. The EXPLAIN plan (if provided) — look for full table scans, poor partition \
   pruning, excessive bytes scanned, sort/spill operations
3. Query execution history (if provided) — real execution time, bytes spilled, \
   partition usage statistics

IMPORTANT: Pay special attention to CORRELATED SUBQUERIES. These appear as \
scalar subqueries in SELECT or WHERE clauses that reference columns from the \
outer query (e.g., WHERE inner.col = outer.col). Correlated subqueries execute \
once per row of the outer query and are almost always a CRITICAL performance \
bottleneck. They should be rewritten as JOINs with GROUP BY, CTEs, or window \
functions.

For each bottleneck you identify, assign:
- A unique ID (B001, B002, etc.)
- A type from: FULL_COLUMN_SCAN, NON_SARGABLE_PREDICATE, UNFILTERED_JOIN, \
  REMOTE_SPILL, LOCAL_SPILL, FULL_TABLE_SCAN, POOR_PARTITION_PRUNING, \
  MISSING_LIMIT, EXPENSIVE_SORT, REDUNDANT_DISTINCT, SUBOPTIMAL_JOIN_ORDER, \
  CARTESIAN_PRODUCT, REPEATED_SUBQUERY, CORRELATED_SUBQUERY
- A severity: CRITICAL (10pts), HIGH (7pts), MEDIUM (4pts), LOW (1pt)
- A clear description
- The location in the query (e.g., SELECT clause, WHERE clause, JOIN clause)

Compute a severity_score as the sum of severity points across all bottlenecks.
Set recommendation to "OPTIMIZE" if any bottlenecks are found, "NO_ACTION" otherwise.

Respond ONLY with valid JSON — no markdown, no prose outside the JSON."""


# ── Analysis prompt template ─────────────────────────────────────────────────

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_SYSTEM_PROMPT),
    ("human", """\
## Task
Analyze the following Snowflake SQL query for performance bottlenecks.

## SQL Query
```sql
{sql}
```

{explain_plan_section}

{query_history_section}

## Response Format
Respond with ONLY a JSON object matching this exact schema — no prose:

```json
{{
  "bottlenecks": [
    {{
      "id": "B001",
      "type": "<bottleneck type>",
      "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
      "description": "<description>",
      "location": "<where in the query>"
    }}
  ],
  "severity_score": <integer sum of severity points>,
  "recommendation": "<OPTIMIZE|NO_ACTION>",
  "reasoning": "<brief chain-of-thought explanation>"
}}
```

IMPORTANT: Return ONLY the JSON."""),
])


# ── Helper functions ─────────────────────────────────────────────────────────

def format_explain_plan_section(plan_text: str) -> str:
    """Format EXPLAIN plan text into a prompt section."""
    if plan_text and plan_text.strip():
        return f"## EXPLAIN Plan\n```\n{plan_text.strip()}\n```"
    return "## EXPLAIN Plan\nNo EXPLAIN plan available."


def format_query_history_section(metadata: dict) -> str:
    """Format QUERY_HISTORY metadata into a prompt section."""
    if not metadata:
        return "## Query Execution History\nNo execution history available."

    lines = ["## Query Execution History"]
    field_map = {
        "execution_time_seconds": "Execution Time (sec)",
        "bytes_scanned": "Bytes Scanned",
        "rows_produced": "Rows Produced",
        "partitions_scanned": "Partitions Scanned",
        "partitions_total": "Partitions Total",
        "bytes_spilled_local": "Bytes Spilled (Local)",
        "bytes_spilled_remote": "Bytes Spilled (Remote)",
        "credits_used": "Credits Used",
    }
    for key, label in field_map.items():
        value = metadata.get(key)
        if value is not None:
            lines.append(f"  - {label}: {value}")

    return "\n".join(lines) if len(lines) > 1 else "## Query Execution History\nNo execution history available."
