# POV3 — Query Auto-Optimization Agent (MVP)

Multi-agent orchestration pipeline using **LangGraph** that simulates the full lifecycle of detecting a slow Snowflake query, analyzing it, optimizing the SQL, validating the result, and generating a Draft Pull Request — all with explicit **Agent-to-Agent (A2A) messaging** and **shared state management**.

---

## Quick Start

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

## Architecture

```
POV4 Alert (mock)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                   LangGraph Pipeline                     │
│                                                          │
│   AnalysisAgent ──→ OptimizationAgent ──→ ValidationAgent│
│                                               │          │
│                                               ▼          │
│                          PRAgent ◄── ReportAgent         │
│                                                          │
│   ┌──────────────────────────────────────────────┐       │
│   │  Shared State (QueryOptimizationState)       │       │
│   │  ├─ input_data    ← POV4 payload             │       │
│   │  ├─ analysis      ← AnalysisAgent            │       │
│   │  ├─ optimization  ← OptimizationAgent        │       │
│   │  ├─ validation    ← ValidationAgent          │       │
│   │  ├─ report        ← ReportAgent              │       │
│   │  ├─ pr            ← PRAgent                  │       │
│   │  └─ messages[]    ← All A2A messages         │       │
│   └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
Draft PR (console output)
```

---

## Project Structure

```
pov3-query-optimizer/
├── main.py                         # Entry point — run this
├── requirements.txt                # Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── messages.py             # AgentMessage Pydantic model (A2A primitive)
    │   └── state.py                # QueryOptimizationState (LangGraph TypedDict)
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py                 # BaseAgent — abstract class with logging & state helpers
    │   ├── analysis.py             # Identifies query bottlenecks (rule-based)
    │   ├── optimization.py         # Rewrites SQL to fix bottlenecks (deterministic)
    │   ├── validation.py           # Verifies semantic equivalence (mock metrics)
    │   ├── report.py               # Assembles human-readable optimization report
    │   └── pr.py                   # Generates Draft PR payload (mock GitHub)
    └── graph/
        ├── __init__.py
        └── workflow.py             # LangGraph StateGraph definition & compilation
```

---

## Agent-to-Agent (A2A) Messaging

Every agent communicates via structured `AgentMessage` objects:

```python
class AgentMessage(BaseModel):
    message_id: str       # Auto-generated UUID
    timestamp: str        # ISO 8601 UTC
    sender: str           # Name of sending agent
    receiver: str         # Name of receiving agent
    task: str             # Human-readable task description
    payload: dict         # Structured data for the receiver
```

The full message chain is accumulated in `state["messages"]` and printed at the end:

```
POV4AlertAgent    → AnalysisAgent      | Analyze slow query Q123
AnalysisAgent     → OptimizationAgent  | Optimize query — 4 bottlenecks (score=28)
OptimizationAgent → ValidationAgent    | Validate optimized query — 3 changes
ValidationAgent   → ReportAgent        | Generate report — validation PASS
ReportAgent       → PRAgent            | Create draft PR with evidence
PRAgent           → HumanReviewer      | Draft PR created — awaiting human review
```

---

## Shared State

LangGraph threads a `QueryOptimizationState` through every node. Each agent reads upstream keys and writes to its own:

| Key            | Written By          | Contains                                         |
|----------------|---------------------|--------------------------------------------------|
| `input_data`   | Runner (main.py)    | POV4 alert payload                               |
| `analysis`     | AnalysisAgent       | Bottlenecks, severity score, recommendation      |
| `optimization` | OptimizationAgent   | Original + optimized SQL, changes applied        |
| `validation`   | ValidationAgent     | Before/after metrics, semantic check result      |
| `report`       | ReportAgent         | Formatted report with summary and evidence       |
| `pr`           | PRAgent             | Branch name, PR title, body (Markdown), labels   |
| `messages`     | All agents          | Append-only list of A2A messages (via reducer)   |

---

## What Each Agent Does

### 1. AnalysisAgent
Parses the SQL text to detect anti-patterns:
- `SELECT *` → full column scan
- `YEAR()` on filter column → non-sargable predicate
- `REMOTE_SPILL` metadata → execution engine issue
- Unfiltered JOIN → large intermediate result

### 2. OptimizationAgent
Applies deterministic rewrite rules:
- `SELECT *` → explicit column list
- `YEAR(col) = 2025` → `col BETWEEN '2025-01-01' AND '2025-12-31'`
- Remote spill → adds `LIMIT 10000` to bound result set

### 3. ValidationAgent
Simulates before/after comparison:
- Execution time reduction
- Credit consumption reduction
- Bytes scanned reduction
- Partition pruning improvement
- Row count match (semantic equivalence)

### 4. ReportAgent
Assembles all findings into a structured report:
- Summary paragraph
- Changes table with rationale
- Performance metrics table

### 5. PRAgent
Generates a complete Draft PR payload:
- Branch name: `ai/optimize-{query_id}`
- PR title with improvement percentage
- Full Markdown body with SQL diff, metrics, and review checklist
- Labels: `ai-generated`, `needs-human-review`
- Auto-merge: **always disabled**

---

## Sample Output

Running `python main.py` produces:

1. **Startup banner** with run timestamp
2. **POV4 alert table** showing the incoming payload
3. **Per-agent panels** with detailed processing logs
4. **A2A messages** logged as each agent hands off
5. **Draft PR preview** rendered in a bordered panel
6. **Message trail table** showing the full agent chain
7. **Final status** with PR details and validation result

---

## Extension Points

This MVP is designed to be extended with:

| Feature                     | Where to Add                                    |
|-----------------------------|-------------------------------------------------|
| **Snowflake integration**   | Replace mock logic in `analysis.py`, `validation.py` with `snowflake-connector-python` |
| **LLM-based optimization**  | Replace rule-based rewrites in `optimization.py` with LangChain LLM calls |
| **GitHub Draft PR**         | Replace console output in `pr.py` with PyGitHub API calls |
| **Conditional routing**     | Add `add_conditional_edges()` in `workflow.py` for retry loops |
| **New agents**              | Create a new file in `agents/`, register in `workflow.py` |
| **Real A2A protocol**       | Swap `AgentMessage` for Google A2A / OpenAI Swarm protocol |
| **Async execution**         | Switch to `graph.ainvoke()` with async agent implementations |

---

## Tech Stack

| Component       | Technology                |
|-----------------|---------------------------|
| Orchestration   | LangGraph `StateGraph`    |
| Data Models     | Pydantic v2               |
| State           | Python `TypedDict`        |
| Console Output  | Rich                      |
| Python          | 3.11+                     |
