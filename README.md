# POV3 — Query Auto-Optimization Agent (Phase 1)

Multi-agent orchestration pipeline using **LangGraph** that detects slow Snowflake queries, analyzes bottlenecks, optimizes the SQL, validates the results, and generates a Draft Pull Request. Features explicit **Agent-to-Agent (A2A) messaging** via Pydantic and **shared state management**, supporting both CLI testing and a production-ready **FastAPI web server** with **Snowflake metadata integration**.

---

## Quick Start

### 1. Installation
```bash
# Clone/navigate to the directory
cd pov3-query-optimizer

# Create virtual environment and activate
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)
Copy `.env.example` to `.env` and fill in your Snowflake credentials:
```bash
cp .env.example .env
```
*Note: If no `.env` is configured or `SNOWFLAKE_ENABLED=False`, the system automatically falls back to dry-run/mock mode.*

### 3. Running the Project

#### CLI Mode (Original - Mock Testing)
Executes a single test run using hardcoded mock data and prints rich visual output to the terminal:
```bash
python main.py
```

#### API Server Mode (New - FastAPI)
Starts the web service to ingest live alerts (e.g., from POV4):
```bash
uvicorn server:app --reload --port 8000
```
* Interactive Swagger UI: Visit [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)
* Ingest Endpoint: `POST /alerts/ingest` (Accepts an `AgentMessage` payload)

---

## Architecture

```
                 POV4 Alert (HTTP POST)
                          │
                          ▼
┌────────────────── server.py (FastAPI) ──────────────────┐
│                                                         │
│                  LangGraph Pipeline                     │
│                                                         │
│   AnalysisAgent ──→ OptimizationAgent ──→ ValidationAgent│
│         │                                     │         │
│         ▼ (Reads Metadata)                    │         │
│     Snowflake DB                               ▼         │
│                              PRAgent ◄── ReportAgent    │
│                                                         │
│   ┌──────────────────────────────────────────────┐      │
│   │  Shared State (QueryOptimizationState)       │      │
│   │  ├─ input_data    ← POV4 payload             │      │
│   │  ├─ analysis      ← AnalysisAgent            │      │
│   │  ├─ optimization  ← OptimizationAgent        │      │
│   │  ├─ validation    ← ValidationAgent          │      │
│   │  ├─ report        ← ReportAgent              │      │
│   │  ├─ pr            ← PRAgent                  │      │
│   │  └─ messages[]    ← All A2A messages         │      │
│   └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
             Draft PR Payload (JSON/Console)
```

---

## Project Structure

```
pov3-query-optimizer/
├── main.py                         # CLI entry point
├── server.py                       # FastAPI server entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md
├── scripts/
│   └── test_snowflake_connection.py # Checks connectivity & credentials
└── src/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── routes.py               # API endpoints (/alerts/ingest, /health)
    ├── config/
    │   ├── __init__.py
    │   └── settings.py             # Config parser (Pydantic BaseSettings)
    ├── connectors/
    │   ├── __init__.py
    │   └── snowflake_manager.py    # Singleton connection pool & query executor
    ├── models/
    │   ├── __init__.py
    │   ├── messages.py             # AgentMessage Pydantic model (A2A primitive)
    │   └── state.py                # QueryOptimizationState (LangGraph TypedDict)
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py                 # Abstract BaseAgent with logging & state helpers
    │   ├── analysis.py             # Identifies query bottlenecks (Snowflake metadata or fallback heuristics)
    │   ├── optimization.py         # Rewrites SQL to fix bottlenecks (deterministic rules)
    │   ├── validation.py           # Verifies semantic equivalence (mock metrics)
    │   ├── report.py               # Assembles human-readable optimization report
    │   └── pr.py                   # Generates Draft PR payload (mock GitHub)
    └── graph/
        ├── __init__.py
        └── workflow.py             # LangGraph StateGraph definition & compilation
```

---

## Snowflake Connection Manager

The `SnowflakeConnectionManager` is a robust connector wrapper providing:
* **Connection Pooling**: Reuses connections safely using a Singleton pattern.
* **Auto-Retry**: Exponential backoff retry logic for query executions.
* **Explains & History**: Fetches explain plans and query execution statistics directly from Snowflake schema and query histories.
* **Schema Introspection**: Pulls table column metadata to aid Optimization and Validation agents.

Test Snowflake connectivity independently:
```bash
python scripts/test_snowflake_connection.py
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

The full message chain is accumulated in `state["messages"]` and returned in the HTTP API response.

---

## Tech Stack

| Component       | Technology                       |
|-----------------|----------------------------------|
| Orchestration   | LangGraph `StateGraph`           |
| Web Framework   | FastAPI + Uvicorn                |
| DB Connection   | `snowflake-connector-python`     |
| Configuration   | `pydantic-settings` + `.env`     |
| Data Models     | Pydantic v2                      |
| State           | Python `TypedDict`               |
| Console Output  | Rich                             |
| Python          | 3.11+ (tested on 3.14)           |

