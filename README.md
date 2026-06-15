# POV3 — Intelligent Query Optimization Platform

A multi-agent orchestration pipeline using **LangGraph** that detects slow Snowflake queries, analyzes bottlenecks, optimizes the SQL using LLMs with RAG context, validates the results with safety checks and semantic screening, and generates a Draft Pull Request.

Features explicit **Agent-to-Agent (A2A) messaging** via Pydantic, **shared state management**, a production-ready **FastAPI web server**, **Snowflake metadata integration**, and **Amazon Bedrock (Nova Pro/Lite)** integration for advanced SQL reasoning.

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

### 2. Configuration
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```
Fill in your **Snowflake** and **AWS** credentials.
*Note: The system supports graceful degradation. If Snowflake is disabled, it falls back to regex-based analysis. If AWS Bedrock is disabled, it falls back to rule-based SQL optimizations.*

### 3. Running the Project

#### API Server Mode (Production)
Starts the FastAPI web service to ingest live alerts (e.g., from POV4):
```bash
python3 -m uvicorn server:app --reload --port 8000
```
* Interactive Swagger UI: Visit [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)
* Ingest Endpoint: `POST /alerts/ingest` (Accepts an `AgentMessage` payload)

#### CLI Mode (Testing)
Executes a single test run using mock data and prints rich visual output to the terminal:
```bash
python3 main.py
```

---

## Architecture & Pipeline

```text
                 POV4 Alert (HTTP POST)
                           │
                           ▼
┌────────────────── server.py (FastAPI) ─────────────────────────┐
│                                                                │
│                  LangGraph Pipeline                            │
│                                                                │
│   AnalysisAgent ──→ OptimizationAgent ──→ ValidationAgent      │
│         │                  │                      │            │
│         ▼ (Metadata)       ▼ (Nova Pro + RAG)     ▼ (Safety)   │
│     Snowflake DB     AWS Bedrock & S3      SQLSafetyEngine     │
│                                                   │            │
│                                                   ▼            │
│                              PRAgent ◄── ReportAgent           │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐   │
│   │  Shared State (QueryOptimizationState)                 │   │
│   │  ├─ input_data    ← POV4 payload                       │   │
│   │  ├─ analysis      ← Bottlenecks & Explain Plans        │   │
│   │  ├─ optimization  ← Optimized SQL & LLM Confidence     │   │
│   │  ├─ validation    ← Decision (APPROVED/REVIEW/REJECTED)│   │
│   │  ├─ report        ← RAG S3 Storage metadata            │   │
│   │  ├─ pr            ← Draft PR payload                   │   │
│   │  └─ messages[]    ← All A2A messages                   │   │
│   └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Draft PR Payload (JSON/Console)
```

---

## Core Components & Engines

### 1. Agents
* **Analysis Agent**: Connects to Snowflake to run `EXPLAIN` and check `QUERY_HISTORY`. Identifies table scans, spillage, and pruning issues. Falls back to regex heuristics if Snowflake is unavailable.
* **Optimization Agent**: Leverages **Amazon Nova Pro** via Bedrock to rewrite SQL. Uses **RAG (Retrieval-Augmented Generation)** to fetch prior successful optimization reports from an S3-backed Bedrock Knowledge Base. Gracefully falls back to deterministic rules if the LLM is unavailable.
* **Validation Agent**: Uses a robust 3-stage validation process:
  1. **SQL Safety Engine**: Deterministic AST-like safety checks (blocks DDL/DML, ensures WHERE/GROUP BY clauses are preserved).
  2. **Explain Plan Diff Engine**: Compares Snowflake EXPLAIN plans before/after optimization, scoring the improvement based on bytes scanned and operations removed.
  3. **Semantic Screener**: Uses **Amazon Nova Lite** to check for semantic equivalence and flag potential edge cases. Output is `APPROVED`, `REVIEW`, or `REJECTED`.
* **Report Agent**: Compiles all pipeline metrics into an `OptimizationReport` and uploads it to S3, continuously feeding the Bedrock Knowledge Base.
* **PR Agent**: Generates the final GitHub Draft Pull Request payload, embedding AI metadata, explain diff insights, and the validation decision.

### 2. Connectors
* `SnowflakeManager`: Singleton connection pool, executes queries, fetches history, and handles auto-retries.
* `BedrockManager`: Amazon Bedrock client for Nova Pro and Nova Lite, handles prompt invocation and JSON parsing.
* `S3Manager`: Uploads optimization reports to S3 to feed the RAG Knowledge Base.
* `RAGManager`: Queries the Bedrock Knowledge Base to retrieve relevant few-shot context for the LLM.

### 3. Engines
* `SQLSafetyEngine`: 9 strict deterministic safety checks (e.g. `NO_DDL_DML`, `LIMIT_NOT_REMOVED`, `WHERE_CLAUSE_PRESERVED`).
* `ExplainPlanDiffEngine`: Extracts metrics and operations from raw Snowflake EXPLAIN plans to calculate performance gains.
* `InsightGenerator`: Converts Explain Diff metrics into human-readable insights for the PR.

---

## Agent-to-Agent (A2A) Messaging

Every agent communicates via structured `AgentMessage` objects. This allows tracing the exact chain of thought and communication throughout the pipeline.

```python
class AgentMessage(BaseModel):
    message_id: str       # Auto-generated UUID
    timestamp: str        # ISO 8601 UTC
    sender: str           # Name of sending agent
    receiver: str         # Name of receiving agent
    task: str             # Human-readable task description
    payload: dict         # Structured data for the receiver
```

---

## Tech Stack

| Component       | Technology                       |
|-----------------|----------------------------------|
| Orchestration   | LangGraph `StateGraph`           |
| Web Framework   | FastAPI + Uvicorn                |
| Cloud AI        | AWS Bedrock (Nova Pro & Lite)    |
| Storage & RAG   | AWS S3 & Bedrock Knowledge Bases |
| DB Connection   | `snowflake-connector-python`     |
| Configuration   | `pydantic-settings` + `.env`     |
| Data Models     | Pydantic v2                      |
| Console Output  | Rich                             |
| Python          | 3.11+                            |
