# POV3 — Intelligent Query Optimization Platform

A multi-agent orchestration pipeline using **LangGraph** and **LangChain** that detects slow Snowflake queries, analyzes bottlenecks, optimizes the SQL using LLMs with RAG context, validates the results with safety checks and semantic screening, and generates a Draft Pull Request.

Features explicit **Agent-to-Agent (A2A) messaging**, **shared state management**, a production-ready **FastAPI web server**, **NATS messaging integration** for decoupled inter-project communication, **Snowflake metadata integration**, **Amazon Bedrock** integration via LangChain for advanced SQL reasoning, and comprehensive **LangSmith** observability.

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
Fill in your **Snowflake**, **AWS**, and **LangSmith** credentials.
*Note: The system supports graceful degradation. If Snowflake is disabled, it uses mock telemetry. If AWS Bedrock is disabled, agents fall back to "Manual Review Required" recommendations without applying deterministic pseudo-optimizations.*

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
                     POV4 Alert (External System)
                                │
                                ▼
                       NATS Subject / Stream
                      (pov4.alerts.optimization)
                                │
          ┌─────────────────────┴──────────────────────┐
          ▼                                            ▼
 [ HTTP POST /alerts/ingest ]                 [ NATS Subscriber ]
          │                                            │
          └─────────────────────┬──────────────────────┘
                                ▼
 ┌───────────────── src/services/pipeline.py ─────────────────────┐
 │                                                                │
 │                      LangGraph Agent Flow                      │
 │                                                                │
 │                       [ AnalysisAgent ]                        │
 │                               │                                │
 │                               ▼                                │
 │                 ┌──► [ OptimizationAgent ]                     │
 │                 │             │                                │
 │      Retry flow │             ▼                                │
 │    (max 2 times)│      [ ValidationAgent ]                     │
 │                 │             │                                │
 │                 └─────── (Rejected)   │ (Approved)             │
 │                                       ▼                        │
 │                                [ ReportAgent ]                 │
 │                                       │                        │
 │                                       ▼                        │
 │                                  [ PRAgent ]                   │
 │                                                                │
 │   ┌────────────────────────────────────────────────────────┐   │
 │   │  Shared State (QueryOptimizationState)                 │   │
 │   │  ├─ input_data       ← POV4 payload                    │   │
 │   │  ├─ analysis         ← Bottlenecks & Explain Plans     │   │
 │   │  ├─ optimization     ← Optimized SQL & LLM Confidence  │   │
 │   │  ├─ validation       ← Decision (APPROVED/REJECTED)    │   │
 │   │  ├─ retry_count      ← Number of validation retries    │   │
 │   │  ├─ feedback_history ← Accumulated failure feedback    │   │
 │   │  ├─ report           ← RAG S3 Storage metadata         │   │
 │   │  ├─ pr               ← Draft PR payload                │   │
 │   │  └─ messages[]       ← All A2A messages                │   │
 │   └────────────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                       Agent Integrations
 ┌────────────────────────────────────────────────────────────────┐
 │ • AnalysisAgent     ──► Connects to Snowflake DB (telemetry)   │
 │ • OptimizationAgent ──► AWS Bedrock LLM + Knowledge Base (RAG) │
 │ • ValidationAgent   ──► Executes internal validation pipeline: │
 │                         1. SQLSafetyEngine (Regex/AST checks)  │
 │                         2. ExplainPlanDiffEngine (Plan compare)│
 │                         3. Semantic Screener (Bedrock equivalent)│
 │ • ReportAgent       ──► Generates HTML report & uploads to S3  │
 │ • PRAgent           ──► Formats and generates the final Draft PR│
 └────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  Draft PR Payload (JSON/Console)
```

---

## Core Components & Engines

### 1. Agents
* **Analysis Agent**: Connects to Snowflake to run `EXPLAIN` and check `QUERY_HISTORY`. Leverages LLM reasoning to identify table scans, spillage, and pruning issues using `ChatBedrock.with_structured_output()`.
* **Optimization Agent**: Leverages **Amazon Bedrock** via LangChain to rewrite SQL. Uses **LangChain RAG (AmazonKnowledgeBasesRetriever)** to fetch prior successful optimization reports from an S3-backed Bedrock Knowledge Base. Fully schema-validated output via `OptimizationResult`. Incorporates dynamic **Validation Feedback** on retry runs to guide the LLM away from past failures.
* **Validation Agent**: Uses a robust 3-stage validation process:
  1. **SQL Safety Engine (Regex Layer)**: Deterministic AST-like safety checks (blocks DDL/DML, ensures WHERE/GROUP BY clauses are preserved).
  2. **Explain Plan Diff Engine**: Compares Snowflake EXPLAIN plans before/after optimization, extracting telemetry cleanly.
  3. **Semantic Screener (LLM Layer)**: Uses **ChatBedrock** to check for semantic equivalence and flag potential edge cases via structured output `SemanticCheckResult`. Output is `APPROVED`, `REVIEW`, or `REJECTED`.
* **Validation Retry Loop**: If validation is not `APPROVED`, the pipeline routes back to the **Optimization Agent** up to 2 times. The failure feedback (performance regression details, safety failures, or semantic mismatch comments) is saved to the state's `feedback_history` and dynamically injected into the optimization prompt.
* **Report Agent**: Compiles all pipeline metrics into an `OptimizationReport` and uploads it to S3, continuously feeding the Bedrock Knowledge Base.
* **PR Agent**: Generates the final GitHub Draft Pull Request payload, embedding AI metadata, explain diff insights, and the validation decision.

### 2. Connectors (LangChain Native)
* `SnowflakeManager`: Singleton connection pool, executes queries, fetches history, and handles auto-retries.
* `ChatBedrock Factory`: Standard LangChain integration for Bedrock models supporting `.invoke()` and `.with_structured_output()`.
* `RAGManager`: Utilizes LangChain's `AmazonKnowledgeBasesRetriever` to seamlessly inject context into optimization prompts.

### 3. Engines
* `SQLSafetyEngine`: Hybrid deterministic + LLM safety checks.
* `ExplainPlanDiffEngine`: Extracts metrics and operations from raw Snowflake EXPLAIN plans to calculate performance gains.
* `PerformanceComparisonEngine`: Compares raw execution telemetry cleanly. Performance metrics are processed in **milliseconds (`ms`)** for execution time and **megabytes (`MB`)** for scanned data to ensure precision.

---

## Agent-to-Agent (A2A) Messaging & Observability

### A2A Messaging
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

### LangSmith Tracing & PII Masking
All agents inherit from `BaseAgent` and use the `@traceable` decorator on their `run()` loops. When `LANGSMITH_API_KEY` is configured, this provides deep, automatic tracking of token usage, latency, chain-of-thought, and state payloads natively inside the LangSmith platform.

To ensure data security, the tracing pipeline integrates a custom client (via `src/config/observability.py`) equipped with a `RuleNodeProcessor` to automatically mask sensitive PII in trace payloads. It matches and redacts:
* Email formats (`[EMAIL_REDACTED]`)
* Credit card formats (`[CC_REDACTED]`)
* Social Security Numbers (`[SSN_REDACTED]`)

---

## Tech Stack

| Component       | Technology                       |
|-----------------|----------------------------------|
| Orchestration   | LangGraph `StateGraph`           |
| LLM Framework   | LangChain `langchain-aws`        |
| Observability   | LangSmith `@traceable`           |
| Web Framework   | FastAPI + Uvicorn                |
| Messaging       | NATS (`nats-py`)                 |
| Cloud AI        | AWS Bedrock                      |
| Storage & RAG   | AWS S3 & Bedrock Knowledge Bases |
| DB Connection   | `snowflake-connector-python`     |
| Configuration   | `pydantic-settings` + `.env`     |
| Data Models     | Pydantic v2                      |
| Console Output  | Rich                             |
| Python          | 3.11+                            |
