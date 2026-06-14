#!/usr/bin/env python3
"""
POV3 — Query Auto-Optimization Agent (FastAPI Server)

Run:
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8000

Or:
    python server.py

Endpoints:
    POST /alerts/ingest  — Receive POV4 alert, run optimization pipeline
    GET  /health         — Health check
    GET  /docs           — Interactive API documentation (Swagger UI)
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from src.api.routes import router
from src.config.settings import get_settings

# ── Logging configuration ──────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI application ─────────────────────────────────────────

app = FastAPI(
    title="POV3 — Query Auto-Optimization Agent",
    description=(
        "Multi-agent pipeline that analyzes slow Snowflake queries, "
        "generates optimized SQL, validates the result, and creates "
        "a draft Pull Request. Powered by LangGraph."
    ),
    version="1.0.0",
)

# Register routes
app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    """Log configuration on startup."""
    settings = get_settings()
    logger.info("POV3 server starting...")
    logger.info("Snowflake enabled: %s", settings.snowflake_enabled)
    logger.info("Snowflake configured: %s", settings.snowflake_configured)

    if settings.snowflake_configured:
        logger.info(
            "Snowflake target: %s / %s.%s (warehouse=%s)",
            settings.snowflake_account,
            settings.snowflake_database,
            settings.snowflake_schema,
            settings.snowflake_warehouse,
        )
    else:
        logger.info("Running in MOCK MODE — Snowflake not configured or disabled.")


# ── Direct execution ───────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
