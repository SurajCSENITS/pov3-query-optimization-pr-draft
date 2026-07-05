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
    """Log configuration and start optional services on startup."""
    settings = get_settings()
    logger.info("POV3 server starting...")
    logger.info("Snowflake enabled: %s", settings.snowflake_enabled)
    logger.info("Snowflake configured: %s", settings.snowflake_configured)
    logger.info("Bedrock configured: %s", settings.bedrock_configured)
    logger.info("RAG configured: %s", settings.rag_configured)

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

    if settings.bedrock_configured:
        logger.info(
            "Bedrock: model=%s screener=%s region=%s",
            settings.bedrock_model_id,
            settings.bedrock_screener_model_id,
            settings.aws_region,
        )
        logger.info("RAG KB: %s (bucket=%s)", settings.bedrock_kb_id, settings.s3_bucket_name)
    else:
        logger.info("Bedrock not configured — LLM optimization disabled.")

    # ── NATS subscriber (inter-project communication) ───────────
    if settings.nats_configured:
        try:
            from src.communication.subscriber import get_nats_subscriber

            subscriber = get_nats_subscriber()
            await subscriber.start()
            logger.info("NATS subscriber started (subject=%s)", settings.nats_subject)
        except Exception as e:
            logger.error("Failed to start NATS subscriber: %s", e)
            logger.info("Continuing in HTTP-only mode.")
    else:
        logger.info("NATS disabled — operating in HTTP-only mode.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Gracefully stop NATS subscriber and close connection on shutdown."""
    settings = get_settings()

    if settings.nats_configured:
        try:
            from src.communication.subscriber import get_nats_subscriber
            from src.communication.nats_client import get_nats_client

            subscriber = get_nats_subscriber()
            await subscriber.stop()

            client = get_nats_client()
            await client.close()

            logger.info("NATS subscriber and connection shut down.")
        except Exception as e:
            logger.warning("Error during NATS shutdown: %s", e)


# ── Direct execution ───────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
