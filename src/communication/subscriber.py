"""
NATS subscriber service for POV3.

Subscribes to POV4 optimization alerts via NATS and bridges
them into the existing LangGraph pipeline. The pipeline is
executed asynchronously in a background task so the subscriber
loop is never blocked.

This module does NOT modify the LangGraph workflow, AgentMessage
schema, or any agent implementation.

Usage (managed by server.py lifecycle):
    subscriber = get_nats_subscriber()
    await subscriber.start()   # on FastAPI startup
    await subscriber.stop()    # on FastAPI shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.communication.nats_client import get_nats_client
from src.config.settings import get_settings
from src.models.messages import AgentMessage
from src.services.pipeline import run_optimization_pipeline

logger = logging.getLogger("pov3.nats.subscriber")

# Thread pool for running the synchronous LangGraph pipeline
# without blocking the asyncio event loop.
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="nats-pipeline")

# ── Module-level singleton ──────────────────────────────────────

_instance: NATSSubscriber | None = None


class NATSSubscriber:
    """
    Subscribes to NATS subject for POV4 optimization alerts.

    On each message:
      1. Deserializes JSON into AgentMessage
      2. Validates required payload fields
      3. Spawns an async background task to run the LangGraph pipeline
      4. Logs the result

    The pipeline runs in a thread pool executor because
    LangGraph's workflow.invoke() is synchronous.
    """

    def __init__(self) -> None:
        self._subscription = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """
        Connect to NATS and subscribe to the optimization alerts subject.

        Uses a queue group so multiple POV3 instances share the load
        (each message is delivered to exactly one subscriber in the group).
        """
        if self._running:
            logger.debug("NATS subscriber already running, skipping.")
            return

        settings = get_settings()
        client = get_nats_client()

        # Ensure NATS is connected
        await client.connect()

        nc = client.connection
        self._subscription = await nc.subscribe(
            subject=settings.nats_subject,
            queue=settings.nats_queue_group,
            cb=self._on_message,
        )
        self._running = True

        logger.info(
            "Subscribed to '%s' (queue=%s)",
            settings.nats_subject,
            settings.nats_queue_group,
        )

    async def stop(self) -> None:
        """Unsubscribe and mark as stopped."""
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None
            logger.info("NATS subscription removed.")

        self._running = False

    # ── Message handler ─────────────────────────────────────────

    async def _on_message(self, msg) -> None:
        """
        NATS message callback.

        Deserializes the message, validates it, and dispatches
        pipeline execution as an async background task.
        """
        subject = msg.subject
        raw = msg.data

        # 1. Deserialize JSON
        try:
            data: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                "Invalid JSON on subject '%s': %s — dropping message.", subject, e
            )
            return

        # 2. Construct AgentMessage
        try:
            message = AgentMessage(**data)
        except Exception as e:
            logger.warning(
                "Invalid AgentMessage on subject '%s': %s — dropping message. Data: %s",
                subject,
                e,
                data,
            )
            return

        # 3. Validate required payload fields
        payload = message.payload
        if not payload.get("query_id") or not payload.get("query_text"):
            logger.warning(
                "Missing required payload fields (query_id/query_text) in message_id=%s — dropping.",
                message.message_id,
            )
            return

        logger.info(
            "Received alert: message_id=%s query_id=%s subject=%s",
            message.message_id,
            payload.get("query_id"),
            subject,
        )

        # 4. Run pipeline asynchronously in thread pool (non-blocking)
        asyncio.get_event_loop().run_in_executor(
            _executor,
            self._run_pipeline_sync,
            message,
        )

    # ── Pipeline execution (runs in thread pool) ────────────────

    @staticmethod
    def _run_pipeline_sync(message: AgentMessage) -> None:
        """
        Execute the LangGraph optimization pipeline synchronously.

        This runs in a thread pool so it doesn't block the
        asyncio event loop or NATS message processing.
        """
        try:
            final_state = run_optimization_pipeline(message)

            validation = final_state.get("validation", {})
            pr = final_state.get("pr", {})

            logger.info(
                "Pipeline complete via NATS: query_id=%s validation=%s pr_state=%s",
                message.payload.get("query_id"),
                validation.get("semantic_check", "N/A"),
                pr.get("pr_state", "N/A"),
            )
        except Exception as e:
            logger.exception(
                "Pipeline failed via NATS for message_id=%s query_id=%s: %s",
                message.message_id,
                message.payload.get("query_id"),
                e,
            )


def get_nats_subscriber() -> NATSSubscriber:
    """
    Return the module-level singleton NATSSubscriber.

    Subscription is NOT active until `start()` is awaited.
    """
    global _instance
    if _instance is None:
        _instance = NATSSubscriber()
    return _instance
