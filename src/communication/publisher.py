"""
NATS publisher utility for POV4 → POV3 communication.

Provides a lightweight publisher that serializes AgentMessage
to JSON and publishes to the configured NATS subject.

This module serves two purposes:
  1. Reference implementation for the POV4 team
  2. Local testing tool for verifying end-to-end NATS flow

Usage (standalone test):
    python -m src.communication.publisher

Usage (programmatic):
    from src.communication.publisher import NATSPublisher

    pub = NATSPublisher()
    await pub.connect()
    await pub.publish(message)
    await pub.close()
"""

from __future__ import annotations

import asyncio
import logging

from src.communication.nats_client import get_nats_client
from src.config.settings import get_settings
from src.models.messages import AgentMessage, AgentRole

logger = logging.getLogger("pov3.nats.publisher")


class NATSPublisher:
    """
    Publishes AgentMessage payloads to a NATS subject.

    Handles connection management, serialization, and
    retry logic for publish failures.
    """

    def __init__(self) -> None:
        self._client = get_nats_client()

    async def connect(self) -> None:
        """Connect to NATS server (delegates to shared client)."""
        await self._client.connect()

    async def publish(
        self,
        message: AgentMessage,
        subject: str | None = None,
    ) -> None:
        """
        Publish an AgentMessage to the configured NATS subject.

        Args:
            message: The AgentMessage to publish.
            subject: Override subject (defaults to NATS_SUBJECT from .env).
        """
        settings = get_settings()
        target_subject = subject or settings.nats_subject
        nc = self._client.connection

        payload = message.model_dump_json().encode("utf-8")

        # Publish with retry (up to 3 attempts)
        last_error = None
        for attempt in range(1, 4):
            try:
                await nc.publish(target_subject, payload)
                await nc.flush()
                logger.info(
                    "Published message_id=%s to '%s' (attempt %d)",
                    message.message_id,
                    target_subject,
                    attempt,
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "Publish attempt %d failed for message_id=%s: %s",
                    attempt,
                    message.message_id,
                    e,
                )
                if attempt < 3:
                    await asyncio.sleep(1 * attempt)  # linear backoff

        raise RuntimeError(
            f"Failed to publish message_id={message.message_id} "
            f"after 3 attempts: {last_error}"
        )

    async def close(self) -> None:
        """Close the NATS connection."""
        await self._client.close()


# ── Standalone test runner ──────────────────────────────────────

async def _main() -> None:
    """Publish a sample POV4 alert for local testing."""
    sample_message = AgentMessage(
        sender=AgentRole.POV4_ALERT.value,
        receiver=AgentRole.ANALYSIS.value,
        task="Analyze slow query Q-TEST-001 — REMOTE_SPILL",
        payload={
            "query_id": "Q-TEST-001",
            "query_text": (
                "SELECT * FROM ORDERS o "
                "JOIN CUSTOMER c ON o.o_custkey = c.c_custkey "
                "WHERE YEAR(o.o_orderdate) = 1995"
            ),
            "warehouse": "WH_LARGE",
            "credits_used": 18,
            "execution_time_seconds": 240,
            "issue_type": "REMOTE_SPILL",
        },
    )

    publisher = NATSPublisher()
    try:
        await publisher.connect()
        print(f"📨 Publishing test alert: {sample_message.summary()}")
        await publisher.publish(sample_message)
        print("✅ Published successfully!")
    finally:
        await publisher.close()


if __name__ == "__main__":
    asyncio.run(_main())
