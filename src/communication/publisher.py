"""
NATS JetStream publisher for POV4 → POV3 communication.

Provides a publisher that serializes AgentMessage to JSON and publishes
to a JetStream stream, receiving a PubAck that confirms the message
has been durably persisted before returning to the caller.

Compared to the previous NATS Core implementation:
  - nc.publish()  →  js.publish()  (returns PubAck with stream + seq)
  - Fire-and-forget → guaranteed receipt by the JetStream server
  - Retry loop preserved for transient publish errors

Usage (programmatic):
    from src.communication.publisher import NATSPublisher

    pub = NATSPublisher()
    await pub.connect()
    await pub.publish(message)
    await pub.close()

Usage (standalone test):
    python -m src.communication.publisher
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
    Publishes AgentMessage payloads to a NATS JetStream stream.

    Handles connection management, stream bootstrap, serialization, and
    retry logic for publish failures.  Each successful publish is confirmed
    by a PubAck containing the stream name and sequence number.
    """

    def __init__(self) -> None:
        self._client = get_nats_client()

    async def connect(self) -> None:
        """
        Connect to NATS and ensure the JetStream stream exists.

        Delegates connection to the shared NATSConnectionManager, then
        calls ensure_stream() so this publisher can be used standalone
        (e.g. for local testing) without depending on the subscriber
        startup path.
        """
        settings = get_settings()
        await self._client.connect()
        await self._client.ensure_stream(
            name=settings.nats_stream_name,
            subjects=[settings.nats_subject],
        )

    async def publish(
        self,
        message: AgentMessage,
        subject: str | None = None,
    ) -> None:
        """
        Publish an AgentMessage to the JetStream stream.

        The call blocks until the NATS server returns a PubAck, confirming
        that the message has been durably stored in the stream.  On failure
        the publish is retried up to 3 times with linear back-off.

        Args:
            message: The AgentMessage to publish.
            subject: Override subject (defaults to NATS_SUBJECT from .env).

        Raises:
            RuntimeError: If all 3 publish attempts fail.
        """
        settings = get_settings()
        target_subject = subject or settings.nats_subject
        js = self._client.jetstream

        payload = message.model_dump_json().encode("utf-8")

        # Publish with retry (up to 3 attempts)
        last_error = None
        for attempt in range(1, 4):
            try:
                ack = await js.publish(target_subject, payload)
                logger.info(
                    "Published message_id=%s to stream='%s' subject='%s' seq=%d (attempt %d)",
                    message.message_id,
                    ack.stream,
                    target_subject,
                    ack.seq,
                    attempt,
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "JetStream publish attempt %d failed for message_id=%s: %s",
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
        """Gracefully close the NATS connection."""
        await self._client.close()


# ── Standalone test runner ──────────────────────────────────────

async def _main() -> None:
    """Publish a sample POV4 alert for local testing."""
    sample_message = AgentMessage(
        sender=AgentRole.POV4_ALERT.value,
        receiver=AgentRole.ANALYSIS.value,
        task="Analyze slow query 01c5a808-0002-3a35-000e-044e0008348e",
        payload={
            "query_id": "01c5a808-0002-3a35-000e-044e0008348e",
            "warehouse": "COMPUTE_WH",
            "credits_used": 0.00002,
            "execution_time_seconds": 1.859,
            "bytes_scanned": 147609264,
            "issue_type": "NON_SARGABLE_PREDICATE",
            "query_text": "SELECT L_ORDERKEY, L_QUANTITY, L_EXTENDEDPRICE FROM LINEITEM WHERE YEAR(L_SHIPDATE) = 1995 AND MONTH(L_SHIPDATE) = 3;"
        }
    )

    publisher = NATSPublisher()
    try:
        await publisher.connect()
        print(f"📨 Publishing test alert: {sample_message.summary()}")
        await publisher.publish(sample_message)
        print("✅ Published successfully with JetStream PubAck!")
    finally:
        await publisher.close()


if __name__ == "__main__":
    asyncio.run(_main())
