"""
NATS JetStream subscriber service for POV3.

Subscribes to the JetStream stream for POV4 optimization alerts and
bridges them into the existing LangGraph pipeline. Every message is
explicitly acknowledged so that:

  - Success  → msg.ack()    (removes from pending; not redelivered)
  - Bad msg  → msg.term()   (dead-lettered; not redelivered)
  - Pipeline error → msg.nak() (redelivered after ACK_WAIT seconds)

The pipeline is executed in a thread pool executor so the subscriber
loop is never blocked.

Compared to the previous NATS Core implementation:
  - nc.subscribe()  →  js.subscribe() with durable push consumer
  - No acknowledgement → explicit ack/nak/term at every exit point
  - Messages are replayed on restart if POV3 was down during delivery

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

from nats.aio.msg import Msg

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
    Subscribes to a JetStream durable push consumer for POV4 optimization alerts.

    On each message:
      1. Deserializes JSON into AgentMessage
      2. Validates required payload fields
      3. Spawns an async background task to run the LangGraph pipeline
      4. Acks / naks / terms the NATS message based on outcome

    Delivery semantics:
      - msg.ack()  — pipeline completed successfully; NATS will not redeliver
      - msg.nak()  — transient pipeline error; NATS redelivers after ACK_WAIT
      - msg.term() — unrecoverable message (bad JSON / missing fields);
                     NATS dead-letters the message without retrying

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
        Connect to NATS, ensure the JetStream stream exists, and subscribe
        with a durable push consumer.

        Uses a queue group so multiple POV3 instances share the load
        (each message is delivered to exactly one subscriber in the group).
        The consumer is durable so the server remembers the last delivered
        sequence even across POV3 restarts.
        """
        if self._running:
            logger.debug("NATS subscriber already running, skipping.")
            return

        settings = get_settings()
        client = get_nats_client()

        # Ensure NATS is connected
        await client.connect()

        # Bootstrap the JetStream stream (idempotent)
        await client.ensure_stream(
            name=settings.nats_stream_name,
            subjects=[settings.nats_subject],
        )

        js = client.jetstream

        # Explicitly pre-create the durable push consumer with deliver_group.
        # In nats-py, when using queue subscriptions, the durable name MUST
        # match the queue group name (library enforced constraint). We pass
        # queue=settings.nats_queue_group only; nats-py sets durable=queue.
        await self._ensure_consumer(
            js=js,
            stream_name=settings.nats_stream_name,
            subject=settings.nats_subject,
            queue=settings.nats_queue_group,
            ack_wait=settings.nats_ack_wait_seconds,
        )

        # Attach to the pre-created consumer. Pass queue only — nats-py
        # derives durable from the queue name automatically.
        self._subscription = await js.subscribe(
            subject=settings.nats_subject,
            queue=settings.nats_queue_group,
            cb=self._on_message,
        )
        self._running = True

        logger.info(
            "JetStream subscriber started — stream='%s' subject='%s' "
            "consumer='%s' queue='%s'",
            settings.nats_stream_name,
            settings.nats_subject,
            settings.nats_consumer_name,
            settings.nats_queue_group,
        )

    @staticmethod
    async def _ensure_consumer(
        js,
        stream_name: str,
        subject: str,
        queue: str,
        ack_wait: int,
    ) -> None:
        """
        Idempotently create a durable push consumer with deliver_group set.

        In nats-py, queue subscription durable names MUST equal the queue
        group name. This method pre-creates the consumer with the correct
        deliver_group so that js.subscribe(queue=...) attaches to it cleanly
        without raising "cannot create queue subscription" errors.

        If the consumer already exists with matching config, this is a no-op.
        If it exists with a mismatched deliver_group (stale), it is deleted
        and recreated so the queue group binding is always correct.
        """
        import nats.js.errors
        from nats.js.api import ConsumerConfig, DeliverPolicy, AckPolicy

        config = ConsumerConfig(
            name=queue,
            durable_name=queue,
            filter_subject=subject,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=float(ack_wait),   # seconds (float) as expected by nats-py 2.x
            deliver_group=queue,
            # Push consumers require a deliver_subject (inbox).
            # Use a well-known inbox derived from the queue name so
            # all instances in the queue group share the same inbox.
            deliver_subject=f"_INBOX.{queue}",
        )

        try:
            await js.add_consumer(stream_name, config)
            logger.info(
                "JetStream consumer '%s' created on stream '%s' "
                "(deliver_group='%s' ack_wait=%ds).",
                queue, stream_name, queue, ack_wait,
            )
        except nats.js.errors.BadRequestError:
            # Consumer exists — verify it has the correct deliver_group.
            # If not (stale config), delete and recreate.
            try:
                info = await js.consumer_info(stream_name, queue)
                existing_group = info.config.deliver_group or ""
                if existing_group != queue:
                    logger.warning(
                        "Consumer '%s' has deliver_group='%s' but expected '%s'. "
                        "Deleting stale consumer and recreating...",
                        queue, existing_group, queue,
                    )
                    await js.delete_consumer(stream_name, queue)
                    await js.add_consumer(stream_name, config)
                    logger.info(
                        "Consumer '%s' recreated with deliver_group='%s'.",
                        queue, queue,
                    )
                else:
                    logger.info(
                        "JetStream consumer '%s' already exists with correct config, reusing.",
                        queue,
                    )
            except Exception as e:
                logger.warning(
                    "Could not verify/fix consumer '%s': %s — proceeding anyway.", durable, e
                )

    async def stop(self) -> None:
        """Unsubscribe from JetStream and mark as stopped."""
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None
            logger.info("JetStream subscription removed.")

        self._running = False

    # ── Message handler ─────────────────────────────────────────

    async def _on_message(self, msg: Msg) -> None:
        """
        JetStream message callback.

        Deserializes the message, validates it, and dispatches pipeline
        execution as an async background task.

        Acknowledgement strategy:
          - Bad JSON or invalid AgentMessage → term() immediately (no retry)
          - Missing payload fields           → term() immediately (no retry)
          - Pipeline dispatched successfully → background task owns ack/nak
        """
        subject = msg.subject
        raw = msg.data

        # 1. Deserialize JSON — unrecoverable if malformed
        try:
            data: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                "Invalid JSON on subject '%s': %s — terminating message (no retry).",
                subject,
                e,
            )
            await msg.term()
            return

        # 2. Construct AgentMessage — unrecoverable if schema invalid
        try:
            message = AgentMessage(**data)
        except Exception as e:
            logger.warning(
                "Invalid AgentMessage on subject '%s': %s — terminating message. Data: %s",
                subject,
                e,
                data,
            )
            await msg.term()
            return

        # 3. Validate required payload fields — unrecoverable
        payload = message.payload
        if not payload.get("query_id") or not payload.get("query_text"):
            logger.warning(
                "Missing required payload fields (query_id/query_text) "
                "in message_id=%s — terminating message (no retry).",
                message.message_id,
            )
            await msg.term()
            return

        logger.info(
            "Received alert via JetStream: message_id=%s query_id=%s subject=%s",
            message.message_id,
            payload.get("query_id"),
            subject,
        )

        # 4. Run pipeline asynchronously in thread pool (non-blocking).
        #    The background task is responsible for calling msg.ack() on
        #    success or msg.nak() on transient failure.
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            _executor,
            self._run_pipeline_sync,
            message,
            msg,
            loop,
        )

    # ── Pipeline execution (runs in thread pool) ────────────────

    @staticmethod
    def _run_pipeline_sync(message: AgentMessage, msg: Msg, loop: asyncio.AbstractEventLoop) -> None:
        """
        Execute the LangGraph optimization pipeline synchronously.

        Runs in a thread pool so it doesn't block the asyncio event loop
        or NATS message processing.

        Acknowledgement:
          - Success  → schedules msg.ack() on the event loop
          - Failure  → schedules msg.nak() so NATS redelivers after ACK_WAIT
        """
        try:
            final_state = run_optimization_pipeline(message)

            validation = final_state.get("validation", {})
            pr = final_state.get("pr", {})

            logger.info(
                "Pipeline complete via JetStream: query_id=%s validation=%s pr_state=%s",
                message.payload.get("query_id"),
                validation.get("semantic_check", "N/A"),
                pr.get("pr_state", "N/A"),
            )

            # Acknowledge successful delivery — NATS will not redeliver
            asyncio.run_coroutine_threadsafe(msg.ack(), loop)

        except Exception as e:
            logger.exception(
                "Pipeline failed via JetStream for message_id=%s query_id=%s: %s",
                message.message_id,
                message.payload.get("query_id"),
                e,
            )
            # Negative-acknowledge — NATS will redeliver after ACK_WAIT seconds
            asyncio.run_coroutine_threadsafe(msg.nak(), loop)


def get_nats_subscriber() -> NATSSubscriber:
    """
    Return the module-level singleton NATSSubscriber.

    Subscription is NOT active until `start()` is awaited.
    """
    global _instance
    if _instance is None:
        _instance = NATSSubscriber()
    return _instance
