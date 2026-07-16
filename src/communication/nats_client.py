"""
NATS JetStream connection manager.

Provides a singleton-style async NATS client with:
  - Automatic reconnection with exponential backoff
  - Graceful shutdown
  - JetStream context access via the `jetstream` property
  - One-shot stream bootstrap via `ensure_stream()`
  - Connection status logging

Usage:
    from src.communication.nats_client import get_nats_client

    client = get_nats_client()
    await client.connect()
    await client.ensure_stream("pov4-alerts", ["pov4.alerts.>"])
    js  = client.jetstream           # nats JetStreamContext
    nc  = client.connection          # raw nats.NATS connection (for admin use)
    await client.close()
"""

from __future__ import annotations

import logging
from typing import List

import nats
import nats.js.errors
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext

from src.config.settings import get_settings

logger = logging.getLogger("pov3.nats.client")

# ── Module-level singleton ──────────────────────────────────────

_instance: NATSConnectionManager | None = None


class NATSConnectionManager:
    """
    Manages the lifecycle of a single NATS connection with JetStream support.

    Handles connect, reconnect callbacks, graceful close, and exposes
    both the raw NATS connection and a JetStream context for pub/sub
    operations that require guaranteed delivery.
    """

    def __init__(self) -> None:
        self._nc: NATSClient | None = None

    # ── Connection properties ────────────────────────────────────

    @property
    def connection(self) -> NATSClient:
        """Return the active NATS connection, or raise if not connected."""
        if self._nc is None or not self._nc.is_connected:
            raise RuntimeError("NATS client is not connected. Call connect() first.")
        return self._nc

    @property
    def is_connected(self) -> bool:
        """Check if the NATS client is currently connected."""
        return self._nc is not None and self._nc.is_connected

    @property
    def jetstream(self) -> JetStreamContext:
        """
        Return the JetStream context for the active connection.

        Use this for all publish/subscribe operations that require
        guaranteed delivery, persistence, or consumer acknowledgement.

        Raises:
            RuntimeError: If NATS is not connected.
        """
        return self.connection.jetstream()

    # ── Lifecycle ───────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Connect to the NATS server using settings from .env.

        Configures automatic reconnection with up to 5 attempts
        and a 2-second wait between retries.
        """
        if self.is_connected:
            logger.debug("NATS client already connected, skipping.")
            return

        settings = get_settings()

        async def _on_disconnect():
            logger.warning("NATS disconnected.")

        async def _on_reconnect():
            logger.info("NATS reconnected.")

        async def _on_error(e: Exception):
            logger.error("NATS error: %s", e)

        try:
            self._nc = await nats.connect(
                servers=[settings.nats_url],
                max_reconnect_attempts=5,
                reconnect_time_wait=2,  # seconds between reconnect attempts
                disconnected_cb=_on_disconnect,
                reconnected_cb=_on_reconnect,
                error_cb=_on_error,
            )
            logger.info("Connected to NATS at %s", settings.nats_url)
        except Exception as e:
            logger.critical("Failed to connect to NATS at %s: %s", settings.nats_url, e)
            self._nc = None
            raise

    async def ensure_stream(self, name: str, subjects: List[str]) -> None:
        """
        Idempotently create a JetStream stream for the given subjects.

        If the stream already exists (possibly with a different config),
        this call is a no-op — the existing stream is left unchanged.

        Args:
            name:     Unique stream name, e.g. "pov4-alerts".
            subjects: List of NATS subject patterns this stream captures,
                      e.g. ["pov4.alerts.>"] or ["pov4.alerts.optimization"].

        Raises:
            RuntimeError: If NATS is not connected.
            Exception:    On unexpected JetStream errors.
        """
        js = self.jetstream
        try:
            info = await js.add_stream(name=name, subjects=subjects)
            logger.info(
                "JetStream stream '%s' created (subjects=%s)",
                info.config.name,
                subjects,
            )
        except nats.js.errors.BadRequestError:
            # Stream already exists — this is the expected steady-state path
            logger.info("JetStream stream '%s' already exists, reusing.", name)
        except Exception as e:
            logger.error("Failed to ensure JetStream stream '%s': %s", name, e)
            raise

    async def close(self) -> None:
        """Gracefully drain and close the NATS connection."""
        if self._nc and not self._nc.is_closed:
            await self._nc.drain()
            logger.info("NATS connection closed.")
            self._nc = None


def get_nats_client() -> NATSConnectionManager:
    """
    Return the module-level singleton NATSConnectionManager.

    Creates the instance on first call. Connection is NOT
    established until `connect()` is awaited.
    """
    global _instance
    if _instance is None:
        _instance = NATSConnectionManager()
    return _instance
