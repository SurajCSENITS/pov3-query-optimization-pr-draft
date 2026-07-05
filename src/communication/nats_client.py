"""
NATS connection manager.

Provides a singleton-style async NATS client with:
  - Automatic reconnection with exponential backoff
  - Graceful shutdown
  - Connection status logging

Usage:
    from src.communication.nats_client import get_nats_client

    client = get_nats_client()
    await client.connect()
    nc = client.connection      # raw nats.NATS connection
    await client.close()
"""

from __future__ import annotations

import logging

import nats
from nats.aio.client import Client as NATSClient

from src.config.settings import get_settings

logger = logging.getLogger("pov3.nats.client")

# ── Module-level singleton ──────────────────────────────────────

_instance: NATSConnectionManager | None = None


class NATSConnectionManager:
    """
    Manages the lifecycle of a single NATS connection.

    Handles connect, reconnect callbacks, and graceful close.
    The raw connection is exposed via the `connection` property
    for pub/sub operations.
    """

    def __init__(self) -> None:
        self._nc: NATSClient | None = None

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

    async def close(self) -> None:
        """Gracefully close the NATS connection."""
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
