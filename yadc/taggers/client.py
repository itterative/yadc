"""Async client for the tagger subprocess.

Wraps the blocking ``TaggerServer.tag()`` in ``asyncio.to_thread`` so it
can be used from async code (API jobs, etc.).
"""

from __future__ import annotations

from typing import Any

from yadc.taggers.base import Tagger, TaggerResult
from yadc.taggers.server import (
    HEARTBEAT_INTERVAL_SECONDS,
    POLL_INTERVAL_SECONDS,
    RESPONSE_TIMEOUT_SECONDS,
    STARTUP_TIMEOUT_SECONDS,
    TaggerServer,
)


class TaggerClient:
    """Async wrapper around :class:`TaggerServer`.

    Usage::

        client = TaggerClient(OnnxTagger, "/path/to/model.onnx")
        await client.start()
        result = await client.tag(image_bytes)  # TaggerResult
        await client.stop()
    """

    def __init__(
        self,
        tagger_cls: type[Tagger],
        model_path: str,
        tagger_kwargs: dict[str, Any] | None = None,
        *,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
        response_timeout: float = RESPONSE_TIMEOUT_SECONDS,
        start_timeout: float = STARTUP_TIMEOUT_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._server: TaggerServer = TaggerServer(
            tagger_cls,
            model_path,
            tagger_kwargs,
            heartbeat_interval=heartbeat_interval,
            response_timeout=response_timeout,
            start_timeout=start_timeout,
            poll_interval=poll_interval,
        )

    async def start(self) -> None:
        """Spawn the tagger process and wait for it to be ready."""
        import asyncio

        await asyncio.to_thread(self._server.start)

    async def tag(self, image_bytes: bytes) -> TaggerResult:
        """Send a tagging request and return the result."""
        import asyncio

        return await asyncio.to_thread(self._server.tag, image_bytes)

    async def stop(self) -> None:
        """Shutdown the tagger process (graceful: lets the worker finish
        its current request)."""
        import asyncio

        await asyncio.to_thread(self._server.stop)

    async def kill(self) -> None:
        """Force-terminate the tagger process immediately, interrupting any
        in-flight inference. See :meth:`TaggerServer.kill`."""
        import asyncio

        await asyncio.to_thread(self._server.kill)

    @property
    def is_alive(self) -> bool:
        """Return ``True`` if the tagger process is running."""
        return self._server.is_alive
