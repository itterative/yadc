"""Tagger process — runs in a separate ``multiprocessing.Process``.

Spawns a worker that loads an ONNX model and serves tagging requests via
``multiprocessing.Queue``.  The main process communicates through a
:class:`TaggerClient` (imported from ``yadc.taggers.client``).

Message protocol (dicts sent over ``Queue``):

    Startup:  worker emits  {``"id"``: 0, ``"ready"``: bool, ``"error"``: str}
              once the model is loaded (``ready=False`` on failure).

    Tag:      Request  {``"id"``: int (>0), ``"image_bytes"``: bytes}
              Response {``"id"``: int, ``"result"``: TaggerResult, ``"error"``: str}

    Heartbeat: worker emits {``"id"``: 0, ``"heartbeat"``: True} while idle
               (no incoming request) every ``HEARTBEAT_INTERVAL_SECONDS``.
               The main process treats any ``id == 0`` message as control
               and drains it when correlating a tag response.

The main process sends ``None`` on the request queue to shut the worker down.
``TaggerResult`` travels through the queue via pickle (the default for
``multiprocessing.Queue``).
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
import traceback
from multiprocessing.context import ForkServerProcess
from typing import Any

from yadc.taggers.base import Tagger, TaggerResult

logger = logging.getLogger(__name__)

# Use ``forkserver`` instead of the platform default (``fork`` on Linux)
# so the worker is forked from a clean, single-threaded server process
# rather than from the main (multi-threaded, async) yadc process.
# ``fork()`` from a threaded parent is deprecated in Python 3.13+ and
# can deadlock — the async event loop holds locks that the forked
# child would also try to acquire.
_ctx = multiprocessing.get_context("forkserver")

# How often the worker pushes a heartbeat while idle (no incoming
# request). The heartbeat lets the main process distinguish a slow but
# healthy worker from a dead process.
HEARTBEAT_INTERVAL_SECONDS: float = 15.0

# Maximum time to wait for a single tag response, or for the startup
# ``ready`` signal. Normal CPU inference completes well under this; the
# cap exists so a wedged (alive but hung) worker is detected rather than
# blocking the request handler forever. Picked generously so legitimate
# slow inference (large images, cold CPU caches) isn't falsely flagged.
RESPONSE_TIMEOUT_SECONDS: float = 120.0

# How often the parent polls the response queue while waiting for a tag
# response or the startup ready signal. This bounds how quickly a dead or
# killed worker is noticed (``is_alive()`` is re-checked each poll). Kept
# short and decoupled from the worker heartbeat so cancelling / killing an
# in-flight request unwinds in roughly this interval rather than a full
# heartbeat.
POLL_INTERVAL_SECONDS: float = 1.0


def _worker(
    tagger_cls: type[Tagger],
    model_path: str,
    tagger_kwargs: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
    heartbeat_interval: float,
) -> None:
    """Worker function that runs in the separate process.

    Loads the model, then loops: receive request → process → send response.
    While idle it pushes a heartbeat so the main process can detect a
    dead or unresponsive worker rather than blocking indefinitely.
    """
    tagger = tagger_cls(**tagger_kwargs)

    try:
        tagger.load_model(model_path)
        response_queue.put({"id": 0, "ready": True})
        logger.info("Tagger process ready.")
    except Exception:  # noqa: BLE001
        response_queue.put({"id": 0, "ready": False, "error": traceback.format_exc()})
        logger.exception("Failed to load tagger model.")
        return

    while True:
        try:
            msg = request_queue.get(timeout=heartbeat_interval)
        except queue.Empty:
            # Idle — affirm the worker is alive and the model is still loaded.
            response_queue.put({"id": 0, "heartbeat": True})
            continue

        if msg is None:
            # Sentinel — shutdown signal.
            break

        msg_id = msg.get("id", 0)
        try:
            result: TaggerResult = tagger.predict(msg["image_bytes"])
            response_queue.put({"id": msg_id, "result": result})
        except Exception:  # noqa: BLE001
            logger.exception("Tagging error.")
            response_queue.put({"id": msg_id, "error": traceback.format_exc()})

    tagger.unload_model()
    logger.info("Tagger process shut down.")


class TaggerServer:
    """Manages the tagger subprocess lifecycle.

    Usage::

        server = TaggerServer(OnnxTagger, "/path/to/model.onnx")
        server.start()          # blocks until model is loaded
        result = server.tag(image_bytes)  # returns a TaggerResult
        server.stop()
    """

    def __init__(
        self,
        tagger_cls: type[Tagger],
        model_path: str,
        tagger_kwargs: dict[str, Any] | None = None,
        *,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
        response_timeout: float = RESPONSE_TIMEOUT_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._tagger_cls: type[Tagger] = tagger_cls
        self._model_path: str = model_path
        self._tagger_kwargs: dict[str, Any] = tagger_kwargs or {}
        self._heartbeat_interval: float = heartbeat_interval
        self._response_timeout: float = response_timeout
        # How often the parent polls the response queue while waiting. This
        # is the granularity at which a dead/killed worker is detected
        # (``is_alive()`` is checked each poll). Decoupled from the worker's
        # heartbeat so cancel/kill unwinds in ~``poll_interval`` rather than
        # waiting a full heartbeat.
        self._poll_interval: float = poll_interval
        self._request_queue: Any = None
        self._response_queue: Any = None
        self._process: ForkServerProcess | None = None

    def start(self) -> None:
        """Spawn the tagger process and wait for it to be ready.

        Bounded by ``response_timeout`` so a worker that dies
        before reporting ready surfaces as a clear error instead of
        hanging the caller.
        """
        self._request_queue = _ctx.Queue()
        self._response_queue = _ctx.Queue()

        self._process = _ctx.Process(
            target=_worker,
            args=(
                self._tagger_cls,
                self._model_path,
                self._tagger_kwargs,
                self._request_queue,
                self._response_queue,
                self._heartbeat_interval,
            ),
            daemon=True,
            name="yadc-tagger",
        )
        self._process.start()

        deadline = time.monotonic() + self._response_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise RuntimeError("Tagger failed to start: no ready signal before timeout")
            # Poll at ``poll_interval`` so a dead process is detected
            # promptly via ``is_alive()`` rather than waiting the full timeout.
            poll = min(remaining, self._poll_interval)
            try:
                msg = self._response_queue.get(timeout=poll)
            except queue.Empty:
                if not self.is_alive:
                    self.stop()
                    raise RuntimeError("Tagger process died during startup") from None
                continue
            # Only ``id == 0`` control messages are expected here (ready).
            if msg.get("id") == 0:
                if msg.get("ready"):
                    return
                err = msg.get("error")
                self.stop()
                raise RuntimeError(f"Tagger failed to start: {err}")
            # Unexpected non-control message during startup — keep waiting.

    def tag(self, image_bytes: bytes) -> TaggerResult:
        """Send a tagging request and return the result (blocking).

        Bounded by ``response_timeout``. Control messages
        (``id == 0`` heartbeats) are drained; a dead process or an
        unresponsive worker surfaces as a ``RuntimeError`` instead of
        an indefinite block.
        """
        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError("Tagger not started")

        msg_id = 1
        self._request_queue.put({"id": msg_id, "image_bytes": image_bytes})
        deadline = time.monotonic() + self._response_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Tagger worker unresponsive (no response before timeout)")
            # Poll at ``poll_interval`` so a dead worker is detected
            # promptly via ``is_alive()`` (and a cancel/kill unwinds in
            # ~this interval) rather than waiting the full timeout.
            poll = min(remaining, self._poll_interval)
            try:
                msg = self._response_queue.get(timeout=poll)
            except queue.Empty:
                if not self.is_alive:
                    raise RuntimeError("Tagger worker process died") from None
                continue
            # Drain control messages (heartbeats); wait for our id.
            if msg.get("id") != msg_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("result") or TaggerResult(tags={}, categories={})

    def stop(self) -> None:
        """Shutdown the tagger process."""
        self._stop(sentinel=True)

    def kill(self) -> None:
        """Force-terminate the tagger process immediately.

        Unlike :meth:`stop`, this skips the graceful sentinel (which a
        worker mid-inference won't service until its current request
        finishes) and goes straight to ``terminate`` / ``kill``. Used by
        the cancel/escalation path to interrupt a wedged or long-running
        inference — there is no way to interrupt an ONNX ``session.run``
        in flight except by ending the process.
        """
        self._stop(sentinel=False)

    def _stop(self, *, sentinel: bool) -> None:
        if sentinel and self._request_queue is not None:
            self._request_queue.put(None)  # graceful shutdown signal
        proc = self._process
        if proc is not None and proc.is_alive():
            if sentinel:
                # Cooperative: give the worker time to finish its current
                # request before escalating.
                proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()  # SIGTERM
                # A worker mid-native ONNX inference won't service SIGTERM
                # until the call returns; keep the force-kill path short.
                proc.join(timeout=2 if not sentinel else 5)
                if proc.is_alive():
                    # Last resort: SIGKILL. Covers a wedged worker.
                    proc.kill()
                    proc.join(timeout=2)

    @property
    def is_alive(self) -> bool:
        """Return ``True`` if the tagger process is running."""
        return self._process is not None and self._process.is_alive()
