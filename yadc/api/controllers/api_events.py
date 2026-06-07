"""SSE event stream endpoint with ``Last-Event-ID`` resumption support (replays from the ring buffer on reconnect)."""

import asyncio
import json

from quart import Response, request

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from ..modules.sse_events import SSEEvents
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import DataclassJSONEncoder

SSE_RETRY_MS = 5000


@controller
def api_events(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory, sse_events: SSEEvents):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    _logger = logging.get_logger(__name__)

    @app.get("/events")
    def events_endpoint():  # pyright: ignore[reportUnusedFunction]
        # Standard SSE Last-Event-ID header (sent automatically by the browser
        # on reconnect).  Also accept a ``lastEventId`` query parameter as a
        # fallback for programmatic clients.
        last_event_id: int | None = None
        raw_id = request.headers.get("Last-Event-ID") or request.args.get("lastEventId")
        if raw_id:
            try:
                last_event_id = int(raw_id)
            except ValueError:
                _logger.warning("Ignoring invalid Last-Event-ID: %r", raw_id)

        async def _retrieve_events():
            try:
                # Tell the browser to wait 5 s before auto-reconnecting.
                yield f"retry: {SSE_RETRY_MS}\n\n"

                async for event_id, event in sse_events.receive(
                    object,
                    last_event_id=last_event_id,
                ):
                    try:
                        parts = [f"event: {event.TYPE}", f"data: {json.dumps(event, cls=DataclassJSONEncoder)}"]
                        if event_id:
                            parts.append(f"id: {event_id}")
                        yield "\n".join(parts) + "\n\n"
                    except GeneratorExit:
                        raise
                    except Exception as e:
                        _logger.warning("Failed to send sse event: %s", e)
            except asyncio.CancelledError:
                # Client disconnected — exhaust the generator cleanly so ASGI
                # sees a complete response rather than an unhandled exception.
                return

        response = Response(_retrieve_events(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response
