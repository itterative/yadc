import json

from flask import Response, request, stream_with_context

from ..configuration import Configuration
from ..json_utils import DataclassJSONEncoder
from ..modules.logging_factory import LoggingFactory
from ..modules.sse_events import SSEEvents
from . import controller
from .blueprints import ApiBlueprint

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

        def _retrieve_events():
            # Tell the browser to wait 5 s before auto-reconnecting.
            yield f"retry: {SSE_RETRY_MS}\n\n"

            for event_id, event in sse_events.receive(object, last_event_id=last_event_id):
                try:
                    parts = [f"event: {event.TYPE}", f"data: {json.dumps(event, cls=DataclassJSONEncoder)}"]
                    if event_id:
                        parts.append(f"id: {event_id}")
                    yield "\n".join(parts) + "\n\n"
                except GeneratorExit:
                    raise
                except Exception as e:
                    _logger.warning("Failed to send sse event: %s", e)

        return Response(stream_with_context(_retrieve_events()), mimetype="text/event-stream")
