import json

from flask import Response, stream_with_context

from ..configuration import Configuration
from ..json_utils import DataclassJSONEncoder
from ..modules.logging_factory import LoggingFactory
from ..modules.sse_events import SSEEvents
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_events(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory, sse_events: SSEEvents):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    _logger = logging.get_logger(__name__)

    @app.get("/events")
    def events_endpoint():  # pyright: ignore[reportUnusedFunction]
        def _retrieve_events():
            for event in sse_events.receive(object):
                try:
                    yield f"event: {event.TYPE}\ndata: {json.dumps(event, cls=DataclassJSONEncoder)}\n\n"
                except GeneratorExit:
                    raise
                except Exception as e:
                    _logger.warning("Failed to send sse event: %s", e)

        return Response(stream_with_context(_retrieve_events()), mimetype="text/event-stream")
