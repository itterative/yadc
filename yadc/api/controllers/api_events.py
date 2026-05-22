import dataclasses
import json
from typing import Any, override

from flask import Response, stream_with_context
from injector import inject

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from ..modules.sse_events import SSEEvents
from .blueprints import ApiBlueprint


@inject
def api_events(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory, sse_events: SSEEvents):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    _logger = logging.get_logger(__name__)

    class _DataclassJSONEncoder(json.JSONEncoder):
        @override
        def default(self, o: Any):
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return dataclasses.asdict(o)  # type: ignore[call-overload]
            return super().default(o)

    @app.get("/events")
    def events_endpoint():  # pyright: ignore[reportUnusedFunction]
        def _retrieve_events():
            for event in sse_events.receive(object):
                try:
                    yield f"event: {event.TYPE}\ndata: {json.dumps(event, cls=_DataclassJSONEncoder)}\n\n"
                except GeneratorExit:
                    raise
                except Exception as e:
                    _logger.warning("Failed to send sse event: %s", e)

        return Response(stream_with_context(_retrieve_events()), mimetype="text/event-stream")
