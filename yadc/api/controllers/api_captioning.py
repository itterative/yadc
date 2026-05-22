from flask import Response, jsonify
from injector import inject

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from .blueprints import ApiBlueprint


@inject
def api_captioning(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    _logger = logging.get_logger(__name__)

    @app.post("/datasets/<name>/caption")
    def start_captioning(name: str):  # pyright: ignore[reportUnusedParameter,reportUnusedFunction]
        """Start a captioning run."""
        # TODO: implement
        return jsonify({"status": "ok"})

    @app.get("/datasets/<name>/caption/status")
    def captioning_status(name: str):  # pyright: ignore[reportUnusedParameter,reportUnusedFunction]
        """SSE stream for captioning progress."""
        # TODO: implement SSE
        return Response("data: {}\n\n", mimetype="text/event-stream")

    @app.delete("/datasets/<name>/caption")
    def stop_captioning(name: str):  # pyright: ignore[reportUnusedParameter,reportUnusedFunction]
        """Stop a captioning run."""
        # TODO: implement
        return jsonify({"status": "ok"})
