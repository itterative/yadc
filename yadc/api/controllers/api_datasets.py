from flask import jsonify

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_datasets(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    _logger = logging.get_logger(__name__)

    @app.get("/datasets")
    def list_datasets():  # pyright: ignore[reportUnusedFunction]
        """List available datasets (scan for .toml configs)."""
        # TODO: implement dataset scanning
        return jsonify([])

    @app.get("/datasets/<name>/images")
    def list_images(name: str):  # pyright: ignore[reportUnusedParameter,reportUnusedFunction]
        """List images with captions/drafts (paginated)."""
        # TODO: implement image listing
        return jsonify([])
