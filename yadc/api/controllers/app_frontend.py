from flask import Response, send_file, send_from_directory

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import AppBlueprint


@controller
def app_frontend(configuration: Configuration, app: AppBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/")
    def index_endpoint():  # pyright: ignore[reportUnusedFunction]
        index_file = f"{configuration.app_frontend_build_path}/index.html"
        import os

        if not os.path.isfile(index_file):
            return Response(
                "<h1>yadc Web UI</h1><p>Frontend not built. Run <code>cd yadc/webui && npm run build</code> first.</p>",
                mimetype="text/html",
            )
        return send_file(index_file)

    @app.get("/robots.txt")
    def robots_endpoint():  # pyright: ignore[reportUnusedFunction]
        import os

        robots_file = f"{configuration.app_frontend_build_path}/robots.txt"
        if os.path.isfile(robots_file):
            return send_file(robots_file)
        return Response("User-agent: *\nDisallow: /\n", mimetype="text/plain")

    @app.get("/_app/<path:path>")
    def app_endpoint(path: str):  # pyright: ignore[reportUnusedFunction]
        response = send_from_directory(f"{configuration.app_frontend_build_path}/_app", path)
        response.headers.set("Cache-Control", configuration.app_frontend_cache_control)
        return response

    @app.errorhandler(404)
    def spa_fallback(e: Exception):  # noqa: ARG001  # pyright: ignore[reportUnusedParameter,reportUnusedFunction]
        """Fall back to index.html for SPA routes."""
        import os

        index_file = f"{configuration.app_frontend_build_path}/index.html"
        if os.path.isfile(index_file):
            return send_file(index_file)
        return Response("Not Found", status=404)
