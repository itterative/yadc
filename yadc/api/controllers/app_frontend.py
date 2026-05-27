from quart import Response, send_file, send_from_directory

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import AppBlueprint


@controller
def app_frontend(configuration: Configuration, app: AppBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/")
    async def index_endpoint():  # pyright: ignore[reportUnusedFunction]
        index_file = f"{configuration.app_frontend_build_path}/index.html"
        import os

        if not os.path.isfile(index_file):
            return Response(
                "<h1>yadc Web UI</h1><p>Frontend not built. Run <code>cd yadc/webui && npm run build</code> first.</p>",
                mimetype="text/html",
            )
        return await send_file(index_file)

    @app.get("/robots.txt")
    async def robots_endpoint():  # pyright: ignore[reportUnusedFunction]
        import os

        robots_file = f"{configuration.app_frontend_build_path}/robots.txt"
        if os.path.isfile(robots_file):
            return await send_file(robots_file)
        return Response("User-agent: *\nDisallow: /\n", mimetype="text/plain")

    @app.get("/site.webmanifest")
    async def webmanifest_endpoint():  # pyright: ignore[reportUnusedFunction]
        import os

        manifest_file = f"{configuration.app_frontend_build_path}/site.webmanifest"
        if os.path.isfile(manifest_file):
            return await send_file(manifest_file, mimetype="application/manifest+json")
        return Response("Not found", status=404)

    @app.get("/<filename>")
    async def static_root_file_endpoint(filename: str):  # pyright: ignore[reportUnusedFunction]
        import os

        # Serve favicon and other static root files from the build directory
        file_path = os.path.join(configuration.app_frontend_build_path, filename)
        if os.path.isfile(file_path):
            return await send_file(file_path)
        return Response("Not found", status=404)

    @app.get("/_app/<path:path>")
    async def app_endpoint(path: str):  # pyright: ignore[reportUnusedFunction]
        response = await send_from_directory(f"{configuration.app_frontend_build_path}/_app", path)
        response.headers.set("Cache-Control", configuration.app_frontend_cache_control)
        return response
