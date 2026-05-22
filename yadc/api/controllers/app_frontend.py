import os

from flask import Response, send_from_directory

from .blueprints import AppBlueprint


def _get_build_path() -> str:
    """Resolve the frontend build path from environment or default."""
    return os.environ.get("YADC_FRONTEND_BUILD_PATH", "")


def _safe_path(base: str, path: str) -> str | None:
    """Return a safe relative path under *base*, or None if it escapes."""
    joined = os.path.normpath(os.path.join(base, path))
    if not joined.startswith(os.path.normpath(base)):
        return None
    return joined


@AppBlueprint.route("/")
def index():
    build_path = _get_build_path()
    index_file = os.path.join(build_path, "index.html")
    if not os.path.isfile(index_file):
        return Response(
            "<h1>yadc Web UI</h1><p>Frontend not built. Run <code>cd yadc/webui && npm run build</code> first.</p>",
            mimetype="text/html",
        )
    return send_from_directory(build_path, "index.html")


@AppBlueprint.route("/robots.txt")
def robots():
    build_path = _get_build_path()
    robots_file = os.path.join(build_path, "robots.txt")
    if os.path.isfile(robots_file):
        return send_from_directory(build_path, "robots.txt")
    return Response("User-agent: *\nDisallow: /\n", mimetype="text/plain")


@AppBlueprint.route("/<path:path>")
def spa_catchall(path: str):
    """Serve static assets directly; fall back to index.html for SPA routes."""
    build_path = _get_build_path()
    safe = _safe_path(build_path, path)
    if safe and os.path.isfile(safe):
        return send_from_directory(build_path, path)
    # SPA fallback
    index_file = os.path.join(build_path, "index.html")
    if os.path.isfile(index_file):
        return send_from_directory(build_path, "index.html")
    return Response("Not Found", status=404)
