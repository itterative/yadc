from flask import Flask, Response

from ..configuration import Configuration


def apply_cors(app: Flask, config: Configuration) -> None:
    """Add CORS headers when enabled (for development)."""

    @app.after_request
    def _add_cors(response: Response) -> Response:
        if config.api_cors_enable:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
