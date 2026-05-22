from flask import Response, request
from injector import inject

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from .blueprints import ApiBlueprint


@inject
def api_cors(configuration: Configuration, app: ApiBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.after_request
    def after_request(response: Response):  # pyright: ignore[reportUnusedFunction]
        if not configuration.api_cors_enable:
            return response

        origin = request.headers.get("Origin")

        if request.method == "OPTIONS":
            response.headers.add("Access-Control-Allow-Credentials", str(configuration.api_cors_allow_credentials).lower())
            response.headers.add("Access-Control-Allow-Methods", configuration.api_cors_allow_methods)

            for header in configuration.api_cors_allow_headers:
                response.headers.add("Access-Control-Allow-Headers", header)
        else:
            response.headers.add("Access-Control-Allow-Credentials", str(configuration.api_cors_allow_credentials).lower())

        if configuration.api_cors_allow_any_origin and origin:
            response.headers.add("Access-Control-Allow-Origin", origin)

        return response
