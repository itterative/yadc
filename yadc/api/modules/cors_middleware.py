from logging import Logger

from flask import Blueprint, Response, request

from ..configuration import Configuration
from .logging_factory import LoggingFactory
from .service import Service


class CORSMiddleware(Service):
    def __init__(self, configuration: Configuration, logging: LoggingFactory):
        self._logger: Logger = logging.get_logger(__name__)
        self._configuration: Configuration = configuration

    def register(self, app: Blueprint):
        """Register CORS after_request handler on a Flask blueprint."""

        @app.after_request
        def after_request(response: Response):  # pyright: ignore[reportUnusedFunction]
            if not self._configuration.api_cors_enable:
                return response

            origin = request.headers.get("Origin")

            if request.method == "OPTIONS":
                response.headers.add("Access-Control-Allow-Credentials", str(self._configuration.api_cors_allow_credentials).lower())
                response.headers.add("Access-Control-Allow-Methods", self._configuration.api_cors_allow_methods)

                for header in self._configuration.api_cors_allow_headers:
                    response.headers.add("Access-Control-Allow-Headers", header)
            else:
                response.headers.add("Access-Control-Allow-Credentials", str(self._configuration.api_cors_allow_credentials).lower())

            if self._configuration.api_cors_allow_any_origin and origin:
                response.headers.add("Access-Control-Allow-Origin", origin)

            return response
