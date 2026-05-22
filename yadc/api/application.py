from __future__ import annotations

import os

from flask import Flask

from .configuration import Configuration


def _register_controllers(app: Flask, config: Configuration) -> None:
    """Import and register all controller modules (Flask Blueprint pattern)."""
    os.environ["YADC_FRONTEND_BUILD_PATH"] = config.app_frontend_build_path

    # Import controllers to trigger @Blueprint.route decorations
    from .controllers import api_captioning, api_datasets, api_events, app_frontend  # noqa: F401
    from .controllers.blueprints import ApiBlueprint, AppBlueprint

    app.register_blueprint(ApiBlueprint)
    app.register_blueprint(AppBlueprint)

    from .controllers.api_cors import apply_cors

    apply_cors(app, config)


class Application:
    """Creates the Flask app and wires everything together."""

    def __init__(self, configuration: Configuration | None = None):
        self._config: Configuration = configuration or Configuration()
        self._app: Flask = Flask(__name__, static_folder=None)
        _register_controllers(self._app, self._config)

    @property
    def flask_app(self) -> Flask:
        return self._app

    def run(self) -> None:
        """Start the server using waitress (production) or Flask dev server (debug)."""
        import waitress

        print(f"yadc web UI starting on http://{self._config.http_host}:{self._config.http_port}")
        waitress.serve(
            self._app,
            host=self._config.http_host,
            port=self._config.http_port,
            threads=self._config.http_threads,
        )
