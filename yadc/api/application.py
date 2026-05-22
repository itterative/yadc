from __future__ import annotations

import sys
import time
from threading import Thread
from typing import Callable, override

from flask import Flask
from injector import Binder, Injector, Module, get_bindings  # pyright: ignore[reportUnknownVariableType]

from .configuration import Configuration
from .controllers.api_captioning import api_captioning
from .controllers.api_cors import api_cors
from .controllers.api_datasets import api_datasets
from .controllers.api_events import api_events
from .controllers.app_frontend import app_frontend
from .controllers.blueprints import ApiBlueprint, AppBlueprint
from .modules.event_dispatcher import EventDispatcher
from .modules.job_scheduler import JobScheduler
from .modules.logging_factory import LoggingFactory
from .modules.service import Service
from .modules.sse_events import SSEEvents


class Application(Module):
    """Creates the Flask app, wires DI, and starts the server."""

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration
        self.app: Flask = Flask(__name__, static_folder=None)

        self.injector: Injector = Injector(self)
        self.services: list[Service] = []

    @override
    def configure(self, binder: Binder):
        binder.bind(Configuration, to=self.configuration)
        binder.bind(Flask, to=self.app)

        binder.bind(
            ApiBlueprint,
            to=ApiBlueprint(
                name="api",
                import_name=__name__,
                url_prefix="/api",
            ),
        )

        binder.bind(
            AppBlueprint,
            to=AppBlueprint(
                name="app",
                import_name=__name__,
            ),
        )

    def configure_services(self):
        services: list[type[Service]] = [
            LoggingFactory,
            EventDispatcher,
            JobScheduler,
            SSEEvents,
        ]

        for service_cls in services:
            self.services.append(self.injector.get(service_cls))

    def configure_controllers(self):
        controllers: list[Callable[..., None]] = [
            api_cors,
            app_frontend,
            api_datasets,
            api_captioning,
            api_events,
        ]

        for controller in controllers:
            controller_deps = {arg: self.injector.get(klass) for arg, klass in get_bindings(controller).items()}  # pyright: ignore[reportUnknownVariableType]

            controller(**controller_deps)

    def configure_app(self):
        app = self.injector.get(Flask)
        app.register_blueprint(self.injector.get(ApiBlueprint))
        app.register_blueprint(self.injector.get(AppBlueprint))

    def run(self) -> None:
        """Configure everything and start the server via waitress."""
        self.configure_services()
        self.configure_controllers()
        self.configure_app()

        def _run():
            import waitress

            print(f"yadc web UI starting on http://{self.configuration.http_host}:{self.configuration.http_port}")
            waitress.serve(
                self.app,
                host=self.configuration.http_host,
                port=self.configuration.http_port,
                threads=self.configuration.http_threads,
            )

        t = Thread(target=_run, daemon=False)
        t.start()

        time.sleep(1)

        # If waitress failed (e.g. port in use), exit cleanly
        if not t.is_alive():
            sys.exit(1)

        try:
            t.join()
        except KeyboardInterrupt:
            pass
