from __future__ import annotations

import sys
import time
from threading import Thread
from typing import override

from flask import Flask
from injector import Binder, Injector, Module, get_bindings, inject, singleton  # pyright: ignore[reportUnknownVariableType]

from . import controllers as controllers_pkg
from . import modules as modules_pkg
from . import services as services_pkg
from .configuration import Configuration
from .controllers.blueprints import ApiBlueprint, AppBlueprint
from .discovery import discover_controllers, discover_services
from .modules.cors_middleware import CORSMiddleware
from .modules.service import Service


class Application(Module):
    """Creates the Flask app, wires DI, and starts the server."""

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration
        self.app: Flask = Flask(__name__, static_folder=None)

        self._discovered_services: list[type[Service]] = []
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

        # Auto-discover services and bind them (inject + singleton scope)
        self._discovered_services = discover_services(modules_pkg) + discover_services(services_pkg)
        for service_cls in self._discovered_services:
            binder.bind(service_cls, to=inject(service_cls), scope=singleton)

    def configure_services(self):
        """Retrieve all discovered services (triggers instantiation with deps fully resolved)."""
        for service_cls in self._discovered_services:
            service = self.injector.get(service_cls)
            self.services.append(service)

        # Register CORS middleware on the API blueprint
        cors = self.injector.get(CORSMiddleware)
        cors.register(self.injector.get(ApiBlueprint))

    def configure_controllers(self):
        """Auto-discover and invoke all @controller functions."""
        for ctrl in discover_controllers(controllers_pkg):
            controller_deps = {arg: self.injector.get(klass) for arg, klass in get_bindings(ctrl).items()}  # pyright: ignore[reportUnknownVariableType]
            ctrl(**controller_deps)

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
