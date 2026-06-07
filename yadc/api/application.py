"""``Application(Module)`` — DI container, auto-discovery, and uvicorn bootstrap.

``configure()`` binds ``Configuration`` and ``Quart`` into the injector,
registers the two ``Blueprint`` subclasses (``ApiBlueprint``,
``AppBlueprint``), then scans ``yadc.api.modules`` and
``yadc.api.services`` for ``Service`` subclasses and binds each one with
``inject(...)`` and ``singleton`` scope. ``configure_services()`` then
triggers instantiation, resolves all dependencies, and registers every
service with the ``EventDispatcher``. ``configure_controllers()`` walks
``yadc.api.controllers`` for functions marked with the ``@controller``
decorator and invokes each one with its injector-resolved arguments.

``run()`` ties the whole thing together: configures services and
controllers, optionally prints the banner, dispatches
``StartupEvent``, then registers the Quart blueprints and starts
``uvicorn``. The uvicorn signal handler is monkey-patched so
``ShutdownEvent`` is dispatched *before* uvicorn starts waiting for
connections to close — this lets SSE generators see the shutdown flag
and exit cleanly.
"""

from __future__ import annotations

from pathlib import Path
from types import FrameType
from typing import override

from injector import Binder, Injector, Module, get_bindings, inject, singleton
from quart import Quart
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from yadc.api.modules import EventDispatcher

from . import controllers as controllers_pkg
from . import modules as modules_pkg
from . import services as services_pkg
from .configuration import Configuration
from .controllers.blueprints import ApiBlueprint, AppBlueprint
from .discovery import discover_controllers, discover_services
from .events import ShutdownEvent, StartupEvent
from .modules.service import Service


class Application(Module):
    """Creates the Quart app, wires DI, and starts the server."""

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration
        self.app: Quart = Quart(__name__, static_folder=None)

        self._discovered_services: list[type[Service]] = []
        self.injector: Injector = Injector(self)
        self.services: list[Service] = []

    @override
    def configure(self, binder: Binder):
        binder.bind(Configuration, to=self.configuration)
        binder.bind(Quart, to=self.app)

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

        # NOTE: need BaseObserver instead of Observer since that changes depending on the platform
        binder.bind(BaseObserver, to=lambda: Observer())

        # Auto-discover services and bind them (inject + singleton scope)
        self._discovered_services = discover_services(modules_pkg) + discover_services(services_pkg)
        for service_cls in self._discovered_services:
            binder.bind(service_cls, to=inject(service_cls), scope=singleton)

    def configure_services(self):
        """Retrieve all discovered services (triggers instantiation with deps fully resolved)."""
        for service_cls in self._discovered_services:
            service = self.injector.get(service_cls)
            self.services.append(service)

        event_dispatcher = self.injector.get(EventDispatcher)

        for service in self.services:
            event_dispatcher.register_service(service)

    def configure_controllers(self):
        """Auto-discover and invoke all @controller functions."""
        for ctrl in discover_controllers(controllers_pkg):
            controller_deps = {arg: self.injector.get(klass) for arg, klass in get_bindings(ctrl).items()}
            ctrl(**controller_deps)

    def configure_app(self):
        import asyncio

        app = self.injector.get(Quart)
        event_dispatcher = self.injector.get(EventDispatcher)

        # Allow request bodies up to the configured upload size limit.
        # Our route handlers perform their own size checks, but Quart's
        # built-in MAX_CONTENT_LENGTH guard fires during body parsing
        # (e.g. await request.form) and defaults to 16 MB.
        app.config["MAX_CONTENT_LENGTH"] = self.configuration.max_upload_size_bytes

        @app.before_serving
        async def _capture_loop():  # pyright: ignore[reportUnusedFunction]
            event_dispatcher.set_loop(asyncio.get_running_loop())

        app.register_blueprint(self.injector.get(ApiBlueprint))
        app.register_blueprint(self.injector.get(AppBlueprint))

    def run(self) -> None:
        """Configure everything and start the server via uvicorn."""
        import uvicorn

        self.configure_services()
        self.configure_controllers()

        if self.configuration.banner_enable:
            banner_path = Path(__file__).parent / "banner.txt"
            try:
                print(banner_path.read_text())
            except FileNotFoundError:
                pass

        print(f"yadc web UI starting on http://{self.configuration.http_host}:{self.configuration.http_port}")

        event_dispatcher = self.injector.get(EventDispatcher)
        event_dispatcher.dispatch(StartupEvent())

        # NOTE: the app must be configured before the cors middleware is set up (on startup event)
        self.configure_app()

        config = uvicorn.Config(
            self.app,
            host=self.configuration.http_host,
            port=self.configuration.http_port,
            log_level="info",
            lifespan="on",
            timeout_graceful_shutdown=self.configuration.graceful_shutdown_timeout,
        )
        server = uvicorn.Server(config)

        # Monkey-patch uvicorn's signal handler so we can dispatch ShutdownEvent
        # *before* uvicorn starts waiting for connections to close. This lets SSE
        # generators see the shutdown flag and exit cleanly.
        _original_handle_exit = server.handle_exit

        def _handle_exit(sig: int, frame: FrameType | None) -> None:
            event_dispatcher.dispatch(ShutdownEvent())
            _original_handle_exit(sig, frame)

        server.handle_exit = _handle_exit  # type: ignore[method-assign]
        server.run()
