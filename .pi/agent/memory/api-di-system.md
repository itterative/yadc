---
name: api-di-system
description: The web UI backend DI system — auto-discovery of services and controllers, injector binding lifecycle, and how to add new ones.
---

# API DI System (Auto-Discovery)

## Overview

The web UI backend (`yadc/api/`) uses the `injector` library with automatic package scanning. There are **no hardcoded service or controller lists** — adding a new service or controller requires editing only the new file.

## Two Discovery Targets

| Kind | Base/Marker | Packages scanned | Discovery method |
|------|------------|------------------|-----------------|
| **Service** | `Service` base class (`modules/service.py`) | `yadc.api.modules` **+ `yadc.api.services`** | `issubclass(obj, Service) and obj is not Service` |
| **Controller** | `_is_controller` attribute (set by `@controller`) | `yadc.api.controllers` | `getattr(obj, "_is_controller", False)` |

## Binding Lifecycle

1. **`Application.configure(binder)`** — called by `Injector.__init__`:
   - Binds `Configuration`, `Flask`, `ApiBlueprint`, `AppBlueprint` explicitly
   - Calls `discover_services(modules_pkg) + discover_services(services_pkg)` to find all `Service` subclasses in both packages
   - Binds each with `binder.bind(cls, to=inject(cls), scope=singleton)` — no per-class `@inject`/`@singleton` decorators needed

2. **`Application.configure_services()`** — called in `run()`:
   - Calls `injector.get()` for every discovered service — triggers instantiation
   - Because all types were bound first, inter-service dependencies are fully resolved regardless of discovery order
   - Calls `EventDispatcher.register_service()` for each service — auto-discovers `@event_handler` methods

3. **`Application.configure_controllers()`** — called in `run()`:
   - Calls `discover_controllers(controllers_pkg)` to find all `@controller` functions
   - Uses `get_bindings(fn)` to resolve parameter types, then `injector.get()` for each
   - Calls each controller function — they register Flask routes as side effects

4. **`Application.configure_app()`** — registers blueprints on the Flask app

## How to Add a New Service

1. Create a class extending `Service` in `yadc/api/modules/` or `yadc/api/services/`
2. Constructor parameters are auto-injected by type — no decorators needed
3. Update the package's `__init__.py` re-exports
4. That's it — discovery finds it, binder registers it, injector resolves deps

```python
# yadc/api/services/my_service.py
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from logging import Logger

class MyService(Service):
    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory):
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
```

**Note**: Use `from logging import Logger` (not `import logging` + `logging.Logger`) when the parameter is also named `logging` to avoid type expression errors in basedpyright.

## How to Add a New Controller

1. Create a function in `yadc/api/controllers/`
2. Decorate with `@controller` (from `controllers/__init__.py`) — this applies `@inject` and sets `_is_controller`
3. Parameters are resolved by type via `get_bindings()`
4. Register Flask routes on the blueprint parameter as side effects

```python
# yadc/api/controllers/api_my_feature.py
from . import controller
from .blueprints import ApiBlueprint
from ..modules.logging_factory import LoggingFactory

@controller
def api_my_feature(app: ApiBlueprint, logging: LoggingFactory):
    @app.get("/my-feature")
    def list_features():
        return jsonify([])
```

## Key Files

| File | Role |
|------|------|
| `yadc/api/discovery.py` | `discover_services()`, `discover_controllers()`, `_walk_package()` — `pkgutil`/`importlib` scanning |
| `yadc/api/controllers/__init__.py` | `@controller` decorator — `setattr(fn, "_is_controller", True)` + `inject(fn)` |
| `yadc/api/modules/service.py` | `Service` — empty base class, marker for discovery |
| `yadc/api/application.py` | `Application(Module)` — wires everything together |
| `yadc/api/configuration.py` | `@dataclass` config bound into the injector |

## Existing Services

| Service | Package | Purpose |
|---------|---------|--------|
| `LoggingFactory` | `modules/` | Per-module named loggers |
| `DBMigrations` | `modules/` | Step-based SQLite migration runner |
| `DBConnectionFactory` | `modules/` | SQLite WAL connections, background init |
| `CORSMiddleware` | `modules/` | Origin-based CORS on ApiBlueprint |
| `EventDispatcher` | `modules/` | Subscribe/dispatch events, `@event_handler` |
| `JobScheduler` | `modules/` | Daemon threads for periodic jobs |
| `SSEEvents` | `modules/` | Condition-based SSE queue, auto-ping, monotonic event IDs, ring buffer history for `Last-Event-ID` resumption |
| `DatasetWatcherService` | `modules/` | watchdog-based filesystem watcher for dataset dirs, debounced `DatasetChangedEvent` emission |
| `SettingsService` | `services/` | KV store over `settings` table (JSON values) |
| `DatasetService` | `services/` | TOML-based datasets, filesystem scanning, SQLite indexing, paginated image queries, caption read/write, import/create/delete/rescan. Injects `DatasetWatcherService` + `Configuration` |
| `CaptioningService` | `services/` | Background captioning jobs (start/stop/status), env/config/template resolution, `CaptioningStatusEvent` emission via `EventDispatcher` |

## Startup Event

A `StartupEvent` is dispatched after all services are instantiated but before `waitress.serve()`. Services that need to start background work (e.g. `DatasetWatcherService` starts its observer thread, `CORSMiddleware` registers its `after_request` handler) do so via `@event_handler(StartupEvent)`. The event is dispatched after `configure_app()` (blueprints registered) so that handlers can interact with the fully-configured Flask app.

## CORS

CORS is handled by `CORSMiddleware` (`modules/cors_middleware.py`) — a `Service`, not a controller. It registers an `after_request` handler on `ApiBlueprint` on `StartupEvent`. Configuration is via `Configuration.api_cors_*` fields.

## Events System

- `events.py` — `Event` base class (has `TYPE: ClassVar[str]`), `StartupEvent`, `PingEvent`, `CaptioningStatusEvent`, `DatasetChangedEvent`, `ResumptionFailedEvent`
- `EventDispatcher` — `subscribe(event_cls, handler)`, `dispatch(event)`, `register_service(service)` (auto-scans for `@event_handler` methods), `@event_handler` decorator
- `@event_handler` uses `typing.get_type_hints()` to resolve annotations — needed because `from __future__ import annotations` stringifies them, causing `issubclass()` to fail on plain `inspect.signature()` annotations
- `SSEEvents` — Condition-based queue, `push(event)` / `receive(event_cls, last_event_id=None)` generator yielding `(event_id, event)` tuples, auto-ping via `JobScheduler`. Assigns monotonic IDs to non-ping events, maintains a configurable ring buffer (`sse_event_history_size`, default 128) for `Last-Event-ID` resumption. On reconnect, replays missed events from history; if the requested ID is too old, yields a `ResumptionFailedEvent`. Handles `CaptioningStatusEvent` and `DatasetChangedEvent`.
- `DatasetWatcherService` — Uses `watchdog.Observer` to watch dataset image directories for filesystem changes (images + sidecars). Debounces events per-dataset (configurable via `Configuration.watcher_debounce_seconds`, default 1s). Dispatches `DatasetChangedEvent` via `EventDispatcher`.
- Controllers receive `SSEEvents` as a dependency and use `receive()` for SSE endpoints
