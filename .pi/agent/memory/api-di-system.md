---
name: api-di-system
description: The web UI backend DI system — auto-discovery of services and controllers, injector binding lifecycle, and how to add new ones.
---

# API DI System (Auto-Discovery)

## Overview

The web UI backend (`yadc/api/`) uses the `injector` library with automatic package scanning. There are **no hardcoded service or controller lists** — adding a new service or controller requires editing only the new file.

## Two Discovery Targets

| Kind | Base/Marker | Package scanned | Discovery method |
|------|------------|-----------------|-----------------|
| **Service** | `Service` base class (`modules/service.py`) | `yadc.api.modules` | `issubclass(obj, Service) and obj is not Service` |
| **Controller** | `_is_controller` attribute (set by `@controller`) | `yadc.api.controllers` | `getattr(obj, "_is_controller", False)` |

## Binding Lifecycle

1. **`Application.configure(binder)`** — called by `Injector.__init__`:
   - Binds `Configuration`, `Flask`, `ApiBlueprint`, `AppBlueprint` explicitly
   - Calls `discover_services(modules_pkg)` to find all `Service` subclasses
   - Binds each with `binder.bind(cls, to=inject(cls), scope=singleton)` — no per-class `@inject`/`@singleton` decorators needed

2. **`Application.configure_services()`** — called in `run()`:
   - Calls `injector.get()` for every discovered service — triggers instantiation
   - Because all types were bound first, inter-service dependencies are fully resolved regardless of discovery order
   - Registers `CORSMiddleware` on `ApiBlueprint` (infrastructure, not a controller)

3. **`Application.configure_controllers()`** — called in `run()`:
   - Calls `discover_controllers(controllers_pkg)` to find all `@controller` functions
   - Uses `get_bindings(fn)` to resolve parameter types, then `injector.get()` for each
   - Calls each controller function — they register Flask routes as side effects

4. **`Application.configure_app()`** — registers blueprints on the Flask app

## How to Add a New Service

1. Create a class extending `Service` in `yadc/api/modules/`
2. Constructor parameters are auto-injected by type — no decorators needed
3. Update `modules/__init__.py` re-exports
4. That's it — discovery finds it, binder registers it, injector resolves deps

```python
# yadc/api/modules/my_service.py
from .logging_factory import LoggingFactory
from .service import Service

class MyService(Service):
    def __init__(self, logging: LoggingFactory):
        self._logger = logging.get_logger(__name__)
```

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

## CORS

CORS is handled by `CORSMiddleware` (`modules/cors_middleware.py`) — a `Service`, not a controller. It registers an `after_request` handler on `ApiBlueprint` during `configure_services()`. Configuration is via `Configuration.api_cors_*` fields.

## Events System

- `events.py` — `Event` base class (has `TYPE: ClassVar[str]`), `PingEvent`, `CaptioningStatusEvent`
- `EventDispatcher` — `subscribe(event_cls, handler)`, `dispatch(event)`, `register_service(service)` (auto-scans for `@event_handler` methods), `@event_handler` decorator
- `SSEEvents` — Condition-based queue, `push(event)` / `receive(event_cls)` generator, auto-ping via `JobScheduler`
- Controllers receive `SSEEvents` as a dependency and use `receive()` for SSE endpoints
