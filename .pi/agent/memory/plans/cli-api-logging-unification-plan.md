---
name: cli-api-logging-unification-plan
description: Unify the two parallel logger systems — `yadc.core.logging` (CLI/captioner) and `yadc.api.modules.logging_factory.LoggingFactory` (API) — into one. After this, both sides call into the same factory; the CLI installs a `ClickHandler`, the API installs a structured `StreamHandler`. Captioner code stops being a third class of code that uses a different logger than the API services that wrap it.
status: Proposed
last_history: 0
---

# Unify CLI + API logging

## Problem

The codebase has three logger call sites, each on a different system:

| Caller | Imports | Returns | `trace()`? | Handlers from |
|---|---|---|---|---|
| CLI commands (`yadc/cli_*.py`) | `from yadc.core import logging` | `_logger` wrapper (has `trace()`) | yes | CLI installs `ClickHandler` at startup |
| Captioner code (`yadc/captioners/api/utils/*.py`, `yadc/core/captioning/runner.py`) | `from yadc.core import logging` | `_logger` wrapper (has `trace()`) | yes | Inherits whatever the host installed; no host = no handler |
| API services (`yadc/api/services/*.py`) | `LoggingFactory.get_logger` (DI) | raw `logging.Logger` | no | `LoggingFactory.__init__` calls `logging.basicConfig` |

The two systems are structurally different:

- `yadc.core.logging.get_logger` returns a `yadc.core.logging._logger` (a thin wrapper around `logging.Logger` that adds `trace()`). The wrapper exposes `trace/debug/info/warning/error/addHandler/setLevel` — but **not** the rest of `logging.Logger`'s surface (`isEnabledFor`, `propagate`, `handlers`, etc.). Callers that need any of those have to reach through `.handlers` or `.setLevel`, which the wrapper does expose, but `.addHandler` works through a `.handlers` indirection (`logger.handlers = logger._logger.handlers`).
- `LoggingFactory.get_logger` returns a raw `logging.Logger`. Different cache, different level semantics (`int` not `str`), no `trace()`.

When the API runs the captioner, both systems are active at once: `yadc.core.logging` (used by captioner code) and `LoggingFactory` (used by API services). They share the global `logging.Logger` registry (same name → same instance), but each factory has its own per-name cache and installs handlers/levels independently. The `LoggingFactory` only attaches a handler via `logging.basicConfig` in `__init__`; per-name loggers in its cache get no handler. So captioner code in the API process gets `propagate=True` from the root logger and ends up on `basicConfig`'s `StreamHandler`, but at the level set by `yadc.core.logging.set_level(...)` (the default `"INFO"` since the API never calls it).

The asymmetry is annoying because captioner code lives *under* API services (the API wraps the captioner in `AsyncCaptionJob`), but the captioner's own log lines go through a different factory than the API's own log lines.

## Goal

One logger factory, two handlers:

- `yadc/core/logging.py` is the canonical logger module. It owns the per-name cache, the default level, the default handler, and the `trace()` extension. Callers get the same thing regardless of whether they're running under the CLI or the API.
- `yadc/cli_logging.py` `ClickHandler` stays in place — it's a CLI concern. The CLI installs it as the default handler at startup.
- `yadc/api/modules/logging_factory.py` becomes a thin DI-managed wrapper around the core factory. The API installs a `StreamHandler` with the structured format from `logging-format` as the default handler.
- API services that today call `LoggingFactory.get_logger(name)` either keep that signature (the wrapper delegates to the core factory) or switch to the core factory directly. Either way they get the same `Logger` interface as the captioner code.
- The `trace()` extension is available everywhere — including API services, once the wrapper returns the core's `PythonLogger` instance.

## Design

### New shape of `yadc/core/logging.py`

```python
# yadc/core/logging.py
import logging
from typing import Any, Protocol, runtime_checkable

TRACE_LEVEL = 5

@runtime_checkable
class Logger(Protocol):
    """Minimum surface every yadc logger exposes. Subset of stdlib logging.Logger."""
    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def addHandler(self, handler: logging.Handler) -> None: ...
    def setLevel(self, level: int | str) -> None: ...
    def isEnabledFor(self, level: int) -> bool: ...


class PythonLogger:
    """Concrete Logger that wraps a stdlib logging.Logger and adds trace()."""
    def __init__(self, logger: logging.Logger): self._logger = logger
    def trace(self, msg, *args, **kw): self._logger.log(TRACE_LEVEL, msg, *args, **kw)
    # ... debug/info/warning/error/exception forward via self._logger.log(level, ...)
    def addHandler(self, h): self._logger.addHandler(h)
    def setLevel(self, level): self._logger.setLevel(level)
    def isEnabledFor(self, level): return self._logger.isEnabledFor(level)


class LoggingFactory:
    """Owns the per-name cache, default handler, default level. The default
    factory lazily created on first get_logger() call is shared across the
    process (singleton)."""
    def __init__(self, default_level: int | str = logging.INFO,
                 default_handler: logging.Handler | None = None): ...
    def get_logger(self, name: str) -> Logger: ...
    def set_level(self, level: int | str) -> None: ...
    def set_handler(self, handler: logging.Handler) -> None: ...


_default_factory: LoggingFactory | None = None

def get_logger(name: str) -> Logger:
    """Module-level convenience — uses the default factory."""
    global _default_factory
    if _default_factory is None:
        _default_factory = LoggingFactory()
    return _default_factory.get_logger(name)

def set_level(level: int | str) -> None: ...
def set_handler(handler: logging.Handler) -> None: ...
```

The `yadc.core.logging.get_logger(name)` call site is preserved (decision #3 below). The change is that `get_logger` now returns a `PythonLogger` (with the same `trace()`/`info()`/etc. surface) instead of the current `_logger` wrapper class — but the call site is identical.

### `yadc/api/modules/logging_factory.py` becomes a thin DI wrapper

```python
# yadc/api/modules/logging_factory.py
import logging
from yadc.core.logging import LoggingFactory, Logger
from .service import Service

class LoggingFactory(Service):
    def __init__(self, configuration: Configuration) -> None:
        self._core = LoggingFactory(
            default_level=configuration.logging_default_level,
            default_handler=logging.StreamHandler(),  # or a structured formatter
        )

    def get_logger(self, name: str) -> Logger:
        return self._core.get_logger(name)

    def set_level(self, level: int) -> None:
        self._core.set_level(level)
```

The public surface of `LoggingFactory` is preserved — controllers and services keep calling `logging_factory.get_logger(name)` — but it now returns a `Logger` (Protocol) which is backed by a `PythonLogger` with `trace()`.

### `ClickHandler` stays put

`yadc/cli_logging.py` is unchanged. The CLI calls `yadc.core.logging.set_handler(ClickHandler())` at startup. The CLI's `cli_common.py --log-level` option continues to call `yadc.core.logging.set_level(...)`.

### Captioner code unchanged at the call site

`yadc/captioners/api/utils/*.py` keep `from yadc.core import logging; _logger = logging.get_logger(__name__)`. The returned object has the same `trace/debug/info/...` surface as today, so no call site changes. In the API process, the per-name cache is now owned by the unified factory, so the API's default handler / level apply automatically.

## Phases

### Phase 1 — Refactor `yadc/core/logging.py`

Add the `Logger` Protocol, `PythonLogger` concrete, `LoggingFactory` class, and `Logger` re-export. Keep the module-level `get_logger` / `set_level` / `set_handler` working as the default-factory facade (decision #3). `PythonLogger` exposes the same surface the current `_logger` does (so the ~17 call sites in `cli_*.py` and `yadc/captioners/api/utils/*.py` keep working without edits).

If decision #4 (backward-compat re-exports) is "preserve", also re-export `TRACE_LEVEL` (already there), `get_logger`, `set_level`, `set_handler` from the module. The change at the call site is zero.

If decision #4 is "break", update all call sites in this phase.

**Files touched:** `yadc/core/logging.py`, `tests/core/test_logging.py` (new).

### Phase 2 — Rewrite `yadc/api/modules/logging_factory.py` as a wrapper

`LoggingFactory(Service)` becomes a thin DI wrapper around `yadc.core.logging.LoggingFactory`. Preserves the constructor (`configuration: Configuration`), the public `get_logger(name) -> Logger` and `set_level(level) -> None` methods. The default handler is a `StreamHandler` with the API's structured formatter (per `logging-format` memory).

API services that do `self._logging_factory.get_logger(name)` get back the core's `PythonLogger` instance. Existing service code that only uses `info/debug/warning/error` keeps working unchanged. Service code that wants `trace()` now has it for free.

The `logging.basicConfig` call in the current `__init__` is removed — the core factory owns handler installation. The existing root logger behavior is preserved (handlers propagate to the root by default; if a service wants non-propagating, it can call `setLevel` per logger).

**Files touched:** `yadc/api/modules/logging_factory.py`.

### Phase 3 — Update API services to drop redundant `addHandler` calls (if any)

A quick scan is needed. If any service currently does `logger.addHandler(...)` after getting the logger from `LoggingFactory`, the calls move to the factory's `set_handler(...)` at startup. If not, this phase is a no-op and is skipped.

**Files touched:** (TBD by the scan) — likely zero in the common case.

### Phase 4 — Cleanup + docs

- Remove the `yadc.core.logging._logger` wrapper class (replaced by `PythonLogger`). Update any remaining references.
- Update `docs/logging-format` and any architecture doc that references the old shape.
- Add `docs/logging-architecture.md` (new) covering the unified factory, the two default handlers (CLI's `ClickHandler`, API's `StreamHandler`), the `Logger` Protocol, and the `trace()` extension.
- Run lint + type-check + full test suite.

**Files touched:** `yadc/core/logging.py`, `docs/logging-architecture.md` (new), possibly `docs/logging-format.md`.

## Open questions

These are the decisions the previous plan deferred. The current state of the codebase (after the captioning shared-runner refactor) gives us enough context to resolve them — but the user should confirm before implementation starts.

1. **Interface style: Protocol vs ABC vs concrete class only.** Recommendation: **Protocol** (`@runtime_checkable`). Duck-compatible with the stdlib `logging.Logger` surface, no inheritance tax, easy to test with `MagicMock(spec=Logger)`. The Protocol is structural — if `PythonLogger` accidentally drops a method, the `get_logger` return type still type-checks because it's `Logger`, not `PythonLogger`. **Decision pending.**

2. **API `LoggingFactory` shape: thin DI wrapper vs delete and inject core factory directly.** Recommendation: **thin DI wrapper**. Keeps existing service constructor signatures unchanged (zero diff in `services/*.py`), and the wrapper is ~10 lines. Deleting it and switching services to `from yadc.core.logging import LoggingFactory` would require either a custom injector binding or a constructor signature change in every service — bigger diff for no real win. **Decision pending.**

3. **Module-level default factory: preserve the `yadc.core.logging.get_logger` facade vs require explicit factory.** Recommendation: **preserve the facade**. Matches the current call-site shape (`from yadc.core import logging; _logger = logging.get_logger(__name__)`) and keeps the diff minimal. The facade delegates to a process-singleton default factory created lazily. **Decision pending.**

4. **Backward-compat re-exports in `yadc.core.logging`.** Recommendation: **preserve all current module-level names** (`get_logger`, `set_level`, `set_handler`, `TRACE_LEVEL`) plus add the new `Logger`, `PythonLogger`, `LoggingFactory`. Zero call-site churn. **Decision pending.**

5. **Where `ClickHandler` lives.** Recommendation: **stays in `yadc/cli_logging.py`**. It's a CLI-only concern (uses `click.secho`). The current location is correct — moving it to `yadc/core/logging/handlers/click.py` would create a `cli` import in `core`, which violates the existing layering. **Decision pending.**

## Risks

- **Silent logging level change in the API.** The current `LoggingFactory.__init__` calls `logging.basicConfig(level=logging.ERROR, format=...)`. The new shape lets the factory install a `StreamHandler` with a structured formatter (per `logging-format` memory) and the configured `logging_default_level` (currently `logging.INFO`). The level might be different from the current implicit behavior — the API currently doesn't expose the configured level to the per-name loggers it returns (it only sets `log_level` on the factory, not on each logger). Mitigated by running the API integration tests after the refactor.
- **`trace()` in API service logs.** Services that adopt `trace()` will route through the API's `StreamHandler`. The handler's level filter will gate the call (per `PythonLogger.trace` → `self._logger.log(TRACE_LEVEL, ...)`). Confirmed safe.
- **Logger name collisions.** Both the CLI and the API use `getLogger(name)` (stdlib). The current code's per-name cache and the stdlib's per-name registry can drift. Mitigated by the new design using one cache and the stdlib registry.
- **StreamHandler contention with the CLI's `ClickHandler`.** The two handlers are in different processes (CLI vs API), so no contention. Confirmed safe.

## Testing

- `tests/core/test_logging.py` (new) — unit tests for `PythonLogger` (forwarding to the wrapped `logging.Logger`, `trace()` uses `TRACE_LEVEL`, `setLevel` / `addHandler` / `isEnabledFor` forward correctly) and `LoggingFactory` (handler swap, level change, per-name caching, `get_logger` returns `Logger`-compatible instance, default factory singleton).
- Existing CLI tests pass unchanged — call sites are preserved.
- Existing API tests pass unchanged — `LoggingFactory` public surface is preserved.
- Manual smoke test: run `yadc caption ...` (CLI handler style), then run the web UI and trigger an action that exercises a captioner log line (API handler style); verify both routes land in the right destination with the right level/format.

## Out of scope

- Replacing the CLI's `click.echo` / `click.secho` calls in the interactive action menu with logger calls. They emit to the user's terminal, not to logs, and the user is the audience.
- Adding structured fields (`[key=value, ...]`) to the API handler's formatter — that's a `logging-format` concern, not a logger factory concern. The handler accepts a `Formatter` from the caller; the actual format string can be tuned separately.
- Per-module log levels. Stdlib supports it via `logging.Logger.setLevel`; we don't need to wrap it.
- Async-aware handlers (e.g. `aiomultiprocess`, log shipping to a remote service). Out of scope until needed.

## Foundation

This plan depends on the shared captioning runner plan (`plans/archive/cli-api-captioning-shared-runner-plan.md`) — specifically, the runner in `yadc/core/captioning/runner.py` uses `yadc.core.logging.get_logger(__name__)`. That surface is preserved by the "preserve facade" decision in question #3, so the runner is unaffected by this refactor. The decision to defer logging unification until *after* the runner landed was deliberate: the runner's logging surface is now visible, and the open questions in this plan can be resolved with that context.

## Decision log

- **2026-06-04 — Proposed.** Extracted from the shared-runner plan (`plans/archive/cli-api-captioning-shared-runner-plan.md` Phase 6). The runner refactor is complete; the logging refactor is now its own plan with the same level of detail and the same decision-deferral policy.
