"""Package scanning — ``discover_services`` and ``discover_controllers`` for injector DI auto-discovery.

``_walk_package`` recursively iterates all sub-modules of a package
using ``pkgutil.iter_modules`` and ``importlib.import_module`` —
when it encounters a sub-package, it recurses into it. This means
classes defined in leaf modules of nested sub-packages (e.g.
``yadc.api.services.captioning.service.CaptioningService``) are
discovered automatically, not just classes in flat files at the
top level. ``discover_services`` collects every class defined in
those modules that is a strict subclass of :class:`Service` *and*
lives in the same module (``obj.__module__ == mod.__name__``) — the
module check prevents importing a foreign class that happens to
extend ``Service`` through ``__all__`` re-exports. ``discover_controllers``
returns every function that has been flagged with the
``_is_controller`` attribute by the ``@controller`` decorator (see
``yadc.api.controllers``).

Used at startup by :class:`Application` to wire DI bindings and
register HTTP routes without an explicit per-service or per-controller
list.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Generator
from types import ModuleType
from typing import Any, Callable

from .modules.service import Service


def _walk_package(package: ModuleType) -> Generator[ModuleType, None, None]:
    """Recursively yield all sub-modules of a package.

    When a sub-package is encountered, recurses into it to yield
    its leaf modules. The sub-package's own ``__init__`` is not
    yielded separately — the convention is to put classes in leaf
    modules (e.g. ``captioning/service.py``) and re-export from
    ``__init__``, so yielding the leaf modules is enough.
    """
    for module_info in pkgutil.iter_modules(package.__path__):
        mod_name = f"{package.__name__}.{module_info.name}"
        if module_info.ispkg:
            sub_pkg = importlib.import_module(mod_name)
            yield from _walk_package(sub_pkg)
        else:
            yield importlib.import_module(mod_name)


def discover_services(package: ModuleType) -> list[type[Service]]:
    """Scan all modules in a package (recursively) for Service subclasses."""
    result: list[type[Service]] = []
    for mod in _walk_package(package):
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Service) and obj is not Service and obj.__module__ == mod.__name__:
                result.append(obj)
    return result


def discover_controllers(package: ModuleType) -> list[Callable[..., Any]]:
    """Scan all modules in a package (recursively) for @controller-marked functions."""
    result: list[Callable[..., Any]] = []
    for mod in _walk_package(package):
        for _, obj in inspect.getmembers(mod, inspect.isfunction):
            if getattr(obj, "_is_controller", False) and obj.__module__ == mod.__name__:
                result.append(obj)
    return result
