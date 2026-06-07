"""Package scanning — ``discover_services`` and ``discover_controllers`` for injector DI auto-discovery.

``_walk_package`` iterates direct sub-modules of a package using
``pkgutil.iter_modules`` and ``importlib.import_module`` (sub-packages
are walked recursively by the caller if needed). ``discover_services``
collects every class defined in those modules that is a strict
subclass of :class:`Service` *and* lives in the same module
(``obj.__module__ == mod.__name__``) — the module check prevents
importing a foreign class that happens to extend ``Service`` through
``__all__`` re-exports. ``discover_controllers`` returns every function
that has been flagged with the ``_is_controller`` attribute by the
``@controller`` decorator (see ``yadc.api.controllers``).

Used at startup by :class:`Application` to wire DI bindings and
register HTTP routes without an explicit per-service or per-controller
list.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any, Callable

from .modules.service import Service


def _walk_package(package: ModuleType):
    """Yield all modules in a package."""
    for module_info in pkgutil.iter_modules(package.__path__):
        yield importlib.import_module(f"{package.__name__}.{module_info.name}")


def discover_services(package: ModuleType) -> list[type[Service]]:
    """Scan all modules in a package for Service subclasses."""
    result: list[type[Service]] = []
    for mod in _walk_package(package):
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Service) and obj is not Service and obj.__module__ == mod.__name__:
                result.append(obj)
    return result


def discover_controllers(package: ModuleType) -> list[Callable[..., Any]]:
    """Scan all modules in a package for @controller-marked functions."""
    result: list[Callable[..., Any]] = []
    for mod in _walk_package(package):
        for _, obj in inspect.getmembers(mod, inspect.isfunction):
            if getattr(obj, "_is_controller", False) and obj.__module__ == mod.__name__:
                result.append(obj)
    return result
