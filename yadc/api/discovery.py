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
