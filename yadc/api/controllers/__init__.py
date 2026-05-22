from typing import Any, Callable

from injector import inject as _inject


def controller(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a function as an auto-discoverable controller with dependency injection."""
    setattr(fn, "_is_controller", True)
    return _inject(fn)
