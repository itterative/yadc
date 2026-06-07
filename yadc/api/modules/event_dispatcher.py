"""``EventDispatcher`` — subscribe/dispatch and ``@event_handler`` decorator for in-process pub/sub."""

import asyncio
import inspect
from inspect import signature
from logging import Logger
from threading import Lock
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, cast, get_type_hints

from ..events import Event
from .logging_factory import LoggingFactory
from .service import Service

Handler = Callable[[Event], Any]
E = TypeVar("E", bound=Event)
P = ParamSpec("P")
R = TypeVar("R")


def event_handler(
    event_cls: type[E],
) -> Callable[[Callable[Concatenate[Any, E, P], R]], Callable[Concatenate[Any, E, P], R]]:
    """Decorator to mark a method as an event handler for a specific event type.

    This decorator provides type safety by ensuring the handler's second parameter
    (after self for instance methods) matches the event type and preserves the
    original function's type signature.

    Args:
        event_cls: The event class to handle (must be a subclass of Event)

    Returns:
        A decorator that preserves the original function's type signature
    """

    def decorator(
        func: Callable[Concatenate[Any, E, P], R],
    ) -> Callable[Concatenate[Any, E, P], R]:
        # Validate parameter count
        sig = signature(func)
        params = list(sig.parameters.values())

        if len(params) < 2:
            raise TypeError(f"Event handler must have at least two parameters (self and event), got {len(params)} parameters")

        # Resolve annotations (handles `from __future__ import annotations` string-ification)
        event_param_name = params[1].name
        try:
            hints = get_type_hints(func)
            ann = hints.get(event_param_name)
        except Exception:
            ann = None

        if ann is not None:
            from typing import get_origin

            origin = get_origin(ann) or ann

            if origin is not event_cls and (origin != Any and not issubclass(origin, event_cls)):
                raise TypeError(f"Event handler's second parameter must be of type {event_cls.__name__}, got {ann}")

        # Add metadata to the function
        setattr(func, "_event_handler", True)
        setattr(func, "_event_class", event_cls)
        setattr(func, "_event_type", event_cls.TYPE)

        return func

    return decorator


class EventDispatcher(Service):
    def __init__(
        self,
        logging: LoggingFactory,
    ):
        self._logger: Logger = logging.get_logger(__name__)
        self._lock: Lock = Lock()
        self._subscribers: dict[str, list[Handler]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture a reference to the running event loop for async handler bridging."""
        self._loop = loop

    def subscribe[T: Event](self, event_cls: type[T], handler: Callable[[T], None]):
        event_type = event_cls.TYPE

        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            self._subscribers[event_type].append(cast(Handler, handler))
            self._logger.debug(
                "Subscribed to event. [type=%s, total_subscribers=%d]",
                event_type,
                len(self._subscribers[event_type]),
            )

    def register_service(self, service: Service):
        """Register all event handlers from a service automatically."""
        service_class = service.__class__
        service_name = service_class.__name__

        # Scan for event handler methods
        for attr_name in dir(service_class):
            attr = getattr(service_class, attr_name)
            if not callable(attr) or not hasattr(attr, "_event_handler"):
                continue

            # Get the event class or type string
            if hasattr(attr, "_event_class"):
                event_cls = getattr(attr, "_event_class")
            else:
                event_type = getattr(attr, "_event_type")
                event_cls = self._get_event_class_by_type(event_type)

            # Get the bound method
            handler_method = getattr(service, attr_name)

            self._logger.debug(
                "Registering event handler for %s in service %s",
                event_cls.TYPE,
                service_name,
            )

            self.subscribe(event_cls, handler_method)

    # This method is no longer needed since we only support class-based event registration
    # but kept for backward compatibility in case string-based event types are used
    # in the metadata of old handlers
    def _get_event_class_by_type(self, event_type: str) -> type[Event]:
        """Get event class by its type string (for backward compatibility)."""
        from .. import events as events_module

        for attr_name in dir(events_module):
            attr = getattr(events_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Event) and hasattr(attr, "TYPE") and attr.TYPE == event_type:
                return attr
        raise ValueError(f"No event class found with type '{event_type}'")

    def dispatch(self, event: Event) -> None:
        """Dispatch an event to all registered handlers.

        Sync handlers are called directly. Async handlers are scheduled on the
        captured event loop via ``create_task`` (same thread) or
        ``run_coroutine_threadsafe`` (background thread).
        """
        event_type = event.__class__.TYPE

        with self._lock:
            handlers = self._subscribers.get(event_type, [])

        for handler in list(handlers):
            try:
                if inspect.iscoroutinefunction(handler):
                    self._dispatch_async(handler, event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(
                    "Error handling event. [type=%s, handler=%s, error=%s]",
                    event_type,
                    handler.__name__,
                    str(e),
                )
                raise

    def _dispatch_async(self, handler: Callable[[Event], Any], event: Event) -> None:
        """Schedule an async handler on the event loop."""
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
            except RuntimeError:
                pass

        if loop is None or not loop.is_running():
            self._logger.warning("Event loop not available for async handler %s, skipping", handler.__name__)
            return

        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is loop:
                asyncio.create_task(handler(event))
                return
        except RuntimeError:
            pass

        asyncio.run_coroutine_threadsafe(handler(event), loop)
