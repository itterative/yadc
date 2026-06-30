"""Tag-catalog service — DI-managed owner of the parsed catalog lifecycle.

Holds the loaded :class:`~yadc.api.services.tag_suggestions.TagCatalog`,
guards its (once-per-process) load with a lock, and exposes the
autocomplete matcher to the controller. State lives in a ``Service``
(rather than module globals) so it has a logger, runs the startup preload
as an ``@event_handler(StartupEvent)``, and matches the pattern of the
other services (see ``docs/api-di-system.md``).

Loading is fire-and-forget at startup so the first ``/tagging/suggest``
keystroke is warm; boot isn't blocked on it. Concurrent requests before
the load finishes share the single in-flight load via :attr:`_lock`; a
failed download is logged and retried lazily on the next request.

The active :class:`CatalogVariant` resolves in layers (persisted user
selection → ``Configuration.tagger_suggestion_variant`` →
:data:`DEFAULT_VARIANT`), mirroring the ``effective_active_tagger``
pattern. :meth:`set_variant` persists a new selection and drops the
cached catalog so the next request loads the new variant; switching
downloads the variant on demand if it isn't already cached.
"""

from __future__ import annotations

import asyncio
import time
from logging import Logger

from yadc.api.configuration import Configuration
from yadc.api.events import StartupEvent
from yadc.api.modules.event_dispatcher import event_handler
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.api.services.settings import SettingsService

from .catalog import (
    CatalogVariant,
    TagCatalog,
    load_catalog,
    resolve_variant,
)
from .suggestions import suggest

# Settings key for the persisted variant selection (mirrors the
# ``tagger.active_model`` namespace used by the tagger-swap flow).
_VARIANT_SETTING_KEY = "tagger.suggestion_variant"


class TagSuggestionsService(Service):
    """Owns the parsed tag catalog and delegates matching to the pure
    :func:`suggest` matcher."""

    def __init__(
        self,
        logging: LoggingFactory,
        settings: SettingsService,
        configuration: Configuration,
    ) -> None:
        self._logger: Logger = logging.get_logger(__name__)
        self._settings: SettingsService = settings
        self._configuration: Configuration = configuration
        self._catalog: TagCatalog | None = None
        # Guards the load so a stampede of first-request coroutines
        # share one in-flight parse instead of each triggering one.
        self._lock: asyncio.Lock = asyncio.Lock()
        self._preload_task: asyncio.Task[None] | None = None

    @property
    def variant(self) -> CatalogVariant:
        """The active variant — persisted user selection wins, else the
        ``Configuration`` default, else :data:`DEFAULT_VARIANT`.

        Read on every ``get_catalog`` so a config change takes effect on
        the next request when no user selection has been persisted.
        """
        persisted = self._settings.get(_VARIANT_SETTING_KEY)
        if isinstance(persisted, str):
            resolved = resolve_variant(persisted)
            if persisted.strip().lower() == resolved.value:
                return resolved
        return resolve_variant(self._configuration.tagger_suggestion_variant)

    async def get_catalog(self) -> TagCatalog:
        """Return the parsed catalog, loading it on first access.

        Concurrent first-callers share the in-flight load via :attr:`_lock`.
        """
        if self._catalog is not None:
            return self._catalog
        async with self._lock:
            # Re-check after acquiring — a concurrent caller (or the startup
            # preload) may have finished while we waited.
            if self._catalog is not None:
                return self._catalog
            self._catalog = await self._load()
            return self._catalog

    async def suggest(self, query: str, *, limit: int = 10) -> list[tuple[str, str]]:
        """Return up to *limit* ``(tag, category)`` pairs for *query*.

        Thin DI wrapper: resolves the catalog (loading if necessary) and
        delegates to the pure matcher.
        """
        catalog = await self.get_catalog()
        return await suggest(query, catalog, limit=limit)

    async def set_variant(self, variant: CatalogVariant) -> CatalogVariant:
        """Persist *variant* as the active selection and drop the cached catalog.

        The next request (or the background warm started here) loads the
        new variant, downloading it on demand if it isn't cached yet.
        Returns the resolved variant — always equal to *variant* once the
        enum coercion has already happened at the controller boundary.
        """
        self._settings.set(_VARIANT_SETTING_KEY, variant.value)
        # Drop the cached bundle so ``get_catalog`` reloads with the new
        # variant. Cancelling an in-flight preload avoids a race where it
        # would re-cache the old variant right after we cleared it.
        self._catalog = None
        if self._preload_task is not None and not self._preload_task.done():
            self._preload_task.cancel()
            try:
                await self._preload_task
            except (asyncio.CancelledError, Exception):
                pass
        self._preload_task = asyncio.create_task(self._warm())
        self._logger.info("Switched tag suggestion variant [variant=%s]", variant.value)
        return variant

    # -- lifecycle -----------------------------------------------------------

    @event_handler(StartupEvent)
    async def on_startup(self, event: StartupEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        """Kick off a background catalog load so the first autocomplete request
        is fast."""
        if self._catalog is not None or self._preload_task is not None:
            return
        self._preload_task = asyncio.create_task(self._warm())

    async def _warm(self) -> None:
        try:
            await self.get_catalog()
        except Exception:
            # Already logged in _load; swallow so the background task
            # isn't flagged as an unhandled-exception warning.
            pass

    async def _load(self) -> TagCatalog:
        variant = self.variant
        self._logger.info("Loading tag catalog [variant=%s]", variant.value)
        t0 = time.perf_counter()
        try:
            catalog = await load_catalog(variant)
        except Exception:
            # Surface but let it propagate so get_catalog doesn't cache a
            # broken state — the next request retries.
            self._logger.exception("Failed to load tag catalog [variant=%s]", variant.value)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._logger.info(
            "Tag catalog loaded in %.0fms [variant=%s, entries=%d]",
            elapsed_ms,
            variant.value,
            len(catalog.entries),
        )
        return catalog
