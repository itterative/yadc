"""Global tag highlights — starred / desired / undesired tiers plus category overrides.

Persists under the existing ``settings`` KV table (key
``tagger.tag_highlights``), alongside the other tagger-global rows
(``tagger.active_model``, ``tagger.suggestion_variant``). Reuses
:class:`SettingsService` for the storage layer so the new
highlights persistence costs nothing in schema or migration weight
— a fresh install has no ``tagger.tag_highlights`` row and the
service returns the hardcoded defaults.

Shape:

- ``starred`` / ``desired`` / ``undesired``: three mutually-exclusive
  tag lists (one tag per tier).
- ``category_overrides``: starred tag → forced section
  (``"character"`` / ``"general"``). Absent entries fall back to the
  model output's category.

Decoding is permissive: a missing row returns the defaults; an
unparseable row logs and returns the defaults, never raises — the
UI always has a usable value even if the stored blob was corrupted
by a manual ``settings`` table edit. The :class:`TagHighlightsPayload`
Pydantic model provides the type-safe shape; permissive default
values ensure missing fields round-trip cleanly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from logging import Logger

import pydantic

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .settings_repository import SettingsRepository


class TagHighlightsPayload(pydantic.BaseModel):
    """Wire shape for the global tag highlights blob.

    Mirrors what we accept on the wire (PUT body) and what we emit
    (GET response). All fields default to empty so a partial blob
    (e.g. an older client that only wrote the tier lists) round-trips
    cleanly through :meth:`TagHighlightsService.get`.
    """

    starred: list[str] = pydantic.Field(default_factory=list)
    desired: list[str] = pydantic.Field(default_factory=list)
    undesired: list[str] = pydantic.Field(default_factory=list)
    category_overrides: dict[str, str] = pydantic.Field(default_factory=dict)


@dataclass
class TagHighlights:
    """User-curated tag tiers used by the visual prune grid + customize UI.

    Mirrors the wire :class:`TagHighlightsPayload` so the dataclass
    travels round-trip without a separate serialised type.
    ``category_overrides`` keys are stored verbatim (the frontend
    enforces the ``str`` value type at the chip flow).
    """

    starred: list[str] = field(default_factory=list)
    desired: list[str] = field(default_factory=list)
    undesired: list[str] = field(default_factory=list)
    category_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: TagHighlightsPayload) -> "TagHighlights":
        """Coerce a wire-shaped payload into a dataclass for the in-memory store."""
        return cls(
            starred=list(payload.starred),
            desired=list(payload.desired),
            undesired=list(payload.undesired),
            category_overrides=dict(payload.category_overrides),
        )


class TagHighlightsService(Service):
    """Global tag highlights accessor (single-row persistence).

    The wire shape omits the ``$version`` marker the old
    ``localStorage`` storable carried — there were no migrations to
    evolve (the schema is small and stable), so the version number
    was pure ceremony on the backend. The frontend still treats the
    shape as versioned against the wire contract it round-trips.
    """

    #: KV key under which the highlights blob is stored.
    _KEY: str = "tagger.tag_highlights"

    def __init__(
        self,
        db: DBConnectionFactory,
        logging: LoggingFactory,
        repo: SettingsRepository,
    ) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
        self._repo: SettingsRepository = repo

    def get(self) -> TagHighlights:
        """Return the persisted highlights, or defaults if missing / corrupt."""
        raw = self._repo.get(self._KEY)
        if raw is None:
            return TagHighlights()
        try:
            payload = TagHighlightsPayload.model_validate(json.loads(raw))
        except (json.JSONDecodeError, pydantic.ValidationError, ValueError):
            # ``ValueError`` covers ``model_validate``'s pre-2.x
            # fallback path; pydantic 2 raises ``ValidationError``,
            # but older yadc docs / tests reference ``ValueError``.
            self._logger.warning("Failed to decode stored tag highlights; returning defaults.")
            return TagHighlights()
        return TagHighlights.from_payload(payload)

    def set(self, value: TagHighlights) -> None:
        """Persist the full highlights blob (single upsert).

        The whole blob is written as one JSON object: the per-tier
        edit frequency doesn't warrant per-key rows, and the size
        budget (a few KB at most) keeps the upsert trivially cheap.
        """
        payload = TagHighlightsPayload(
            starred=list(value.starred),
            desired=list(value.desired),
            undesired=list(value.undesired),
            category_overrides=dict(value.category_overrides),
        )
        self._repo.upsert(self._KEY, payload.model_dump_json())
