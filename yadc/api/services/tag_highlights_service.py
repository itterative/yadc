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
  tag lists (one entry per tier). Each entry carries a ``name`` and
  a ``canonical_form`` flag — ``canonical_form: True`` means the
  name is the canonical (model-output) form and the frontend
  applies the user's ``replace_underscores`` preference; ``False``
  means the name should be rendered verbatim (free-text user input,
  or a kaomoji whose underscores must be preserved). The backend
  auto-flips the flag to ``False`` for names in the kaomoji set
  (see :class:`TaggedEntry`'s validator) so the curated-tier
  storage doesn't need kaomoji-specific knowledge at every call
  site.
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

from ...taggers.postprocessing import kaomojis
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .settings_repository import SettingsRepository


class TaggedEntry(pydantic.BaseModel):
    """One curated tag entry: the canonical identity plus a display-form signal.

    ``name`` is the canonical tag identity (``speech_bubble``, not the
    display form) — stored and returned verbatim, so toggling the user's
    ``replace_underscores`` preference re-derives the display form on
    the fly rather than baking it in at insert time.

    ``canonical_form`` is ``True`` for entries whose name is the
    canonical (model-output) form — the frontend applies the user's
    ``replace_underscores`` preference when rendering. ``False`` for
    entries whose literal identity is pinned (free-text user input,
    or a kaomoji whose underscores must be preserved) — the
    frontend renders verbatim.

    The validator auto-flips the flag to ``False`` for names in the
    :data:`yadc.taggers.postprocessing.kaomojis` set so the backend
    becomes the authoritative source for the kaomoji carve-out —
    the frontend never has to special-case kaomojis. The default
    for unspecified flags is ``True`` (catalog / model-output
    entries are the common case).

    The GET boundary returns this shape as-is; the frontend applies
    the display transform at render time.
    """

    name: str
    # Default True (canonical = model-output form). The validator
    # below overrides to False for kaomojis regardless of what the
    # caller sent, so a frontend mistake can't accidentally let a
    # kaomoji through with the transform applied.
    canonical_form: bool = True

    @pydantic.field_validator("canonical_form", mode="after")
    @classmethod
    def _override_canonical_for_kaomojis(cls, v: bool, info: pydantic.ValidationInfo) -> bool:
        """Force ``canonical_form=False`` for any name in the kaomoji set.

        Kaomojis (``0_0``, ``(o)_(o)``, ``>_<`` …) contain underscores
        that the user's ``replace_underscores`` preference would
        corrupt (``0_0`` → ``0 0``). The backend already guards the
        model-output path via :func:`replace_underscore_for_tag`; this
        validator extends the carve-out to the curated-tier storage
        so the frontend can stay kaomoji-unaware.
        """
        name = info.data.get("name")
        if isinstance(name, str) and name in kaomojis:
            return False
        return v


class TagHighlightsPayload(pydantic.BaseModel):
    """Wire shape for the global tag highlights blob.

    Mirrors what we accept on the wire (PUT body) and what we emit
    (GET response). All fields default to empty so a partial blob
    (e.g. an older client that only wrote the tier lists) round-trips
    cleanly through :meth:`TagHighlightsService.get`.
    """

    starred: list[TaggedEntry] = pydantic.Field(default_factory=list)
    desired: list[TaggedEntry] = pydantic.Field(default_factory=list)
    undesired: list[TaggedEntry] = pydantic.Field(default_factory=list)
    category_overrides: dict[str, str] = pydantic.Field(default_factory=dict)


@dataclass
class TagHighlights:
    """User-curated tag tiers used by the visual prune grid + customize UI.

    Mirrors the wire :class:`TagHighlightsPayload` so the dataclass
    travels round-trip without a separate serialised type.
    ``category_overrides`` keys are stored verbatim (the frontend
    enforces the ``str`` value type at the chip flow).
    """

    starred: list[TaggedEntry] = field(default_factory=list)
    desired: list[TaggedEntry] = field(default_factory=list)
    undesired: list[TaggedEntry] = field(default_factory=list)
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
