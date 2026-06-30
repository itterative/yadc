"""Tag-catalog data layer — pure download / parse / build helpers.

Owns the catalog's data shapes and the pure functions that produce them,
but **no process-wide state**: lifecycle (caching, locking, startup
preloading) is the job of :class:`yadc.api.services.TagSuggestionsService`.
The matcher (:mod:`~yadc.api.services.tag_suggestions.suggestions`)
consumes a :class:`TagCatalog` directly — its per-canonical jump lists
(:attr:`TagCatalog.char_indices`) and inverted char index
(:attr:`TagCatalog.char_index`) skip the bulk of the catalog before the
per-entry fuzzy score.

Source CSVs come from BetaDoggo's danbooru-tag-list releases and are
cached at ``~/.cache/yadc/tagging/catalogs/<variant>.csv``. Adding a
variant is a one-line change in :class:`CatalogVariant` +
:data:`VARIANT_URL` + :data:`VARIANT_LABEL`; the matcher is
format-agnostic and the cache key is the variant (not the URL, so
upstream renames don't strand cached files).

CSV columns: ``tag`` (canonical, underscored), ``category_id`` (danbooru
taxonomy — only the IDs in :data:`_KNOWN_CATEGORIES` are kept), ``count``
(post count, unused by the parser but kept for popularity scoring), and
``aliases`` (quoted comma-separated alternate spellings).
"""

from __future__ import annotations

import asyncio
import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx


class CatalogVariant(str, Enum):
    """Catalog variants — each is a different tagger's reference list.

    Adding a new variant needs a URL in :data:`VARIANT_URL` and a display
    label in :data:`VARIANT_LABEL`; the parser and matcher are
    variant-agnostic once the CSV is on disk.
    """

    ANIMA = "anima"
    ILLUSTRIOUS = "illustrious"
    NOOBAIXL = "noobaixl"


# Display labels for the picker UI. Kept here (not title-cased from the enum
# value) so brand spellings (``NoobAIXL``) render correctly.
VARIANT_LABEL: dict[CatalogVariant, str] = {
    CatalogVariant.ANIMA: "Anima",
    CatalogVariant.ILLUSTRIOUS: "Illustrious",
    CatalogVariant.NOOBAIXL: "NoobAIXL",
}


# GitHub release URLs from BetaDoggo/danbooru-tag-list. Pinned to a
# specific file per variant so a future release of one doesn't break
# another. Filenames in the release are exactly as listed.
VARIANT_URL: dict[CatalogVariant, str] = {
    CatalogVariant.ANIMA: "https://github.com/BetaDoggo/danbooru-tag-list/releases/download/Model-Tags/anima-1.0.csv",
    CatalogVariant.ILLUSTRIOUS: "https://github.com/BetaDoggo/danbooru-tag-list/releases/download/Model-Tags/illustriousV1.0_underscore.csv",
    CatalogVariant.NOOBAIXL: "https://github.com/BetaDoggo/danbooru-tag-list/releases/download/Model-Tags/NoobAIXL1.1_underscore.csv",
}

# danbooru category taxonomy — only IDs the matcher surfaces. Other IDs
# (2=focus, 9=rating on the live site) are dropped: rating isn't in the
# source files, and the rest aren't useful for autocomplete.
_KNOWN_CATEGORIES: dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

# Separators normalized to a single space during matching. Spaces and
# underscores are equivalent at the matching layer, so ``long_hair``,
# ``long hair``, and ``long-hair`` all reduce to ``"long hair"``.
_TAG_NORMALIZE_SEPARATORS = "_()[]{}:;,."


def normalize_tag_name(s: str) -> str:
    """Lowercase *s* and collapse separators to single spaces.

    Used on both sides of the matcher's jump-list walk so a query ``"long hair"``
    and a canonical ``"long_hair"`` resolve to the same positions. A divergence
    between the indexed canonical and the normalized query silently breaks
    the pre-check, so there must be exactly one normalization point.
    """
    lowered = s.lower()
    translated = lowered.translate({ord(c): ord(" ") for c in _TAG_NORMALIZE_SEPARATORS})
    return " ".join(translated.split())


def _normalize_with_char_index(s: str) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    """Normalize *s* and build a per-char position index for it.

    Returns ``(normalized, sorted ((char, positions)) pairs)`` where
    ``positions`` lists the offsets of ``char`` in the normalized string.
    The matcher walks this greedily for its subsequence pre-check: for
    each query char it jumps to the smallest position > last_pos via
    ``bisect_right`` on the matching tuple, and sorted ``char`` order
    lets the walk bail early when the query char sorts past the current
    entry. Fixed-length per tag (one entry per unique char) so the
    matcher skips the bulk of the catalog in O(query_length) per entry
    before the (much more expensive) fuzzy score.
    """
    normalized = normalize_tag_name(s)
    if not normalized:
        return "", ()
    char_to_positions: dict[str, list[int]] = {}
    for i, c in enumerate(normalized):
        char_to_positions.setdefault(c, []).append(i)
    return normalized, tuple((c, tuple(positions)) for c, positions in sorted(char_to_positions.items()))


def parse_betadoggo_csv(text: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Parse a BetaDoggo ``danbooru-tag-list`` CSV body.

    Returns ``(canonical, category, aliases)`` tuples. Categories outside
    :data:`_KNOWN_CATEGORIES` are dropped. Aliases are returned raw (the
    matcher normalizes them at match time) so this stays a pure
    CSV-shape transform.
    """
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        try:
            category_id = int(row[1])
        except ValueError:
            continue
        label = _KNOWN_CATEGORIES.get(category_id)
        if label is None:
            continue
        canonical = row[0].strip()
        if not canonical:
            continue
        # Danbooru tags can't contain commas (they're a meta-character in the
        # tag system), and csv.reader keeps a quoted alias field as one cell,
        # so splitting on commas inside it is unambiguous.
        aliases = tuple(alias.strip() for cell in row[3:] for alias in cell.split(",") if alias.strip())
        out.append((canonical, label, aliases))
    return out


# ---------- Cache location & disk layout ----------

# ``~/.cache/yadc/tagging/catalogs/`` — the tagging-feature slice of the
# project-wide cache root (``~/.cache/yadc/``).
_TAG_CACHE_SUBDIR = "tagging/catalogs"


def get_cache_dir() -> Path:
    """User-cache directory for downloaded tag catalogs —
    ``~/.cache/yadc/tagging/catalogs/``."""
    return Path(f"~/.cache/yadc/{_TAG_CACHE_SUBDIR}").expanduser()


def get_cache_path(variant: CatalogVariant) -> Path:
    """Local cache file for *variant* — ``<cache_dir>/<variant>.csv``.

    Filename is the enum *value*, not the source URL's filename, so the
    cache survives upstream release renames.
    """
    return get_cache_dir() / f"{variant.value}.csv"


def is_cached(variant: CatalogVariant) -> bool:
    """True if the on-disk cache file exists and is non-empty."""
    path = get_cache_path(variant)
    return path.exists() and path.stat().st_size > 0


# ---------- Download ----------


async def download_catalog(variant: CatalogVariant, *, force: bool = False) -> Path:
    """Stream the CSV from :data:`VARIANT_URL` to its cache path.

    Idempotent — skips the network round-trip when the file exists unless
    *force* is set. Streams in 64KB chunks; follows redirects (the GitHub
    release URL redirects to S3).
    """
    cache_path = get_cache_path(variant)
    if cache_path.exists() and not force:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = VARIANT_URL[variant]
    # 120s covers slow networks on a 4MB download; bursts above that
    # are more likely a stalled connection than a healthy transfer.
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with cache_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
    return cache_path


# ---------- In-memory catalog shape & builder ----------


CatalogEntry = tuple[str, str, tuple[str, ...]]


@dataclass(slots=True, frozen=True)
class TagCatalog:
    """Parsed catalog held in memory, with per-canonical jump lists and
    post counts.

    :attr:`normalized`, :attr:`char_indices`, :attr:`counts`, and
    :attr:`normalized_aliases` are parallel to :attr:`entries`.
    :attr:`char_index` is the inverted index over the union of each tag's
    canonical + alias chars (see :func:`_build_char_index`) — it restricts
    the matcher's per-tag scan to candidates containing every distinct
    query char. Frozen so accidental mutation blows up loudly; the
    matcher's stable-order guarantee relies on :attr:`entries` being stable.
    """

    entries: tuple[CatalogEntry, ...]
    normalized: tuple[str, ...]
    char_indices: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]
    counts: tuple[int, ...]
    char_index: dict[str, tuple[int, ...]]
    # Per-tag pre-normalized alias forms, parallel to :attr:`entries`.
    # Built once at load (alias normalization was ~55% of the matcher's
    # alias-scoring cost) so the matcher doesn't re-normalize every alias
    # on every keystroke.
    normalized_aliases: tuple[tuple[str, ...], ...]


def _parse_and_sort_from_csv_text(text: str) -> tuple[list[CatalogEntry], list[int]]:
    """Parse *text* and return ``(entries, counts)`` sorted by count desc.

    The counts are returned alongside so the matcher's popularity boost
    (:func:`yadc.api.services.tag_suggestions.suggest`) scores against
    real usage volume. BetaDoggo's CSVs ship pre-sorted by count desc, so
    this is defensive — but the sort is what makes the matcher's
    early-stop-at-N-candidates optimization sound (the cap is reached on
    the highest-confidence hits first).
    """
    raw: list[tuple[str, str, int, tuple[str, ...]]] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        try:
            category_id = int(row[1])
        except ValueError:
            continue
        label = _KNOWN_CATEGORIES.get(category_id)
        if label is None:
            continue
        canonical = row[0].strip()
        if not canonical:
            continue
        try:
            count = int(row[2])
        except (ValueError, IndexError):
            count = 0
        aliases = tuple(alias.strip() for cell in row[3:] for alias in cell.split(",") if alias.strip())
        raw.append((canonical, label, count, aliases))
    raw.sort(key=lambda entry: entry[2], reverse=True)
    return (
        [(canonical, category, aliases) for canonical, category, _, aliases in raw],
        [entry[2] for entry in raw],
    )


def _parse_catalog_from_disk(cache_path: Path) -> tuple[list[CatalogEntry], list[int]]:
    """Sync helper — read + parse + sort. Called in a worker thread by
    :func:`load_catalog` so the parser doesn't block the event loop."""
    text = cache_path.read_text(encoding="utf-8", errors="replace")
    return _parse_and_sort_from_csv_text(text)


def _build_catalog(
    entries: list[CatalogEntry],
    counts: list[int] | None = None,
) -> TagCatalog:
    """Build the in-memory bundle from parsed entries.

    Splits each canonical into its normalized form + a char→positions jump
    list (see :func:`_normalize_with_char_index`), pre-normalizes aliases,
    and builds the inverted char index. *counts* defaults every count to 1,
    which disables the popularity boost — the right shape for unit-test
    fixtures that don't care about ranking.
    """
    normalized_list: list[str] = []
    char_indices_list: list[tuple[tuple[str, tuple[int, ...]], ...]] = []
    normalized_aliases_list: list[tuple[str, ...]] = []
    if counts is None:
        counts = [1] * len(entries)
    for canonical, _, aliases in entries:
        norm, idx = _normalize_with_char_index(canonical)
        normalized_list.append(norm)
        char_indices_list.append(idx)
        # Aliases are static; pay ``normalize_tag_name`` once at load
        # rather than per keystroke.
        normalized_aliases_list.append(tuple(normalize_tag_name(a) for a in aliases))
    normalized_aliases = tuple(normalized_aliases_list)
    return TagCatalog(
        entries=tuple(entries),
        normalized=tuple(normalized_list),
        char_indices=tuple(char_indices_list),
        counts=tuple(counts),
        char_index=_build_char_index(normalized_list, normalized_aliases),
        normalized_aliases=normalized_aliases,
    )


def _build_char_index(
    normalized_canonicals: Sequence[str],
    normalized_aliases: Sequence[tuple[str, ...]],
) -> dict[str, tuple[int, ...]]:
    """Build the inverted char → tag-index map used by the matcher.

    For each tag the char set is the union of the normalized canonical
    *and* all normalized aliases — a tag is a candidate for a query iff
    its unioned char set contains every distinct query char. Indexing the
    union (not the canonical alone) keeps alias-driven matches (e.g.
    ``high_res`` → ``highres``, whose canonical lacks ``_``) from being
    pruned out. Tag indexes are appended in :attr:`entries` order
    (post-count desc at load), so every char's tuple is popularity-sorted
    — the matcher intersects these in place and gets a popularity-sorted
    candidate list for free.
    """
    buckets: dict[str, list[int]] = {}
    for i, (canonical, aliases) in enumerate(zip(normalized_canonicals, normalized_aliases)):
        seen: set[str] = set()
        seen.update(canonical)
        for alias in aliases:
            seen.update(alias)
        for c in seen:
            buckets.setdefault(c, []).append(i)
    return {c: tuple(idxs) for c, idxs in buckets.items()}


async def load_catalog(variant: CatalogVariant) -> TagCatalog:
    """Download (if needed) + parse + preprocess the catalog bundle.

    The parse + per-tag jump-list build run in worker threads via
    :func:`asyncio.to_thread` so they don't pin the event loop on first
    load. Does **not** cache — lifecycle is the service's concern.
    """
    cache_path = await download_catalog(variant)
    raw_entries, raw_counts = await asyncio.to_thread(_parse_catalog_from_disk, cache_path)
    return await asyncio.to_thread(_build_catalog, raw_entries, raw_counts)


# Default variant — the ultimate fallback when neither a persisted user
# selection nor ``Configuration.tagger_suggestion_variant`` yields a valid
# variant. The service owns variant resolution; this constant is only the
# floor.
DEFAULT_VARIANT = CatalogVariant.NOOBAIXL


def resolve_variant(raw: str | None) -> CatalogVariant:
    """Map a free-form string to a :class:`CatalogVariant`, falling back to
    :data:`DEFAULT_VARIANT` on anything unknown.

    Centralizes the ``str`` → enum coercion so a bad value in config or
    settings downgrades gracefully rather than crashing boot or the
    request handler.
    """
    if raw is None:
        return DEFAULT_VARIANT
    try:
        return CatalogVariant(raw.strip().lower())
    except ValueError:
        return DEFAULT_VARIANT


def list_variants() -> list[tuple[CatalogVariant, str]]:
    """Return ``(variant, label)`` pairs for every known variant, in enum order.

    Used by the picker endpoint to populate its dropdown.
    """
    return [(v, VARIANT_LABEL[v]) for v in CatalogVariant]


__all__ = [
    "CatalogVariant",
    "CatalogEntry",
    "TagCatalog",
    "VARIANT_URL",
    "VARIANT_LABEL",
    "DEFAULT_VARIANT",
    "parse_betadoggo_csv",
    "normalize_tag_name",
    "get_cache_dir",
    "get_cache_path",
    "is_cached",
    "download_catalog",
    "load_catalog",
    "resolve_variant",
    "list_variants",
]  # noqa: F401 — public API surface, symbols re-exported
