"""Tests for ``yadc.api.services.tag_suggestions.catalog``.

Pure-function tests for the parser + cache-path helpers (no network,
no asyncio). The async loader is tested with monkeypatched download
+ parse so the network is never touched during the test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yadc.api.services.tag_suggestions import (
    DEFAULT_VARIANT,
    VARIANT_URL,
    CatalogVariant,
    get_cache_dir,
    get_cache_path,
    is_cached,
    parse_betadoggo_csv,
)

# ---------- parse_betadoggo_csv ----------

# A small fragment copied from the actual BetaDoggo file so the parser
# is locked against the real CSV layout, not an idealized version.
_REAL_FRAGMENT = (
    '1girl,0,6160038,"1girls,sole_female"\n'
    'highres,5,5441561,"high_res,high_resolution,hires"\n'
    "solo,0,7530606,\n"
    'large_breasts,0,1621770,"big_breasts,large_boobs,large_breast,large_tits,/lb"\n'
    'rating_explicit,9,100,"explicit"'
)


def test_parse_keeps_canonical_aliases_and_category() -> None:
    """Sanity check: parser preserves all three output fields per row.

    The order is what the matcher relies on for ranking stability.
    """
    entries = parse_betadoggo_csv(_REAL_FRAGMENT)
    assert ("1girl", "general", ("1girls", "sole_female")) in entries
    assert ("highres", "meta", ("high_res", "high_resolution", "hires")) in entries
    assert ("solo", "general", ()) in entries


def test_parse_splits_multi_alias_fields() -> None:
    """Comma-separated aliases inside the quoted field become separate
    entries. The CSV reader keeps the quoted value as one cell; the
    parser must split it.
    """
    entries = parse_betadoggo_csv(_REAL_FRAGMENT)
    highres = [entry for entry in entries if entry[0] == "highres"]
    assert highres == [("highres", "meta", ("high_res", "high_resolution", "hires"))]

    large = [entry for entry in entries if entry[0] == "large_breasts"]
    assert large == [
        (
            "large_breasts",
            "general",
            ("big_breasts", "large_boobs", "large_breast", "large_tits", "/lb"),
        )
    ]


def test_parse_drops_unknown_categories() -> None:
    """Only the five known category IDs land in the output.

    9 (rating) appears in the fragment and must be filtered out;
    the matcher surfaces ``rating`` for explicit / sensitive via a
    separate channel, not via the catalog.
    """
    entries = parse_betadoggo_csv(_REAL_FRAGMENT)
    names = {entry[0] for entry in entries}
    assert "1girl" in names
    assert "highres" in names
    assert "solo" in names
    assert "large_breasts" in names
    # rating_explicit has category 9 — not in _KNOWN_CATEGORIES — must drop.
    assert "rating_explicit" not in names


def test_parse_drops_rows_with_unparseable_category() -> None:
    """Bad data must not blow up the catalog."""
    text = "tag_ok,0,100,\nbad,not_a_number,100,\ntag2,4,50,"
    entries = parse_betadoggo_csv(text)
    assert ("tag_ok", "general", ()) in entries
    assert ("tag2", "character", ()) in entries
    assert all(entry[0] != "bad" for entry in entries)


def test_parse_handles_empty_canonical() -> None:
    """An empty tag string is a malformed row and gets dropped."""
    text = ",0,100,\nreal,4,50,"
    entries = parse_betadoggo_csv(text)
    assert entries == [("real", "character", ())]


def test_parse_handles_empty_input() -> None:
    """No rows in, no rows out."""
    assert parse_betadoggo_csv("") == []
    assert parse_betadoggo_csv("\n\n\n") == []


# ---------- Cache layout ----------


def test_get_cache_dir_is_under_tagging_subdir() -> None:
    """The cache must live under the project-wide cache root in a
    feature-specific subdirectory so other features' caches
    (``api_requests/``, ``api-debug/``) don't collide with it.
    """
    cache_dir = get_cache_dir()
    parts = cache_dir.expanduser().parts
    # Suffix must be ``.cache/yadc/tagging/catalogs`` regardless of the
    # exact home-dir prefix (XDG / Apple / Win all differ).
    assert parts[-4:] == (".cache", "yadc", "tagging", "catalogs")


def test_get_cache_path_uses_variant_value() -> None:
    """Cache filename is the enum *value*, not the URL's release file.

    So renaming a release upstream (e.g. ``NoobAIXL1.2_underscore.csv``)
    doesn't strand files; old cache still works under its known name.
    """
    assert get_cache_path(CatalogVariant.NOOBAIXL).name == "noobaixl.csv"
    assert get_cache_path(CatalogVariant.ANIMA).name == "anima.csv"
    assert get_cache_path(CatalogVariant.ILLUSTRIOUS).name == "illustrious.csv"


def test_variant_urls_are_pinned() -> None:
    """Each variant has a known download URL.

    URLs are external so we can't test liveness reliably, but the
    contract is that the dict is populated for every variant — adding
    a new variant must update this map.
    """
    assert set(VARIANT_URL.keys()) == set(CatalogVariant)
    for url in VARIANT_URL.values():
        assert url.startswith("https://github.com/BetaDoggo/danbooru-tag-list/releases/download/")


def test_default_variant_is_noobaixl() -> None:
    """Until user-switching lands, NOOBAIXL is the active default.

    A follow-up will let the user select a variant per-dataset via
    Configuration; flipping the default here shouldn't be needed
    beyond that.
    """
    assert DEFAULT_VARIANT == CatalogVariant.NOOBAIXL


# ---------- is_cached / download ----------


def test_is_cached_false_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: tmp_path / "x.csv")
    assert is_cached(CatalogVariant.NOOBAIXL) is False


def test_is_cached_false_for_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "x.csv"
    empty.write_text("")
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: empty)
    assert is_cached(CatalogVariant.NOOBAIXL) is False


def test_is_cached_true_for_nonempty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "x.csv"
    f.write_text("a,0,1,\n")
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: f)
    assert is_cached(CatalogVariant.NOOBAIXL) is True


# ---------- Service: load / cache / download ----------


@pytest.fixture
def tag_suggestions(
    logging_factory,
    settings_service,
    test_configuration,
):
    """A freshly-constructed ``TagSuggestionsService`` wired to a real
    ``SettingsService`` + the test ``Configuration``.

    Per-test construction (not session-scoped) so each test starts with an
    empty in-memory catalog cache and the same clean settings DB.
    """
    from yadc.api.services.tag_suggestions import TagSuggestionsService

    return TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )


@pytest.mark.asyncio
async def test_service_loads_catalog_when_called(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tag_suggestions,
) -> None:
    """First ``get_catalog`` parses the cached CSV file.

    The cache file is pre-populated at a tmp_path so the loader's file
    read is observable. ``download_catalog`` isn't patched — the test
    verifies that a populated cache file produces the expected parsed
    catalog, which is the loader's responsibility.
    """
    target = tmp_path / "noobaixl.csv"
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: target)
    target.write_text(
        "alpha,0,10,alias_a\nbeta,4,5,\ngamma,5,1,alt_gamma\n",
        encoding="utf-8",
    )

    catalog = await tag_suggestions.get_catalog()
    by_name = {entry[0]: entry for entry in catalog.entries}
    assert by_name["alpha"] == ("alpha", "general", ("alias_a",))
    assert by_name["beta"] == ("beta", "character", ())
    assert by_name["gamma"] == ("gamma", "meta", ("alt_gamma",))


@pytest.mark.asyncio
async def test_service_caches_after_first_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tag_suggestions,
) -> None:
    """Second ``get_catalog`` hits the in-memory cache without re-parsing.

    Counts parse invocations. Re-parsing 140k rows on every call would
    defeat the entire point of the service holding the parsed bundle.
    """
    target = tmp_path / "noobaixl.csv"
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: target)
    target.write_text("alpha,0,10,\n", encoding="utf-8")

    from yadc.api.services.tag_suggestions import catalog as tag_catalog

    parse_count = 0
    real_parse = tag_catalog._parse_and_sort_from_csv_text

    def counting_parse(text: str) -> list:
        nonlocal parse_count
        parse_count += 1
        return real_parse(text)

    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog._parse_and_sort_from_csv_text", counting_parse)

    a = await tag_suggestions.get_catalog()
    b = await tag_suggestions.get_catalog()
    assert a is b
    assert parse_count == 1


@pytest.mark.asyncio
async def test_service_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tag_suggestions,
) -> None:
    """No file on disk → ``download_catalog`` runs, the file ends up
    parsed, and the result is cached by the service.
    """
    target = tmp_path / "noobaixl.csv"
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: target)

    download_calls: list[tuple[CatalogVariant, bool]] = []

    async def fake_download(variant: CatalogVariant, *, force: bool = False) -> Path:
        download_calls.append((variant, force))
        target.write_text("downloaded,4,1,alt_dl\n", encoding="utf-8")
        return target

    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.download_catalog", fake_download)

    catalog = await tag_suggestions.get_catalog()
    assert catalog.entries == (("downloaded", "character", ("alt_dl",)),)
    assert download_calls == [(DEFAULT_VARIANT, False)]
    assert target.exists()


@pytest.mark.asyncio
async def test_fresh_service_reads_current_disk_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logging_factory,
    settings_service,
    test_configuration,
) -> None:
    """A new service instance re-reads the cache file — the previous
    module-global ``invalidate_cache`` semantics are now just "construct
    a new service". The in-memory cache is per-instance, so a stale
    bundle is never served across services.
    """
    from yadc.api.services.tag_suggestions import TagSuggestionsService

    target = tmp_path / "noobaixl.csv"
    monkeypatch.setattr("yadc.api.services.tag_suggestions.catalog.get_cache_path", lambda v: target)
    target.write_text("once,0,1,\n", encoding="utf-8")

    svc1 = TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )
    assert [entry[0] for entry in (await svc1.get_catalog()).entries] == ["once"]

    # Simulate the cache file changing underneath us — a new service
    # sees the new data, but the old service still serves its cache.
    target.write_text("twice,0,1,\n", encoding="utf-8")
    assert [entry[0] for entry in (await svc1.get_catalog()).entries] == ["once"]

    svc2 = TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )
    assert [entry[0] for entry in (await svc2.get_catalog()).entries] == ["twice"]


# ---------- Service: variant resolution + switching ----------


@pytest.mark.asyncio
async def test_variant_defaults_to_config_when_unset(
    settings_service,
    test_configuration,
    logging_factory,
) -> None:
    """No persisted selection → ``Configuration.tagger_suggestion_variant``
    wins; an invalid config value falls back to ``noobaixl``."""
    from yadc.api.services.tag_suggestions import TagSuggestionsService

    test_configuration.tagger_suggestion_variant = "illustrious"
    svc = TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )
    assert svc.variant == CatalogVariant.ILLUSTRIOUS

    # An unknown config value degrades to the default rather than raising.
    test_configuration.tagger_suggestion_variant = "flux"
    assert svc.variant == DEFAULT_VARIANT


@pytest.mark.asyncio
async def test_variant_persisted_selection_beats_config(
    settings_service,
    test_configuration,
    logging_factory,
) -> None:
    """A persisted user selection takes precedence over the config default."""
    from yadc.api.services.tag_suggestions import TagSuggestionsService

    test_configuration.tagger_suggestion_variant = "anima"
    settings_service.set("tagger.suggestion_variant", "illustrious")

    svc = TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )
    assert svc.variant == CatalogVariant.ILLUSTRIOUS


@pytest.mark.asyncio
async def test_variant_persisted_garbage_falls_back_to_config(
    settings_service,
    test_configuration,
    logging_factory,
) -> None:
    """A corrupted persisted selection is ignored, landing on the config value."""
    from yadc.api.services.tag_suggestions import TagSuggestionsService

    test_configuration.tagger_suggestion_variant = "anima"
    settings_service.set("tagger.suggestion_variant", "flux")

    svc = TagSuggestionsService(
        logging=logging_factory,
        settings=settings_service,
        configuration=test_configuration,
    )
    assert svc.variant == CatalogVariant.ANIMA


@pytest.mark.asyncio
async def test_set_variant_drops_cache_and_reloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tag_suggestions,
    settings_service,
) -> None:
    """``set_variant`` persists the selection and reloads the new variant on
    the next ``get_catalog`` — a distinct cache file per variant proves the
    bundle was rebuilt, not served from the old one."""
    # Per-variant cache files so the reload reads different content.
    cache_files = {
        CatalogVariant.NOOBAIXL: tmp_path / "noobaixl.csv",
        CatalogVariant.ILLUSTRIOUS: tmp_path / "illustrious.csv",
    }
    cache_files[CatalogVariant.NOOBAIXL].write_text("noob_tag,0,1,\n", encoding="utf-8")
    cache_files[CatalogVariant.ILLUSTRIOUS].write_text("illust_tag,0,1,\n", encoding="utf-8")
    monkeypatch.setattr(
        "yadc.api.services.tag_suggestions.catalog.get_cache_path",
        lambda v: cache_files[v],
    )

    first = await tag_suggestions.get_catalog()
    assert [e[0] for e in first.entries] == ["noob_tag"]

    await tag_suggestions.set_variant(CatalogVariant.ILLUSTRIOUS)
    # Persisted for the next service construction.
    assert settings_service.get("tagger.suggestion_variant") == "illustrious"

    second = await tag_suggestions.get_catalog()
    assert [e[0] for e in second.entries] == ["illust_tag"]
