"""Tag-suggestions package — catalog loading + fuzzy autocomplete matching.

Public surface re-exported so consumers import from the package root
without reaching into submodules. The split mirrors the ``captioning``
package: ``service.py`` is the DI ``Service``, the other modules are
pure helpers it composes.
"""

from .catalog import (
    DEFAULT_VARIANT,
    VARIANT_LABEL,
    VARIANT_URL,
    CatalogEntry,
    CatalogVariant,
    TagCatalog,
    download_catalog,
    get_cache_dir,
    get_cache_path,
    is_cached,
    list_variants,
    load_catalog,
    normalize_tag_name,
    parse_betadoggo_csv,
    resolve_variant,
)
from .service import TagSuggestionsService
from .suggestions import suggest

__all__ = [
    "DEFAULT_VARIANT",
    "CatalogEntry",
    "CatalogVariant",
    "TagCatalog",
    "TagSuggestionsService",
    "VARIANT_LABEL",
    "VARIANT_URL",
    "download_catalog",
    "get_cache_dir",
    "get_cache_path",
    "is_cached",
    "list_variants",
    "load_catalog",
    "normalize_tag_name",
    "parse_betadoggo_csv",
    "resolve_variant",
    "suggest",
]
