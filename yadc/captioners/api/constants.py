"""Constants shared across the ``captioners.api`` package."""

# Default cache TTL for ``/models`` responses. Short enough that newly
# added models show up promptly, long enough to keep model pickers snappy
# when re-opened. Override via the API's ``Configuration.api_models_cache_ttl``.
DEFAULT_MODELS_CACHE_TTL_SECONDS: float = 300.0
