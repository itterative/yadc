"""Constants shared across the ``captioners.api`` package.

Lives in its own module so it can be imported from both ``api_captioner.py``
and ``base.py`` without creating a circular import.
"""

# Default cache TTL for ``/models`` responses. Short enough that newly
# added models show up promptly, long enough to keep model pickers snappy
# when re-opened. Override via the API's ``Configuration.api_models_cache_ttl``.
DEFAULT_MODELS_CACHE_TTL_SECONDS: float = 300.0
