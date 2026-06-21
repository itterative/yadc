"""Constants used by the LLM client layer.

Separate from ``yadc/captioners/api/constants.py`` which holds
captioner-orchestration constants (env timeout etc.).
"""

# Default cache TTL for ``/models`` responses. Short enough that newly
# added models show up promptly, long enough to keep model pickers snappy
# when re-opened. Override via the API's ``Configuration.api_models_cache_ttl``.
DEFAULT_MODELS_CACHE_TTL_SECONDS: float = 300.0
