"""Constants shared across the ``captioners.api`` package."""

# Default cache TTL for ``/models`` responses. Short enough that newly
# added models show up promptly, long enough to keep model pickers snappy
# when re-opened. Override via the API's ``Configuration.api_models_cache_ttl``.
DEFAULT_MODELS_CACHE_TTL_SECONDS: float = 300.0

# Default upper bound for the entire ``list_models`` operation (env
# decryption + API-type inference probes + the list call itself). Caps
# the time the WebUI's model picker can spend spinning on a misconfigured
# or unreachable env so the user gets a clean error instead of a
# permanent spinner. Override via the API's
# ``Configuration.list_models_timeout`` (or pass ``timeout=None`` from
# the cmd layer to disable). 10s is short enough to fail fast on a
# dead/slow API, long enough to tolerate a sluggish but working one.
DEFAULT_LIST_MODELS_TIMEOUT_SECONDS: float = 10.0
