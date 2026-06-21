"""Constants used by the captioner layer and its env-orchestrator.

The model-list cache TTL lives in :mod:`yadc.llm.constants` (alongside
the cache behaviour) — only the orchestrator timeout stays here.
"""

# Default upper bound for the entire ``list_models`` operation (env
# decryption + API-type inference probes + the list call itself). Caps
# the time the WebUI's model picker can spend spinning on a misconfigured
# or unreachable env so the user gets a clean error instead of a
# permanent spinner. Override via the API's
# ``Configuration.list_models_timeout`` (or pass ``timeout=None`` from
# the cmd layer to disable). 10s is short enough to fail fast on a
# dead/slow API, long enough to tolerate a sluggish but working one.
DEFAULT_LIST_MODELS_TIMEOUT_SECONDS: float = 10.0
