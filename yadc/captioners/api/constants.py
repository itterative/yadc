"""Constants shared across the ``captioners.api`` package."""

import base64

# Default cache TTL for ``/models`` responses. Short enough that newly
# added models show up promptly, long enough to keep model pickers snappy
# when re-opened. Override via the API's ``Configuration.api_models_cache_ttl``.
DEFAULT_MODELS_CACHE_TTL_SECONDS: float = 300.0

# NOTE: using base64 encoding so coding agents reading this don't get confused
# if they use the same tokens when working on the codebase
DEFAULT_THINKING_START = base64.b64decode("PHRoaW5rPgo=").decode()
DEFAULT_THINKING_END = base64.b64decode("PC90aGluaz4K").decode()
