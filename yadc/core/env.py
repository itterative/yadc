import os

YADC_PASSWORD: str | None = os.environ.get("YADC_PASSWORD") or None
DEBUG_CAPTION_RESPONSES: bool = os.environ.get("YADC_DEBUG_CAPTION_RESPONSES", "0").strip() == "1"
DEBUG_CAPTION_REQUESTS_BODY: bool = os.environ.get("YADC_DEBUG_CAPTION_REQUESTS_BODY", "0").strip() == "1"
