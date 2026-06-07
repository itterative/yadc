"""Caption job options — the single source of truth for a captioning run's knobs."""

from typing import ClassVar

import pydantic


class CaptionJobOptions(pydantic.BaseModel):
    """All configurable parameters for a single captioning run.

    Extra keys in the input dict are silently ignored (``extra="ignore"``).
    """

    # API connection
    api_url: str = ""
    api_token: str = ""
    api_model_name: str = ""

    # Environment / config resolution
    env: str = "default"

    # Captioning behaviour
    prompt_template: str = ""
    prompt_name: str = ""
    max_tokens: int = 512
    image_quality: str = "auto"
    store_conversation: bool = False
    overwrite: bool = False
    rounds: int = 1
    draft: str = ""

    # Concurrency
    # Number of in-flight `predict_stream` requests to allow at once.
    # 1 = sequential (default; matches the original behavior). Values
    # > 1 enable parallel captioning via the runner's semaphore-gated
    # gather. Job-level only — not a per-dataset TOML field.
    max_concurrent: int = 1

    # Reasoning
    reasoning: bool = False
    reasoning_effort: str = "low"
    reasoning_exclude_output: bool = True

    # Password for decrypting password-mode environment settings
    password: str | None = None

    # If set, only caption these specific image IDs (single-image mode)
    image_ids: list[int] | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="ignore")
