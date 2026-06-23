"""Pydantic models + typed errors for the prompt-generation domain.

Lives in the neutral :mod:`yadc.prompt_generation` package so both the
CLI and the API import from one place (no ``api → cmd → api`` cycle
when the API service delegates into the streaming core).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pydantic

PromptGenerationFocus = Literal["system", "user", "both"]


class ExamplePair(pydantic.BaseModel):
    """A single few-shot example: subject + caption + pre-encoded image.

    The frontend is responsible for acquiring the image bytes (manual
    uploads OR dataset fetches via the existing
    ``GET /datasets/<name>/images/<id>/image`` endpoint) and
    base64-encoding them as a data URL. The backend stays stateless
    about image sources.
    """

    subject: str
    caption: str
    # ``data:<mime>;base64,...`` — the frontend always sends this.
    image_data_url: str


class PromptGenerationRequest(pydantic.BaseModel):
    """Inputs to :func:`yadc.prompt_generation.stream_template_chunks`."""

    env: str
    intent: str
    examples: list[ExamplePair] = pydantic.Field(default_factory=list)
    focus: PromptGenerationFocus = "both"
    api_model_name: str | None = None
    # Generation limits. ``max_tokens`` is the output cap sent to the
    # model (the client default of 4096 truncates longer templates);
    # ``image_quality`` is the few-shot example image fidelity — it
    # maps to OpenAI's ``image_url.detail`` and Gemini's
    # ``mediaResolution``.
    max_tokens: int = pydantic.Field(default=16384, ge=100, le=65536)
    image_quality: Literal["auto", "low", "high"] = "auto"
    # When set, runs in refine mode: the model applies the requested
    # changes to this existing template instead of inventing a new one
    # from scratch. No separate ``mode`` field — presence is the signal.
    template_content: str | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("template_content")
    @classmethod
    def _template_content_must_be_non_empty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("template_content must be a non-empty string when provided")
        return value


class PromptGenerationError(Exception):
    """Base class for prompt-generation failures surfaced to callers.

    The streaming core raises :class:`PromptGenerationConfigError` for
    configuration problems and lets ``PasswordRequiredError`` from
    ``cmd_envs.load_env`` propagate untouched (callers map it to the
    right UX — TTY prompt for CLI, 403 for API).
    """


class PromptGenerationConfigError(PromptGenerationError, ValueError):
    """The resolved env config is missing required fields (api_url, api_model_name).

    Inherits from both :class:`PromptGenerationError` (the domain
    hierarchy) and :class:`ValueError` (so generic ``except ValueError``
    handlers and existing test assertions keep working — bad config
    values are value errors by Python convention).
    """
