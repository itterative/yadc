"""``PromptGenerationService`` — generate a Jinja2 prompt template from an intent.

Calls any configured env's model directly via :func:`yadc.llm.create_client`
(no captioner — there's no image involved) and yields the generated template
body as :class:`StreamChunk`\\s. Reasoning is disabled (``reasoning=None``)
because the artifact is the template body itself; thinking wastes tokens.

Examples are sent as a multi-turn conversation with priming assistant
turns so the LLM learns to acknowledge each example briefly and wait
for the final "now generate" before emitting the template.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from importlib.resources import files
from logging import Logger
from typing import ClassVar, Literal

import pydantic

from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.cmd import envs as cmd_envs
from yadc.llm import ImageUrl, ImageUrlPart, Message, StreamChunk, TextPart, create_client
from yadc.llm.async_session import AsyncSession
from yadc.llm.cache import HTTPResponseCache
from yadc.llm.response_logger import ResponseLogger


def _load_prompt(prompt: str):
    path = files("yadc.api.services.prompt_generation") / "prompts" / prompt
    return path.read_text(encoding="utf-8").strip()


_GENERATE_SYSTEM_PROMPT: str = _load_prompt("generate.txt")
_REFINE_SYSTEM_PROMPT: str = _load_prompt("refine.txt")

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
    """Inputs to :meth:`PromptGenerationService.generate`."""

    env: str
    intent: str
    examples: list[ExamplePair] = pydantic.Field(default_factory=list)
    focus: PromptGenerationFocus = "both"
    api_model_name: str | None = None
    # Generation limits. ``max_tokens`` is the output cap sent to the
    # model (the client default of 4096 truncates longer templates);
    # ``image_quality`` is the few-shot example image fidelity — it
    # maps to OpenAI's ``image_url.detail`` and Gemini's
    # ``mediaResolution`` (see ``PromptGenerationService.generate``).
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


class PromptGenerationService(Service):
    """Generate a Jinja2 prompt template from an intent + few-shot examples.

    One-shot: instantiates a fresh LLM client per ``generate`` call and
    closes it when the stream is exhausted.
    """

    def __init__(self, logging: LoggingFactory) -> None:
        self._logger: Logger = logging.get_logger(__name__)

    async def generate(
        self,
        *,
        request: PromptGenerationRequest,
        password: str | None = None,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        async_session: AsyncSession | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk`\\s for the generated template body.

        ``PasswordRequiredError`` from ``cmd_envs.load_env`` propagates
        (the controller maps it to HTTP 403). ``cache`` and
        ``response_logger`` are forwarded to the LLM client; ``None``
        uses the client's defaults. ``async_session`` is exposed for tests.
        """
        config = cmd_envs.load_env(request.env, password=password)

        if not config.api.url:
            raise ValueError(f"environment '{request.env}' has no api_url configured")

        model_name = request.api_model_name or config.api.model_name
        if not model_name:
            raise ValueError(f"environment '{request.env}' has no api_model_name configured; pass api_model_name explicitly")

        if async_session is None:
            headers: dict[str, str] = {}
            if config.api.token:
                headers["Authorization"] = f"Bearer {config.api.token}"
            async_session = AsyncSession(config.api.url, headers=headers, cache=cache, response_logger=response_logger)

        client = await create_client(
            api_url=config.api.url,
            api_token=config.api.token,
            cache=cache,
            response_logger=response_logger,
            async_session=async_session,
        )

        self._logger.info(
            "Generating prompt template via env '%s' (model=%s, focus=%s, examples=%d).",
            request.env,
            model_name,
            request.focus,
            len(request.examples),
        )

        try:
            await client.load_model(model_name)
            messages = _build_messages(
                request,
                system_prompt=_GENERATE_SYSTEM_PROMPT,
                refine_system_prompt=_REFINE_SYSTEM_PROMPT,
                image_quality=request.image_quality,
            )
            stream = await client.predict_next_message_stream(
                messages,
                reasoning=None,
                max_tokens=request.max_tokens,
                # Gemini reads ``image_quality`` from opts and maps it to
                # ``mediaResolution``; OpenAI-compatible clients pick it up
                # from the message's ``image_url.detail`` instead. Passing
                # it here too covers both backends (mirrors the captioner).
                image_quality=request.image_quality,
            )
            async for chunk in stream:
                yield chunk
        finally:
            # ``client.aclose`` also closes the underlying ``async_session``
            # (which is owned by the client unless injected). httpx
            # ``aclose`` is idempotent, so re-closing is safe either way.
            await client.aclose()


# ---------------------------------------------------------------------------
# Meta-conversation construction
# ---------------------------------------------------------------------------

# Priming assistant text (sent as FAKE assistant turns in the request,
# not real LLM responses). Sets the conversational mode so the LLM
# acknowledges each example briefly instead of generating a template
# mid-conversation.
_EXAMPLE_ACK = "Got it. Send the next example."

# Refine-mode priming ack for the existing-template user turn (between
# the priming and the first example). "next item" is generic because
# the next item could be an example, but the user is sending the
# existing template first — using "item" instead of "example" covers
# both.
_REFINE_TEMPLATE_ACK = "Got it. Send the next item."


def _build_messages(
    request: PromptGenerationRequest,
    *,
    system_prompt: str,
    refine_system_prompt: str,
    image_quality: Literal["auto", "low", "high"] = "auto",
) -> list[Message]:
    """Build the multi-turn meta-conversation for the template-generation LLM call.

    Two modes (signalled by ``request.template_content is not None`` —
    no separate ``mode`` field, keeps the API surface small):

    - **Generate** (default): the model invents a Jinja2 template from
      scratch based on the intent (+ optional style-reference examples).
    - **Refine**: the model applies the requested changes to the
      user-provided existing template, preserving what works and
      emitting a full refined template (not a diff).

    Generate mode structure with N examples::

        System:    <generate_system_prompt>
        User:      intent + "I'll send N examples, then ask you to generate"
        Assistant: "Understood. Send your examples."              (fake priming)
        User:      example 1 + image
        Assistant: "Got it. Send the next example."                (fake priming)
        ...
        User:      example N + image
        User:      "Now generate the Jinja2 template."

    Refine mode structure with N examples::

        System:    <refine_system_prompt>
        User:      intent + "I'll send the existing template + N examples..."
        Assistant: "Understood. Send your items."                 (fake priming)
        User:      <existing template>
        Assistant: "Got it. Send the next item."                   (fake priming)
        User:      example 1 + image
        Assistant: "Got it. Send the next example."                (fake priming)
        ...
        User:      example N + image
        User:      "Now generate the refined Jinja2 template."

    With 0 examples, the priming is skipped and the first user message
    asks directly for the template. In refine mode the existing
    template content is folded into that single user message (no
    separate template turn — there's no "before the examples" slot
    to insert into when there are no examples).
    """
    is_refine = request.template_content is not None
    messages: list[Message] = [
        Message(
            role="system",
            content=refine_system_prompt if is_refine else system_prompt,
        )
    ]

    if not request.intent:
        request.intent = "(none provided)"

    if not request.examples:
        if is_refine:
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Refine my existing Jinja2 prompt template for image captioning."
                        f"\n\nMy current template:\n{request.template_content}"
                        f"\n\nIntent for refinement:\n{request.intent.strip()}"
                        f"\n\n{_format_hints(request.focus)}."
                        "\n\nOutput only the template, without any markdown or instructions on how to use the template."
                    ),
                )
            )
        else:
            last_message = (
                "Generate a Jinja2 prompt template for image captioning."
                "Output only the template, without any markdown or instructions on how to use the template."
            ) + _format_hints(request.focus)
            messages.append(
                Message(
                    role="user",
                    content=(f"{last_message}.\n\nIntent:\n{request.intent.strip()}."),
                )
            )
        return messages

    n = len(request.examples)

    if is_refine:
        messages.append(
            Message(
                role="user",
                content=(
                    f"I need to refine my existing Jinja2 prompt template for image captioning. "
                    f"Intent: {request.intent.strip()}. "
                    f"I'll send you {n} example(s) — each as a subject + caption + image — "
                    f"plus my current template, then ask you to generate. "
                    f"Briefly acknowledge each item. "
                    f'When I say "now generate", emit the refined Jinja2 template.'
                ),
            )
        )
        messages.append(Message(role="assistant", content="Understood. Send your items."))
        # The existing template is its own user turn between the priming
        # and the first example, so the LLM can acknowledge it before
        # processing the examples.
        messages.append(
            Message(
                role="user",
                content=(f"My current template (please refine it according to the intent):\n\n{request.template_content}"),
            )
        )
        messages.append(Message(role="assistant", content=_REFINE_TEMPLATE_ACK))
    else:
        messages.append(
            Message(
                role="user",
                content=(
                    f"I need a Jinja2 prompt template for image captioning.\n\n"
                    f"Intent:\n{request.intent.strip()}\n\n"
                    f"I'll send you {n} example(s) — each as a "
                    f"subject + caption + image. Briefly acknowledge each one. "
                    f'When I say "now generate", emit the Jinja2 template.'
                ),
            )
        )
        messages.append(Message(role="assistant", content="Understood. Send your examples."))

    for i, ex in enumerate(request.examples):
        is_last = i == n - 1
        messages.append(
            Message(
                role="user",
                content=[
                    TextPart(text=f"Example {i + 1}/{n}: {ex.subject}\n{ex.caption}"),
                    ImageUrlPart(image_url=ImageUrl(url=ex.image_data_url, detail=image_quality)),
                ],
            )
        )
        if not is_last:
            messages.append(Message(role="assistant", content=_EXAMPLE_ACK))

    template_word = "refined Jinja2 template" if is_refine else "Jinja2 template"
    last_message = (
        f"Now generate the {template_word}. "
        f"Output only the template, without any markdown or instructions on how to use the template. "
        f"{_format_hints(request.focus)}"
    )
    messages.append(Message(role="user", content=last_message))

    return messages


def _format_hints(focus: PromptGenerationFocus) -> str:
    """The Jinja2 block format hint that follows the imperative sentence.

    Shared between the 0-example (folded into the first user turn) and
    the N-example (appended to the final "go" user turn) paths so the
    wording is identical across both.
    """
    match focus:
        case "both":
            return (
                "Output both the system_prompt and user_prompt top-level blocks.\n\n"
                "Use the following format:\n"
                "{% set system_prompt %}\n...\n{% endset %}\n\n"
                "{% set user_prompt %}\n...\n{% endset %}"
            )
        case "system":
            return "Output only the system_prompt top-level block.Use the following format:\n{% set system_prompt %}\n...\n{% endset %}"
        case "user":
            return "Output only the user_prompt top-level block.Use the following format:\n{% set user_prompt %}\n...\n{% endset %}"


__all__ = [
    "ExamplePair",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "PromptGenerationService",
]
