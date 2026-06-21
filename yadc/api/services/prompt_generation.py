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
from logging import Logger
from typing import ClassVar, Literal

import pydantic

from yadc.cmd import envs as cmd_envs
from yadc.llm import ImageUrl, ImageUrlPart, Message, StreamChunk, TextPart, create_client
from yadc.llm.async_session import AsyncSession
from yadc.llm.cache import HTTPResponseCache
from yadc.llm.response_logger import ResponseLogger

from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service

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

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


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
            messages = _build_messages(request)
            stream = await client.predict_next_message_stream(messages, reasoning=None, max_tokens=25000)
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


def _build_messages(request: PromptGenerationRequest) -> list[Message]:
    """Build the multi-turn meta-conversation for the template-generation LLM call.

    With N examples:

        User:      intent + "I'll send N examples, then ask you to generate"
        Assistant: "Understood. Send your examples."              (fake priming)
        User:      example 1 + image
        Assistant: "Got it. Send the next example."                (fake priming)
        User:      example 2 + image
        ...
        User:      example N + image
        User:      "Now generate the Jinja2 template."

    With 0 examples the priming is skipped (the first user message asks
    directly for the template).
    """
    if not request.intent:
        request.intent = "(none provided)"

    if not request.examples:
        last_message = (
            "Generate a Jinja2 prompt template for image captioning."
            "Output only the template, without any markdown or instructions on how to use the template."
        )

        match request.focus:
            case "both":
                last_message += "Output both the system_prompt and user_prompt top-level blocks.\n\n"
                last_message += "Use the following format:\n"
                last_message += "{% set system_prompt %}\n...\n{% endset %}\n\n"
                last_message += "{% set user_prompt %}\n...\n{% endset %}"
            case "system":
                last_message += "Output only the system_prompt top-level block."
                last_message += "Use the following format:\n"
                last_message += "{% set system_prompt %}\n...\n{% endset %}"
            case "user":
                last_message += "Output only the user_prompt top-level block."
                last_message += "Use the following format:\n"
                last_message += "{% set user_prompt %}\n...\n{% endset %}"

        return [
            Message(
                role="user",
                content=(
                    f"{last_message}.\n\n"
                    f"Intent:\n{request.intent.strip()}."),
            )
        ]

    messages: list[Message] = [
        Message(
            role="user",
            content=(
                f"I need a Jinja2 prompt template for image captioning.\n\n"
                f"Intent:\n{request.intent.strip()}\n\n"
                f"I'll send you {len(request.examples)} example(s) — each as a "
                f"subject + caption + image. Briefly acknowledge each one. "
                f'When I say "now generate", emit the Jinja2 template.'
            ),
        ),
        Message(role="assistant", content="Understood. Send your examples."),
    ]

    n = len(request.examples)
    for i, ex in enumerate(request.examples):
        is_last = i == n - 1
        messages.append(
            Message(
                role="user",
                content=[
                    TextPart(text=f"Example {i + 1}/{n}: {ex.subject}\n{ex.caption}"),
                    ImageUrlPart(image_url=ImageUrl(url=ex.image_data_url, detail="auto")),
                ],
            )
        )
        if not is_last:
            messages.append(Message(role="assistant", content=_EXAMPLE_ACK))

    last_message = "Now generate the Jinja2 template. Output only the template, without any markdown or instructions on how to use the template. "

    match request.focus:
        case "both":
            last_message += "Output both the system_prompt and user_prompt top-level blocks.\n\n"
            last_message += "Use the following format:\n"
            last_message += "{% set system_prompt %}\n...\n{% endset %}\n\n"
            last_message += "{% set user_prompt %}\n...\n{% endset %}"
        case "system":
            last_message += "Output only the system_prompt top-level block."
            last_message += "Use the following format:\n"
            last_message += "{% set system_prompt %}\n...\n{% endset %}"
        case "user":
            last_message += "Output only the user_prompt top-level block."
            last_message += "Use the following format:\n"
            last_message += "{% set user_prompt %}\n...\n{% endset %}"

    messages.append(Message(role="user", content=last_message))

    return messages


_SYSTEM_PROMPT = """\
You are a prompt engineer for an image captioning system. Your job is to
generate a complete Jinja2 prompt template from a stated intent and
few-shot examples (subject + caption + image) sent across multiple turns.

## Jinja2 Template
The template MUST contain exactly two top-level blocks:

    {% set system_prompt %}...{% endset %}
    {% set user_prompt %}...{% endset %}

`system_prompt` is the system message sent to the captioning model.
`user_prompt` is the user message; it should contain the instructions
for the model, optionally referencing variables that the user is
directly referencing.

## Context
The generated template is rendered once per dataset image and feeds a
vision-capable captioning model. At runtime:

- `system_prompt` is sent verbatim as the model's `system` message.
- `user_prompt` is sent as the `user` message *alongside the image*
  (the image is attached as a base64 data URL by the runtime, not by
  the template — so the template just needs to write the instructions
  for the model).
- The model's response text is captured as the caption. Any thinking
  content is stripped, so don't ask for JSON, tags, or other wrappers.

## Jinja variables
The template may contains variables which the user can reference during
the captioning. Refrain from adding your own variables unless explictly
asked by the user.

*Note: the user_prompt contains only text, images are send during the
captioning process as a separate attachment.*

## Rules
- Emit ONLY the Jinja2 template body — no commentary, no markdown fences,
  no explanations, no surrounding prose.
- Always emit the top-level blocks.
- The template is saved verbatim to a `.jinja` file, so every character
  outside the two blocks will be rendered as-is.
- Use Jinja2 syntax (`{% ... %}`, `{{ ... }}`, `{# ... #}`).
- Keep the template robust: prefer `default('...')` filters for any
  optional variables.
- Mirror the tone and structure of the few-shot examples (subject
  phrasing, caption length, level of detail, vocabulary).
"""


__all__ = [
    "ExamplePair",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "PromptGenerationService",
]
