"""Meta-conversation construction for the prompt-generation LLM call.

Owns the system prompts + the multi-turn scaffolding that primes the
LLM to acknowledge examples briefly before emitting the template
body. ``_build_messages`` is the single source of truth for the
generate vs. refine branching and the 0-vs-N example format.

System prompts are loaded from ``.txt`` files in this package
(``generate.txt`` / ``refine.txt``) at import time. ``stream_template_chunks``
in :mod:`yadc.prompt_generation.streaming` is the only consumer of the
``_build_messages`` helper.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

from yadc.llm import ImageUrl, ImageUrlPart, Message, TextPart

from .models import PromptGenerationFocus, PromptGenerationRequest


def _load_prompt(prompt: str) -> str:
    path = files("yadc.prompt_generation") / "prompts" / prompt
    return path.read_text(encoding="utf-8").strip()


# Public system-prompt constants — exposed via ``__init__.py`` so tests
# can pin the constant-to-constant equality that caught the original
# "dead system prompt" bug (the old ``_SYSTEM_PROMPT`` was never sent
# to the LLM until Phase 5b fixed the wire-up).
_GENERATE_SYSTEM_PROMPT: str = _load_prompt("generate.txt")
_REFINE_SYSTEM_PROMPT: str = _load_prompt("refine.txt")


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
    "_GENERATE_SYSTEM_PROMPT",
    "_REFINE_SYSTEM_PROMPT",
    "_build_messages",
]
