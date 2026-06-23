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


# Brief ack sent after each non-last example (as a FAKE assistant turn,
# not a real LLM response). Demonstrates the "acknowledge briefly, wait
# for the next" pattern the model mirrors before the go-ahead triggers
# generation.
_EXAMPLE_ACK = "Got it. Send the next example."


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

    The intent is always its own first user message. The conversation
    then walks the optional pieces (existing template in refine mode,
    then examples), each acknowledged by a brief fake assistant turn,
    and ends with a single go-ahead (``last_message``) asking the model
    to emit the template. The intent ack only establishes the go-ahead
    trigger — it deliberately doesn't reference the template or
    examples, so the assistant isn't scripted to know what (if
    anything) arrives before the go-ahead. The model follows the
    demonstrated acknowledgement pattern and emits the template on the
    go-ahead.

    Generate mode with N examples::

        System:    <generate_system_prompt>
        User:      "Intent: <intent>"
        Assistant: "Understood. I'm waiting for your go-ahead."
        User:      example 1 + image
        Assistant: "Got it. Send the next example."
        ...
        User:      example N + image
        User:      "Now generate the Jinja2 template."   (last_message)

    Refine mode with N examples::

        System:    <refine_system_prompt>
        User:      "Intent: <intent>"
        Assistant: "Understood. I'm waiting for your go-ahead."
        User:      <existing template>
        Assistant: "Got it."
        User:      example 1 + image
        Assistant: "Got it. Send the next example."
        ...
        User:      example N + image
        User:      "Now generate the refined Jinja2 template."  (last_message)

    With 0 examples the example turns are skipped; the go-ahead
    (``last_message``) follows the intent ack directly (generate) or
    the template ack (refine). The intent ack is identical in every
    case — the assistant isn't told what context (if any) will arrive
    before the go-ahead.
    """
    is_refine = request.template_content is not None
    n = len(request.examples)
    messages: list[Message] = [Message(role="system", content=refine_system_prompt if is_refine else system_prompt)]

    intent = request.intent.strip() or "(no specific intent)"

    # Intent: first user message, on its own. The assistant can't know
    # yet whether a template or examples will follow, so this ack only
    # establishes the go-ahead trigger — it reads like a natural chat
    # rather than a scripted setup that already knows the turn count.
    messages.append(Message(role="user", content=f"Intent: {intent}"))
    messages.append(Message(role="assistant", content="Understood. I'm waiting for your go-ahead."))

    # Existing template (refine mode): its own user turn + a brief ack.
    if is_refine:
        messages.append(Message(role="user", content=f"My current template (refine it per the intent):\n\n{request.template_content}"))
        messages.append(Message(role="assistant", content="Got it."))

    # Examples: each a multimodal user turn; non-last ones get an ack.
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

    # Go-ahead — the single "emit the template" message, identical in
    # wording across 0/N examples (only "generate" vs "refined" differs).
    template_word = "refined Jinja2 template" if is_refine else "Jinja2 template"
    last_message = (
        f"Now generate the {template_word}. "
        f"Output only the template, without any markdown or instructions on how to use the template. "
        f"{_format_hints(request.focus)}"
    )
    messages.append(Message(role="user", content=last_message))

    return messages


def _format_hints(focus: PromptGenerationFocus) -> str:
    """The Jinja2 block format hint appended to the final go-ahead user turn.

    The same wording reaches the model regardless of example count,
    since every path now ends with ``last_message``.
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
            return "Output only the system_prompt top-level block. Use the following format:\n{% set system_prompt %}\n...\n{% endset %}"
        case "user":
            return "Output only the user_prompt top-level block. Use the following format:\n{% set user_prompt %}\n...\n{% endset %}"


__all__ = [
    "_GENERATE_SYSTEM_PROMPT",
    "_REFINE_SYSTEM_PROMPT",
    "_build_messages",
]
