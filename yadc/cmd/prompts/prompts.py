"""Pure logic for ``yadc prompts`` — prompt template generation via LLM.

No click imports. The CLI layer (`cli_prompts.py`) wraps this module.
The actual env-load → validate → client-build → stream → cleanup
wiring lives in :mod:`yadc.prompt_generation.streaming`; this module
adds a CLI-friendly ``generate()`` wrapper that

- accepts an ``on_chunk`` callback (so the CLI owns stdout vs. a
  library consumer can route chunks elsewhere)
- buffers reasoning across the stream and emits it once via the
  ``on_reasoning`` callback (or the CLI's logger), instead of letting
  reasoning leak through the chunk callback alongside template text
- flushes any buffered reasoning in ``finally`` so it isn't silently
  dropped when the stream raises mid-flight

Raises typed exceptions (:class:`PromptGenerationConfigError`,
:class:`PasswordRequiredError` from ``cmd_envs.load_env``) instead of
calling ``sys.exit`` — the CLI layer maps them to exit codes.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from yadc.cmd import templates as cmd_templates
from yadc.core import logging
from yadc.core.constants import IMAGE_EXTENSIONS, SIDECAR_EXTENSIONS
from yadc.prompt_generation import (
    ExamplePair,
    PromptGenerationConfigError,
    PromptGenerationRequest,
    stream_template_chunks,
)

_logger = logging.get_logger(__name__)


def _resolve_password(explicit: str | None) -> str | None:
    """Resolve password: explicit --password, then ``YADC_PASSWORD`` env var.

    Reads ``os.environ`` directly (rather than the
    :data:`yadc.core.env.YADC_PASSWORD` module constant) so callers
    that set the env var after import — including tests using
    ``monkeypatch.setenv`` / ``patch.dict(os.environ, ...)`` — see
    the current value.
    """
    if explicit:
        return explicit
    return os.environ.get("YADC_PASSWORD") or None


def _resolve_refine_target(refine: str) -> str:
    """Resolve a ``--refine`` target to template content.

    Accepts either a path to an existing template file or a user
    template name (looked up via ``cmd_templates.load_user_template``,
    matching the bare-name convention used by the ``templates`` CLI).
    A file path takes precedence — if the arg matches an existing
    file we read it directly without checking the template store.

    Raises ``FileNotFoundError`` with a clear, actionable message if
    neither a file nor a known template matches.
    """
    refine_path = Path(refine)
    if refine_path.is_file():
        return refine_path.read_text(encoding="utf-8")

    try:
        return cmd_templates.load_user_template(refine)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"refine target not found: '{refine}' is neither a file path nor a known user template (see `yadc templates list`)") from e


def _pair_with_sidecar(image_path: Path) -> ExamplePair:
    """Build an :class:`ExamplePair` from an image file + its ``.txt`` sidecar.

    ``subject`` is the filename stem (matches the webui's
    ``ExamplesPanel.svelte`` seed), ``caption`` is the trimmed
    sidecar contents, ``image_data_url`` is the base64-encoded image
    as a data URL. Raises ``FileNotFoundError`` if the sidecar is
    missing, ``ValueError`` if the sidecar is empty or the MIME type
    can't be determined.
    """
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.exists():
        raise FileNotFoundError(f"caption sidecar not found for '{image_path.name}': expected '{sidecar.name}'")

    caption = sidecar.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"caption sidecar is empty: '{sidecar}'")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"could not determine image MIME type for: '{image_path}' (supported extensions: {sorted(IMAGE_EXTENSIONS)})")

    image_bytes = image_path.read_bytes()
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    return ExamplePair(
        subject=image_path.stem,
        caption=caption,
        image_data_url=data_url,
    )


def resolve_examples_targets(targets: tuple[str, ...]) -> tuple[list[ExamplePair], list[str]]:
    """Resolve positional example paths to ``(examples, skipped)``.

    A usable target is an image file (supported extension, exists)
    paired with its ``<stem>.txt`` sidecar. Files whose extension is
    in :data:`SIDECAR_EXTENSIONS` (``.txt`` caption sidecars, ``.toml``
    configs, …) are dropped silently — a broad glob sweeps them in
    alongside their image and they aren't user errors. Other
    non-images and missing paths (e.g. an unexpanded ``*.png`` glob)
    go in ``skipped`` for the caller to warn about. A real image
    whose sidecar is missing or empty still raises.

    Raises ``ValueError`` if ``targets`` is non-empty but resolved to
    no images — passing examples and getting zero is a user error.
    """
    examples: list[ExamplePair] = []
    skipped: list[str] = []
    for target in targets:
        path = Path(target)
        suffix = path.suffix.lower()
        if suffix in SIDECAR_EXTENSIONS:
            continue
        if suffix not in IMAGE_EXTENSIONS or not path.is_file():
            skipped.append(target)
            continue
        examples.append(_pair_with_sidecar(path))

    if targets and not examples:
        raise ValueError(f"no valid example images found among {len(targets)} argument(s) provided")
    return examples, skipped


async def generate(
    *,
    env: str,
    intent: str,
    focus: Literal["system", "user", "both"] = "both",
    api_model_name: str | None = None,
    max_tokens: int = 16384,
    image_quality: Literal["auto", "low", "high"] = "auto",
    refine: str | None = None,
    examples: list[ExamplePair] | tuple[ExamplePair, ...] | None = None,
    password: str | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> str:
    """Generate a Jinja2 prompt template via the named env's model.

    ``on_chunk`` receives each text fragment as it streams. ``on_reasoning``
    receives the buffered reasoning once (after any text fragment
    arrives). Both default to no-ops; the CLI passes functions that
    write to ``sys.stdout`` / ``_logger.info`` respectively.

    Returns the accumulated template body on success. Raises
    :class:`PromptGenerationConfigError` for missing config fields,
    :class:`PasswordRequiredError` for password-protected envs, and
    any exception from the underlying LLM client — the CLI catches
    these and maps them to exit codes.
    """
    template_content: str | None = None
    if refine is not None:
        template_content = _resolve_refine_target(refine)

    password = _resolve_password(password)

    request = PromptGenerationRequest(
        env=env,
        intent=intent,
        focus=focus,
        api_model_name=api_model_name,
        max_tokens=max_tokens,
        image_quality=image_quality,
        template_content=template_content,
        examples=list(examples) if examples else [],
    )

    _logger.info(
        "Generating prompt template via env '%s' (focus=%s).",
        env,
        focus,
    )

    def _noop(_text: str) -> None:
        return None

    sink = on_chunk or _noop
    reasoning_sink = on_reasoning or _noop
    accumulated: list[str] = []
    reasoning_buffer: list[str] = []
    reasoning_emitted = False

    def _emit_reasoning() -> None:
        nonlocal reasoning_emitted
        if reasoning_emitted or not reasoning_buffer:
            return
        reasoning_sink("".join(reasoning_buffer))
        reasoning_emitted = True

    try:
        async for chunk in stream_template_chunks(request, password=password):
            if chunk.reasoning:
                reasoning_buffer.append(chunk.reasoning)
            if chunk.reasoning_summary:
                reasoning_buffer.append(chunk.reasoning_summary)
            if chunk.text:
                _emit_reasoning()
                sink(chunk.text)
                accumulated.append(chunk.text)
    except PromptGenerationConfigError:
        # Config errors are already user-actionable messages; let the
        # CLI log them with context.
        raise
    finally:
        # Flush any pending reasoning so it's not silently dropped when
        # the stream raises mid-flight or completes without text.
        if not reasoning_emitted and reasoning_buffer:
            reasoning_sink("".join(reasoning_buffer))

    return "".join(accumulated)
