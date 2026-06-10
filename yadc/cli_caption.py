"""Caption command — caption a dataset using a remote vision-capable AI.

The captioning loop (model-create / stream / save) lives in
``yadc.core.captioning.CaptioningRunner``. This module builds a
runner from the click args, supplies click-emitting callbacks, and
implements the interactive action menu on top.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from yadc.captioners.api import APITypes
from yadc.captioners.api.utils.cache import HTTPResponseCache
from yadc.captioners.api.utils.response_logger import ResponseLogger
from yadc.cmd import cache as cmd_cache
from yadc.cmd import configs as cmd_configs
from yadc.cmd import status as cmd_status
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core import logging
from yadc.core.captioner import (
    ROLE_ASSISTANT,
    ROLE_USER,
    CaptionerRound,
    PromptRenderer,
    ReplyRound,
)
from yadc.core.captioning import (
    CaptioningRunner,
    CaptionJobOptions,
    load_dataset_config,
)
from yadc.core.config import Config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import reapply_dataset_extras
from yadc.core.env import DEBUG_CAPTION_REQUESTS_BODY, DEBUG_CAPTION_RESPONSES, YADC_PASSWORD
from yadc.core.prediction import PredictionContext
from yadc.utils.dict_utils import load_toml, load_toml_file

from . import cli_common
from .core import utils

_logger = logging.get_logger(__name__)


# --- CLI callbacks implementing the CaptioningCallbacks Protocol ---


class CLICallbacks:
    """Click-emitting implementation of :class:`CaptioningCallbacks`.

    Each ``on_token`` call is routed to ``click.echo`` (streaming) or
    buffered for end-of-call output (no-stream). Per-image lifecycle
    events are logged but don't print directly — the outer action loop
    prints the image metadata before invoking the runner.
    """

    def __init__(self, do_stream: bool):
        self._do_stream: bool = do_stream
        self._buffer: list[str] = []

    async def on_token(self, token: str) -> None:
        if self._do_stream:
            click.echo(token, nl=False)
        else:
            self._buffer.append(token)

    async def on_image_started(self, image: DatasetImage) -> None:  # pyright: ignore[reportUnusedParameter]
        # No-op: the outer action loop prints metadata before invoking the runner.
        pass

    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None:  # pyright: ignore[reportUnusedParameter]
        # Not called for ``caption_image_dry_run``; the outer loop logs per-image timing.
        pass

    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None:  # pyright: ignore[reportUnusedParameter]
        _logger.warning("Failed to caption %s: %s", image.path, error)

    async def on_before_save(self, paths: list[str]) -> None:  # pyright: ignore[reportUnusedParameter]
        pass


class CLIPrintCallbacks:
    """Minimal callbacks for the non-interactive parallel captioning path.

    Used by ``_caption_async`` when ``--max-concurrent > 1`` is passed
    with ``--non-interactive``. Per-token output is suppressed — it
    would interleave across images and be unreadable. Per-image
    completion (with duration) and per-image errors are logged; the
    saved caption is on disk and can be reviewed there.
    """

    async def on_token(self, token: str) -> None:  # pyright: ignore[reportUnusedParameter]
        # Suppressed: in parallel mode tokens from N images would
        # interleave and produce unreadable output.
        pass

    async def on_image_started(self, image: DatasetImage) -> None:  # pyright: ignore[reportUnusedParameter]
        # No per-image output in parallel mode; the per-image summary
        # in on_image_captioned is enough.
        pass

    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None:
        _logger.info("Captioned %s (%.1f sec)", image.path, duration_ms / 1000)

    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None:  # pyright: ignore[reportUnusedParameter]
        _logger.warning("Failed to caption %s: %s", image.path, error)

    async def on_before_save(self, paths: list[str]) -> None:  # pyright: ignore[reportUnusedParameter]
        pass


# --- Interactive prompts ---


def _resolve_initial_password(explicit: str | None) -> str | None:
    """Resolve the initial password for ``CaptionJobOptions``.

    Order: explicit ``--password`` value, then ``YADC_PASSWORD`` env
    var, then ``None`` (the loader will surface a ``PasswordRequiredError``
    and the caller will prompt the user). The interactive prompt is
    deferred until we know the env actually needs a password — that
    way, non-password envs don't trigger an unnecessary prompt.
    """
    if explicit:
        return explicit
    return YADC_PASSWORD


def _prompt_for_password(had_previous: bool) -> str | None:
    """Prompt the user for a password if stdin is a TTY.

    Returns ``None`` when stdin isn't a TTY (piped/redirected) — the
    caller is expected to fail with a clear error in that case rather
    than blocking on a prompt that nobody can answer. *had_previous*
    controls the prompt text so the user knows whether their prior
    password was wrong or they simply didn't supply one.
    """
    if not _is_tty():
        return None
    prompt = (
        "Enter password to decrypt environment settings (previous password was incorrect): "
        if had_previous
        else "Enter password to decrypt environment settings: "
    )
    return click.prompt(prompt, hide_input=True, default="", show_default=False) or None


def _is_tty() -> bool:
    """Whether stdin is a TTY.

    Extracted so tests can patch it via ``monkeypatch.setattr``. Click's
    ``CliRunner`` does not call ``sys.stdin.isatty()`` through the same
    object the test sees, so we use a thin module-level helper that's
    trivial to swap.
    """
    return sys.stdin.isatty()


def _prompt_for_yes(prompt: str, default: bool, interactive: bool) -> bool:
    if not interactive:
        return default
    return click.confirm(prompt, default=default)


def _prompt_for_override(value: str, default: str, interactive: bool) -> str:
    if not interactive:
        return default
    response: str = click.prompt(
        f"Override {value}? ({default}) " if default else f"Override {value}? ",
        show_default=False,
        default=default,
    )
    return response or default


def _prompt_for_action(prompt: str, actions: dict[str, str], default_action: str, interactive: bool) -> str:
    assert default_action in actions
    if not interactive:
        return actions[default_action]
    prompt_str = f"{prompt} ({' '.join(f'{k}={v}' for k, v in actions.items())}) [{actions[default_action]}] "
    click.echo(prompt_str, nl=False)
    response = None
    while response not in actions:
        response = click.getchar() or default_action
    click.echo(response)
    return actions[response]


def _print_dataset_image_meta(dataset_image: DatasetImage):
    click.echo(f"Path: {dataset_image.path}")
    for key, value in (dataset_image.__pydantic_extra__ or {}).items():
        _logger.info("%s: %s", key.capitalize(), value)
    if drafts := dataset_image.read_all_drafts():
        for name, content in drafts.items():
            _logger.info("Draft (%s): %s", name, content[:200] + "..." if len(content) > 200 else content)
    if caption := dataset_image.read_caption():
        _logger.info("Caption:")
        _logger.info(caption)
        _logger.info("------------")
        _logger.info("")


# --- Captioning (interactive loop) ---


async def _stream_one_round(
    runner: CaptioningRunner,
    image: DatasetImage,
    *,
    caption_rounds: list[CaptionerRound] | None = None,
    extra_messages: list[ReplyRound] | None = None,
    prediction_context: PredictionContext | None = None,
) -> tuple[str, float]:
    """Stream one caption, return ``(caption_text, elapsed_seconds)``.

    Wraps :meth:`CaptioningRunner.caption_image_dry_run` with the
    per-image timer the CLI has always emitted.
    """
    with utils.Timer() as timer:
        caption = await runner.caption_image_dry_run(
            image,
            caption_rounds=caption_rounds,
            extra_messages=extra_messages,
            prediction_context=prediction_context,
        )
    click.echo("")  # newline after streaming
    return caption, timer.elapsed


async def _caption(
    runner: CaptioningRunner,
    dataset: list[DatasetImage],
    config: Config,
    interactive: bool,
    rounds: int,
    save_draft: str = "",
):
    do_quit = False
    do_print_separator = False
    return_code = cmd_status.STATUS_OK

    if config.settings.advanced.assistant_prefill and runner.model.api_type in (APITypes.GEMINI, APITypes.OPENAI, APITypes.OPENROUTER):
        _logger.warning("Warning: assistant prefill is set, but the API might not support it")

    # The "prompts" action renders the same Jinja2 template the model
    # would render, so we don't need to invoke the model for it.
    renderer = PromptRenderer(prompt_template=config.prompt.template)

    for dataset_image in dataset:
        if do_quit:
            break

        if do_print_separator:
            _logger.info("")
            _logger.info("------------")
            _logger.info("")
        else:
            do_print_separator = True

        _print_dataset_image_meta(dataset_image)

        dataset_image_current = DatasetImage.model_validate(dataset_image.model_dump())
        drafts = dataset_image_current.read_all_drafts() or None

        caption = ""
        caption_rounds: list[CaptionerRound] = []
        reply_history: list[ReplyRound] = []
        last_prediction_context: PredictionContext | None = None
        pending_reply: tuple[ReplyRound, ReplyRound] | None = None
        do_prompt = True

        while do_prompt:
            return_code = cmd_status.STATUS_OK

            try:
                action = _prompt_for_action(
                    "Next action",
                    dict(q="quit", s="skip", c="continue", r="retry", e="edit", p="prompts", y="reply", x="clear replies"),
                    "c",
                    interactive,
                )
            except (KeyboardInterrupt, click.Abort):
                click.echo("")
                action = "quit"

            match action:
                case "quit":
                    caption = ""
                    do_prompt = False
                    do_quit = True
                    break

                case "skip":
                    caption = ""
                    do_prompt = False
                    break

                case "continue":
                    do_prompt = not caption

                case "retry":
                    pass

                case "clear replies":
                    if not reply_history:
                        _logger.info("No reply history to clear.")
                        continue
                    reply_history = []
                    last_prediction_context = None

                case "reply":
                    if not caption:
                        _logger.info("No caption to reply to. Generate one first.")
                        continue

                    user_message = click.prompt("Reply", default="", show_default=False)
                    if not user_message:
                        continue

                    pending_reply = (
                        ReplyRound(
                            role=ROLE_ASSISTANT,
                            content=caption,
                            reasoning=last_prediction_context.reasoning if last_prediction_context else None,
                            reasoning_encrypted=last_prediction_context.reasoning_encrypted if last_prediction_context else None,
                        ),
                        ReplyRound(
                            role=ROLE_USER,
                            content=user_message,
                        ),
                    )
                    pass

                case "edit":
                    dataset_image_tmp_edited = None
                    while True:
                        dataset_image_tmp_edited = click.edit(
                            dataset_image_tmp_edited or dataset_image_current.dump_toml(),
                            extension=".toml",
                            require_save=True,
                        )

                        if dataset_image_tmp_edited is None:
                            _logger.info("Dataset image toml editing was cancelled.")
                            break

                        try:
                            dataset_image_toml = load_toml(dataset_image_tmp_edited)
                            dataset_image_current = DatasetImage(
                                path=dataset_image_current.path,
                                caption=dataset_image_current.caption,
                                caption_suffix=dataset_image_current.caption_suffix,
                                toml_suffix=dataset_image_current.toml_suffix,
                                history_suffix=dataset_image_current.history_suffix,
                                **dataset_image_toml,
                            )
                            reapply_dataset_extras(dataset_image_current)
                        except Exception:
                            _logger.warning("Warning: toml is not valid")
                            if not _prompt_for_yes("Retry?", True, interactive):
                                break
                            continue

                        with open(dataset_image_current.toml_path, "w") as f:
                            f.write(dataset_image_current.dump_toml())

                        _logger.info("Dataset image toml was updated.")
                    caption_rounds = []
                    reply_history = []
                    last_prediction_context = None
                    continue

                case "prompts":
                    system_prompt, user_prompt = renderer.render(dataset_image_current, drafts=drafts)
                    _logger.info("SYSTEM PROMPT")
                    _logger.info(system_prompt)
                    _logger.info("")
                    _logger.info("USER PROMPT")
                    _logger.info(user_prompt)
                    continue

                case _:
                    raise AssertionError(f"bad action: {action}")

            if not do_prompt and caption:
                break

            previous_caption = caption

            extra_messages: list[ReplyRound] = []
            if reply_history:
                extra_messages.extend(reply_history)
            if pending_reply:
                extra_messages.extend(pending_reply)

            try:
                prediction_context = PredictionContext()

                if rounds <= 1 or extra_messages:
                    caption, elapsed = await _stream_one_round(
                        runner,
                        dataset_image_current,
                        extra_messages=extra_messages,
                        prediction_context=prediction_context,
                    )
                    _logger.info("Captioning done (%.3f sec)", elapsed)
                    _logger.info("")
                else:
                    j = 0
                    if caption_rounds:
                        j = rounds  # reuse accepted rounds, skip to final
                    else:
                        _logger.info("Doing %d rounds...", rounds)

                    while j < rounds:
                        j += 1
                        new_caption = _prompt_for_override(f"round #{j}", "", interactive)

                        if not new_caption:
                            new_caption, elapsed = await _stream_one_round(
                                runner,
                                dataset_image_current,
                                caption_rounds=None,
                                prediction_context=PredictionContext(),
                            )
                            if interactive:
                                _logger.info(new_caption)

                            _logger.info("Round #%d done. (%.3f sec)", j, elapsed)

                            if not _prompt_for_yes("Accept caption?", True, interactive):
                                j -= 1
                                if caption_rounds:
                                    caption_rounds.pop()
                                continue

                        caption_rounds.append(CaptionerRound(iteration=j, caption=new_caption))

                    _logger.info("")
                    caption, elapsed = await _stream_one_round(
                        runner,
                        dataset_image_current,
                        caption_rounds=caption_rounds,
                        prediction_context=PredictionContext(),
                    )
                    _logger.info("End round done. (%.3f sec)", elapsed)
                    _logger.info("")

                last_prediction_context = prediction_context

                if pending_reply is not None:
                    reply_history.extend(pending_reply)
                    pending_reply = None
            except (KeyboardInterrupt, click.Abort):
                if not interactive:
                    caption = ""
                    caption_rounds = []
                    do_quit = True
                    do_prompt = False
                    return_code = cmd_status.STATUS_ERROR
                    break

                _logger.info("Cancelled captioning.")
                caption = previous_caption

        if not caption:
            continue

        # Commit the accepted caption. The dry-run already streamed it;
        # save_caption writes the files (caption + TOML + history, or
        # the draft file in draft mode) via the runner. We pass
        # ``dataset_image_current`` (not the original) so any TOML
        # edits made via the "edit" action are preserved in the
        # re-serialized sidecar.
        try:
            await runner.save_caption(dataset_image_current, caption)
        except Exception as e:
            _logger.warning("Failed to save caption for %s: %s", dataset_image.path, e)
            continue

        if save_draft:
            _logger.info("Draft saved as %s.", save_draft)
            _logger.info("")
        else:
            _logger.info("")

        caption = ""

    return return_code


# --- CLI entry point ---


@click.command(
    short_help="Caption a dataset",
    help="Caption a dataset. A dataset config is necessary in order to start captioning. See documentation for details: https://github.com/itterative/yadc",
)
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--env", type=str, default=None, help="Configuration environment")
@click.option("--api-url", type=str, default=None, help="Override API url")
@click.option("--api-token", type=str, default=None, help="Override API auth token")
@click.option("--api-model-name", type=str, default=None, help="Override API model")
@click.option("--user-config", type=str, default=None, help="Base user config")
@click.option("--user-template", type=str, default=None, help="Override user template")
@click.option("--stream/--no-stream", is_flag=True, default=None, help="Enable the streaming of captions")
@click.option("--interactive/--non-interactive", "interactive", is_flag=True, default=None, help="Enable interactive mode")
@click.option("--overwrite/--no-overwrite", "overwrite", is_flag=True, default=None, help="Overwrite existing caption")
@click.option("--cache/--no-cache", "cache", is_flag=True, default=True, help="Cache API requests")
@click.option("--rounds", type=click.IntRange(min=1, max_open=True), default=None, required=False, help="How many captioning rounds to do")
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=None,
    show_default="from env or 1",
    required=False,
    help=(
        "How many captioning requests can be in flight at once. Defaults to the selected env's "
        "max_concurrent, or 1 if unset. Only valid with --non-interactive."
    ),
)
@click.option("--draft", type=str, default=None, required=False, help="Save caption as a named draft instead of the final caption")
@click.option(
    "--password",
    "password",
    type=str,
    default=None,
    help=("Password for decrypting password-mode env settings. Defaults to the YADC_PASSWORD env var. Prompts interactively (TTY) if neither is set."),
)
@cli_common.log_level
def caption(dataset: str, **kwargs: Any):
    return asyncio.run(_caption_async(dataset, **kwargs))


def _resolve_config_defaults(dataset_path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve click kwargs against the config's top-level defaults.

    The loader needs a fully-resolved ``CaptionJobOptions`` (the
    overwrite/draft filter reads ``options.overwrite``). For values
    the user didn't pass on the command line, fall back to the
    config's top-level field.
    """
    raw = load_toml_file(dataset_path.open())
    return {
        "overwrite": kwargs["overwrite"] if kwargs.get("overwrite") is not None else raw.get("overwrite_captions", True),
        "rounds": kwargs["rounds"] if kwargs.get("rounds") is not None else raw.get("rounds", 1),
        "interactive": kwargs["interactive"] if kwargs.get("interactive") is not None else raw.get("interactive", False),
    }


async def _caption_async(dataset: str, **kwargs: Any):
    _logger.info("Using python %d.%d.%d.", sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    dataset_path = Path(dataset)

    # Resolve top-level defaults from the config so the loader's
    # overwrite/draft filter sees the user's intent.
    defaults = _resolve_config_defaults(dataset_path, kwargs)

    user_config = kwargs.get("user_config")
    if user_config is not None:
        _logger.info("Using %s user config.", user_config)

    options = CaptionJobOptions(
        api_url=kwargs.get("api_url") or "",
        api_token=kwargs.get("api_token") or "",
        api_model_name=kwargs.get("api_model_name") or "",
        env=kwargs.get("env") or "default",
        prompt_name=kwargs.get("user_template") or "",
        overwrite=defaults["overwrite"],
        rounds=defaults["rounds"],
        # ``None`` lets the loader resolve to the env's max_concurrent
        # (if any) and finally to 1. See ``apply_config_overrides``.
        max_concurrent=kwargs.get("max_concurrent"),
        draft=kwargs.get("draft") or "",
        # Explicit ``--password`` wins; fall back to ``YADC_PASSWORD``
        # so scripts can set the env var without touching the CLI.
        password=_resolve_initial_password(kwargs.get("password")),
    )

    try:
        config, dataset_to_do = load_dataset_config(
            dataset_path,
            options,
            user_config=user_config,
        )
    except PasswordRequiredError:
        # The env's token is password-encrypted. We either had no
        # password at all, or the one we had was wrong. Try to prompt
        # the user for a fresh password (TTY only) and retry once;
        # outside a TTY we fail with a clear actionable error.
        had_password = options.password is not None
        password = _prompt_for_password(had_password)
        if password is None:
            if had_password:
                _logger.error("Error: password is incorrect.")
            else:
                _logger.error(
                    "Error: environment is password-encrypted but no password was provided. "
                    "Set YADC_PASSWORD, pass --password, or run in a terminal for an interactive prompt."
                )
            sys.exit(cmd_status.STATUS_USER_ERROR)
        options = options.model_copy(update={"password": password})
        try:
            config, dataset_to_do = load_dataset_config(
                dataset_path,
                options,
                user_config=user_config,
            )
        except PasswordRequiredError:
            _logger.error("Error: password is incorrect.")
            sys.exit(cmd_status.STATUS_USER_ERROR)
    except FileNotFoundError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)
    except ValueError as e:
        if user_config is not None:
            if user_configs := cmd_configs.list_user_config():
                _logger.error(
                    "Error: failed to load user config: %s; available configs: %s",
                    user_config,
                    ", ".join(user_configs),
                )
            else:
                _logger.error("Error: failed to load user config: %s", user_config)
            sys.exit(cmd_status.STATUS_USER_ERROR)
        _logger.error("Error loading dataset: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    do_stream = kwargs.get("stream") or False
    interactive = defaults["interactive"]

    if config.prompt.name:
        _logger.info("Using prompt template: %s", config.prompt.name)

    _logger.info("Found %d images.", len(dataset_to_do))
    if not dataset_to_do:
        _logger.info("Nothing to do.")
        sys.exit(cmd_status.STATUS_OK)

    # Interactive mode + parallel captioning is not supported — the
    # interactive flow reviews one caption at a time and there is no
    # natural way to interleave N parallel reviews. Users who want
    # parallelism should use --non-interactive. ``max_concurrent`` may
    # be ``None`` here if the loader was mocked in tests; the
    # truthy guard avoids a TypeError on ``None > 1``.
    if interactive and options.max_concurrent and options.max_concurrent > 1:
        _logger.error("Error: --max-concurrent > 1 is not supported in interactive mode; use --non-interactive.")
        sys.exit(cmd_status.STATUS_ERROR)

    cache: HTTPResponseCache | None = None
    if kwargs.get("cache", True):
        cache = HTTPResponseCache(cache_dir=cmd_cache.api_requests_cache_dir())

    response_logger: ResponseLogger | None = None
    if DEBUG_CAPTION_RESPONSES:
        log_dir = cmd_cache.debug_log_dir(str(dataset_path))
        response_logger = ResponseLogger(log_dir, log_body=DEBUG_CAPTION_REQUESTS_BODY)
        _logger.info("API response debug logging enabled: %s", response_logger.log_dir)

    _logger.info("Loading model...")

    # Callbacks differ between interactive and non-interactive parallel
    # mode. Choose early so the runner constructor gets the right one.
    max_concurrent = options.max_concurrent or 1
    if not interactive and max_concurrent > 1:
        callbacks = CLIPrintCallbacks()
    else:
        callbacks = CLICallbacks(do_stream=do_stream)

    async with CaptioningRunner(
        config,
        options,
        callbacks,
        cache=cache,
        response_logger=response_logger,
    ) as runner:
        _logger.info("")
        _logger.info("Captioning...")

        save_draft = options.draft
        if save_draft:
            _logger.info("Saving captions as draft: %s", save_draft)

        with utils.Timer() as timer:
            # ``options.max_concurrent`` is ``None`` for the CLI default
            # but the loader (``apply_config_overrides``) resolves it to
            # an ``int`` before we get here.
            if not interactive and max_concurrent > 1:
                # Non-interactive parallel path: skip the action menu
                # and let the runner stream + save all images
                # concurrently. Per-token output is suppressed (would
                # interleave across images); per-image completion and
                # errors are logged via CLIPrintCallbacks.
                _logger.info("Running with up to %d concurrent requests.", max_concurrent)
                await runner.caption_images(
                    dataset_to_do,
                    max_concurrent=max_concurrent,
                )
                return_code = cmd_status.STATUS_OK
            else:
                return_code = await _caption(
                    runner=runner,
                    dataset=dataset_to_do,
                    config=config,
                    interactive=interactive,
                    rounds=options.rounds,
                    save_draft=save_draft,
                )

    # Runner's __aexit__ already called model.log_usage() and aclose.
    _logger.info("Done. (%.1f sec)", timer.elapsed)

    sys.exit(return_code)
