import sys
from typing import Optional, TextIO

import click
import pydantic
import toml

from yadc.captioners.api import APICaptioner, APITypes
from yadc.captioners.api.utils.cache import HTTPResponseCache
from yadc.cmd import app as yadc_app
from yadc.cmd import configs as cmd_configs
from yadc.cmd import envs as cmd_envs
from yadc.cmd import status as cmd_status
from yadc.cmd import templates as cmd_templates
from yadc.core import logging
from yadc.core.captioner import ROLE_ASSISTANT, ROLE_USER, CaptionerRound, ReplyRound
from yadc.core.config import ConfigSettings, parse_config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import reapply_dataset_extras, resolve_dataset
from yadc.core.prediction import PredictionContext

from . import cli_common
from .core import utils

_logger = logging.get_logger(__name__)


# --- Template resolution ---


def _resolve_template(prompt_name: str, prompt_template: str) -> str:
    """Resolve a prompt template through the fallback chain: user → builtin → default."""
    if prompt_template:
        return prompt_template

    for loader, label in [
        (cmd_templates.load_user_template, "user"),
        (cmd_templates.load_builtin_template, "built-in"),
    ]:
        try:
            return loader(prompt_name)
        except Exception:
            _logger.debug("No %s template found: %s", label, prompt_name)

    if prompt_name:
        if user_templates := cmd_templates.list_user_template():
            _logger.error(
                "Error: prompt template could not be loaded: %s; available templates: %s",
                prompt_name,
                ", ".join(user_templates),
            )
        else:
            _logger.error("Error: prompt template could not be loaded: %s", prompt_name)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    _logger.warning("No prompt template defined. Will use the default.")

    try:
        return cmd_templates.default_template()
    except Exception:
        _logger.error("Error: default prompt template could not be loaded")
        sys.exit(cmd_status.STATUS_ERROR)


# --- Dataset loading ---


def _load_dataset(
    dataset_stream: TextIO,
    env: Optional[str],
    user_config: Optional[str],
    user_template: Optional[str],
    api_url: Optional[str],
    api_token: Optional[str],
    api_model_name: Optional[str],
):
    dataset_toml_raw = toml.load(dataset_stream)

    try:
        if user_config is not None:
            _logger.info("Using %s user config.", user_config)
            dataset_toml_raw = cmd_configs.merge_user_config(user_config, dataset_toml_raw)
    except ValueError:
        if user_configs := cmd_configs.list_user_config():
            _logger.error(
                "Error: failed to load user config: %s; available configs: %s",
                user_config,
                ", ".join(user_configs),
            )
        else:
            _logger.error("Error: failed to load user config: %s", user_config)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    # merge with user env
    env = env or dataset_toml_raw.get("env", "default")
    assert isinstance(env, str), "invalid dataset toml env"

    _logger.info("Using %s user environment.", env)
    user_env = cmd_envs.load_env(env=env)

    dataset_toml_raw.setdefault("api", {})
    dataset_toml_raw_api = dataset_toml_raw["api"]
    assert isinstance(dataset_toml_raw_api, dict), "invalid dataset toml api section"

    dataset_toml_raw_api["url"] = api_url or user_env.api.url or dataset_toml_raw_api.get("url", "")
    dataset_toml_raw_api["token"] = api_token or user_env.api.token or dataset_toml_raw_api.get("token", "")
    dataset_toml_raw_api["model_name"] = api_model_name or user_env.api.model_name or dataset_toml_raw_api.get("model_name", "")

    dataset_toml_raw.setdefault("prompt", {})

    if user_template is not None:
        dataset_toml_raw["prompt"]["name"] = user_template
        dataset_toml_raw["prompt"].pop("template", None)

    # parse config with v1/v2 duck-typing
    try:
        dataset_toml = parse_config(dataset_toml_raw)
    except pydantic.ValidationError as e:
        raise ValueError(f"invalid configuration: {e}")

    # resolve dataset entries into images
    dataset_toml._resolved_images = resolve_dataset(  # type: ignore[attr-defined]
        dataset_toml.dataset,
        dataset_toml.caption_suffix,
    )

    return dataset_toml


# --- Interactive prompts ---


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


# --- Captioning ---


def _predict_caption_one_shot(
    model: APICaptioner,
    dataset_image: DatasetImage,
    settings: ConfigSettings,
    do_stream: bool,
    conversation_overrides: dict,
    drafts: dict[str, str] | None = None,
    extra_messages: list[ReplyRound] | None = None,
    prediction_context: PredictionContext | None = None,
) -> str:
    """Single-round caption prediction with streaming output."""
    caption_parts = []

    try:
        with utils.Timer() as timer:
            if do_stream:
                tokens = model.predict_stream(
                    dataset_image,
                    max_new_tokens=settings.max_tokens,
                    conversation_overrides=conversation_overrides,
                    prefill=settings.advanced.assistant_prefill,
                    drafts=drafts,
                    extra_messages=extra_messages,
                    prediction_context=prediction_context,
                )
            else:
                tokens = [
                    model.predict(
                        dataset_image,
                        max_new_tokens=settings.max_tokens,
                        conversation_overrides=conversation_overrides,
                        prefill=settings.advanced.assistant_prefill,
                        drafts=drafts,
                        extra_messages=extra_messages,
                        prediction_context=prediction_context,
                    )
                ]

            for token in tokens:
                caption_parts.append(token)
                click.echo(token, nl=False)
    except ValueError as e:
        _logger.error("Error: %s", e)
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        if do_stream:
            click.echo("")
        raise

    click.echo("")
    _logger.info("Captioning done (%.3f sec)", timer.elapsed)
    _logger.info("")

    return "".join(caption_parts).strip()


def _predict_caption_rounds(
    model: APICaptioner,
    dataset_image: DatasetImage,
    settings: ConfigSettings,
    do_stream: bool,
    conversation_overrides: dict,
    rounds: int,
    caption_rounds: list[CaptionerRound],
    interactive: bool,
    drafts: dict[str, str] | None = None,
) -> str:
    """Multi-round caption prediction with intermediate acceptance prompts."""
    j = 0
    if caption_rounds:
        j = rounds  # reuse accepted rounds, skip to final
    else:
        _logger.info("Doing %d rounds...", rounds)

    try:
        while j < rounds:
            j += 1
            new_caption = _prompt_for_override(f"round #{j}", "", interactive)

            if not new_caption:
                with utils.Timer() as timer_round:
                    new_caption = model.predict(
                        dataset_image,
                        max_new_tokens=settings.max_tokens,
                        use_cache=True,
                        conversation_overrides=conversation_overrides,
                        prefill=settings.advanced.assistant_prefill,
                        drafts=drafts,
                    ).strip()

                if interactive:
                    _logger.info(new_caption)

                _logger.info("Round #%d done. (%.3f sec)", j, timer_round.elapsed)

                if not _prompt_for_yes("Accept caption?", True, interactive):
                    j -= 1
                    caption_rounds.pop()
                    continue

            caption_rounds.append(CaptionerRound(iteration=j, caption=new_caption))

        # final round using accepted caption_rounds
        fresh_image = DatasetImage(path=dataset_image.path)
        caption_parts = []

        _logger.info("")

        predict_kwargs = dict(
            caption_rounds=caption_rounds,
            max_new_tokens=settings.max_tokens,
            conversation_overrides=conversation_overrides,
        )
        if drafts:
            predict_kwargs["drafts"] = drafts

        with utils.Timer() as timer_end_round:
            if do_stream:
                tokens = model.predict_stream(fresh_image, **predict_kwargs)
            else:
                tokens = [model.predict(fresh_image, **predict_kwargs)]

            for token in tokens:
                caption_parts.append(token)
                click.echo(token, nl=False)
    except ValueError as e:
        _logger.error("Error: %s", e)
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        if do_stream:
            click.echo("")
        raise

    click.echo("")
    _logger.info("End round done. (%.3f sec)", timer_end_round.elapsed)
    _logger.info("")

    return "".join(caption_parts).strip()


def _caption(
    dataset: list[DatasetImage],
    model: APICaptioner,
    settings: ConfigSettings,
    do_stream: bool,
    interactive: bool,
    rounds: int,
    save_draft: str = "",
):
    do_quit = False
    do_print_separator = False
    return_code = cmd_status.STATUS_OK

    if settings.advanced.assistant_prefill and model.api_type in (APITypes.GEMINI, APITypes.OPENAI, APITypes.OPENROUTER):
        _logger.warning("Warning: assistant prefill is set, but the API might not support it")

    conversation_overrides = settings.advanced.model_dump()

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

        dataset_image_current = DatasetImage(**dataset_image.model_dump())
        drafts = dataset_image_current.read_all_drafts() or None

        caption = ""
        caption_rounds: list[CaptionerRound] = []
        reply_history: list[ReplyRound] = []
        last_prediction_context: PredictionContext | None = None
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

                    reasoning = last_prediction_context.reasoning if last_prediction_context else None
                    reasoning_encrypted = last_prediction_context.reasoning_encrypted if last_prediction_context else None
                    reply_history.append(
                        ReplyRound(
                            role=ROLE_ASSISTANT,
                            content=caption,
                            reasoning=reasoning,
                            reasoning_encrypted=reasoning_encrypted,
                        )
                    )
                    reply_history.append(
                        ReplyRound(
                            role=ROLE_USER,
                            content=user_message,
                        )
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
                            dataset_image_toml = toml.loads(dataset_image_tmp_edited)
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
                        break

                    caption_rounds = []
                    reply_history = []
                    last_prediction_context = None
                    continue

                case "prompts":
                    system_prompt, user_prompt = model.prompts_from_image(dataset_image_current, drafts=drafts)
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

            try:
                prediction_context = PredictionContext()

                if rounds <= 1:
                    caption = _predict_caption_one_shot(
                        model,
                        dataset_image_current,
                        settings,
                        do_stream,
                        conversation_overrides,
                        drafts=drafts,
                        extra_messages=reply_history or None,
                        prediction_context=prediction_context,
                    )
                else:
                    caption = _predict_caption_rounds(
                        model,
                        dataset_image_current,
                        settings,
                        do_stream,
                        conversation_overrides,
                        rounds,
                        caption_rounds,
                        interactive,
                        drafts=drafts,
                    )

                last_prediction_context = prediction_context
            except (KeyboardInterrupt, click.Abort):
                if not interactive:
                    caption = ""
                    caption_rounds = []
                    do_quit = True
                    do_prompt = False
                    return_code = cmd_status.STATUS_ERROR
                    break

                _logger.info("Cancelled captioning.")
                caption = ""

        if not caption:
            continue

        if save_draft:
            dataset_image.write_draft(save_draft, caption)
            _logger.info("Draft saved as %s.", save_draft)
            _logger.info("")
        else:
            # save current toml history if it hasn't been saved before
            if dataset_image.caption:
                dataset_image.save_history(when_not_exists=True)

            dataset_image_current.update_caption(caption)
            dataset_image_current.save_history(when_not_exists=False)

            _logger.info("")

        caption = ""

    return return_code


# --- CLI entry point ---


@click.command(
    short_help="Caption a dataset",
    help="Caption a dataset. A dataset config is necessary in order to start captioning. See documentation for details: https://github.com/itterative/yadc",
)
@click.argument("dataset", type=click.File("r"))
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
@click.option("--draft", type=str, default=None, required=False, help="Save caption as a named draft instead of the final caption")
@cli_common.log_level
def caption(dataset: TextIO, **kwargs):
    _logger.info("Using python %d.%d.%d.", sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    try:
        dataset_toml = _load_dataset(
            dataset,
            env=kwargs.get("env"),
            user_config=kwargs.get("user_config"),
            user_template=kwargs.get("user_template"),
            api_url=kwargs.get("api_url"),
            api_token=kwargs.get("api_token"),
            api_model_name=kwargs.get("api_model_name"),
        )
    except (AssertionError, ValueError) as e:
        _logger.error("Error loading dataset: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    # resolve CLI overrides with config defaults
    do_stream = kwargs.get("stream") or False
    interactive = kwargs.get("interactive") or dataset_toml.interactive
    rounds = kwargs.get("rounds") or dataset_toml.rounds
    overwrite_captions = kwargs.get("overwrite") or dataset_toml.overwrite_captions
    cache_flag = kwargs.get("cache", True)
    save_draft = kwargs.get("draft") or ""

    # resolve prompt template
    dataset_toml.prompt.template = _resolve_template(dataset_toml.prompt.name, dataset_toml.prompt.template)

    if dataset_toml.prompt.name:
        _logger.info("Using prompt template: %s", dataset_toml.prompt.name)

    resolved_images: list[DatasetImage] = dataset_toml._resolved_images  # type: ignore[attr-defined]

    # filter out already-captioned images
    dataset_to_do: list[DatasetImage] = []
    skipped = 0

    for dataset_image in resolved_images:
        if save_draft:
            if not overwrite_captions and dataset_image.draft_path(save_draft).exists():
                skipped += 1
                continue
        else:
            if not overwrite_captions and dataset_image.caption_path.exists():
                skipped += 1
                continue
        dataset_to_do.append(dataset_image)

    _logger.info("Found %d images.", len(resolved_images))

    if skipped:
        _logger.info("Skipped %d images.", skipped)

    if not dataset_to_do:
        _logger.info("Nothing to do.")
        sys.exit(cmd_status.STATUS_OK)

    cache: HTTPResponseCache | None = None
    if cache_flag:
        cache = HTTPResponseCache(cache_dir=yadc_app.CACHE_PATH / "api_requests")

    _logger.info("Loading model...")

    try:
        model = APICaptioner(
            api_url=dataset_toml.api.url,
            api_token=dataset_toml.api.token,
            prompt_template=dataset_toml.prompt.template,
            store_conversation=dataset_toml.settings.store_conversation,
            image_quality=dataset_toml.settings.image_quality,
            reasoning=dataset_toml.reasoning.enable,
            reasoning_effort=dataset_toml.reasoning.thinking_effort,
            reasoning_exclude_output=dataset_toml.reasoning.exclude_from_output,
            reasoning_start_token=dataset_toml.reasoning.advanced.thinking_start,
            reasoning_end_token=dataset_toml.reasoning.advanced.thinking_end,
            cache=cache,
        )
        model.load_model(dataset_toml.api.model_name)
    except ValueError as e:
        _logger.error("Error: failed to load model: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    _logger.info("")
    _logger.info("Captioning...")

    if save_draft:
        _logger.info("Saving captions as draft: %s", save_draft)

    with utils.Timer() as timer:
        return_code = _caption(
            dataset=dataset_to_do,
            model=model,
            settings=dataset_toml.settings,
            do_stream=do_stream,
            interactive=interactive,
            rounds=rounds,
            save_draft=save_draft,
        )

    model.log_usage()
    _logger.info("Done. (%.1f sec)", timer.elapsed)

    sys.exit(return_code)
