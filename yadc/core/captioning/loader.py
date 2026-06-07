"""Config loading + override application + template resolution.

``load_dataset_config`` is the canonical entry point for loading a
dataset config, applying overrides, parsing the result, resolving the
dataset, and filtering the images to caption.
"""

from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import Any

import pydantic

from yadc.cmd import configs as cmd_configs
from yadc.cmd import envs as cmd_envs
from yadc.cmd import templates as cmd_templates
from yadc.core.config import Config, parse_config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import resolve_dataset
from yadc.utils.dict_utils import load_toml_file, toml_to_plain

from .options import CaptionJobOptions


def apply_config_overrides(raw: dict[str, Any], opts: CaptionJobOptions) -> dict[str, Any]:
    """Merge env/config overrides from *opts* into the raw TOML dict.

    Precedence (highest to lowest): ``opts`` > env > existing TOML.
    """
    env_name = opts.env or raw.get("env", "default")
    user_env = cmd_envs.load_env(env_name, password=opts.password)

    raw.setdefault("api", {})
    api: dict[str, Any] = raw["api"]

    # Apply overrides: CLI option > env > existing TOML
    api["url"] = opts.api_url or user_env.api.url or api.get("url", "")
    api["token"] = opts.api_token or user_env.api.token or api.get("token", "")
    api["model_name"] = opts.api_model_name or user_env.api.model_name or api.get("model_name", "")

    # Apply prompt overrides
    raw.setdefault("prompt", {})
    if opts.prompt_name:
        raw["prompt"]["name"] = opts.prompt_name
        raw["prompt"].pop("template", None)
    if opts.prompt_template:
        raw["prompt"]["template"] = opts.prompt_template
        raw["prompt"].pop("name", None)

    # Apply other option overrides
    if opts.max_tokens != 512:
        raw.setdefault("settings", {})
        raw["settings"]["max_tokens"] = opts.max_tokens

    if opts.image_quality != "auto":
        raw.setdefault("settings", {})
        raw["settings"]["image_quality"] = opts.image_quality

    if opts.reasoning:
        raw.setdefault("reasoning", {})
        raw["reasoning"]["enable"] = True
        raw["reasoning"]["thinking_effort"] = opts.reasoning_effort
        raw["reasoning"]["exclude_from_output"] = opts.reasoning_exclude_output

    if opts.rounds != 1:
        raw["rounds"] = opts.rounds

    return raw


def resolve_template(prompt_name: str, prompt_template: str, logger: Logger | None = None) -> str:
    """Resolve a prompt template through the fallback chain.

    Tries ``opts.prompt_template`` first, then user templates, then
    built-in templates, then the default. Raises ``ValueError`` if a
    named template can't be found.
    """
    if prompt_template:
        return prompt_template

    for loader in (cmd_templates.load_user_template, cmd_templates.load_builtin_template):
        try:
            return loader(prompt_name)
        except Exception:
            continue

    if prompt_name:
        available = cmd_templates.list_user_template()
        raise ValueError(
            f"Prompt template '{prompt_name}' not found. Available: {', '.join(available)}" if available else f"Prompt template '{prompt_name}' not found."
        )

    # No template specified — use default
    if logger:
        logger.warning("No prompt template specified, using default.")
    return cmd_templates.default_template()


def load_dataset_config(
    config_path: str | Path,
    options: CaptionJobOptions,
    *,
    user_config: str | None = None,
    image_path_resolver: Callable[[int], Path | None] | None = None,
) -> tuple[Config, list[DatasetImage]]:
    """Load a dataset config, apply options, parse, resolve, and filter images.

    Args:
        config_path: Path to the dataset TOML config file. Must exist.
        options: ``CaptionJobOptions`` with API/env/prompt/reasoning/
            rounds overrides. Use ``CaptionJobOptions()`` for defaults
            (which is a no-op pass-through).
        user_config: Optional name of a user config to merge. The merge's
            ``ValueError`` is re-raised; callers decide how to surface it.
        image_path_resolver: Optional callable that maps an ``image_id``
            to its filesystem ``Path``. Required when ``options.image_ids``
            is set; ignored otherwise.

    Returns:
        ``(config, images)`` where ``images`` is the resolved list
        filtered for ``image_ids`` (if any) and for overwrite/draft.
        Order matches ``resolve_dataset`` (filesystem iteration) — the
        caller is responsible for reordering if a specific order is
        needed (e.g. id DESC for parallel captioning).

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: On TOML parse failure, config validation failure,
            user_config merge failure, template resolution failure, or
            when ``image_ids`` resolves to an empty set.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = load_toml_file(f)

    if user_config is not None:
        raw = cmd_configs.merge_user_config(user_config, raw)

    raw = apply_config_overrides(raw, options)

    try:
        config = parse_config(toml_to_plain(raw))
    except pydantic.ValidationError as e:
        raise ValueError(f"invalid configuration: {e}")

    config.prompt.template = resolve_template(config.prompt.name, config.prompt.template)

    images = resolve_dataset(
        config.dataset,
        config.caption_suffix,
        base_dir=str(config_path.parent),
    )

    # Single-image mode bypasses the overwrite/draft filter — the caller
    # asked for these specific images and expects them to be re-captioned.
    if options.image_ids and image_path_resolver is not None:
        target_paths: set[Path] = set()
        for image_id in options.image_ids:
            path = image_path_resolver(image_id)
            if path is not None:
                target_paths.add(Path(path))
        if not target_paths:
            raise ValueError("Specified image(s) not found in dataset")
        images = [img for img in images if Path(img.path) in target_paths]
        return config, images

    to_do: list[DatasetImage] = []
    for img in images:
        if options.draft:
            if not options.overwrite and img.draft_path(options.draft).exists():
                continue
        else:
            if not options.overwrite and img.caption_path.exists():
                continue
        to_do.append(img)

    return config, to_do
