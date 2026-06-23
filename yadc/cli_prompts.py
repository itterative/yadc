"""``yadc prompts`` — generate Jinja2 prompt templates via LLM."""

import asyncio
import sys

import click

from yadc.cmd import status as cmd_status
from yadc.cmd import templates as cmd_templates
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core import logging
from yadc.prompt_generation import PromptGenerationConfigError

from . import cli_common
from .cmd import prompts as cmd_prompts

_logger = logging.get_logger(__name__)


@click.group("prompts", short_help="Generate Jinja2 prompt templates", help="Generate Jinja2 prompt templates from an intent using an LLM.")
def prompts():
    pass


@prompts.command(
    "generate",
    short_help="Generate a prompt template from an intent",
    help=(
        "Generate a Jinja2 prompt template from an intent using the named env's model. "
        "Template body is written to stdout. Progress and reasoning go to stderr.\n\n"
        "Pass ``--save-as NAME`` to also write the result to the user template store "
        "under ``NAME``. Combine with ``--force`` to overwrite an existing template. "
        "In a pipe, output goes to stdout and nothing is saved unless ``--save-as`` is given.\n\n"
        "EXAMPLES: pass image files as positional arguments. "
        "Each image must have a ``<stem>.txt`` sidecar with the caption. "
        "Arguments that aren't supported images (wrong extension, a directory, "
        "or an unexpanded glob) are warned about and skipped. "
        "Use ``--`` before examples to separate them from flags if a filename starts with ``-``."
    ),
)
@click.option("--env", type=str, default="default", help="Configuration environment")
@click.option("--intent", type=str, required=True, help="What the prompt should do (e.g. 'caption cats concisely')")
@click.option(
    "--focus",
    type=click.Choice(["system", "user", "both"]),
    default="both",
    help="Which template blocks to populate",
)
@click.option("--api-model-name", type=str, default=None, help="Override the env's default model")
@click.option(
    "--max-tokens",
    type=click.IntRange(min=100, max=65536),
    default=16384,
    help="Maximum output tokens",
)
@click.option(
    "--image-quality",
    type=click.Choice(["auto", "low", "high"]),
    default="auto",
    help="Few-shot example image fidelity (OpenAI: image_url.detail, Gemini: mediaResolution)",
)
@click.option(
    "--refine",
    type=str,
    default=None,
    help=(
        "Template to refine instead of generating from scratch. "
        "Either a path to an existing file, or a user template name "
        "(see `yadc templates list`). A file path takes precedence."
    ),
)
@click.option(
    "--save-as",
    "save_as",
    type=str,
    default=None,
    help="Save the generated template to the user template store under this name.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Overwrite an existing user template without prompting (used with --save-as).",
)
@click.option(
    "--password",
    type=str,
    default=None,
    help="Password for decrypting password-mode env settings (defaults to YADC_PASSWORD)",
)
@click.argument(
    "examples",
    nargs=-1,
    type=str,
)
@cli_common.log_level
def generate(
    env: str,
    intent: str,
    focus: str,
    api_model_name: str | None,
    max_tokens: int,
    image_quality: str,
    refine: str | None,
    save_as: str | None,
    force: bool,
    password: str | None,
    examples: tuple[str, ...],
):
    asyncio.run(_run_generate(env, intent, focus, api_model_name, max_tokens, image_quality, refine, save_as, force, password, examples))


async def _run_generate(
    env: str,
    intent: str,
    focus: str,
    api_model_name: str | None,
    max_tokens: int,
    image_quality: str,
    refine: str | None,
    save_as: str | None,
    force: bool,
    password: str | None,
    examples: tuple[str, ...],
) -> None:
    try:
        resolved_examples, skipped = cmd_prompts.resolve_examples_targets(examples)
    except (FileNotFoundError, ValueError) as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    for target in skipped:
        _logger.warning("Skipping example '%s'.", target)

    try:
        content = await cmd_prompts.generate(
            env=env,
            intent=intent,
            focus=focus,  # pyright: ignore[reportArgumentType]
            api_model_name=api_model_name,
            max_tokens=max_tokens,
            image_quality=image_quality,  # pyright: ignore[reportArgumentType]
            refine=refine,
            examples=resolved_examples,
            password=password,
            on_chunk=_write_stdout,
            on_reasoning=_log_reasoning,
        )
    except PasswordRequiredError:
        _logger.error(
            "Error: environment '%s' is password-encrypted but no password was provided. "
            "Set YADC_PASSWORD, pass --password, or run in a terminal for an interactive prompt.",
            env,
        )
        sys.exit(cmd_status.STATUS_USER_ERROR)
    except FileNotFoundError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)
    except PromptGenerationConfigError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)
    except ValueError as e:
        # Catch-all for unexpected value errors (e.g. malformed args,
        # client config). Surfaces a clear message instead of a
        # traceback for users.
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)
    except Exception as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    if save_as is not None:
        _save_user_template(save_as, content, force=force)


def _write_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _log_reasoning(reasoning: str) -> None:
    quoted = "\n".join(f"> {line}" for line in reasoning.splitlines())
    _logger.info(quoted)


def _user_template_exists(name: str) -> bool:
    try:
        cmd_templates.load_user_template(name)
    except FileNotFoundError:
        return False
    return True


def _save_user_template(name: str, content: str, *, force: bool) -> None:
    """Save ``content`` to the user template store under ``name``.

    If a template with the same name already exists:
    - With ``--force``: overwrite without prompting.
    - In a TTY: prompt the user to confirm.
    - Otherwise: error out.
    """
    if _user_template_exists(name) and not force:
        if not sys.stdin.isatty():
            _logger.error(
                "Error: template '%s' already exists. Use --force to overwrite, or run in a terminal for an interactive prompt.",
                name,
            )
            sys.exit(cmd_status.STATUS_USER_ERROR)
        if not click.confirm(f"Template '{name}' already exists. Overwrite?", default=False):
            _logger.info("Skipped.")
            return

    try:
        cmd_templates.save_user_template(name, content)
    except PermissionError:
        _logger.error("Error: failed to save user template: permissions error")
        sys.exit(cmd_status.STATUS_ERROR)
    _logger.info("Saved template '%s'.", name)
