"""Tests for the ``yadc caption`` CLI command.

Focus on the new ``--max-concurrent`` flag:
- Argument parsing and validation
- The interactive+max-concurrent early error
- The non-interactive parallel path that uses ``runner.caption_images``
- The default behaviour (max_concurrent=1) is preserved
"""

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from yadc.cli_caption import CLIPrintCallbacks, caption
from yadc.cmd.envs.keystorage_password import PasswordRequiredError  # noqa: E402

# ---- helpers ----

# Patch targets for cli_caption. Single source of truth — update here
# if the import layout changes in yadc/cli_caption.py.
_PATCH_LOAD_DATASET_CONFIG = "yadc.cli_caption.load_dataset_config"
_PATCH_CLICK_PROMPT = "yadc.cli_caption.click.prompt"
_PATCH_RUNNER = "yadc.cli_caption.CaptioningRunner"
_PATCH_YADC_PASSWORD = "yadc.cli_caption.YADC_PASSWORD"
_PATCH_IS_TTY = "yadc.cli_caption._is_tty"


def _make_image(path: Path):
    # Create a minimal valid JPEG (1x1 pixel)
    from PIL import Image

    img = Image.new("RGB", (1, 1), color="red")
    img.save(path)


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def dataset_toml(tmp_path: Path) -> Path:
    """Create a real dataset TOML with one fake image."""
    img = tmp_path / "img001.png"
    _make_image(img)

    toml = textwrap.dedent(
        f"""\
        [api]
        url = "http://localhost:8080"
        model_name = "test"

        [prompt]
        template = "test"

        [[dataset]]
        path = "{img}"
        """
    )
    config = tmp_path / "dataset.toml"
    config.write_text(toml)
    return config


def _fake_loaded_config() -> tuple[MagicMock, list[MagicMock]]:
    """Return a (config, images) tuple mocking load_dataset_config output."""
    config = MagicMock()
    config.prompt.name = ""
    config.prompt.template = "t"
    config.api.url = "http://x"
    config.api.model_name = "m"
    config.api.token = ""
    config.settings.advanced.model_dump.return_value = {}
    config.settings.max_tokens = 512
    config.settings.image_quality = "auto"
    config.settings.store_conversation = False
    config.reasoning.enable = False
    config.reasoning.thinking_effort = "low"
    config.reasoning.exclude_from_output = True
    images = [MagicMock(spec=["path", "caption"])]
    images[0].path = "/tmp/img001.png"
    images[0].caption = ""
    return config, images


@pytest.fixture
def patched_load_dataset_config():
    """Patch ``load_dataset_config`` in ``yadc.cli_caption``."""
    with patch(_PATCH_LOAD_DATASET_CONFIG) as mock_load:
        yield mock_load


@pytest.fixture
def patched_click_prompt():
    """Patch ``click.prompt`` in ``yadc.cli_caption``.

    Tests that want a specific return value should use
    ``patched_click_prompt.return_value = "..."`` or pass
    ``return_value=`` to the ``patch(...)`` call.
    """
    with patch(_PATCH_CLICK_PROMPT) as mock_prompt:
        yield mock_prompt


@pytest.fixture
def patched_runner():
    """Patch ``CaptioningRunner`` in ``yadc.cli_caption``.

    Yields a (MockRunner_class, mock_instance) tuple. The class mock is
    what callers receive from the patch; the instance is the value
    returned by the context manager (``__aenter__``), wired with the
    async methods the CLI uses.
    """
    with patch(_PATCH_RUNNER) as mock_runner_cls:
        mock_instance = MagicMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.caption_image_dry_run = AsyncMock()
        mock_instance.caption_images = AsyncMock()
        mock_runner_cls.return_value = mock_instance
        yield mock_runner_cls, mock_instance


@pytest.fixture
def yadc_password():
    """Factory fixture that patches ``YADC_PASSWORD`` in ``yadc.cli_caption``
    to a given value. Used to simulate the env var being set or unset.
    """

    def _patch(value):
        return patch(_PATCH_YADC_PASSWORD, value)

    return _patch


@pytest.fixture
def is_tty():
    """Factory fixture that patches ``_is_tty`` in ``yadc.cli_caption``
    to return a given value. ``CliRunner`` doesn't pipe through the
    real ``sys.stdin``, so this is the test-friendly equivalent.
    """

    def _patch(value: bool):
        return patch(_PATCH_IS_TTY, lambda: value)

    return _patch


# ---- argument parsing ----


class TestMaxConcurrentArgument:
    """The ``--max-concurrent`` click option is parsed correctly."""

    def test_default_is_none(self, cli_runner, dataset_toml):
        """--max-concurrent defaults to ``None`` (sentinel) when the user doesn't pass it.

        The CLI passes ``None`` through to ``CaptionJobOptions``; the
        loader (``apply_config_overrides``) resolves ``None`` to the
        env's ``max_concurrent`` or to ``1``. This test asserts the
        CLI-side contract: the CLI does not force a value.
        """
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_image_dry_run = AsyncMock()
            MockRunner.return_value = mock_instance

            cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive"])

        # The runner is built with the CaptionJobOptions as the second arg
        MockRunner.assert_called_once()
        opts = MockRunner.call_args.args[1]
        assert opts.max_concurrent is None

    def test_explicit_value(self, cli_runner, dataset_toml):
        """--max-concurrent 4 sets options.max_concurrent=4."""
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance

            result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "4"])

        assert result.exit_code == 0, result.stderr
        opts = MockRunner.call_args.args[1]
        assert opts.max_concurrent == 4
        # Parallel path used caption_images, not _caption's per-image dry-run
        mock_instance.caption_images.assert_awaited_once()
        assert mock_instance.caption_images.call_args.kwargs["max_concurrent"] == 4

    def test_zero_is_rejected(self, cli_runner, dataset_toml):
        """--max-concurrent 0 is rejected by click's IntRange(min=1)."""
        result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "0"])
        assert result.exit_code != 0

    def test_negative_is_rejected(self, cli_runner, dataset_toml):
        """Negative values are rejected by click's IntRange(min=1)."""
        result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "-1"])
        assert result.exit_code != 0


# ---- interactive + max-concurrent rejection ----


class TestInteractiveIncompatibleWithParallel:
    """--max-concurrent > 1 with --interactive is rejected with a clear error."""

    def test_interactive_with_max_concurrent_exits_error(self, cli_runner, dataset_toml, caplog):
        """The early check fires and the runner is never created."""
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()

            with caplog.at_level("ERROR", logger="yadc.cli_caption"):
                result = cli_runner.invoke(caption, [str(dataset_toml), "--interactive", "--max-concurrent", "4"])

        assert result.exit_code != 0
        # The error message names the flag and the workaround.
        # The yadc logger writes via a StreamHandler in unit tests
        # (no ClickHandler installed), so we inspect caplog rather
        # than result.stderr.
        message_text = " ".join(record.message.lower() for record in caplog.records)
        assert "max-concurrent" in message_text
        assert "non-interactive" in message_text
        # And the runner was never instantiated — we error out before
        # any model setup.
        MockRunner.assert_not_called()

    def test_interactive_with_default_max_concurrent_runs_legacy_path(self, cli_runner, dataset_toml):
        """--interactive (with default --max-concurrent 1) does NOT take the parallel path.

        The legacy _caption path has its own coverage; here we just
        verify that the parallel ``caption_images`` method is NOT used
        and the runner is instantiated.
        """
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("stop after __aenter__"))
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_image_dry_run = AsyncMock()
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance

            cli_runner.invoke(caption, [str(dataset_toml), "--interactive"])

        # The runner was instantiated (we get past the early check)
        MockRunner.assert_called_once()
        # But the parallel method was never used — the legacy _caption
        # path was selected. (The __aenter__ side_effect terminates
        # the path early; we don't need to actually run the legacy
        # logic in this unit test.)
        mock_instance.caption_images.assert_not_called()


# ---- non-interactive parallel path ----


class TestNonInteractiveParallelPath:
    """The non-interactive path with --max-concurrent > 1 uses caption_images."""

    def test_uses_caption_images_with_max_concurrent(self, cli_runner, dataset_toml):
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance

            result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "3"])

        assert result.exit_code == 0, result.stderr
        mock_instance.caption_images.assert_awaited_once()
        call = mock_instance.caption_images.call_args
        assert call.kwargs["max_concurrent"] == 3

    def test_sequential_non_interactive_does_not_use_caption_images(self, cli_runner, dataset_toml):
        """--non-interactive without --max-concurrent still uses the old _caption path."""
        with patch("yadc.cli_caption.load_dataset_config") as mock_load, patch("yadc.cli_caption.CaptioningRunner") as MockRunner:
            mock_load.return_value = _fake_loaded_config()
            mock_instance = MagicMock()
            # Stop after the runner context is entered; we don't need
            # to actually run the legacy _caption path here — it has
            # its own coverage. We just want to verify the path
            # selection.
            mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("stop after __aenter__"))
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_image_dry_run = AsyncMock()
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance

            cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive"])

        # The runner was instantiated
        MockRunner.assert_called_once()
        # And the parallel method was not used — the legacy _caption
        # path was selected for max_concurrent=1.
        mock_instance.caption_images.assert_not_called()


# ---- CLIPrintCallbacks ----


class TestCLIPrintCallbacks:
    """The minimal callback class used by the non-interactive parallel path."""

    @pytest.mark.asyncio
    async def test_on_token_is_noop(self, capsys):
        cb = CLIPrintCallbacks()
        await cb.on_token("hello")
        await cb.on_token(" world")
        # Nothing should have been printed to stdout/stderr
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    @pytest.mark.asyncio
    async def test_on_image_captioned_logs_duration(self, caplog):
        cb = CLIPrintCallbacks()
        image = MagicMock()
        image.path = "/tmp/img.jpg"
        with caplog.at_level("INFO", logger="yadc.cli_caption"):
            await cb.on_image_captioned(image, duration_ms=1500)
        # The log line includes the path and the duration
        assert any("img.jpg" in record.message and "1.5" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_on_image_error_logs_warning(self, caplog):
        cb = CLIPrintCallbacks()
        image = MagicMock()
        image.path = "/tmp/img.jpg"
        with caplog.at_level("WARNING", logger="yadc.cli_caption"):
            await cb.on_image_error(image, error="boom", duration_ms=0)
        assert any("img.jpg" in record.message and "boom" in record.message for record in caplog.records)


# ---- password handling ----


class TestPasswordResolution:
    """``yadc caption`` resolves the env-decryption password in priority order:

    ``--password`` > ``YADC_PASSWORD`` env > interactive prompt (TTY only) > fail.
    """

    def test_explicit_password_flag_is_forwarded(self, cli_runner, dataset_toml, patched_load_dataset_config, patched_runner):
        """``--password secret`` sets ``CaptionJobOptions.password``."""
        patched_load_dataset_config.return_value = _fake_loaded_config()
        MockRunner, _ = patched_runner

        # ``--max-concurrent 2`` forces the non-interactive parallel path,
        # which is simpler to drive with a MagicMock image than the
        # legacy interactive loop. Password resolution happens before
        # the runner is created, so the path doesn't affect what
        # we're testing.
        result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "2", "--password", "supersecret"])

        assert result.exit_code == 0, result.stderr
        opts = MockRunner.call_args.args[1]
        assert opts.password == "supersecret"

    def test_yadc_password_env_is_default(self, cli_runner, dataset_toml, patched_load_dataset_config, patched_runner, yadc_password):
        """``YADC_PASSWORD`` env var is used when ``--password`` is not given."""
        with yadc_password("envpass"):
            patched_load_dataset_config.return_value = _fake_loaded_config()
            MockRunner, _ = patched_runner

            result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "2"])

        assert result.exit_code == 0, result.stderr
        opts = MockRunner.call_args.args[1]
        assert opts.password == "envpass"

    def test_explicit_password_overrides_env(self, cli_runner, dataset_toml, patched_load_dataset_config, patched_runner, yadc_password):
        """``--password`` wins over ``YADC_PASSWORD``."""
        with yadc_password("envpass"):
            patched_load_dataset_config.return_value = _fake_loaded_config()
            MockRunner, _ = patched_runner

            result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "2", "--password", "cliwins"])

        assert result.exit_code == 0, result.stderr
        opts = MockRunner.call_args.args[1]
        assert opts.password == "cliwins"


class TestPasswordRequiredErrorHandling:
    """``PasswordRequiredError`` from the loader is caught and the user is prompted."""

    def test_prompts_and_retries_on_first_failure(
        self,
        cli_runner,
        dataset_toml,
        caplog,
        patched_load_dataset_config,
        patched_click_prompt,
        patched_runner,
        yadc_password,
        is_tty,
    ):
        """When load_env raises PasswordRequiredError with no prior password, the
        CLI prompts the user (TTY mocked true), retries with the new
        password, and the runner is created successfully."""
        with yadc_password(None), is_tty(True):
            patched_click_prompt.return_value = "typed-password"
            # First call raises, second call succeeds.
            patched_load_dataset_config.side_effect = [PasswordRequiredError("test"), _fake_loaded_config()]
            MockRunner, _ = patched_runner

            with caplog.at_level("ERROR", logger="yadc.cli_caption"):
                result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive", "--max-concurrent", "2"])

        assert result.exit_code == 0, result.stderr
        patched_click_prompt.assert_called_once()
        # The prompt's text mentions password (don't be brittle about exact wording).
        prompt_text = patched_click_prompt.call_args.args[0].lower()
        assert "password" in prompt_text
        # The runner received the prompted password.
        opts = MockRunner.call_args.args[1]
        assert opts.password == "typed-password"
        # load_dataset_config was called twice (initial + retry).
        assert patched_load_dataset_config.call_count == 2
        # The second call's options carry the typed password.
        retry_opts = patched_load_dataset_config.call_args_list[1].args[1]
        assert retry_opts.password == "typed-password"

    def test_fails_with_clear_error_when_not_a_tty(
        self,
        cli_runner,
        dataset_toml,
        caplog,
        patched_load_dataset_config,
        patched_click_prompt,
        yadc_password,
        is_tty,
    ):
        """When load_env raises PasswordRequiredError in a non-TTY context and
        no password was supplied, the CLI fails with a clear error and
        does not retry."""
        with yadc_password(None), is_tty(False):
            patched_load_dataset_config.side_effect = PasswordRequiredError("test")

            with caplog.at_level("ERROR", logger="yadc.cli_caption"):
                result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive"])

        assert result.exit_code != 0
        patched_click_prompt.assert_not_called()
        assert patched_load_dataset_config.call_count == 1
        message_text = " ".join(record.message.lower() for record in caplog.records)
        assert "password-encrypted" in message_text
        # Actionable hints are present
        assert "yadc_password" in message_text or "--password" in message_text

    def test_wrong_password_fails_in_non_tty(
        self,
        cli_runner,
        dataset_toml,
        caplog,
        patched_load_dataset_config,
        patched_click_prompt,
        yadc_password,
        is_tty,
    ):
        """A wrong password from YADC_PASSWORD in a non-TTY context gives a
        clear 'incorrect' error and does not prompt."""
        with yadc_password("wrong"), is_tty(False):
            patched_load_dataset_config.side_effect = PasswordRequiredError("test")

            with caplog.at_level("ERROR", logger="yadc.cli_caption"):
                result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive"])

        assert result.exit_code != 0
        patched_click_prompt.assert_not_called()
        assert patched_load_dataset_config.call_count == 1
        message_text = " ".join(record.message.lower() for record in caplog.records)
        assert "incorrect" in message_text

    def test_wrong_password_prompts_retry_then_fails(
        self,
        cli_runner,
        dataset_toml,
        caplog,
        patched_load_dataset_config,
        patched_click_prompt,
        yadc_password,
        is_tty,
    ):
        """A wrong password from YADC_PASSWORD in a TTY context prompts for
        a fresh password, retries, and fails with 'incorrect' when the
        retry also fails."""
        with yadc_password("wrong"), is_tty(True):
            patched_click_prompt.return_value = "also-wrong"
            patched_load_dataset_config.side_effect = PasswordRequiredError("test")

            with caplog.at_level("ERROR", logger="yadc.cli_caption"):
                result = cli_runner.invoke(caption, [str(dataset_toml), "--non-interactive"])

        assert result.exit_code != 0
        # The prompt text hints that the prior password was wrong.
        prompt_text = patched_click_prompt.call_args.args[0].lower()
        assert "incorrect" in prompt_text or "previous" in prompt_text
        # The retry's password was forwarded to the loader.
        retry_opts = patched_load_dataset_config.call_args_list[1].args[1]
        assert retry_opts.password == "also-wrong"
        # And the final error message names it as incorrect.
        message_text = " ".join(record.message.lower() for record in caplog.records)
        assert "incorrect" in message_text
