"""Tests for the ``yadc caption`` CLI command.

Focus on the new ``--max-concurrent`` flag:
- Argument parsing and validation
- The interactive+max-concurrent early error
- The non-interactive parallel path that uses ``runner.caption_images``
- The default behaviour (max_concurrent=1) is preserved
"""

import struct
import textwrap
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from yadc.cli_caption import CLIPrintCallbacks, caption

# ---- helpers ----


def _make_png() -> bytes:
    """Create a minimal valid 1×1 PNG."""
    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
    idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return header + ihdr + idat + iend


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def dataset_toml(tmp_path: Path) -> Path:
    """Create a real dataset TOML with one fake image."""
    img = tmp_path / "img001.png"
    img.write_bytes(_make_png())
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
