"""Tests for ``yadc prompts`` CLI — generate and save subcommands.

Two layers:

- **Subprocess tests** (the ``Test*Help`` classes +
  ``TestPromptsGenerateCliValidation``) shell out to the real
  ``uv run yadc`` binary via the ``cli(isolated=True)`` fixture from
  ``tests/cli/conftest.py``. These verify the click surface, the
  XDG-isolated save path, and click-level validation (required
  options, ``Choice`` rejection). No LLM is touched.

- **Unit tests** (``TestPromptsGenerateCmd``, ``TestSaveUserTemplate``)
  call :func:`yadc.cmd.prompts.generate` and ``_save_user_template``
  directly with the streaming core patched out via
  :mod:`unittest.mock`. These verify the CLI-friendly wrapper's
  behavior — chunk streaming via ``on_chunk``, reasoning
  buffering/flush via ``on_reasoning``, finally-block reasoning flush
  on stream errors, password forwarding, and the save-as / force
  logic — without needing a real backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yadc.llm import StreamChunk

# Centralized patch target paths — see ``testing-conventions.md``.
_PATCH_STREAM_TEMPLATE_CHUNKS = "yadc.cmd.prompts.prompts.stream_template_chunks"


class TestPromptsHelp:
    """Smoke tests for --help output (subprocess)."""

    def test_prompts_group_help(self, cli):
        runner = cli(isolated=True)
        result = runner("prompts --help")
        assert "generate" in result.stdout

    def test_generate_help(self, cli):
        runner = cli(isolated=True)
        result = runner("prompts generate --help")
        assert "--env" in result.stdout
        assert "--intent" in result.stdout
        assert "--focus" in result.stdout
        assert "--image-quality" in result.stdout
        assert "--refine" in result.stdout
        assert "--save-as" in result.stdout
        assert "--force" in result.stdout
        assert "user template" in result.stdout.lower()


class TestPromptsGenerateCliValidation:
    """Click-level validation for ``prompts generate`` — runs the real CLI."""

    def test_intent_required(self, cli):
        runner = cli(isolated=True)
        result = runner("prompts generate", should_fail=True)
        assert "--intent" in result.stderr or "Missing option" in result.stderr

    def test_focus_rejects_invalid_value(self, cli):
        runner = cli(isolated=True)
        result = runner("prompts generate --intent 'x' --focus bogus", should_fail=True)
        assert "Invalid value" in result.stderr or "--focus" in result.stderr

    def test_image_quality_rejects_invalid_value(self, cli):
        runner = cli(isolated=True)
        result = runner("prompts generate --intent 'x' --image-quality ultra", should_fail=True)
        assert "Invalid value" in result.stderr or "--image-quality" in result.stderr

    def test_refine_accepts_template_name_without_file_check(self, cli):
        """``--refine <name>`` is accepted by click even when no file matches.

        Click no longer validates the path exists (the cmd layer does
        file-vs-template resolution), so a template name that's not a
        file passes click validation. The actual lookup error surfaces
        from the cmd layer later — this test only confirms click
        doesn't reject the option upfront.
        """
        runner = cli(isolated=True)
        result = runner(
            "prompts generate --intent 'x' --refine nonexistent-template-name",
            should_fail=True,
        )
        # Click didn't reject (no "Invalid value" / "does not exist");
        # the failure is from the cmd layer's template lookup.
        assert "does not exist" not in result.stderr


def _async_iter(chunks: list[StreamChunk]) -> AsyncIterator[StreamChunk]:
    """Build an async iterator that yields the given chunks."""

    async def _iter() -> AsyncIterator[StreamChunk]:
        for c in chunks:
            yield c

    return _iter()


def _stream_template_chunks_mock(chunks: list[StreamChunk] | None = None) -> MagicMock:
    """Build a mock of ``stream_template_chunks`` that yields ``chunks``."""
    chunks = chunks or []
    cm = MagicMock()
    cm.return_value = _async_iter(chunks)
    return cm


class TestPromptsGenerateCmd:
    """``yadc.cmd.prompts.generate`` — chunk streaming, reasoning flush, error mapping.

    Patches out :func:`yadc.prompt_generation.stream_template_chunks` so
    no real LLM is contacted; verifies the CLI-friendly wrapper's
    callback wiring and error handling.
    """

    @pytest.mark.asyncio
    async def test_streams_chunks_via_on_chunk_and_returns_full_text(self):
        chunks = [
            StreamChunk(text="{% set "),
            StreamChunk(text="system_prompt %}hi{% endset %}"),
            StreamChunk(reasoning="thinking..."),
            StreamChunk(text="{% set user_prompt %}body{% endset %}"),
        ]
        received_text: list[str] = []
        received_reasoning: list[str] = []

        def _on_chunk(text: str) -> None:
            received_text.append(text)

        def _on_reasoning(reasoning: str) -> None:
            received_reasoning.append(reasoning)

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, _stream_template_chunks_mock(chunks)):
            from yadc.cmd.prompts.prompts import generate

            result = await generate(
                env="default",
                intent="caption cats",
                on_chunk=_on_chunk,
                on_reasoning=_on_reasoning,
            )

        assert result == "{% set system_prompt %}hi{% endset %}{% set user_prompt %}body{% endset %}"
        assert received_text == ["{% set ", "system_prompt %}hi{% endset %}", "{% set user_prompt %}body{% endset %}"]

    @pytest.mark.asyncio
    async def test_reasoning_emitted_once_on_first_text_chunk(self):
        chunks = [
            StreamChunk(reasoning="step 1\n"),
            StreamChunk(reasoning_summary="summary"),
            StreamChunk(reasoning="step 2"),
            StreamChunk(text="hello"),
            StreamChunk(reasoning="ignored after text"),
        ]
        received_reasoning: list[str] = []

        def _on_reasoning(reasoning: str) -> None:
            received_reasoning.append(reasoning)

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, _stream_template_chunks_mock(chunks)):
            from yadc.cmd.prompts.prompts import generate

            await generate(env="default", intent="x", on_reasoning=_on_reasoning)

        # Reasoning received exactly once, and not the post-text chunk.
        assert received_reasoning == ["step 1\nsummarystep 2"]

    @pytest.mark.asyncio
    async def test_reasoning_flushed_in_finally_when_buffered_before_raise(self):
        """Reasoning buffered BEFORE a mid-stream raise reaches ``on_reasoning`` exactly once.

        Mirrors :func:`test_reasoning_emitted_once_on_first_text_chunk`
        but with the stream raising after buffering reasoning + before
        any text — the ``_emit_reasoning`` branch never fires, so the
        ``finally`` block has to flush.
        """
        received_reasoning: list[str] = []

        async def _chunks_then_raise() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(reasoning="thinking...")
            raise ValueError("upstream died")

        cm = MagicMock()
        cm.return_value = _chunks_then_raise()

        def _on_reasoning(reasoning: str) -> None:
            received_reasoning.append(reasoning)

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
            from yadc.cmd.prompts.prompts import generate

            with pytest.raises(ValueError, match="upstream died"):
                await generate(env="default", intent="x", on_reasoning=_on_reasoning)

        assert received_reasoning == ["thinking..."]

    @pytest.mark.asyncio
    async def test_password_resolved_from_env_var_when_no_explicit(self):
        """``YADC_PASSWORD`` env var is used when no explicit ``password`` is passed."""
        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with (
            patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm),
            patch.dict("os.environ", {"YADC_PASSWORD": "from-env"}),
        ):
            from yadc.cmd.prompts.prompts import generate

            await generate(env="default", intent="x")

        # The request was passed to ``stream_template_chunks`` with the env-derived password.
        call_kwargs = cm.call_args.kwargs
        assert call_kwargs["password"] == "from-env"

    @pytest.mark.asyncio
    async def test_explicit_password_overrides_env_var(self):
        """Explicit ``password=`` argument wins over the env var."""
        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with (
            patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm),
            patch.dict("os.environ", {"YADC_PASSWORD": "from-env"}),
        ):
            from yadc.cmd.prompts.prompts import generate

            await generate(env="default", intent="x", password="explicit")

        assert cm.call_args.kwargs["password"] == "explicit"

    @pytest.mark.asyncio
    async def test_refine_reads_existing_template_from_disk(self):
        """``refine=<path>`` reads the file and forwards it in the request."""
        import tempfile
        from pathlib import Path

        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("{% set system_prompt %}existing{% endset %}")
            template_path = f.name

        try:
            with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
                from yadc.cmd.prompts.prompts import generate

                await generate(env="default", intent="tighten wording", refine=template_path)

            # The streaming core received a request with template_content set.
            request = cm.call_args.args[0]
            assert request.template_content == "{% set system_prompt %}existing{% endset %}"
        finally:
            Path(template_path).unlink()

    @pytest.mark.asyncio
    async def test_refine_resolves_user_template_by_name(self, monkeypatch):
        """``refine=<name>`` looks up the user template when no file matches."""
        import tempfile
        from pathlib import Path

        # Create a user template in the isolated template dir.
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", Path(tmpdir))
            (Path(tmpdir) / "my-template.jinja").write_text("{% set user_prompt %}from-store{% endset %}")

            cm = _stream_template_chunks_mock([StreamChunk(text="x")])

            with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
                from yadc.cmd.prompts.prompts import generate

                await generate(env="default", intent="tighten wording", refine="my-template")

            request = cm.call_args.args[0]
            assert request.template_content == "{% set user_prompt %}from-store{% endset %}"

    @pytest.mark.asyncio
    async def test_refine_missing_file_and_template_raises_clear_error(self, monkeypatch):
        """Neither a matching file nor a user template → ``FileNotFoundError`` with hint."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty template store — no user templates match.
            monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", Path(tmpdir))

            with patch(_PATCH_STREAM_TEMPLATE_CHUNKS) as cm:
                from yadc.cmd.prompts.prompts import generate

                with pytest.raises(FileNotFoundError, match="neither a file path nor a known user template"):
                    await generate(env="default", intent="x", refine="nonexistent-target")

        cm.assert_not_called()


# 1x1 transparent PNG — used in the webui test fixtures; included
# here so the test for the data URL format has a reference value.
_TINY_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="


def _write_png_pair(directory: Path, stem: str, caption: str) -> Path:
    """Create ``<stem>.png`` + ``<stem>.txt`` in ``directory``. Returns the image path."""
    from PIL import Image

    image_path = directory / f"{stem}.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    (directory / f"{stem}.txt").write_text(caption)
    return image_path


class TestResolveExamplesTargets:
    """``yadc.cmd.prompts.resolve_examples_targets`` — image file → (examples, skipped)."""

    def test_empty_targets_returns_empty(self):
        from yadc.cmd.prompts import resolve_examples_targets

        assert resolve_examples_targets(()) == ([], [])

    def test_single_file_with_sidecar_produces_example_pair(self, tmp_path):
        from yadc.cmd.prompts import resolve_examples_targets

        image_path = _write_png_pair(tmp_path, "cat", "a tabby cat")

        examples, skipped = resolve_examples_targets((str(image_path),))

        assert skipped == []
        assert len(examples) == 1
        ex = examples[0]
        assert ex.subject == "cat"
        assert ex.caption == "a tabby cat"
        assert ex.image_data_url.startswith("data:image/png;base64,")

    def test_nonexistent_image_reported_in_skipped(self, tmp_path):
        """A non-existent path with an image extension (e.g. an unexpanded glob) is reported in skipped alongside valid images."""
        from yadc.cmd.prompts import resolve_examples_targets

        image_path = _write_png_pair(tmp_path, "cat", "a cat")

        examples, skipped = resolve_examples_targets((str(image_path), "/nonexistent/glob-*.png"))

        assert [ex.subject for ex in examples] == ["cat"]
        assert skipped == ["/nonexistent/glob-*.png"]

    def test_unsupported_extension_reported_in_skipped(self, tmp_path):
        from yadc.cmd.prompts import resolve_examples_targets

        image_path = _write_png_pair(tmp_path, "cat", "a cat")
        bad_path = tmp_path / "image.psd"
        bad_path.write_bytes(b"fake psd")

        examples, skipped = resolve_examples_targets((str(image_path), str(bad_path)))

        assert [ex.subject for ex in examples] == ["cat"]
        assert skipped == [str(bad_path)]

    def test_sidecar_swept_in_by_glob_is_dropped_silently(self, tmp_path):
        """A ``.txt`` sidecar passed alongside its image is not warned (not in skipped) — it's a glob artifact, not a user error."""
        from yadc.cmd.prompts import resolve_examples_targets

        image_path = _write_png_pair(tmp_path, "cat", "a cat")
        sidecar = image_path.with_suffix(".txt")

        examples, skipped = resolve_examples_targets((str(image_path), str(sidecar)))

        assert skipped == []
        assert [ex.subject for ex in examples] == ["cat"]

    def test_raises_when_no_valid_images(self, tmp_path):
        """Non-empty targets that resolve to zero images is a user error."""
        from yadc.cmd.prompts import resolve_examples_targets

        bad_path = tmp_path / "image.psd"
        bad_path.write_bytes(b"fake psd")

        with pytest.raises(ValueError, match="no valid example images"):
            resolve_examples_targets((str(bad_path),))

    def test_raises_when_only_sidecars_passed(self, tmp_path):
        """Passing only sidecars (e.g. a ``*.txt`` glob) resolves to nothing and errors."""
        from yadc.cmd.prompts import resolve_examples_targets

        sidecar = tmp_path / "cat.txt"
        sidecar.write_text("a cat")

        with pytest.raises(ValueError, match="no valid example images"):
            resolve_examples_targets((str(sidecar),))

    def test_missing_sidecar_raises_filenotfound(self, tmp_path):
        from PIL import Image

        from yadc.cmd.prompts import resolve_examples_targets

        image_path = tmp_path / "orphan.png"
        Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)
        # No sidecar created.

        with pytest.raises(FileNotFoundError, match="caption sidecar not found"):
            resolve_examples_targets((str(image_path),))

    def test_empty_sidecar_raises_value_error(self, tmp_path):
        from yadc.cmd.prompts import resolve_examples_targets

        image_path = _write_png_pair(tmp_path, "blank", "   \n  ")

        with pytest.raises(ValueError, match="caption sidecar is empty"):
            resolve_examples_targets((str(image_path),))


class TestPromptsGenerateWithExamples:
    """End-to-end: examples are resolved and forwarded to the streaming core."""

    @pytest.mark.asyncio
    async def test_examples_resolved_and_forwarded(self, tmp_path):
        from yadc.cmd.prompts import resolve_examples_targets
        from yadc.cmd.prompts.prompts import generate

        cat = _write_png_pair(tmp_path, "cat", "a tabby cat")
        dog = _write_png_pair(tmp_path, "dog", "a brown dog")

        examples, skipped = resolve_examples_targets((str(cat), str(dog)))
        assert skipped == []
        assert len(examples) == 2

        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
            await generate(env="default", intent="caption animals", examples=examples)

        request = cm.call_args.args[0]
        assert len(request.examples) == 2
        assert [ex.subject for ex in request.examples] == ["cat", "dog"]
        assert [ex.caption for ex in request.examples] == ["a tabby cat", "a brown dog"]
        assert all(ex.image_data_url.startswith("data:image/png;base64,") for ex in request.examples)

    @pytest.mark.asyncio
    async def test_empty_examples_list_forwarded(self):
        """No examples → empty list (not None) so the request shape is consistent."""
        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
            from yadc.cmd.prompts.prompts import generate

            await generate(env="default", intent="x", examples=[])

        request = cm.call_args.args[0]
        assert request.examples == []

    @pytest.mark.asyncio
    async def test_examples_none_means_no_examples(self):
        """``examples=None`` (default) is equivalent to ``examples=[]``."""
        cm = _stream_template_chunks_mock([StreamChunk(text="x")])

        with patch(_PATCH_STREAM_TEMPLATE_CHUNKS, cm):
            from yadc.cmd.prompts.prompts import generate

            await generate(env="default", intent="x")

        request = cm.call_args.args[0]
        assert request.examples == []


class TestSaveUserTemplate:
    """``yadc.cli_prompts._save_user_template`` — --save-as / --force logic."""

    def test_saves_to_user_template_store(self, tmp_path, monkeypatch):
        from yadc.cli_prompts import _save_user_template

        monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", tmp_path / "templates")
        _save_user_template("my-template", "hello world", force=False)

        assert (tmp_path / "templates" / "my-template.jinja").read_text() == "hello world"

    def test_existing_template_without_force_and_tty_prompts(self, tmp_path, monkeypatch):
        """Existing template in TTY mode without --force → prompt for overwrite."""
        from yadc.cli_prompts import _save_user_template

        template_path = tmp_path / "templates"
        template_path.mkdir()
        (template_path / "existing.jinja").write_text("old content")
        monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", template_path)
        # Pretend we're in a TTY so the prompt fires (it would, in real use).
        monkeypatch.setattr("yadc.cli_prompts.sys.stdin.isatty", lambda: True)
        # Decline the overwrite prompt.
        monkeypatch.setattr("yadc.cli_prompts.click.confirm", lambda *a, **kw: False)

        _save_user_template("existing", "new content", force=False)

        # Unchanged because user declined.
        assert (template_path / "existing.jinja").read_text() == "old content"

    def test_existing_template_with_force_overwrites(self, tmp_path, monkeypatch):
        from yadc.cli_prompts import _save_user_template

        template_path = tmp_path / "templates"
        template_path.mkdir()
        (template_path / "existing.jinja").write_text("old content")
        monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", template_path)

        _save_user_template("existing", "new content", force=True)

        assert (template_path / "existing.jinja").read_text() == "new content"

    def test_existing_template_non_tty_without_force_errors(self, tmp_path, monkeypatch):
        from yadc.cli_prompts import _save_user_template

        template_path = tmp_path / "templates"
        template_path.mkdir()
        (template_path / "existing.jinja").write_text("old content")
        monkeypatch.setattr("yadc.cmd.templates.templates.TEMPLATE_PATH", template_path)
        monkeypatch.setattr("yadc.cli_prompts.sys.stdin.isatty", lambda: False)

        with pytest.raises(SystemExit) as exc_info:
            _save_user_template("existing", "new content", force=False)
        assert exc_info.value.code == 2  # STATUS_USER_ERROR

        # Unchanged.
        assert (template_path / "existing.jinja").read_text() == "old content"
