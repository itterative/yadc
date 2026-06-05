"""Tests for yadc.core.captioning.runner — the shared captioning loop.

The runner is exercised in isolation: ``APICaptioner.create`` and
``AsyncSession`` are mocked so the runner can run its stream/save
logic without network calls. Callbacks are exercised via
``MagicMock(spec=CaptioningCallbacks)`` so the mock's auto-assertions
catch any missing/extra invocations.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from PIL import Image

from yadc.core.captioning.options import CaptionJobOptions
from yadc.core.captioning.runner import (
    CaptioningCallbacks,
    CaptioningRunner,
    HTTPTTimeouts,
)
from yadc.core.config import Config, parse_config
from yadc.core.dataset import DatasetImage

# Patch target paths centralized so renames only need updates here.
_PATCH_APICAPTIONER_CREATE = "yadc.core.captioning.runner.APICaptioner.create"
_PATCH_ASYNC_SESSION = "yadc.core.captioning.runner.AsyncSession"


def _config(token: str = "") -> Config:
    raw = {
        "api": {"url": "http://test", "model_name": "test-model", "token": token},
        "prompt": {"template": "t"},
        "dataset": [],
    }
    return parse_config(raw)


def _real_image(path: Path) -> DatasetImage:
    Image.new("RGB", (1, 1), color="red").save(path, format="JPEG")
    return DatasetImage(path=str(path))


async def _fake_stream(*tokens: str):
    for token in tokens:
        yield token


def _model_yielding(*tokens: str) -> MagicMock:
    """Build a mock model whose ``predict_stream`` yields the given tokens."""
    model = MagicMock()
    model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream(*tokens))
    model.load_model = AsyncMock()
    model.log_usage = MagicMock()
    return model


@pytest.fixture
def patched_async_session():
    """Patched ``AsyncSession`` class. The constructed instance is ``.return_value``."""
    with patch(_PATCH_ASYNC_SESSION) as mock_session_cls:
        session = MagicMock()
        session.aclose = AsyncMock()
        mock_session_cls.return_value = session
        yield mock_session_cls


@pytest.fixture
def patched_model():
    """Mock model with a default ``predict_stream`` yielding ``"hello world"``."""
    with patch(_PATCH_APICAPTIONER_CREATE, new_callable=AsyncMock) as mock_create:
        model = _model_yielding("hello world")
        mock_create.return_value = model
        yield model


class TestContextManager:
    """__aenter__ creates session+model, __aexit__ tears them down."""

    @pytest.mark.asyncio
    async def test_aenter_creates_session_and_loads_model(self, patched_async_session, patched_model):
        async with CaptioningRunner(_config(), CaptionJobOptions()):
            pass

        patched_async_session.assert_called_once()
        patched_model.load_model.assert_awaited_once_with("test-model")

    @pytest.mark.parametrize(
        "token,expected",
        [
            ("secret", "Bearer secret"),
            ("", None),  # no Authorization header
        ],
    )
    @pytest.mark.asyncio
    async def test_aenter_authorization_header(self, patched_async_session, patched_model, token, expected):
        async with CaptioningRunner(_config(token=token), CaptionJobOptions()):
            pass

        headers = patched_async_session.call_args.kwargs["headers"]
        if expected is None:
            assert "Authorization" not in headers
        else:
            assert headers["Authorization"] == expected

    @pytest.mark.asyncio
    async def test_aenter_passes_http_timeouts_to_session(self, patched_async_session, patched_model):
        timeouts = HTTPTTimeouts(connect=5.0, read=10.0, write=15.0, pool=20.0)
        async with CaptioningRunner(_config(), CaptionJobOptions(), http_timeouts=timeouts):
            pass

        kwargs = patched_async_session.call_args.kwargs
        assert kwargs["connect_timeout"] == 5.0
        assert kwargs["read_timeout"] == 10.0
        assert kwargs["write_timeout"] == 15.0
        assert kwargs["pool_timeout"] == 20.0

    @pytest.mark.asyncio
    async def test_aexit_closes_session_and_logs_usage(self, patched_async_session, patched_model):
        async with CaptioningRunner(_config(), CaptionJobOptions()):
            pass

        patched_model.log_usage.assert_called_once()
        patched_async_session.return_value.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aenter_propagates_value_error_from_model_creation(self, patched_async_session):
        with patch(_PATCH_APICAPTIONER_CREATE, new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = ValueError("bad config")
            with pytest.raises(ValueError, match="bad config"):
                async with CaptioningRunner(_config(), CaptionJobOptions()):
                    pass


class TestCaptionImage:
    """caption_image streams, saves, and fires the right callbacks."""

    @pytest.mark.asyncio
    async def test_streams_tokens_fires_callbacks_and_saves(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream("foo", " ", "bar"))
        image = _real_image(tmp_path / "img.jpg")
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
            caption = await runner.caption_image(image, callbacks)

        assert caption == "foo bar"
        callbacks.on_image_started.assert_called_once_with(image)
        callbacks.on_token.assert_has_calls([call("foo"), call(" "), call("bar")])
        callbacks.on_image_captioned.assert_called_once()
        assert image.caption_path.read_text() == "foo bar"
        assert image.history_path.exists()

    @pytest.mark.asyncio
    async def test_draft_mode_writes_to_draft_path_only(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), CaptionJobOptions(draft="gemma")) as runner:
            await runner.caption_image(image, callbacks)

        assert image.draft_path("gemma").read_text() == "hello world"
        assert not image.caption_path.exists()
        assert not image.history_path.exists()

    @pytest.mark.asyncio
    async def test_empty_caption_does_not_save_or_fire_captioned(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream())
        image = _real_image(tmp_path / "img.jpg")
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        registered: list[list[str]] = []
        async with CaptioningRunner(
            _config(),
            CaptionJobOptions(),
            expected_change_registrar=lambda paths: registered.append(list(paths)),
        ) as runner:
            caption = await runner.caption_image(image, callbacks)

        assert caption == ""
        callbacks.on_image_captioned.assert_not_called()
        assert registered == []
        assert not image.caption_path.exists()

    @pytest.mark.asyncio
    async def test_registrar_receives_normal_mode_paths(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        registered: list[list[str]] = []
        async with CaptioningRunner(
            _config(),
            CaptionJobOptions(),
            expected_change_registrar=lambda paths: registered.append(list(paths)),
        ) as runner:
            await runner.caption_image(image, AsyncMock(spec=CaptioningCallbacks))

        assert set(registered[0]) == {str(image.caption_path), str(image.toml_path), str(image.history_path)}

    @pytest.mark.asyncio
    async def test_registrar_receives_draft_mode_path(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        registered: list[list[str]] = []
        async with CaptioningRunner(
            _config(),
            CaptionJobOptions(draft="gemma"),
            expected_change_registrar=lambda paths: registered.append(list(paths)),
        ) as runner:
            await runner.caption_image(image, AsyncMock(spec=CaptioningCallbacks))

        assert registered[0] == [str(image.draft_path("gemma"))]

    @pytest.mark.asyncio
    async def test_registrar_called_before_writes(self, patched_async_session, patched_model, tmp_path):
        """expected_change_registrar must fire BEFORE the file writes."""
        image = _real_image(tmp_path / "img.jpg")
        call_order: list[str] = []

        def registrar(paths):
            call_order.append("registrar")

        original_update = image.update_caption

        def tracked_update(caption):
            call_order.append("write_caption")
            original_update(caption)

        image.update_caption = tracked_update

        async with CaptioningRunner(
            _config(),
            CaptionJobOptions(),
            expected_change_registrar=registrar,
        ) as runner:
            await runner.caption_image(image, AsyncMock(spec=CaptioningCallbacks))

        assert call_order == ["registrar", "write_caption"]

    @pytest.mark.parametrize("exc_message", ["api error", "boom"])
    @pytest.mark.asyncio
    async def test_fires_on_image_error_and_re_raises(self, patched_async_session, patched_model, tmp_path, exc_message):
        image = _real_image(tmp_path / "img.jpg")
        exc = RuntimeError(exc_message)

        async def _raise(*a, **k):
            raise exc
            yield  # pragma: no cover

        patched_model.predict_stream = MagicMock(side_effect=_raise)
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        with pytest.raises(RuntimeError, match=exc_message):
            async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
                await runner.caption_image(image, callbacks)

        callbacks.on_image_error.assert_called_once()
        # First arg is the image, second is the error string
        assert callbacks.on_image_error.call_args.args[0] == image
        assert callbacks.on_image_error.call_args.args[1] == exc_message

    @pytest.mark.asyncio
    async def test_cancellation_propagates_without_firing_on_image_error(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")

        async def _cancel(*a, **k):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        patched_model.predict_stream = MagicMock(side_effect=_cancel)
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        with pytest.raises(asyncio.CancelledError):
            async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
                await runner.caption_image(image, callbacks)

        # Cancellation is not a per-image error — on_image_error must not fire
        callbacks.on_image_error.assert_not_called()


class TestCaptionImageDryRun:
    """caption_image_dry_run streams without saving."""

    @pytest.mark.asyncio
    async def test_returns_caption_text_and_streams_callbacks(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream("a", "b", "c"))
        image = _real_image(tmp_path / "img.jpg")
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        registered: list[list[str]] = []
        async with CaptioningRunner(
            _config(),
            CaptionJobOptions(),
            expected_change_registrar=lambda paths: registered.append(list(paths)),
        ) as runner:
            caption = await runner.caption_image_dry_run(image, callbacks)

        assert caption == "abc"
        callbacks.on_image_started.assert_called_once_with(image)
        callbacks.on_token.assert_has_calls([call("a"), call("b"), call("c")])
        callbacks.on_image_captioned.assert_not_called()
        assert registered == []
        assert not image.caption_path.exists()
        assert not image.history_path.exists()


class TestExtrasForwarded:
    """caption_rounds, extra_messages, prediction_context are forwarded to the model."""

    @pytest.mark.asyncio
    async def test_extra_messages_forwarded(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.captioner import ReplyRound

        image = _real_image(tmp_path / "img.jpg")
        messages = [ReplyRound(role="user", content="hi")]
        async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
            await runner.caption_image_dry_run(image, AsyncMock(spec=CaptioningCallbacks), extra_messages=messages)

        assert patched_model.predict_stream.call_args.kwargs["extra_messages"] is messages

    @pytest.mark.asyncio
    async def test_caption_rounds_forwarded(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.captioner import CaptionerRound

        image = _real_image(tmp_path / "img.jpg")
        rounds = [CaptionerRound(iteration=1, caption="prev")]
        async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
            await runner.caption_image_dry_run(image, AsyncMock(spec=CaptioningCallbacks), caption_rounds=rounds)

        assert patched_model.predict_stream.call_args.kwargs["caption_rounds"] is rounds

    @pytest.mark.asyncio
    async def test_prediction_context_forwarded_when_provided(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.prediction import PredictionContext

        image = _real_image(tmp_path / "img.jpg")
        ctx = PredictionContext()
        async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
            await runner.caption_image_dry_run(image, AsyncMock(spec=CaptioningCallbacks), prediction_context=ctx)

        assert patched_model.predict_stream.call_args.kwargs["prediction_context"] is ctx

    @pytest.mark.asyncio
    async def test_prediction_context_default_when_not_provided(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.prediction import PredictionContext

        image = _real_image(tmp_path / "img.jpg")
        async with CaptioningRunner(_config(), CaptionJobOptions()) as runner:
            await runner.caption_image_dry_run(image, AsyncMock(spec=CaptioningCallbacks))

        assert isinstance(patched_model.predict_stream.call_args.kwargs["prediction_context"], PredictionContext)


class TestUsedWithoutContextManager:
    """caption_image and caption_image_dry_run require the async context manager."""

    @pytest.mark.asyncio
    async def test_caption_image_asserts_without_aenter(self, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        with pytest.raises(AssertionError):
            await CaptioningRunner(_config(), CaptionJobOptions()).caption_image(image, AsyncMock(spec=CaptioningCallbacks))

    @pytest.mark.asyncio
    async def test_caption_image_dry_run_asserts_without_aenter(self, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        with pytest.raises(AssertionError):
            await CaptioningRunner(_config(), CaptionJobOptions()).caption_image_dry_run(image, AsyncMock(spec=CaptioningCallbacks))
