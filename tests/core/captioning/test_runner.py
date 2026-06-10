"""Tests for yadc.core.captioning.runner — the shared captioning loop.

The runner is exercised in isolation: ``APICaptioner.create`` and
``AsyncSession`` are mocked so the runner can run its stream/save
logic without network calls. Callbacks are exercised via
``MagicMock(spec=CaptioningCallbacks)`` so the mock's auto-assertions
catch any missing/extra invocations.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
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
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)):
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
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(token=token), options, callbacks):
            pass

        headers = patched_async_session.call_args.kwargs["headers"]
        if expected is None:
            assert "Authorization" not in headers
        else:
            assert headers["Authorization"] == expected

    @pytest.mark.asyncio
    async def test_aenter_passes_http_timeouts_to_session(self, patched_async_session, patched_model):
        timeouts = HTTPTTimeouts(connect=5.0, read=10.0, write=15.0, pool=20.0)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks, http_timeouts=timeouts):
            pass

        kwargs = patched_async_session.call_args.kwargs
        assert kwargs["connect_timeout"] == 5.0
        assert kwargs["read_timeout"] == 10.0
        assert kwargs["write_timeout"] == 15.0
        assert kwargs["pool_timeout"] == 20.0

    @pytest.mark.asyncio
    async def test_aexit_closes_session_and_logs_usage(self, patched_async_session, patched_model):
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)):
            pass

        patched_model.log_usage.assert_called_once()
        patched_async_session.return_value.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aenter_propagates_value_error_from_model_creation(self, patched_async_session):
        with patch(_PATCH_APICAPTIONER_CREATE, new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = ValueError("bad config")
            with pytest.raises(ValueError, match="bad config"):
                async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)):
                    pass


class TestCaptionImage:
    """caption_image streams, saves, and fires the right callbacks."""

    @pytest.mark.asyncio
    async def test_streams_tokens_fires_callbacks_and_saves(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream("foo", " ", "bar"))
        image = _real_image(tmp_path / "img.jpg")
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            caption = await runner.caption_image(image)

        assert caption == "foo bar"
        callbacks.on_image_started.assert_called_once_with(image)
        callbacks.on_token.assert_has_calls([call("foo"), call(" "), call("bar")])
        callbacks.on_image_captioned.assert_called_once()
        assert image.caption_path.read_text() == "foo bar"
        assert image.history_path.exists()

    @pytest.mark.asyncio
    async def test_draft_mode_writes_to_draft_path_only(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        options = CaptionJobOptions(draft="gemma")
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_image(image)

        assert image.draft_path("gemma").read_text() == "hello world"
        assert not image.caption_path.exists()
        assert not image.history_path.exists()

    @pytest.mark.asyncio
    async def test_empty_caption_does_not_save_or_fire_captioned(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream())
        image = _real_image(tmp_path / "img.jpg")
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        async with CaptioningRunner(_config(), options, callbacks) as runner:
            caption = await runner.caption_image(image)

        assert caption == ""
        callbacks.on_image_captioned.assert_not_called()
        callbacks.on_before_save.assert_not_called()
        assert not image.caption_path.exists()

    @pytest.mark.asyncio
    async def test_before_save_receives_normal_mode_paths(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_image(image)

        callbacks.on_before_save.assert_called_once()
        paths = callbacks.on_before_save.call_args[0][0]
        assert set(paths) == {str(image.caption_path), str(image.toml_path), str(image.history_path)}

    @pytest.mark.asyncio
    async def test_before_save_receives_draft_mode_path(self, patched_async_session, patched_model, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        options = CaptionJobOptions(draft="gemma")
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_image(image)

        callbacks.on_before_save.assert_called_once_with([str(image.draft_path("gemma"))])

    @pytest.mark.asyncio
    async def test_before_save_called_before_writes(self, patched_async_session, patched_model, tmp_path):
        """on_before_save must fire BEFORE the file writes."""
        image = _real_image(tmp_path / "img.jpg")
        call_order: list[str] = []

        callbacks = AsyncMock(spec=CaptioningCallbacks)
        callbacks.on_before_save = AsyncMock(side_effect=lambda paths: call_order.append("before_save"))

        original_update = image.update_caption

        def tracked_update(caption):
            call_order.append("write_caption")
            original_update(caption)

        image.update_caption = tracked_update

        async with CaptioningRunner(_config(), CaptionJobOptions(), callbacks) as runner:
            await runner.caption_image(image)

        assert call_order == ["before_save", "write_caption"]

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
            async with CaptioningRunner(_config(), CaptionJobOptions(), callbacks) as runner:
                await runner.caption_image(image)

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
            async with CaptioningRunner(_config(), CaptionJobOptions(), callbacks) as runner:
                await runner.caption_image(image)

        # Cancellation is not a per-image error — on_image_error must not fire
        callbacks.on_image_error.assert_not_called()


class TestCaptionImageDryRun:
    """caption_image_dry_run streams without saving."""

    @pytest.mark.asyncio
    async def test_returns_caption_text_and_streams_callbacks(self, patched_async_session, patched_model, tmp_path):
        patched_model.predict_stream = MagicMock(side_effect=lambda *a, **k: _fake_stream("a", "b", "c"))
        image = _real_image(tmp_path / "img.jpg")
        callbacks = AsyncMock(spec=CaptioningCallbacks)
        async with CaptioningRunner(_config(), CaptionJobOptions(), callbacks) as runner:
            caption = await runner.caption_image_dry_run(image)

        assert caption == "abc"
        callbacks.on_image_started.assert_called_once_with(image)
        callbacks.on_token.assert_has_calls([call("a"), call("b"), call("c")])
        callbacks.on_image_captioned.assert_not_called()
        callbacks.on_before_save.assert_not_called()
        assert not image.caption_path.exists()
        assert not image.history_path.exists()


class TestExtrasForwarded:
    """caption_rounds, extra_messages, prediction_context are forwarded to the model."""

    @pytest.mark.asyncio
    async def test_extra_messages_forwarded(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.captioner import ReplyRound

        image = _real_image(tmp_path / "img.jpg")
        messages = [ReplyRound(role="user", content="hi")]
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)) as runner:
            await runner.caption_image_dry_run(image, extra_messages=messages)

        assert patched_model.predict_stream.call_args.kwargs["extra_messages"] is messages

    @pytest.mark.asyncio
    async def test_caption_rounds_forwarded(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.captioner import CaptionerRound

        image = _real_image(tmp_path / "img.jpg")
        rounds = [CaptionerRound(iteration=1, caption="prev")]
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)) as runner:
            await runner.caption_image_dry_run(image, caption_rounds=rounds)

        assert patched_model.predict_stream.call_args.kwargs["caption_rounds"] is rounds

    @pytest.mark.asyncio
    async def test_prediction_context_forwarded_when_provided(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.prediction import PredictionContext

        image = _real_image(tmp_path / "img.jpg")
        ctx = PredictionContext()
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)) as runner:
            await runner.caption_image_dry_run(image, prediction_context=ctx)

        assert patched_model.predict_stream.call_args.kwargs["prediction_context"] is ctx

    @pytest.mark.asyncio
    async def test_prediction_context_default_when_not_provided(self, patched_async_session, patched_model, tmp_path):
        from yadc.core.prediction import PredictionContext

        image = _real_image(tmp_path / "img.jpg")
        async with CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)) as runner:
            await runner.caption_image_dry_run(image)

        assert isinstance(patched_model.predict_stream.call_args.kwargs["prediction_context"], PredictionContext)


class TestUsedWithoutContextManager:
    """caption_image and caption_image_dry_run require the async context manager."""

    @pytest.mark.asyncio
    async def test_caption_image_asserts_without_aenter(self, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        with pytest.raises(AssertionError):
            await CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)).caption_image(image)

    @pytest.mark.asyncio
    async def test_caption_image_dry_run_asserts_without_aenter(self, tmp_path):
        image = _real_image(tmp_path / "img.jpg")
        with pytest.raises(AssertionError):
            await CaptioningRunner(_config(), CaptionJobOptions(), AsyncMock(spec=CaptioningCallbacks)).caption_image_dry_run(image)


class _Gate:
    """Test helper: per-call count + a shared event that controls when streams complete.

    Each ``predict_stream`` call increments ``in_flight`` and awaits
    ``release`` before yielding. Lets tests block N streams open at
    once to assert the runner's concurrency bound, then release them
    all in one go.
    """

    def __init__(self, release: asyncio.Event, tokens: tuple[str, ...] = ("ok",)):
        self.in_flight = 0
        self.max_in_flight = 0
        self._release = release
        self._tokens = tokens

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await self._release.wait()
            for token in self._tokens:
                yield token
        finally:
            self.in_flight -= 1

    def attach(self, model: MagicMock) -> None:
        model.predict_stream = MagicMock(side_effect=self.stream)


def _many_images(tmp_path: Path, n: int) -> list[DatasetImage]:
    return [_real_image(tmp_path / f"img_{i:03d}.jpg") for i in range(n)]


class TestCaptionImages:
    """caption_images: parallel captioning with bounded concurrency."""

    @pytest.mark.asyncio
    async def test_empty_images_is_a_noop(self, patched_async_session, patched_model, tmp_path):
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_images([], max_concurrent=4)
        patched_model.predict_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_concurrent_must_be_at_least_one(self, patched_async_session, patched_model, tmp_path):
        images = _many_images(tmp_path, 2)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
                await runner.caption_images(images, max_concurrent=0)
            with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
                await runner.caption_images(images, max_concurrent=-1)

    @pytest.mark.asyncio
    async def test_max_concurrent_one_processes_serially(self, patched_async_session, patched_model, tmp_path):
        """max_concurrent=1 is equivalent to a sequential loop."""
        release = asyncio.Event()
        gate = _Gate(release, tokens=("a",))
        gate.attach(patched_model)
        images = _many_images(tmp_path, 5)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            task = asyncio.create_task(runner.caption_images(images, max_concurrent=1))
            # Give the runner a chance to start the first stream
            await asyncio.sleep(0.01)
            assert gate.in_flight == 1
            release.set()
            await task

        assert gate.max_in_flight == 1
        assert patched_model.predict_stream.call_count == 5
        for image in images:
            assert image.caption_path.read_text() == "a"
        assert callbacks.on_image_captioned.call_count == 5

    @pytest.mark.asyncio
    async def test_max_concurrent_bounds_in_flight_count(self, patched_async_session, patched_model, tmp_path):
        """With max_concurrent=N, at most N streams are in flight at once."""
        release = asyncio.Event()
        gate = _Gate(release, tokens=("ok",))
        gate.attach(patched_model)
        images = _many_images(tmp_path, 10)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            task = asyncio.create_task(runner.caption_images(images, max_concurrent=3))
            # Wait for the gate to fill
            for _ in range(100):
                if gate.in_flight == 3:
                    break
                await asyncio.sleep(0.005)
            assert gate.in_flight == 3
            assert gate.max_in_flight == 3
            release.set()
            await task

        assert gate.max_in_flight == 3
        assert patched_model.predict_stream.call_count == 10
        assert callbacks.on_image_captioned.call_count == 10

    @pytest.mark.asyncio
    async def test_per_image_error_does_not_fail_batch(self, patched_async_session, patched_model, tmp_path):
        """A failing image does not abort siblings; only the failing image is reported."""
        images = _many_images(tmp_path, 3)
        # Image at index 1 raises a per-image error (not API-attributable)
        call_count = {"n": 0}

        def maybe_raise(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 2:

                async def _raise():
                    raise ValueError("api did not return text")
                    yield  # pragma: no cover

                return _raise()
            return _fake_stream("ok")

        patched_model.predict_stream = MagicMock(side_effect=maybe_raise)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_images(images, max_concurrent=3)

        # All three were attempted
        assert call_count["n"] == 3
        # The two succeeding images wrote captions
        assert images[0].caption_path.read_text() == "ok"
        assert images[2].caption_path.read_text() == "ok"
        # The failing image was reported via on_image_error, NOT on_image_captioned
        assert callbacks.on_image_error.call_count == 1
        failing_call = callbacks.on_image_error.call_args
        assert failing_call.args[0] is images[1]
        assert "api did not return text" in failing_call.args[1]
        # Two successful captions
        assert callbacks.on_image_captioned.call_count == 2

    @pytest.mark.asyncio
    async def test_callbacks_fire_per_image_out_of_order_is_ok(self, patched_async_session, patched_model, tmp_path):
        """Callbacks may fire in completion order, not submission order."""
        release = asyncio.Event()
        gate = _Gate(release, tokens=("x",))
        gate.attach(patched_model)
        images = _many_images(tmp_path, 4)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            task = asyncio.create_task(runner.caption_images(images, max_concurrent=4))
            for _ in range(100):
                if gate.in_flight == 4:
                    break
                await asyncio.sleep(0.005)
            assert gate.in_flight == 4
            release.set()
            await task

        # All four were started and completed
        assert {id(c.args[0]) for c in callbacks.on_image_captioned.call_args_list} == {id(img) for img in images}

    @pytest.mark.asyncio
    async def test_cancellation_cancels_siblings_and_propagates(self, patched_async_session, patched_model, tmp_path):
        """Cancelling the awaiting task cancels in-flight siblings and re-raises."""
        release = asyncio.Event()
        gate = _Gate(release, tokens=("x",))
        gate.attach(patched_model)
        images = _many_images(tmp_path, 5)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            task = asyncio.create_task(runner.caption_images(images, max_concurrent=3))
            # Wait until the semaphore is full
            for _ in range(100):
                if gate.in_flight == 3:
                    break
                await asyncio.sleep(0.005)
            assert gate.in_flight == 3
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # All in-flight streams were released (gate drained)
        assert gate.in_flight == 0
        # The 2 queued images never started a stream
        assert patched_model.predict_stream.call_count == 3

    @pytest.mark.asyncio
    async def test_api_error_threshold_aborts_batch(self, patched_async_session, patched_model, tmp_path):
        """N consecutive identical API errors abort the batch."""
        from yadc.core.captioning.runner import BatchAbortedError

        images = _many_images(tmp_path, 5)
        call_count = {"n": 0}

        def raise_api_error(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1

            async def _raise():
                raise ValueError("api returned an error (http 401): authentication failure")
                yield  # pragma: no cover

            return _raise()

        patched_model.predict_stream = MagicMock(side_effect=raise_api_error)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            with pytest.raises(BatchAbortedError, match="3 consecutive API errors"):
                await runner.caption_images(images, max_concurrent=5)

        # Threshold is 3: at most 3 in flight when the abort fires; the
        # 4th and 5th never start. We don't assert the exact number
        # because timing is scheduler-dependent, but at least 3 and at
        # most 5 should have been attempted.
        assert 3 <= call_count["n"] <= 5
        # The errors were reported via on_image_error
        assert callbacks.on_image_error.call_count >= 3

    @pytest.mark.asyncio
    async def test_mixed_errors_dont_trigger_abort(self, patched_async_session, patched_model, tmp_path):
        """Image errors between API errors reset the consecutive counter.

        Uses ``max_concurrent=1`` so the per-image error order is
        deterministic; the abort heuristic is independent of
        concurrency and the behaviour is easier to reason about when
        sequenced.
        """

        images = _many_images(tmp_path, 6)
        # Sequence: API, API, image, API, API, image -> no abort
        sequence = [
            "api returned an error (http 401): authentication failure",
            "api returned an error (http 401): authentication failure",
            "api did not return text",  # image-attributable; resets counter
            "api returned an error (http 401): authentication failure",
            "api returned an error (http 401): authentication failure",
            "image not found",  # image-attributable
        ]
        call_count = {"n": 0}

        def per_call(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            message = sequence[call_count["n"] - 1]

            if "401" in message:

                async def _raise_api():
                    raise ValueError(message)
                    yield  # pragma: no cover

                return _raise_api()

            async def _raise_image():
                raise ValueError(message)
                yield  # pragma: no cover

            return _raise_image()

        patched_model.predict_stream = MagicMock(side_effect=per_call)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            # Should NOT raise BatchAbortedError
            await runner.caption_images(images, max_concurrent=1)

        assert call_count["n"] == 6
        assert callbacks.on_image_error.call_count == 6
        assert callbacks.on_image_captioned.call_count == 0

    @pytest.mark.asyncio
    async def test_changing_api_signature_resets_consecutive_counter(self, patched_async_session, patched_model, tmp_path):
        """Different API error signatures don't accumulate toward the threshold.

        ``max_concurrent=1`` keeps the per-image error order
        deterministic; this test is about the signature counter, not
        concurrency.
        """

        images = _many_images(tmp_path, 4)
        # 2x 401, 2x 404 — neither pair reaches the threshold of 3
        sequence = [
            "api returned an error (http 401): authentication failure",
            "api returned an error (http 401): authentication failure",
            "api returned an error (http 404): model not found",
            "api returned an error (http 404): model not found",
        ]
        call_count = {"n": 0}

        def per_call(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            message = sequence[call_count["n"] - 1]

            async def _raise():
                raise ValueError(message)
                yield  # pragma: no cover

            return _raise()

        patched_model.predict_stream = MagicMock(side_effect=per_call)
        options = CaptionJobOptions()
        callbacks = AsyncMock(spec=CaptioningCallbacks)

        async with CaptioningRunner(_config(), options, callbacks) as runner:
            await runner.caption_images(images, max_concurrent=1)

        assert call_count["n"] == 4
        assert callbacks.on_image_error.call_count == 4
