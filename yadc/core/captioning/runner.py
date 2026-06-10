"""Captioning runner — the shared model-create / stream / save loop.

Used by both the CLI's ``yadc caption`` command and the API's
``AsyncCaptionJob`` to do the actual captioning work. The runner is a
pure async class — no DI, no click, no Quart. Callers plug in their own
callbacks for output / event emission / cancellation.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from logging import Logger
from typing import Any, Protocol, runtime_checkable

from yadc.captioners.api import APICaptioner
from yadc.captioners.api.async_session import AsyncSession
from yadc.captioners.api.utils.cache import HTTPResponseCache
from yadc.captioners.api.utils.response_logger import ResponseLogger
from yadc.core.captioner import CaptionerRound, ReplyRound
from yadc.core.config import Config
from yadc.core.dataset import DatasetImage
from yadc.core.prediction import PredictionContext

from .options import CaptionJobOptions

# ---------------------------------------------------------------------------
# Batch error handling
# ---------------------------------------------------------------------------


class BatchAbortedError(Exception):
    """Raised by :meth:`CaptioningRunner.caption_images` when the batch is aborted.

    The runner aborts the batch when it sees ``_BATCH_API_ERROR_THRESHOLD``
    consecutive API-attributable errors with the same signature (e.g. three
    ``http 401`` errors in a row). The signature is included in the
    exception message so callers can show a clear, actionable error to
    the user.
    """


# Substrings in the normalised error message that indicate the API
# itself is at fault (as opposed to a per-image issue). The runner
# tracks consecutive errors matching these patterns and aborts the
# batch once the threshold is reached.
#
# Note: we match by substring on the message returned from
# ``ErrorNormalizationMixin._normalize_error`` (see
# ``yadc/captioners/api/utils/error_normalization.py``). The strings
# are deliberately conservative — anything not in this list is treated
# as image-attributable and does NOT count toward the abort threshold.
#
# Each pattern is a (compiled regex, signature-format) pair. The
# signature returned by ``_classify_error_message`` includes the
# captured group (e.g. the HTTP status code) so that a 401 and a 404
# are treated as distinct API problems and don't accumulate toward the
# same abort threshold.
_API_ERROR_SIGNATURE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^api returned an error \(http (\d+)\)"), "http {0}"),
    (re.compile(r"^api returned an error \(generation (\d+)\)"), "generation {0}"),
    (re.compile(r"^api returned an error \(moderation (\d+)\)"), "moderation {0}"),
    (re.compile(r"^Connection closed unexpectedly"), "connection closed"),
    (re.compile(r"^api unavailable"), "api unavailable"),
)

_BATCH_API_ERROR_THRESHOLD: int = 3


def _classify_error_message(message: str) -> str | None:
    """Return a stable signature for an API-attributable error, or ``None``.

    Used to group consecutive API errors so the runner can detect
    persistent backend issues (e.g. three 401s in a row) and abort the
    batch. Image-attributable errors return ``None`` and do not affect
    the consecutive-API-error counter.

    Distinct signatures per error type — e.g. ``"http 401"`` and
    ``"http 404"`` — so an interleaving of unrelated API errors does
    not accumulate toward the abort threshold.
    """
    for pattern, signature_format in _API_ERROR_SIGNATURE_PATTERNS:
        match = pattern.match(message)
        if match is None:
            continue
        groups = match.groups()
        if groups:
            return signature_format.format(*groups)
        return signature_format
    return None


@dataclass
class _BatchErrorTracker:
    """Counts consecutive API errors of the same signature.

    A success or an image-attributable error resets the counter. When
    the counter reaches ``_BATCH_API_ERROR_THRESHOLD`` identical
    signatures, ``record_error`` raises :class:`BatchAbortedError` to
    abort the batch.
    """

    _consecutive: int = 0
    _last_signature: str | None = None

    def record_success(self) -> None:
        self._consecutive = 0
        self._last_signature = None

    def record_error(self, exc: BaseException) -> None:
        """Record an error from a finished per-image task.

        Image-attributable errors (no signature match) reset the
        counter — they break the chain of identical API errors. A
        new API signature (e.g. switching from 401 to 404) also
        resets the counter to 1. The same API signature increments
        the counter. Raises :class:`BatchAbortedError` once the
        threshold is reached.
        """
        signature = _classify_error_message(str(exc))
        if signature is None:
            # Image-attributable: not a persistent API problem, reset
            # the consecutive-API-error chain.
            self._consecutive = 0
            self._last_signature = None
            return
        if signature == self._last_signature:
            self._consecutive += 1
        else:
            self._consecutive = 1
            self._last_signature = signature
        if self._consecutive >= _BATCH_API_ERROR_THRESHOLD:
            raise BatchAbortedError(f"Aborting batch after {self._consecutive} consecutive API errors with signature '{signature}'. Last error: {exc}")


@runtime_checkable
class CaptioningCallbacks(Protocol):
    """Callback interface for captioning events.

    Any class with the methods below (matching signatures)
    satisfies this protocol structurally — callers can pass ``self`` of
    a service class (e.g. the API's ``AsyncCaptionJob``) directly as
    the ``callbacks`` argument to the runner.

    All methods are ``async def`` so implementations can do async work
    directly (e.g. ``await asyncio.Lock``) without scheduling it as a
    separate task. The runner ``await``s each method in turn, so
    callbacks run sequentially in the runner's task.

    ``on_image_captioned`` is called only when the caption is non-empty
    and was successfully saved. ``on_image_error`` is called on any
    non-cancellation exception, immediately before the exception is
    re-raised so the caller can decide whether to continue or abort.
    """

    async def on_token(self, token: str) -> None: ...
    async def on_image_started(self, image: DatasetImage) -> None: ...
    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None: ...
    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None: ...
    async def on_before_save(self, paths: list[str]) -> None: ...


@dataclass
class HTTPTTimeouts:
    """HTTP client timeouts passed to ``AsyncSession``.

    Mirrors the relevant ``Configuration`` fields from the API. The CLI
    uses ``AsyncSession``'s built-in defaults by leaving this at its
    constructed defaults.
    """

    connect: float = 30.0
    read: float | None = None
    write: float = 30.0
    pool: float = 30.0


class CaptioningRunner:
    """Async captioning loop shared by the CLI and the API.

    Use as an async context manager — model + session are created on
    entry and torn down on exit (``log_usage`` + ``aclose``).

    The runner does NOT loop over images itself; callers iterate and
    call :meth:`caption_image` (or :meth:`caption_image_dry_run`) for
    each. This keeps the runner simple and lets each side plug in its
    own batch control (CLI's interactive prompts, API's stop event,
    etc.).
    """

    def __init__(
        self,
        config: Config,
        options: CaptionJobOptions,
        callbacks: CaptioningCallbacks,
        *,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        http_timeouts: HTTPTTimeouts | None = None,
        logger: Logger | None = None,
    ):
        self._config: Config = config
        self._options: CaptionJobOptions = options
        self._callbacks: CaptioningCallbacks = callbacks
        self._cache: HTTPResponseCache | None = cache
        self._response_logger: ResponseLogger | None = response_logger
        self._http_timeouts: HTTPTTimeouts = http_timeouts or HTTPTTimeouts()
        self._logger: Logger = logger or logging.getLogger(__name__)

        self._model: APICaptioner | None = None
        self._async_session: AsyncSession | None = None
        self._conversation_overrides: dict[str, Any] = config.settings.advanced.model_dump()

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> "CaptioningRunner":
        async_headers: dict[str, str] = {}
        if self._config.api.token:
            async_headers["Authorization"] = f"Bearer {self._config.api.token}"

        self._async_session = AsyncSession(
            self._config.api.url,
            headers=async_headers,
            connect_timeout=self._http_timeouts.connect,
            read_timeout=self._http_timeouts.read,
            write_timeout=self._http_timeouts.write,
            pool_timeout=self._http_timeouts.pool,
        )
        self._model = await APICaptioner.create(
            api_url=self._config.api.url,
            api_token=self._config.api.token,
            prompt_template=self._config.prompt.template,
            store_conversation=self._config.settings.store_conversation,
            image_quality=self._config.settings.image_quality,
            reasoning=self._config.reasoning.enable,
            reasoning_effort=self._config.reasoning.thinking_effort,
            reasoning_exclude_output=self._config.reasoning.exclude_from_output,
            cache=self._cache,
            response_logger=self._response_logger,
            async_session=self._async_session,
        )
        await self._model.load_model(self._config.api.model_name)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._model is not None:
            self._model.log_usage()
        if self._async_session is not None:
            await self._async_session.aclose()

    @property
    def model(self) -> APICaptioner:
        """The underlying ``APICaptioner``. Available inside the ``async with`` block.

        Exposed for callers that need direct model access (e.g. the CLI's
        ``api_type`` prefill warning, or showing prompts before calling).
        """
        assert self._model is not None, "CaptioningRunner used outside 'async with'"
        return self._model

    # -- per-image captioning -----------------------------------------------

    async def caption_image(
        self,
        image: DatasetImage,
        *,
        caption_rounds: list[CaptionerRound] | None = None,
        extra_messages: list[ReplyRound] | None = None,
        prediction_context: PredictionContext | None = None,
    ) -> str:
        """Stream + accumulate + save. Returns the saved caption text.

        Returns ``""`` if the model produced an empty caption (in which
        case no save happens and ``on_image_captioned`` is not called).

        ``prediction_context`` is exposed as a parameter so callers
        that need to read the populated ``PredictionContext`` after the
        call (e.g. the CLI's reply flow, which reads ``reasoning``) can
        pass their own. If ``None``, a fresh ``PredictionContext`` is
        created and the caller has no way to read it — fine for the
        API, which discards the data.

        Before writing caption/TOML/history (or the draft file in
        draft mode), the runner calls
        ``callbacks.on_before_save([...])`` with the absolute paths so
        the caller can register expected file changes with a watcher.
        """
        assert self._model is not None, "CaptioningRunner used outside 'async with'"

        await self._callbacks.on_image_started(image)
        t0 = time.monotonic()

        try:
            caption = await self._stream_image(
                image,
                caption_rounds=caption_rounds,
                extra_messages=extra_messages,
                prediction_context=prediction_context,
            )
        except asyncio.CancelledError:
            self._logger.debug("Captioning of %s was cancelled", image.path)
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            await self._callbacks.on_image_error(image, str(exc), duration_ms)
            raise

        if not caption:
            return ""

        await self.save_caption(image, caption)

        duration_ms = int((time.monotonic() - t0) * 1000)
        await self._callbacks.on_image_captioned(image, duration_ms)
        return caption

    async def caption_image_dry_run(
        self,
        image: DatasetImage,
        *,
        caption_rounds: list[CaptionerRound] | None = None,
        extra_messages: list[ReplyRound] | None = None,
        prediction_context: PredictionContext | None = None,
    ) -> str:
        """Stream + accumulate without saving. Returns the caption text.

        Used by the CLI's interactive flow where the user may
        retry/reject before committing the caption to disk. Returns
        ``""`` if the model produced an empty caption.

        Does not call ``on_before_save`` or ``on_image_captioned`` — there is nothing to save.
        """
        assert self._model is not None, "CaptioningRunner used outside 'async with'"

        await self._callbacks.on_image_started(image)

        return await self._stream_image(
            image,
            caption_rounds=caption_rounds,
            extra_messages=extra_messages,
            prediction_context=prediction_context,
        )

    async def caption_images(
        self,
        images: list[DatasetImage],
        *,
        max_concurrent: int = 1,
        caption_rounds: list[CaptionerRound] | None = None,
        extra_messages: list[ReplyRound] | None = None,
    ) -> None:
        """Caption all images, up to ``max_concurrent`` in flight at once.

        Equivalent to iterating and calling :meth:`caption_image` for
        each image, but with a semaphore gating how many requests are
        in flight at once. ``max_concurrent == 1`` reproduces the
        original sequential behaviour.

        Per-image errors (already reported via
        ``callbacks.on_image_error``) are caught and do not fail the
        batch. Cancellation propagates: when the awaiting task is
        cancelled, all in-flight siblings are cancelled and the
        :class:`asyncio.CancelledError` is re-raised.

        The batch is aborted with :class:`BatchAbortedError` when
        ``_BATCH_API_ERROR_THRESHOLD`` consecutive API-attributable
        errors share the same signature (e.g. three 401 responses in
        a row). Image-attributable errors do not count toward this
        threshold. Sibling tasks are cancelled and the abort error is
        re-raised; in-flight HTTP requests receive
        :class:`asyncio.CancelledError` and unwind via the underlying
        ``httpx`` client.

        Callbacks fire on each per-image task independently and may
        be invoked out of submission order (callers should match
        results by ``DatasetImage``, not position).
        """
        assert self._model is not None, "CaptioningRunner used outside 'async with'"
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        if not images:
            return

        sem = asyncio.Semaphore(max_concurrent)
        tracker = _BatchErrorTracker()

        async def _caption_one(image: DatasetImage) -> None:
            async with sem:
                try:
                    caption = await self.caption_image(
                        image,
                        caption_rounds=caption_rounds,
                        extra_messages=extra_messages,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # ``on_image_error`` was already fired by
                    # ``caption_image``; record for the abort heuristic
                    # and continue. May raise ``BatchAbortedError`` if
                    # the API-error threshold is reached.
                    tracker.record_error(exc)
                    return
                if caption:
                    tracker.record_success()

        tasks = [asyncio.create_task(_caption_one(img)) for img in images]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Cancellation or abort: cancel in-flight siblings and
            # drain their unwinds, then re-raise the original.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    # -- internals ----------------------------------------------------------

    async def _stream_image(
        self,
        image: DatasetImage,
        *,
        caption_rounds: list[CaptionerRound] | None = None,
        extra_messages: list[ReplyRound] | None = None,
        prediction_context: PredictionContext | None = None,
    ) -> str:
        """Stream tokens from the model, accumulate, return the caption text.

        Re-raises :class:`asyncio.CancelledError` and any other
        exception without invoking ``on_image_error`` — the caller
        (``caption_image``) handles that.
        """
        assert self._model is not None, "CaptioningRunner used outside 'async with'"

        predict_kwargs: dict[str, Any] = dict(
            max_new_tokens=self._config.settings.max_tokens,
            conversation_overrides=self._conversation_overrides,
            prefill=self._config.settings.advanced.assistant_prefill,
            drafts=image.read_all_drafts() or None,
            prediction_context=prediction_context or PredictionContext(),
        )
        if extra_messages is not None:
            predict_kwargs["extra_messages"] = extra_messages
        if caption_rounds is not None:
            predict_kwargs["caption_rounds"] = caption_rounds

        caption_parts: list[str] = []
        async for token in self._model.predict_stream(image, **predict_kwargs):
            caption_parts.append(token)
            await self._callbacks.on_token(token)

        return "".join(caption_parts).strip()

    async def save_caption(self, image: DatasetImage, caption: str) -> None:
        """Write caption (or draft) to disk, notifying callbacks before writes.

        Public API so callers that used :meth:`caption_image_dry_run`
        (e.g. the CLI's interactive flow) can commit the caption after
        the user accepts it. Honors ``options.draft`` (writes the
        draft file instead).

        Calls ``callbacks.on_before_save`` with the file paths before
        any writes.
        """
        if self._options.draft:
            paths = [str(image.draft_path(self._options.draft))]
        else:
            paths = [
                str(image.caption_path),
                str(image.toml_path),
                str(image.history_path),
            ]

        await self._callbacks.on_before_save(paths)

        if self._options.draft:
            image.write_draft(self._options.draft, caption)
            return

        if image.caption:
            image.save_history(when_not_exists=True)
        image.update_caption(caption)
        image.save_history(when_not_exists=False)
