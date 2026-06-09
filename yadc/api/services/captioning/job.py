"""Async captioning job — pure state machine and ``CaptioningCallbacks`` implementation.

``AsyncCaptionJob`` owns job state (processed/total/errors, status
transitions), the asyncio task lifecycle (start/stop/wait), and
implements the ``CaptioningCallbacks`` Protocol for the runner.

All API infrastructure (preflight, SSE events, watcher registration)
lives in ``AsyncCaptionJobRunner`` — the job delegates to it via the
``_runner`` reference and never imports DI services directly.
"""

import asyncio
import inspect
import time
from typing import Any, Callable

from yadc.core import Config, DatasetImage
from yadc.core.captioning import CaptionJobOptions

from .job_runner import AsyncCaptionJobRunner
from .models import JobInfo, JobStatus, RefineOptions


class AsyncCaptionJob:
    """Runs a single captioning pass over a dataset as an asyncio task.

    Implements the ``CaptioningCallbacks`` Protocol — pass ``self`` to
    ``CaptioningRunner.caption_image`` and the job's state tracking
    hooks into the runner's lifecycle.

    This class has **no DI imports** — all infrastructure is provided
    by ``AsyncCaptionJobRunner`` via ``self._runner``.
    """

    def __init__(
        self,
        dataset_name: str,
        options: CaptionJobOptions,
        job_id: str,
        on_done: Callable[[], Any],
        runner: AsyncCaptionJobRunner,
        refine: RefineOptions | None = None,
    ):
        self._dataset_name: str = dataset_name
        self._opts: CaptionJobOptions = options
        self._job_id: str = job_id
        self._on_done: Callable[[], Any] = on_done
        self._runner: AsyncCaptionJobRunner = runner
        self._refine: RefineOptions | None = refine

        # State (guarded by _state_lock). An asyncio.Lock because the
        # CaptioningCallbacks methods are async and run on the event
        # loop thread.
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._status: JobStatus = "running"
        self._stop_event: asyncio.Event = asyncio.Event()
        self._processed: int = 0
        self._total: int = 0
        self._errors: int = 0
        self._error: str | None = None
        self._error_messages: list[str] = []

        self._api_url: str = ""
        self._api_model_name: str = ""
        self._config: Any | None = None  # set by preflight via runner

        # Cached preflight result. ``CaptioningService.start_job_async``
        # does a synchronous preflight to fail-fast on config errors
        # and to return an accurate initial ``JobInfo``; the result is
        # stashed here so ``_ado_run`` doesn't re-parse the config and
        # re-query the database. ``None`` when the job was created
        # outside the service (e.g. in tests) — ``_ado_run`` falls
        # back to ``self._runner.preflight(...)`` in that case.
        self._preflighted: tuple[Config, list[DatasetImage]] | None = None

        # ``time.monotonic()`` at the moment the job actually starts
        # running (set in ``_arun``).  Used to compute ``elapsed`` for
        # the status event.  ``None`` until then so a status snapshot
        # taken before the job runs (e.g. from ``snapshot()``) reports
        # ``elapsed=0`` rather than a huge negative number.
        self._started_at: float | None = None

        self._task: asyncio.Task[Any] | None = None
        self._finished_at: float | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def finished_at(self) -> float | None:
        return self._finished_at

    def start(self) -> None:
        self._task = asyncio.create_task(self._arun())

    def request_stop(self) -> None:
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def set_preflight(self, config: Config, to_do: list[DatasetImage]) -> None:
        """Cache the preflight result so ``_ado_run`` can reuse it.

        Must be called before :meth:`start`. Populates the snapshot
        fields (``api_url``, ``api_model_name``, ``total``)
        synchronously so the response from ``start_job_async`` is
        accurate without waiting for the background task to run.
        """
        self._preflighted = (config, to_do)
        self._api_url = config.api.url
        self._api_model_name = config.api.model_name
        self._total = len(to_do)

    async def wait(self, timeout: float | None = None) -> None:
        """Wait for the underlying task to finish."""
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass

    # -- snapshot ------------------------------------------------------------

    async def snapshot(self) -> JobInfo:
        return await self._snapshot_locked()

    async def _snapshot_locked(self) -> JobInfo:
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = time.monotonic() - self._started_at
        async with self._state_lock:
            return JobInfo(
                status=self._status,
                dataset_name=self._dataset_name,
                job_id=self._job_id,
                processed=self._processed,
                total=self._total,
                errors=self._errors,
                error=self._error,
                error_messages=self._error_messages.copy(),
                api_url=self._api_url,
                api_model_name=self._api_model_name,
                elapsed=elapsed,
                # ``opts.max_concurrent`` is ``None`` for the API default
                # but the loader resolves it to an ``int`` before
                # ``_arun`` runs. Coalesce defensively for type-check.
                max_concurrent=self._opts.max_concurrent or 1,
            )

    # -- CaptioningCallbacks Protocol ----------------------------------------

    async def on_token(self, token: str) -> None:  # pyright: ignore[reportUnusedParameter]
        # The API doesn't stream tokens to clients; the final caption is
        # included in ImageCaptionedEvent.caption.
        pass

    async def on_before_save(self, paths: list[str]) -> None:
        self._runner.expect_file_changes(self._dataset_name, paths)

    async def on_image_started(self, image: DatasetImage) -> None:
        self._runner.emit_image_started(self._dataset_name, self._job_id, image)

    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None:
        async with self._state_lock:
            self._processed += 1
        self._runner.emit_image_captioned(self._dataset_name, self._job_id, image, self._api_url, self._api_model_name, duration_ms)
        await self._emit_status()

    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None:
        async with self._state_lock:
            self._errors += 1
            self._error_messages.append(error)
        self._runner.emit_image_error(self._dataset_name, self._job_id, self._api_url, self._api_model_name, image, error, duration_ms)
        await self._emit_status()

    # -- main loop -----------------------------------------------------------

    async def _arun(self) -> None:
        """Entry point for the asyncio task."""
        # Mark the start of the job *before* ``_ado_run`` so the first
        # status emit (right after preflight) already has a meaningful
        # ``elapsed`` value.  ``_arun`` is the natural place because
        # it's the async entry point — ``_started_at`` reflects "when
        # did the runner actually take over", not "when was the job
        # scheduled".
        self._started_at = time.monotonic()
        try:
            await self._ado_run()
        except asyncio.CancelledError:
            await self._set_state(status="cancelled")
            await self._emit_status()
        except Exception as exc:
            await self._set_state(error=str(exc), status="error", set_error=True)
            async with self._state_lock:
                self._error_messages.append(str(exc))
            await self._emit_status()
        finally:
            self._finished_at = time.monotonic()
            # Schedule cleanup as a background task so _arun returns
            # promptly and alive becomes False — this lets callers start
            # a new job immediately after cancellation.
            if inspect.iscoroutinefunction(self._on_done):
                asyncio.create_task(self._on_done())
            else:
                self._on_done()

    async def _ado_run(self) -> None:
        if self._preflighted is not None:
            config, to_do = self._preflighted
        else:
            # Direct callers (tests) bypass the service's synchronous
            # preflight, so resolve here.
            config, to_do = self._runner.preflight(self._dataset_name, self._opts)

        self._api_url = config.api.url
        self._api_model_name = config.api.model_name
        self._config = config

        await self._set_state(total=len(to_do))
        await self._emit_status()

        if not to_do:
            await self._set_state(status="done")
            await self._emit_status()
            return

        runner = self._runner.build_runner(config, self._opts, callbacks=self)

        async with runner:
            if self._refine is not None:
                assert len(to_do) == 1, f"Refine jobs must have exactly one image, got {len(to_do)}"
                img = to_do[0]

                caption = await runner.caption_image_dry_run(img, extra_messages=self._refine.extra_messages)

                async with self._state_lock:
                    self._processed += 1

                if caption:
                    self._runner.emit_image_refined(self._dataset_name, self._job_id, img, caption, self._refine)
            else:
                # Coalesce ``None`` (default) to 1 — the loader
                # normally resolves this earlier, but the runner takes
                # ``int`` and basedpyright doesn't know about the
                # loader side-effect.
                await runner.caption_images(
                    to_do,
                    max_concurrent=self._opts.max_concurrent or 1,
                )

        if self._check_stop():
            await self._set_state(status="cancelled")
        else:
            await self._set_state(status="done")
        await self._emit_status()

    # -- state helpers -------------------------------------------------------

    async def _set_state(
        self,
        *,
        status: JobStatus | None = None,
        total: int | None = None,
        error: str | None = None,
        set_error: bool = False,
    ) -> None:
        async with self._state_lock:
            if status is not None:
                self._status = status
            if total is not None:
                self._total = total
            if set_error:
                self._error = error

    def _check_stop(self) -> bool:
        return self._stop_event.is_set()

    # -- event emission (delegates to runner) --------------------------------

    async def _emit_status(self) -> None:
        snap = await self._snapshot_locked()
        await self._runner.emit_status(self._dataset_name, self._job_id, snap, self._started_at)
