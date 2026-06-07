"""Captioning service — manages background captioning jobs for datasets.

Each dataset can have at most one active captioning job at a time. Jobs run as
``asyncio.Task`` instances and emit ``CaptioningStatusEvent`` via the
``EventDispatcher`` as images are processed.

The actual model-create / stream / save loop lives in
``yadc.core.captioning.CaptioningRunner``. This module orchestrates the
runner: preflight (config + image resolution), per-image event
emission, state tracking, and lifecycle.

Usage from the API layer::

    await svc.start_job_async("my_dataset", options)
    await svc.stop_job_async("my_dataset")
    await svc.get_status_async("my_dataset")
"""

import asyncio
import inspect
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Literal

from yadc.core.captioner import ReplyRound
from yadc.core.captioning import (
    CaptionJobOptions,
    HTTPTTimeouts,
    load_dataset_config,
)
from yadc.core.captioning.runner import CaptioningRunner
from yadc.core.config import Config
from yadc.core.dataset import DatasetImage
from yadc.utils import LRU

from ..configuration import Configuration
from ..events import (
    CaptioningStatusEvent,
    ImageCaptionedEvent,
    ImageCaptionErrorEvent,
    ImageCaptionStartedEvent,
    ImageRefinedEvent,
    ShutdownEvent,
    StartupEvent,
)
from ..modules.dataset_watcher import DatasetWatcherService
from ..modules.event_dispatcher import EventDispatcher, event_handler
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .datasets import DatasetService


@dataclass
class RefineOptions:
    """Parameters for a dry-run refine job (feedback-driven caption refinement).

    Running with refine options active implies a dry run."""

    extra_messages: list[ReplyRound] | None = None
    refine_source: Literal["caption", "draft"] = "caption"
    refine_draft_name: str = ""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

type JobStatus = Literal["idle", "running", "stopping", "error", "done", "cancelled"]


@dataclass
class JobInfo:
    """Snapshot of a running (or recently finished) captioning job."""

    status: JobStatus
    dataset_name: str
    job_id: str = ""
    processed: int = 0
    total: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    api_url: str = ""
    api_model_name: str = ""
    # Seconds since the job started. 0 before the job has actually
    # started.  Returned to the frontend on the job-start / status
    # endpoints for the "elapsed" display.
    elapsed: float = 0.0
    # Configured concurrency for the job.  Returned to the frontend so
    # it can correctly factor the ETA under parallel runs.
    max_concurrent: int = 1


# ---------------------------------------------------------------------------
# Async job runner
# ---------------------------------------------------------------------------


class AsyncCaptionJob:
    """Runs a single captioning pass over a dataset as an asyncio task.

    Implements the ``CaptioningCallbacks`` Protocol — pass ``self`` to
    ``CaptioningRunner.caption_image`` and the job's event emission /
    state tracking hooks into the runner's lifecycle.
    """

    def __init__(
        self,
        dataset_name: str,
        configuration: Configuration,
        dataset_service: DatasetService,
        dataset_watcher: DatasetWatcherService,
        event_dispatcher: EventDispatcher,
        logger: Logger,
        options: CaptionJobOptions,
        on_done: Callable[[], Any],
        job_id: str = "",
        refine: RefineOptions | None = None,
    ):
        self._dataset_name: str = dataset_name
        self._configuration: Configuration = configuration
        self._dataset_service: DatasetService = dataset_service
        self._dataset_watcher: DatasetWatcherService = dataset_watcher
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._logger: Logger = logger
        self._opts: CaptionJobOptions = options
        self._on_done: Callable[[], Any] = on_done
        self._job_id: str = job_id
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
        self._config: Config | None = None  # set by preflight_images

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

    async def on_image_started(self, image: DatasetImage) -> None:
        self._emit_image_started(image)

    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None:
        async with self._state_lock:
            self._processed += 1
        self._emit_image_captioned(image, duration_ms)
        await self._emit_status()

    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None:
        self._logger.warning("Failed to caption %s: %s", image.path, error)
        async with self._state_lock:
            self._errors += 1
            self._error_messages.append(error)
        self._emit_image_error(image, error, duration_ms)
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
            self._logger.info("Captioning job for '%s' cancelled", self._dataset_name)
            await self._set_state(status="cancelled")
            await self._emit_status()
        except Exception as exc:
            self._logger.exception("Captioning job for '%s' failed: %s", self._dataset_name, exc)
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

    def preflight_images(self) -> list[DatasetImage]:
        """Synchronously resolve the dataset and filter images.

        Performs config loading, override application, and dataset
        resolution via :func:`load_dataset_config`. The returned list
        is reordered by database id descending via a single SQL
        query so concurrent captioning starts with the newest images
        first. Stores the parsed :class:`Config` on ``self._config``
        for use by :meth:`_ado_run` (which builds the runner from it).

        Returns the list of images to be processed. Callers can use
        ``len(result)`` to know the total before the async task starts.

        Raises ``ValueError`` on configuration / resolution errors.
        """
        ds_info = self._dataset_service.get_dataset(self._dataset_name)
        if ds_info is None or ds_info.config_path is None:
            raise ValueError(f"Dataset '{self._dataset_name}' not found")

        config_path = Path(ds_info.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        config, images = load_dataset_config(
            config_path,
            self._opts,
            image_path_resolver=self._image_path_resolver,
        )

        self._api_url = config.api.url
        self._api_model_name = config.api.model_name
        self._config = config

        # Reorder the filesystem-resolved list by DB id DESC (newest
        # first). Single SQL query — no N+1. Images not in the DB
        # (e.g. inline extras in the config that haven't been
        # scanned) sort to the end.
        ordered_paths = self._dataset_service.get_image_paths_in_desc_order(self._dataset_name)
        if ordered_paths:
            position = {path: i for i, path in enumerate(ordered_paths)}
            sentinel = len(ordered_paths)
            images.sort(key=lambda img: position.get(img.path, sentinel))

        return images

    def _image_path_resolver(self, image_id: int) -> Path | None:
        """Resolve an image ID to its filesystem path for the loader's image_ids filter."""
        info = self._dataset_service.get_image(self._dataset_name, image_id)
        return Path(info.path) if info else None

    async def _ado_run(self) -> None:
        to_do = self.preflight_images()

        await self._set_state(total=len(to_do))
        await self._emit_status()

        if not to_do:
            self._logger.info("No images to caption for dataset '%s'.", self._dataset_name)
            await self._set_state(status="done")
            await self._emit_status()
            return

        assert self._config is not None, "preflight_images must be called first"

        runner = CaptioningRunner(
            self._config,
            self._opts,
            expected_change_registrar=self._expected_change_registrar,
            logger=self._logger,
            http_timeouts=HTTPTTimeouts(
                connect=self._configuration.http_timeout_connect,
                read=self._configuration.http_timeout_read,
                write=self._configuration.http_timeout_write,
                pool=self._configuration.http_timeout_pool,
            ),
        )

        async with runner:
            if self._refine is not None:
                assert len(to_do) == 1, f"Refine jobs must have exactly one image, got {len(to_do)}"
                img = to_do[0]

                caption = await runner.caption_image_dry_run(img, self, extra_messages=self._refine.extra_messages)

                async with self._state_lock:
                    self._processed += 1

                if caption:
                    self._emit_image_refined(img, caption)
            else:
                # Coalesce ``None`` (default) to 1 — the loader
                # normally resolves this earlier, but the runner takes
                # ``int`` and basedpyright doesn't know about the
                # loader side-effect.
                await runner.caption_images(
                    to_do,
                    self,
                    max_concurrent=self._opts.max_concurrent or 1,
                )

        if self._check_stop():
            await self._set_state(status="cancelled")
        else:
            await self._set_state(status="done")
        await self._emit_status()

        self._logger.info(
            "Captioning finished for '%s': %d/%d processed, %d errors.",
            self._dataset_name,
            self._processed,
            self._total,
            self._errors,
        )

    def _expected_change_registrar(self, paths: list[str]) -> None:
        """Register upcoming file writes with the watcher so it suppresses inotify events.

        Called by ``CaptioningRunner._save_caption`` before writing
        caption/TOML/history (or the draft file in draft mode).
        """
        for path in paths:
            self._dataset_watcher.expect_file_change(self._dataset_name, path)

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

    # -- event emission ------------------------------------------------------

    async def _emit_status(self) -> None:
        snap = await self._snapshot_locked()
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = time.monotonic() - self._started_at
        event = CaptioningStatusEvent(
            status=snap.status,
            dataset_name=snap.dataset_name,
            processed=snap.processed,
            total=snap.total,
            errors=snap.errors,
            job_id=snap.job_id,
            error=snap.error,
            error_messages=snap.error_messages,
            api_url=snap.api_url,
            api_model_name=snap.api_model_name,
            elapsed=elapsed,
            # Coalesce ``None`` (default) to 1 — see ``_snapshot_locked``.
            max_concurrent=self._opts.max_concurrent or 1,
        )
        self._event_dispatcher.dispatch(event)

    def _emit_image_started(self, dataset_image: DatasetImage) -> None:
        info = self._dataset_service.get_image_by_path(self._dataset_name, dataset_image.path)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageCaptionStartedEvent(
                dataset_name=self._dataset_name,
                job_id=self._job_id,
                image_id=info.id,
                file_name=info.file_name,
            )
        )

    def _emit_image_captioned(self, dataset_image: DatasetImage, duration_ms: int) -> None:
        info = self._dataset_service.get_image_by_path(self._dataset_name, dataset_image.path)
        if info is None:
            return
        self._dataset_service.refresh_image_index(self._dataset_name, info.id)
        info = self._dataset_service.get_image(self._dataset_name, info.id)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageCaptionedEvent(
                dataset_name=self._dataset_name,
                job_id=self._job_id,
                id=info.id,
                file_name=info.file_name,
                path=info.path,
                has_caption=info.has_caption,
                has_toml=info.has_toml,
                width=info.width,
                height=info.height,
                draft_names=info.draft_names,
                last_modified_t=info.last_modified_t,
                caption=dataset_image.caption,
                duration_ms=duration_ms,
                api_url=self._api_url,
                api_model_name=self._api_model_name,
            )
        )

    def _emit_image_error(self, dataset_image: DatasetImage, error: str, duration_ms: int) -> None:
        info = self._dataset_service.get_image_by_path(self._dataset_name, dataset_image.path)
        image_id = info.id if info is not None else -1
        self._event_dispatcher.dispatch(
            ImageCaptionErrorEvent(
                dataset_name=self._dataset_name,
                job_id=self._job_id,
                image_id=image_id,
                error=error,
                duration_ms=duration_ms,
                api_url=self._api_url,
                api_model_name=self._api_model_name,
            )
        )

    def _emit_image_refined(self, dataset_image: DatasetImage, caption: str) -> None:
        assert self._refine is not None, "emit image refined called without refine active"

        info = self._dataset_service.get_image_by_path(self._dataset_name, dataset_image.path)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageRefinedEvent(
                dataset_name=self._dataset_name,
                job_id=self._job_id,
                image_id=info.id,
                caption=caption,
                source=self._refine.refine_source,
                draft_name=self._refine.refine_draft_name,
            )
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CaptioningService(Service):
    """Manages async captioning jobs — start, stop, query status."""

    def __init__(
        self,
        dataset_service: DatasetService,
        event_dispatcher: EventDispatcher,
        dataset_watcher: DatasetWatcherService,
        logging: LoggingFactory,
        configuration: Configuration,
    ):
        self._dataset_service: DatasetService = dataset_service
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._dataset_watcher: DatasetWatcherService = dataset_watcher
        self._logger: Logger = logging.get_logger(__name__)
        self._configuration: Configuration = configuration

        self._async_lock: asyncio.Lock = asyncio.Lock()
        self._async_jobs: dict[str, AsyncCaptionJob] = {}

        # Bounded LRU cache for dry-run refine results (key = "dataset/image_id/source_key").
        self._refine_lock: asyncio.Lock = asyncio.Lock()
        self._refine_results: LRU[str, str] = LRU(self._configuration.refine_result_buffer_size)

        self._cleanup_task: asyncio.Task[Any] | None = None

    # -- event handlers ------------------------------------------------------

    @event_handler(StartupEvent)
    async def on_startup(self, event: StartupEvent):  # pyright: ignore[reportUnusedParameter]
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._logger.warning("Cleanup task already running, not starting another")
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    @event_handler(ShutdownEvent)
    async def on_shutdown(self, event: ShutdownEvent):  # pyright: ignore[reportUnusedParameter]
        if self._cleanup_task is None or self._cleanup_task.done():
            return
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass

    # FIXME: event is pushed to sse and to this
    #        an issue might arise if the frontend tries to grab the status before this gets triggered
    #        (in practice, unlikely to happen since this runs in memory, while the frontend is behind network)
    @event_handler(ImageRefinedEvent)
    async def on_image_refined(self, event: ImageRefinedEvent) -> None:
        source_key = f"{event.source}/{event.draft_name}" if event.source == "draft" else "caption"
        key = f"{event.dataset_name}/{event.image_id}/{source_key}"
        async with self._refine_lock:
            self._refine_results[key] = event.caption

    # -- public API ----------------------------------------------------------

    def _mark_expected_changes(self, dataset_name: str, job_id: str) -> None:
        """Tag the next watcher events for this dataset with the captioning job ID."""
        self._dataset_watcher.expect_changes(dataset_name, job_id)

    async def start_job_async(
        self,
        dataset_name: str,
        options: CaptionJobOptions,
        refine: RefineOptions | None = None,
    ) -> JobInfo:
        """Start an async captioning job for *dataset_name*."""
        job_id = uuid.uuid4().hex[:12]

        async with self._async_lock:
            if dataset_name in self._async_jobs and self._async_jobs[dataset_name].alive:
                raise ValueError(f"A captioning job is already running for dataset '{dataset_name}'")

            async def _on_done() -> None:
                await self._cleanup_async(dataset_name)

            job = AsyncCaptionJob(
                dataset_name=dataset_name,
                configuration=self._configuration,
                dataset_service=self._dataset_service,
                dataset_watcher=self._dataset_watcher,
                event_dispatcher=self._event_dispatcher,
                logger=self._logger,
                options=options,
                on_done=_on_done,
                job_id=job_id,
                refine=refine,
            )
            self._async_jobs[dataset_name] = job
            # Synchronous preflight: compute total so the initial status is
            # accurate (avoids returning 0/0 when images are present).  If
            # there is nothing to do we return done immediately without
            # starting a background task.
            try:
                to_do = job.preflight_images()
            except ValueError as exc:
                # NOTE: drop possible zombie process
                del self._async_jobs[dataset_name]
                return JobInfo(
                    status="error",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    error=str(exc),
                )

            if not to_do:
                # NOTE: drop possible zombie process
                del self._async_jobs[dataset_name]
                return JobInfo(
                    status="done",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    total=0,
                    processed=0,
                    errors=0,
                )

            await job._set_state(total=len(to_do))
            if refine is None:
                self._mark_expected_changes(dataset_name, job_id)
            job.start()

        return await self.get_status_async(dataset_name)

    async def stop_job_async(self, dataset_name: str) -> bool:
        """Request cancellation of the running async job for *dataset_name*.

        Waits for the task to actually finish (up to 30 s) so that the
        caller can safely start a new job immediately afterwards.
        """
        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)
            if job is None or not job.alive:
                return False
            job.request_stop()

        # Wait for the task to finish so start_job_async won't reject
        # with "already running".  30 s is a generous ceiling for the
        # current API request to be aborted by the cancellation.
        await job.wait(timeout=30.0)
        return True

    def is_captioning(self, dataset_name: str) -> bool:
        """Return ``True`` if a captioning job is actively running for *dataset_name*."""
        job = self._async_jobs.get(dataset_name)
        return job is not None and job.alive

    async def get_status_async(self, dataset_name: str) -> JobInfo:
        """Return a snapshot of the async job status for *dataset_name*."""
        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)
            if job is None:
                return JobInfo(status="idle", dataset_name=dataset_name)
            return await job.snapshot()

    async def get_refine_result(
        self,
        dataset_name: str,
        image_id: int,
        source: Literal["caption", "draft"] = "caption",
        draft_name: str = "",
    ) -> str | None:
        """Return the latest dry-run refine result for an image, if any."""
        source_key = f"{source}/{draft_name}" if source == "draft" else "caption"
        key = f"{dataset_name}/{image_id}/{source_key}"
        async with self._refine_lock:
            return self._refine_results.get(key)

    # -- private helpers -----------------------------------------------------

    async def _cleanup_async(self, dataset_name: str, sleep_time: float = 5) -> None:
        """Remove finished async jobs after a short delay."""
        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)
            job_id = (await job.snapshot()).job_id if job is not None else ""

        await asyncio.sleep(sleep_time)

        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)

            if job is None or job.alive:
                return

            del self._async_jobs[dataset_name]
        self._dataset_watcher.clear_expected_changes_for_job(dataset_name, job_id)

        # Final rescan to catch any edge cases (files added/removed by
        # other processes during captioning). Per-image refresh_image_index
        # calls already handled the captioning writes, so this is cheap
        # when there are no external changes.
        self._dataset_service.rescan_dataset(dataset_name)

    async def _cleanup_loop(self) -> None:
        """Periodic background task that removes dead jobs after a grace period."""
        while True:
            await asyncio.sleep(self._configuration.captioning_cleanup_interval_seconds)

            now = time.monotonic()
            dead: list[str] = []
            grace = self._configuration.captioning_cleanup_grace_seconds

            async with self._async_lock:
                dataset_jobs = list(self._async_jobs.items())

            for dataset_name, job in dataset_jobs:
                if not job.alive:
                    finished_at = job.finished_at
                    if finished_at is not None and now - finished_at > grace:
                        dead.append(dataset_name)

            for dataset_name in dead:
                await self._cleanup_async(dataset_name, sleep_time=0)
