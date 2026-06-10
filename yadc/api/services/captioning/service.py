"""Captioning service — manages background captioning jobs for datasets.

Each dataset can have at most one active captioning job at a time. Jobs run as
``asyncio.Task`` instances and emit ``CaptioningStatusEvent`` via the
``EventDispatcher`` as images are processed.

The actual model-create / stream / save loop lives in
``yadc.core.captioning.CaptioningRunner``. This module orchestrates job
lifecycle: start/stop/status queries, cleanup, and final dataset rescans.

Usage from the API layer::

    await svc.start_job_async("my_dataset", options)
    await svc.stop_job_async("my_dataset")
    await svc.get_status_async("my_dataset")
"""

import asyncio
import time
import uuid
from logging import Logger
from typing import Any, Literal

from yadc.api.configuration import Configuration
from yadc.api.events import ImageRefinedEvent, ShutdownEvent, StartupEvent
from yadc.api.modules import DatasetWatcherService, EventDispatcher, LoggingFactory, Service
from yadc.api.modules.event_dispatcher import event_handler
from yadc.api.services.datasets import DatasetService
from yadc.core.captioning import (
    CaptionJobOptions,
)
from yadc.utils import LRU

from .job import AsyncCaptionJob
from .job_runner import AsyncCaptionJobRunner
from .models import JobInfo, RefineOptions


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

    def _make_job_runner(self) -> AsyncCaptionJobRunner:
        """Create an ``AsyncCaptionJobRunner`` wired with this service's DI deps."""
        return AsyncCaptionJobRunner(
            configuration=self._configuration,
            dataset_service=self._dataset_service,
            dataset_watcher=self._dataset_watcher,
            event_dispatcher=self._event_dispatcher,
            logger=self._logger,
        )

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

            job_runner = self._make_job_runner()
            job = AsyncCaptionJob(
                dataset_name=dataset_name,
                options=options,
                job_id=job_id,
                on_done=_on_done,
                runner=job_runner,
                refine=refine,
            )
            self._async_jobs[dataset_name] = job
            # Synchronous preflight: compute total so the initial status is
            # accurate (avoids returning 0/0 when images are present).  If
            # there is nothing to do we return done immediately without
            # starting a background task.
            try:
                config, to_do = job_runner.preflight(dataset_name, options)
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

            # Store preflight results on the job so _ado_run doesn't
            # re-run preflight.
            job._api_url = config.api.url
            job._api_model_name = config.api.model_name
            job._config = config
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
