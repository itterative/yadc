"""Async captioning job runner — API glue between the service and the captioning core.

``AsyncCaptionJobRunner`` holds DI dependencies (``DatasetService``,
``EventDispatcher``, etc.) and provides the API integration that
``AsyncCaptionJob`` needs: preflight (config loading, image resolution,
DB ordering), SSE event emission, and watcher registration.

``AsyncCaptionJob`` is a pure state machine + ``CaptioningCallbacks``
implementation with no DI awareness — it delegates to the runner for
all infrastructure work.
"""

import time
from logging import Logger
from pathlib import Path

from yadc.api.configuration import Configuration
from yadc.api.events import (
    CaptioningStatusEvent,
    ImageCaptionedEvent,
    ImageCaptionErrorEvent,
    ImageCaptionStartedEvent,
    ImageRefinedEvent,
)
from yadc.api.modules import DatasetWatcherService, EventDispatcher
from yadc.api.services.datasets import DatasetService
from yadc.core import Config, DatasetImage
from yadc.core.captioning import CaptioningCallbacks, CaptioningRunner, CaptionJobOptions, HTTPTTimeouts, load_dataset_config

from .models import JobInfo, RefineOptions


class AsyncCaptionJobRunner:
    """API glue for a single captioning job — preflight, event emission, watcher registration.

    Created by ``CaptioningService`` per ``start_job_async`` call. Holds the
    DI dependencies so the job itself stays free of service imports.
    """

    def __init__(
        self,
        configuration: Configuration,
        dataset_service: DatasetService,
        dataset_watcher: DatasetWatcherService,
        event_dispatcher: EventDispatcher,
        logger: Logger,
    ):
        self._configuration: Configuration = configuration
        self._dataset_service: DatasetService = dataset_service
        self._dataset_watcher: DatasetWatcherService = dataset_watcher
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._logger: Logger = logger

    # -- preflight -----------------------------------------------------------

    def preflight(
        self,
        dataset_name: str,
        options: CaptionJobOptions,
    ) -> tuple[Config, list[DatasetImage], int]:
        """Synchronously resolve the dataset and filter images.

        Performs config loading, override application, and dataset
        resolution via :func:`load_dataset_config`. The returned list
        is reordered by database id descending via a single SQL
        query so concurrent captioning starts with the newest images
        first.

        Returns ``(config, images, skipped)`` where ``skipped`` is the
        number of images dropped by the overwrite/draft filter. Raises
        ``ValueError`` on configuration / resolution errors.
        """
        ds_info = self._dataset_service.get_dataset(dataset_name)
        if ds_info is None or ds_info.config_path is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        config_path = Path(ds_info.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        config, images, skipped = load_dataset_config(
            config_path,
            options,
            image_path_resolver=lambda image_id: self._image_path_resolver(dataset_name, image_id),
        )

        # Reorder the filesystem-resolved list by DB id DESC (newest
        # first). Single SQL query — no N+1. Images not in the DB
        # (e.g. inline extras in the config that haven't been
        # scanned) sort to the end.
        ordered_paths = self._dataset_service.get_image_paths_in_desc_order(dataset_name)
        if ordered_paths:
            position = {path: i for i, path in enumerate(ordered_paths)}
            sentinel = len(ordered_paths)
            images.sort(key=lambda img: position.get(img.path, sentinel))

        return config, images, skipped

    def _image_path_resolver(self, dataset_name: str, image_id: int) -> Path | None:
        """Resolve an image ID to its filesystem path for the loader's image_ids filter."""
        info = self._dataset_service.get_image(dataset_name, image_id)
        return Path(info.path) if info else None

    # -- CaptioningRunner construction ---------------------------------------

    def build_runner(
        self,
        config: Config,
        options: CaptionJobOptions,
        *,
        callbacks: CaptioningCallbacks,
    ) -> CaptioningRunner:
        """Build a ``CaptioningRunner`` wired with API timeouts."""
        return CaptioningRunner(
            config,
            options,
            callbacks,
            logger=self._logger,
            http_timeouts=HTTPTTimeouts(
                connect=self._configuration.http_timeout_connect,
                read=self._configuration.http_timeout_read,
                write=self._configuration.http_timeout_write,
                pool=self._configuration.http_timeout_pool,
            ),
        )

    # -- watcher registration -----------------------------------------------

    def expect_file_changes(self, dataset_name: str, paths: list[str]) -> None:
        """Register upcoming file writes with the watcher so it suppresses inotify events."""
        for path in paths:
            self._dataset_watcher.expect_file_change(dataset_name, path)

    # -- event emission ------------------------------------------------------

    async def emit_status(
        self,
        dataset_name: str,
        job_id: str,
        snapshot: JobInfo,
        started_at: float | None,
    ) -> None:
        """Dispatch a ``CaptioningStatusEvent`` from a job snapshot."""
        elapsed = 0.0
        if started_at is not None:
            elapsed = time.monotonic() - started_at
        self._event_dispatcher.dispatch(
            CaptioningStatusEvent(
                status=snapshot.status,
                dataset_name=dataset_name,
                processed=snapshot.processed,
                total=snapshot.total,
                errors=snapshot.errors,
                job_id=job_id,
                error=snapshot.error,
                error_messages=snapshot.error_messages,
                api_url=snapshot.api_url,
                api_model_name=snapshot.api_model_name,
                elapsed=elapsed,
                max_concurrent=snapshot.max_concurrent,
            )
        )

    def emit_image_started(self, dataset_name: str, job_id: str, dataset_image: DatasetImage) -> None:
        """Dispatch ``ImageCaptionStartedEvent`` after looking up the image in the DB."""
        info = self._dataset_service.get_image_by_path(dataset_name, dataset_image.path)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageCaptionStartedEvent(
                dataset_name=dataset_name,
                job_id=job_id,
                image_id=info.id,
                file_name=info.file_name,
            )
        )

    def emit_image_captioned(
        self,
        dataset_name: str,
        job_id: str,
        dataset_image: DatasetImage,
        api_url: str,
        api_model_name: str,
        duration_ms: int,
    ) -> None:
        """Dispatch ``ImageCaptionedEvent`` with a fresh image index lookup."""
        info = self._dataset_service.get_image_by_path(dataset_name, dataset_image.path)
        if info is None:
            return
        self._dataset_service.refresh_image_index(dataset_name, info.id)
        info = self._dataset_service.get_image(dataset_name, info.id)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageCaptionedEvent(
                dataset_name=dataset_name,
                job_id=job_id,
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
                api_url=api_url,
                api_model_name=api_model_name,
            )
        )

    def emit_image_error(
        self,
        dataset_name: str,
        job_id: str,
        api_url: str,
        api_model_name: str,
        dataset_image: DatasetImage,
        error: str,
        duration_ms: int,
    ) -> None:
        """Dispatch ``ImageCaptionErrorEvent`` (image_id = -1 if image not in DB)."""
        info = self._dataset_service.get_image_by_path(dataset_name, dataset_image.path)
        image_id = info.id if info is not None else -1
        self._event_dispatcher.dispatch(
            ImageCaptionErrorEvent(
                dataset_name=dataset_name,
                job_id=job_id,
                image_id=image_id,
                error=error,
                duration_ms=duration_ms,
                api_url=api_url,
                api_model_name=api_model_name,
            )
        )

    def emit_image_refined(
        self,
        dataset_name: str,
        job_id: str,
        dataset_image: DatasetImage,
        caption: str,
        refine: RefineOptions,
    ) -> None:
        """Dispatch ``ImageRefinedEvent`` for a dry-run refine result."""
        info = self._dataset_service.get_image_by_path(dataset_name, dataset_image.path)
        if info is None:
            return
        self._event_dispatcher.dispatch(
            ImageRefinedEvent(
                dataset_name=dataset_name,
                job_id=job_id,
                image_id=info.id,
                caption=caption,
                source=refine.refine_source,
                draft_name=refine.refine_draft_name,
            )
        )
