"""Captioning service — manages background captioning jobs for datasets.

Each dataset can have at most one active captioning job at a time. Jobs run as
``asyncio.Task`` instances and emit ``CaptioningStatusEvent`` via the
``EventDispatcher`` as images are processed.

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
from typing import Any, ClassVar, Literal

import pydantic
import toml

from yadc.captioners.api import APICaptioner
from yadc.captioners.api.async_session import AsyncSession
from yadc.cmd import envs as cmd_envs
from yadc.cmd import templates as cmd_templates
from yadc.core.config import ConfigSettings, parse_config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import resolve_dataset
from yadc.core.prediction import PredictionContext

from ..configuration import Configuration
from ..events import CaptioningStatusEvent, ImageCaptionedEvent, ImageCaptionErrorEvent, ImageCaptionStartedEvent
from ..modules.dataset_watcher import DatasetWatcherService
from ..modules.event_dispatcher import EventDispatcher
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .datasets import DatasetService

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


class CaptionJobOptions(pydantic.BaseModel):
    """All configurable parameters for a single captioning run.

    Extra keys in the input dict are silently ignored (``extra="ignore"``).
    """

    # API connection
    api_url: str = ""
    api_token: str = ""
    api_model_name: str = ""

    # Environment / config resolution
    env: str = "default"

    # Captioning behaviour
    prompt_template: str = ""
    prompt_name: str = ""
    max_tokens: int = 512
    image_quality: str = "auto"
    store_conversation: bool = False
    overwrite: bool = False
    rounds: int = 1
    draft: str = ""

    # Reasoning
    reasoning: bool = False
    reasoning_effort: str = "low"
    reasoning_exclude_output: bool = True

    # Password for decrypting password-mode environment settings
    password: str | None = None

    # If set, only caption these specific image IDs (single-image mode)
    image_ids: list[int] | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------


def apply_config_overrides(raw: dict[str, Any], opts: CaptionJobOptions) -> dict[str, Any]:
    """Merge env/config overrides from *opts* into the raw TOML dict."""
    env_name = opts.env or raw.get("env", "default")
    user_env = cmd_envs.load_env(env_name, password=opts.password)

    raw.setdefault("api", {})
    api: dict[str, Any] = raw["api"]

    # Apply overrides: CLI option > env > existing TOML
    api["url"] = opts.api_url or user_env.api.url or api.get("url", "")
    api["token"] = opts.api_token or user_env.api.token or api.get("token", "")
    api["model_name"] = opts.api_model_name or user_env.api.model_name or api.get("model_name", "")

    # Apply prompt overrides
    raw.setdefault("prompt", {})
    if opts.prompt_name:
        raw["prompt"]["name"] = opts.prompt_name
        raw["prompt"].pop("template", None)
    if opts.prompt_template:
        raw["prompt"]["template"] = opts.prompt_template
        raw["prompt"].pop("name", None)

    # Apply other option overrides
    if opts.max_tokens != 512:
        raw.setdefault("settings", {})
        raw["settings"]["max_tokens"] = opts.max_tokens

    if opts.image_quality != "auto":
        raw.setdefault("settings", {})
        raw["settings"]["image_quality"] = opts.image_quality

    if opts.reasoning:
        raw.setdefault("reasoning", {})
        raw["reasoning"]["enable"] = True
        raw["reasoning"]["thinking_effort"] = opts.reasoning_effort
        raw["reasoning"]["exclude_from_output"] = opts.reasoning_exclude_output

    if opts.rounds != 1:
        raw["rounds"] = opts.rounds

    return raw


def resolve_template(prompt_name: str, prompt_template: str, logger: Logger | None = None) -> str:
    """Resolve a prompt template through the fallback chain."""
    if prompt_template:
        return prompt_template

    for loader in (cmd_templates.load_user_template, cmd_templates.load_builtin_template):
        try:
            return loader(prompt_name)
        except Exception:
            continue

    if prompt_name:
        available = cmd_templates.list_user_template()
        raise ValueError(
            f"Prompt template '{prompt_name}' not found. Available: {', '.join(available)}" if available else f"Prompt template '{prompt_name}' not found."
        )

    # No template specified — use default
    if logger:
        logger.warning("No prompt template specified, using default.")
    return cmd_templates.default_template()


# ---------------------------------------------------------------------------
# Async job runner
# ---------------------------------------------------------------------------


class AsyncCaptionJob:
    """Runs a single captioning pass over a dataset as an asyncio task."""

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

        # State (guarded by _state_lock)
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

        self._task: asyncio.Task[Any] | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

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
            )

    # -- main loop -----------------------------------------------------------

    async def _arun(self) -> None:
        """Entry point for the asyncio task."""
        try:
            await self._ado_run()
        except asyncio.CancelledError:
            self._logger.info("Captioning job for '%s' cancelled", self._dataset_name)
            await self._set_state(status="cancelled")
            await self._emit_status()
        except Exception as exc:
            self._logger.exception("Captioning job for '%s' failed: %s", self._dataset_name, exc)
            msg = str(exc)
            await self._set_state(error=msg, status="error", set_error=True)
            async with self._state_lock:
                self._error_messages.append(msg)
            await self._emit_status()
        finally:
            # Schedule cleanup as a background task so _arun returns
            # promptly and alive becomes False — this lets callers start
            # a new job immediately after cancellation.
            if inspect.iscoroutinefunction(self._on_done):
                asyncio.create_task(self._on_done())
            else:
                self._on_done()

    def preflight_images(self) -> list[DatasetImage]:
        """Synchronously resolve the dataset and filter images.

        Performs steps 1-6 of captioning (config → images → filtering).
        Returns the list of images that would be processed.  Callers can use
        ``len(result)`` to know the total before the async task starts.

        Raises ``ValueError`` on configuration / resolution errors.
        """
        ds_info = self._dataset_service.get_dataset(self._dataset_name)
        if ds_info is None or ds_info.config_path is None:
            raise ValueError(f"Dataset '{self._dataset_name}' not found")

        config_path = Path(ds_info.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw = toml.load(f)

        raw = self._apply_overrides(raw)

        try:
            config = parse_config(raw)
        except Exception as exc:
            raise ValueError(f"Invalid dataset config: {exc}") from exc

        config.prompt.template = self._resolve_template(config.prompt.name, config.prompt.template)

        self._api_url = config.api.url
        self._api_model_name = config.api.model_name

        images = resolve_dataset(config.dataset, config.caption_suffix, base_dir=str(config_path.parent))
        if not images:
            return []

        if self._opts.image_ids:
            target_paths: set[Path] = set()
            for image_id in self._opts.image_ids:
                info = self._dataset_service.get_image(self._dataset_name, image_id)
                if info:
                    target_paths.add(Path(info.path))
            images = [img for img in images if Path(img.path) in target_paths]
            if not images:
                raise ValueError(f"Specified image(s) not found in dataset '{self._dataset_name}'")

        to_do: list[DatasetImage] = []
        for img in images:
            if self._opts.image_ids:
                to_do.append(img)
                continue
            if self._opts.draft:
                if not self._opts.overwrite and img.draft_path(self._opts.draft).exists():
                    continue
            else:
                if not self._opts.overwrite and img.caption_path.exists():
                    continue
            to_do.append(img)

        return to_do

    async def _ado_run(self) -> None:
        to_do = self.preflight_images()

        await self._set_state(total=len(to_do))
        await self._emit_status()

        if not to_do:
            self._logger.info("No images to caption for dataset '%s'.", self._dataset_name)
            await self._set_state(status="done")
            await self._emit_status()
            return

        # Re-parse config for the model creation (cheap).
        ds_info = self._dataset_service.get_dataset(self._dataset_name)
        config_path = Path(ds_info.config_path) if ds_info and ds_info.config_path else Path()
        with open(config_path) as f:
            raw = toml.load(f)
        raw = self._apply_overrides(raw)
        config = parse_config(raw)
        config.prompt.template = self._resolve_template(config.prompt.name, config.prompt.template)

        # Create the captioner with an async session
        async_headers: dict[str, str] = {}
        if config.api.token:
            async_headers["Authorization"] = f"Bearer {config.api.token}"

        model = await APICaptioner.create(
            api_url=config.api.url,
            api_token=config.api.token,
            prompt_template=config.prompt.template,
            store_conversation=config.settings.store_conversation,
            image_quality=config.settings.image_quality,
            reasoning=config.reasoning.enable,
            reasoning_effort=config.reasoning.thinking_effort,
            reasoning_exclude_output=config.reasoning.exclude_from_output,
            async_session=AsyncSession(
                config.api.url,
                headers=async_headers,
                connect_timeout=self._configuration.http_timeout_connect,
                read_timeout=self._configuration.http_timeout_read,
                write_timeout=self._configuration.http_timeout_write,
                pool_timeout=self._configuration.http_timeout_pool,
            ),
        )
        await model.load_model(config.api.model_name)

        # Caption each image
        conversation_overrides = config.settings.advanced.model_dump()

        for img in to_do:
            if self._check_stop():
                break

            self._emit_image_started(img)
            t0 = time.monotonic()

            try:
                await self._acaption_one(model, img, config.settings, conversation_overrides)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                self._logger.warning("Failed to caption %s: %s", img.path, exc)
                await self._record_error(str(exc))
                await self._emit_status()
                self._emit_image_error(img, str(exc), duration_ms)
                continue

            duration_ms = int((time.monotonic() - t0) * 1000)
            await self._increment_processed()
            await self._emit_status()
            self._emit_image_captioned(img, duration_ms)

        model.log_usage()

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

    async def _acaption_one(
        self,
        model: APICaptioner,
        dataset_image: DatasetImage,
        settings: ConfigSettings,
        conversation_overrides: dict[str, Any],
    ) -> str:
        """Caption a single image, saving the result."""
        caption_parts: list[str] = []
        try:
            async for token in model.predict_stream(
                dataset_image,
                max_new_tokens=settings.max_tokens,
                conversation_overrides=conversation_overrides,
                prefill=settings.advanced.assistant_prefill,
                drafts=dataset_image.read_all_drafts() or None,
                prediction_context=PredictionContext(),
            ):
                caption_parts.append(token)
        except asyncio.CancelledError:
            self._logger.debug("Captioning of %s was cancelled", dataset_image.path)
            raise

        caption = "".join(caption_parts).strip()

        if not caption:
            return ""

        if self._opts.draft:
            self._dataset_watcher.expect_file_change(self._dataset_name, str(dataset_image.draft_path(self._opts.draft)))
            dataset_image.write_draft(self._opts.draft, caption)
        else:
            # Register all files that will be written so the watcher suppresses
            # the resulting inotify events for this captioning job.
            self._dataset_watcher.expect_file_change(self._dataset_name, str(dataset_image.caption_path))
            self._dataset_watcher.expect_file_change(self._dataset_name, str(dataset_image.toml_path))
            self._dataset_watcher.expect_file_change(self._dataset_name, str(dataset_image.history_path))

            # Save history of previous caption if present
            if dataset_image.caption:
                dataset_image.save_history(when_not_exists=True)
            dataset_image.update_caption(caption)
            dataset_image.save_history(when_not_exists=False)

        return caption

    # -- config resolution helpers -------------------------------------------

    def _apply_overrides(self, raw: dict[str, Any]) -> dict[str, Any]:
        return apply_config_overrides(raw, self._opts)

    def _resolve_template(self, prompt_name: str, prompt_template: str) -> str:
        return resolve_template(prompt_name, prompt_template, self._logger)

    # -- state helpers -------------------------------------------------------

    async def _set_state(
        self,
        *,
        status: JobStatus | None = None,
        total: int | None = None,
        error: str | None | None = None,
        set_error: bool = False,
    ) -> None:
        async with self._state_lock:
            if status is not None:
                self._status = status
            if total is not None:
                self._total = total
            if set_error:
                self._error = error

    async def _increment_processed(self) -> None:
        async with self._state_lock:
            self._processed += 1

    async def _record_error(self, message: str) -> None:
        async with self._state_lock:
            self._errors += 1
            self._error_messages.append(message)

    def _check_stop(self) -> bool:
        return self._stop_event.is_set()

    async def _emit_status(self) -> None:
        snap = await self.snapshot()
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

    def _emit_image_captioned(self, dataset_image: DatasetImage, duration_ms: int = 0) -> None:
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

    def _emit_image_error(self, dataset_image: DatasetImage, error: str, duration_ms: int = 0) -> None:
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

    # -- public API ----------------------------------------------------------

    def _mark_expected_changes(self, dataset_name: str, job_id: str) -> None:
        """Tag the next watcher events for this dataset with the captioning job ID."""
        self._dataset_watcher.expect_changes(dataset_name, job_id)

    async def start_job_async(self, dataset_name: str, options: CaptionJobOptions) -> JobInfo:
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
            )
            self._async_jobs[dataset_name] = job
            # Synchronous preflight: compute total so the initial status is
            # accurate (avoids returning 0/0 when images are present).  If
            # there is nothing to do we return done immediately without
            # starting a background task.
            try:
                to_do = job.preflight_images()
            except ValueError as exc:
                return JobInfo(
                    status="error",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    error=str(exc),
                )

            if not to_do:
                return JobInfo(
                    status="done",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    total=0,
                    processed=0,
                    errors=0,
                )

            await job._set_state(total=len(to_do))
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

    # -- private helpers -----------------------------------------------------

    async def _cleanup_async(self, dataset_name: str) -> None:
        """Remove finished async jobs after a short delay."""
        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)
            job_id = (await job.snapshot()).job_id if job is not None else ""

        await asyncio.sleep(5)

        async with self._async_lock:
            job = self._async_jobs.get(dataset_name)
            if job is not None and not job.alive:
                del self._async_jobs[dataset_name]
        self._dataset_watcher.clear_expected_changes_for_job(dataset_name, job_id)

        # Final rescan to catch any edge cases (files added/removed by
        # other processes during captioning). Per-image refresh_image_index
        # calls already handled the captioning writes, so this is cheap
        # when there are no external changes.
        self._dataset_service.rescan_dataset(dataset_name)
