"""Captioning service — manages background captioning jobs for datasets.

Each dataset can have at most one active captioning job at a time. Jobs run in
dedicated daemon threads and emit ``CaptioningStatusEvent`` via the
``EventDispatcher`` as images are processed.

Usage from the API layer::

    svc.start_job("my_dataset", options)
    svc.stop_job("my_dataset")
    svc.get_status("my_dataset")
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, ClassVar, Literal

import pydantic
import toml

from yadc.captioners.api import APICaptioner
from yadc.cmd import envs as cmd_envs
from yadc.cmd import templates as cmd_templates
from yadc.core.config import ConfigSettings, parse_config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import resolve_dataset
from yadc.core.prediction import PredictionContext

from ..events import CaptioningStatusEvent
from ..modules.event_dispatcher import EventDispatcher
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .datasets import DatasetService

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

type JobStatus = Literal["idle", "running", "stopping", "error", "done"]


@dataclass
class JobInfo:
    """Snapshot of a running (or recently finished) captioning job."""

    status: JobStatus
    dataset_name: str
    processed: int = 0
    total: int = 0
    errors: int = 0
    error: str | None = None


class CaptionJobOptions(pydantic.BaseModel):
    """All configurable parameters for a single captioning run.

    Extra keys in the input dict are silently ignored (``extra=\"ignore\"``).
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
    draft: str = ""

    # Reasoning
    reasoning: bool = False
    reasoning_effort: str = "low"
    reasoning_exclude_output: bool = True

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Config resolution helpers (shared by CaptionJob and CaptioningService)
# ---------------------------------------------------------------------------


def apply_config_overrides(raw: dict[str, Any], opts: CaptionJobOptions) -> dict[str, Any]:
    """Merge env/config overrides from *opts* into the raw TOML dict."""
    env_name = opts.env or raw.get("env", "default")
    try:
        user_env = cmd_envs.load_env(env_name)
    except Exception:
        user_env = cmd_envs.load_env("default")

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
# Job runner
# ---------------------------------------------------------------------------


class CaptionJob:
    """Runs a single captioning pass over a dataset in a background thread."""

    def __init__(
        self,
        dataset_name: str,
        dataset_service: DatasetService,
        event_dispatcher: EventDispatcher,
        logger: Logger,
        options: CaptionJobOptions,
        on_done: Callable[[], None],
    ):
        self._dataset_name: str = dataset_name
        self._dataset_service: DatasetService = dataset_service
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._logger: Logger = logger
        self._opts: CaptionJobOptions = options
        self._on_done: Callable[[], None] = on_done

        # State (guarded by _state_lock)
        self._state_lock: threading.Lock = threading.Lock()
        self._status: JobStatus = "running"
        self._stop_requested: bool = False
        self._processed: int = 0
        self._total: int = 0
        self._errors: int = 0
        self._error: str | None = None

        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        with self._state_lock:
            self._stop_requested = True
            self._status = "stopping"

    # -- snapshot ------------------------------------------------------------

    @property
    def snapshot(self) -> JobInfo:
        with self._state_lock:
            return JobInfo(
                status=self._status,
                dataset_name=self._dataset_name,
                processed=self._processed,
                total=self._total,
                errors=self._errors,
                error=self._error,
            )

    # -- main loop -----------------------------------------------------------

    def _run(self) -> None:
        """Entry point for the background thread."""
        try:
            self._do_run()
        except Exception as exc:
            self._logger.exception("Captioning job for '%s' failed: %s", self._dataset_name, exc)
            self._set_state(error=str(exc), status="error", set_error=True)
            self._emit_status()
        finally:
            self._on_done()

    def _do_run(self) -> None:
        # 1. Resolve the dataset config path
        ds_info = self._dataset_service.get_dataset(self._dataset_name)
        if ds_info is None or ds_info.config_path is None:
            raise ValueError(f"Dataset '{self._dataset_name}' not found")

        config_path = Path(ds_info.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        # 2. Load raw TOML and merge env/api overrides
        with open(config_path) as f:
            raw = toml.load(f)

        raw = self._apply_overrides(raw)

        # 3. Parse into a Config object
        try:
            config = parse_config(raw)
        except Exception as exc:
            raise ValueError(f"Invalid dataset config: {exc}") from exc

        # 4. Resolve prompt template
        config.prompt.template = self._resolve_template(config.prompt.name, config.prompt.template)

        # 5. Resolve images
        images = resolve_dataset(config.dataset, config.caption_suffix)
        if not images:
            self._logger.info("No images to caption for dataset '%s'.", self._dataset_name)
            self._set_state(status="done")
            self._emit_status()
            return

        # 6. Filter already-captioned images
        to_do: list[DatasetImage] = []
        for img in images:
            if self._opts.draft:
                if not self._opts.overwrite and img.draft_path(self._opts.draft).exists():
                    continue
            else:
                if not self._opts.overwrite and img.caption_path.exists():
                    continue
            to_do.append(img)

        self._set_state(total=len(to_do))
        self._emit_status()

        if not to_do:
            self._logger.info("All images already captioned for dataset '%s'.", self._dataset_name)
            self._set_state(status="done")
            self._emit_status()
            return

        # 7. Create the captioner
        model = APICaptioner(
            api_url=config.api.url,
            api_token=config.api.token,
            prompt_template=config.prompt.template,
            store_conversation=config.settings.store_conversation,
            image_quality=config.settings.image_quality,
            reasoning=config.reasoning.enable,
            reasoning_effort=config.reasoning.thinking_effort,
            reasoning_exclude_output=config.reasoning.exclude_from_output,
        )
        model.load_model(config.api.model_name)

        # 8. Caption each image
        conversation_overrides = config.settings.advanced.model_dump()

        for img in to_do:
            if self._check_stop():
                break

            try:
                self._caption_one(model, img, config.settings, conversation_overrides)
            except Exception as exc:
                self._logger.warning("Failed to caption %s: %s", img.path, exc)
                self._increment_errors()
                self._emit_status()
                continue

            self._increment_processed()
            self._emit_status()

        model.log_usage()

        # Final status — "done" regardless of whether stop was requested
        # (the stop has been completed at this point)
        self._set_state(status="done")
        self._emit_status()

        self._logger.info(
            "Captioning finished for '%s': %d/%d processed, %d errors.",
            self._dataset_name,
            self._processed,
            self._total,
            self._errors,
        )

    def _caption_one(
        self,
        model: APICaptioner,
        dataset_image: DatasetImage,
        settings: ConfigSettings,
        conversation_overrides: dict[str, Any],
    ) -> str:
        """Caption a single image, saving the result."""
        caption = model.predict(
            dataset_image,
            max_new_tokens=settings.max_tokens,
            use_cache=True,
            conversation_overrides=conversation_overrides,
            prefill=settings.advanced.assistant_prefill,
            drafts=dataset_image.read_all_drafts() or None,
            prediction_context=PredictionContext(),
        ).strip()

        if not caption:
            return ""

        if self._opts.draft:
            dataset_image.write_draft(self._opts.draft, caption)
        else:
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

    def _set_state(
        self,
        *,
        status: JobStatus | None = None,
        total: int | None = None,
        error: str | None | None = None,
        set_error: bool = False,
    ) -> None:
        with self._state_lock:
            if status is not None:
                self._status = status
            if total is not None:
                self._total = total
            if set_error:
                self._error = error

    def _increment_processed(self) -> None:
        with self._state_lock:
            self._processed += 1

    def _increment_errors(self) -> None:
        with self._state_lock:
            self._errors += 1

    def _check_stop(self) -> bool:
        with self._state_lock:
            return self._stop_requested

    def _emit_status(self) -> None:
        snap = self.snapshot
        event = CaptioningStatusEvent(
            status=snap.status,
            dataset_name=snap.dataset_name,
            processed=snap.processed,
            total=snap.total,
            errors=snap.errors,
            error=snap.error,
        )
        self._event_dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CaptioningService(Service):
    """Manages captioning jobs — start, stop, query status.

    Thread-safety: all mutable state is guarded by ``_lock``.
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
    ):
        self._dataset_service: DatasetService = dataset_service
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._logger: Logger = logging.get_logger(__name__)

        self._lock: threading.Lock = threading.Lock()
        self._jobs: dict[str, CaptionJob] = {}

    # -- public API ----------------------------------------------------------

    def start_job(self, dataset_name: str, options: CaptionJobOptions) -> JobInfo:
        """Start a captioning job for *dataset_name*.

        Args:
            dataset_name: Name of the dataset to caption.
            options: Validated captioning options.

        Raises:
            ValueError: If the dataset is unknown or a job is already running.
        """
        with self._lock:
            if dataset_name in self._jobs and self._jobs[dataset_name].alive:
                raise ValueError(f"A captioning job is already running for dataset '{dataset_name}'")

            job = CaptionJob(
                dataset_name=dataset_name,
                dataset_service=self._dataset_service,
                event_dispatcher=self._event_dispatcher,
                logger=self._logger,
                options=options,
                on_done=lambda: self._cleanup(dataset_name),
            )
            self._jobs[dataset_name] = job
            job.start()

        return self.get_status(dataset_name)

    def stop_job(self, dataset_name: str) -> bool:
        """Request cancellation of the running job for *dataset_name*.

        Returns ``True`` if a job was running and cancellation was requested.
        """
        with self._lock:
            job = self._jobs.get(dataset_name)
            if job is None or not job.alive:
                return False
            job.request_stop()
            return True

    def get_status(self, dataset_name: str) -> JobInfo:
        """Return a snapshot of the job status for *dataset_name*."""
        with self._lock:
            job = self._jobs.get(dataset_name)
            if job is None:
                return JobInfo(status="idle", dataset_name=dataset_name)
            return job.snapshot

    def caption_single(self, dataset_name: str, image_id: int, options: CaptionJobOptions) -> dict[str, Any]:
        """Caption a single image synchronously.

        Resolves the dataset config, creates a captioner, and captions the
        specified image. Returns a dict with the resulting caption.

        Raises:
            ValueError: If the dataset or image is not found, or a batch job is running.
        """
        # Block if a batch job is running for this dataset
        with self._lock:
            if dataset_name in self._jobs and self._jobs[dataset_name].alive:
                raise ValueError(f"A batch captioning job is running for dataset '{dataset_name}'")

        # Resolve dataset config
        ds_info = self._dataset_service.get_dataset(dataset_name)
        if ds_info is None or ds_info.config_path is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        config_path = Path(ds_info.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw = toml.load(f)

        # Build a temporary job-like object just for _apply_overrides
        raw = apply_config_overrides(raw, options)

        try:
            config = parse_config(raw)
        except Exception as exc:
            raise ValueError(f"Invalid dataset config: {exc}") from exc

        # Resolve template
        config.prompt.template = resolve_template(config.prompt.name, config.prompt.template, self._logger)

        # Resolve the specific image
        info = self._dataset_service.get_image(dataset_name, image_id)
        if info is None:
            raise ValueError(f"Image {image_id} not found in dataset '{dataset_name}'")

        image_path = Path(info.path)
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        # Build DatasetImage with extras
        dataset_image = DatasetImage(path=str(image_path))
        dataset_image.caption = dataset_image.read_caption()

        # Load TOML extras
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = toml.loads(f.read())
            except Exception:
                pass
        if extras:
            dataset_image = DatasetImage.model_validate({"path": str(image_path), "caption": dataset_image.caption, **extras})

        # Create the captioner
        model = APICaptioner(
            api_url=config.api.url,
            api_token=config.api.token,
            prompt_template=config.prompt.template,
            store_conversation=config.settings.store_conversation,
            image_quality=config.settings.image_quality,
            reasoning=config.reasoning.enable,
            reasoning_effort=config.reasoning.thinking_effort,
            reasoning_exclude_output=config.reasoning.exclude_from_output,
        )
        model.load_model(config.api.model_name)

        # Caption
        conversation_overrides = config.settings.advanced.model_dump()
        caption = model.predict(
            dataset_image,
            max_new_tokens=config.settings.max_tokens,
            use_cache=True,
            conversation_overrides=conversation_overrides,
            prefill=config.settings.advanced.assistant_prefill,
            drafts=dataset_image.read_all_drafts() or None,
            prediction_context=PredictionContext(),
        ).strip()

        if caption:
            if options.draft:
                dataset_image.write_draft(options.draft, caption)
            else:
                if dataset_image.caption:
                    dataset_image.save_history(when_not_exists=True)
                dataset_image.update_caption(caption)
                dataset_image.save_history(when_not_exists=False)

            # Update the image index
            self._dataset_service.refresh_image_index(dataset_name, image_id)

        model.log_usage()

        return {"caption": caption}

    # -- private helpers -----------------------------------------------------

    def _cleanup(self, dataset_name: str) -> None:
        """Remove finished jobs after a short delay (so clients can read final status)."""

        def _delayed():
            import time

            time.sleep(5)
            with self._lock:
                job = self._jobs.get(dataset_name)
                if job is not None and not job.alive:
                    del self._jobs[dataset_name]

        t = threading.Thread(target=_delayed, daemon=True)
        t.start()
