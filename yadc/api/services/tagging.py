"""Tagging service — manages the tagger subprocess for the API backend.

The subprocess is spawned **lazily** on the first tagging request and
torn down after ``Configuration.tagger_idle_timeout_seconds`` of
inactivity (default 15 minutes). A periodic background job
(``IDLE_CHECK_INTERVAL_SECONDS``) checks the idle timer and stops the
subprocess when it expires. The next request after teardown respawns
it.

Lifecycle transitions (start / stop / idle-stop) are serialized
through ``self._lifecycle_lock`` (``threading.Lock``) so the sync
idle check, async request handlers, and the shutdown event handler
never race on the subprocess state.

The service dispatches ``TaggerStatusEvent`` (subprocess state
changes), ``ImageTaggedEvent`` (per-image success), and
``ImageTagErrorEvent`` (per-image failure) via the
:class:`EventDispatcher` so SSE clients see what happened.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import pydantic
import tomlkit

from yadc.api.configuration import Configuration
from yadc.api.events import (
    ImageTagErrorEvent,
    ImageTaggedEvent,
    ImageTagStartedEvent,
    ShutdownEvent,
    StartupEvent,
    TaggerStatusEvent,
    TagJobStatusEvent,
)
from yadc.api.modules import DatasetWatcherService, EventDispatcher, JobScheduler, LoggingFactory, Service
from yadc.api.modules.dataset_watcher import SELF_JOB_ID
from yadc.api.modules.event_dispatcher import event_handler
from yadc.api.services.dataset_jobs import DatasetJobService, JobClaim
from yadc.api.services.dataset_repository import ImageInfo
from yadc.api.services.datasets import DatasetService
from yadc.taggers import OnnxTagger, TaggerResult, apply_thresholds, extras_tags, format_draft
from yadc.taggers.client import TaggerClient
from yadc.taggers.postprocessing import replace_underscores as replace_underscores_in
from yadc.utils import LRU

# How often the idle-check job wakes up. Worst-case shutdown latency
# after the timeout elapses is one interval (e.g. 30s for a 15min
# timeout). Small enough not to waste CPU; large enough to be
# negligible relative to the timeout.
IDLE_CHECK_INTERVAL_SECONDS: float = 30.0

TaggerState = Literal["starting", "ready", "stopping", "stopped", "failed"]


type TagJobStatus = Literal["idle", "running", "stopping", "error", "done", "cancelled"]


# Outcome of a cancel request (graceful-first, kill-as-fallback). "stopped"
# means a job's stop_event was set (or nothing was running) without needing to
# kill the subprocess; "killed" means the subprocess was force-terminated
# because an inference was still in flight after the grace window.
type CancelOutcome = Literal["stopped", "killed", "stale_job", "nothing_running"]


@dataclass
class TaggingThresholds:
    """Per-category score thresholds applied to a tagging result."""

    rating: float = 0.0
    general: float = 0.35
    character: float = 0.85


class TagSaveOptions(pydantic.BaseModel):
    """Where and how to persist a tag result for an image.

    - ``none`` — tag only (emit events, write nothing). Default.
    - ``draft`` — write formatted text to a named draft file
      (``<image>.<name>.draft~``). Reuses the caption draft sidecar.
    - ``extras`` — merge a ``[tags]`` sub-table into the image TOML
      (``general`` / ``character`` lists + a single ``rating`` string).

    ``overwrite`` governs the batch save path only: when ``False`` (the
    default), images that already carry the target artifact (a draft of
    the same name, or a ``[tags]`` sub-table) are skipped entirely — no
    tag, no save. The interactive save endpoint always writes.
    """

    mode: Literal["none", "draft", "extras"] = "none"
    draft_name: str = "tags"
    draft_format: str = "comma"
    overwrite: bool = False


class TagJobOptions(pydantic.BaseModel):
    """Body for ``POST /datasets/<name>/tag`` — start a batch tagging job.

    ``image_ids`` omitted/empty → tag the whole dataset; otherwise tag
    just the listed images. Thresholds fall back to the server config
    when unset. ``save`` controls per-image persistence.
    """

    image_ids: list[int] | None = None
    rating_threshold: float | None = None
    general_threshold: float | None = None
    character_threshold: float | None = None
    replace_underscores: bool | None = None
    save: TagSaveOptions = pydantic.Field(default_factory=TagSaveOptions)
    source: str | None = None


@dataclass(frozen=True)
class TaggerResultKey:
    """Cache key for a single tagged image result.

    Two keys compare equal iff they name the same image AND were
    produced under the same model + threshold + post-processing
    settings. When any of those change, the new key hashes to a
    different bucket and the old entry ages out via LRU eviction —
    no explicit invalidation needed.

    Frozen so the dataclass-generated ``__hash__`` and ``__eq__`` are
    reliable: ``LRU`` uses the key as an ``OrderedDict`` key, which
    requires hashable + equality support.
    """

    dataset_name: str
    image_id: int
    # The configured model source: ``tagger_repo_id`` (HF download) or
    # ``tagger_model_path`` (local file). Only one is non-empty per
    # :attr:`TaggingService.is_configured`.
    model_id: str
    rating_threshold: float
    general_threshold: float
    character_threshold: float
    replace_underscores: bool


@dataclass
class TagJobInfo:
    """Snapshot of a running (or recently finished) tagging job."""

    status: TagJobStatus
    dataset_name: str
    job_id: str = ""
    processed: int = 0
    total: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    source: str = ""
    elapsed: float = 0.0


@dataclass
class CancelResult:
    """Outcome of a cancel request (see :meth:`TaggingService.cancel_async`)."""

    outcome: CancelOutcome
    killed: bool = False
    job_id: str = ""


class TaggingService(Service):
    """Manages the tagger subprocess lifecycle and tagging requests.

    The subprocess is spawned on demand and torn down after
    ``Configuration.tagger_idle_timeout_seconds`` of inactivity. Set
    the timeout to ``0`` to keep the subprocess up indefinitely once
    it has been started.
    """

    def __init__(
        self,
        configuration: Configuration,
        logging: LoggingFactory,
        event_dispatcher: EventDispatcher,
        dataset_service: DatasetService,
        dataset_watcher: DatasetWatcherService,
        dataset_jobs: DatasetJobService,
        job_scheduler: JobScheduler | None = None,
    ):
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._dataset_service: DatasetService = dataset_service
        self._dataset_watcher: DatasetWatcherService = dataset_watcher
        self._dataset_jobs: DatasetJobService = dataset_jobs
        self._job_scheduler: JobScheduler | None = job_scheduler

        self._tagger_client: TaggerClient | None = None
        # ``threading.Lock`` (not ``asyncio.Lock``) so the sync idle
        # check thread can acquire it without going through
        # ``run_coroutine_threadsafe``. Async request handlers acquire
        # it across an ``await``; that's acceptable because the actual
        # tag inference happens off-thread via ``asyncio.to_thread``,
        # so the event loop stays free for non-locking work.
        self._lifecycle_lock: threading.Lock = threading.Lock()
        # ``None`` until the first request arrives. Used as the
        # "last touched" timestamp by the idle-check job.
        self._last_used_t: float | None = None

        # Batch tagging jobs (one per dataset). ``asyncio.Lock`` guards
        # the dict; the jobs themselves run as ``asyncio.Task``s.
        self._tag_jobs_lock: asyncio.Lock = asyncio.Lock()
        self._tag_jobs: dict[str, _TagJobState] = {}
        # Holds in-flight deferred expected-changes clears so the event
        # loop can't GC them mid-sleep. Entries self-remove on completion.
        self._pending_clears: set[asyncio.Task[None]] = set()

        # Bounded LRU cache for tag results, keyed by
        # :class:`TaggerResultKey` (image identity + model + thresholds +
        # post-processing). Persists across requests and browser sessions
        # so a fresh Tags tab loaded after a batch job sees the existing
        # result without re-running the model. The fingerprint baked into
        # the key means model / threshold changes naturally hash to a
        # different bucket — old entries age out via LRU eviction without
        # needing explicit invalidation.
        self._tag_lock: asyncio.Lock = asyncio.Lock()
        self._tag_results: LRU[TaggerResultKey, TaggerResult] = LRU(self._configuration.tagger_result_buffer_size)

    # --- event handlers ---------------------------------------------------

    @event_handler(StartupEvent)
    async def on_startup(self, event: StartupEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        if not self.is_configured:
            self._logger.debug("No tagger model configured (both tagger_model_path and tagger_repo_id empty); tagging endpoints will return 503.")
        if self._job_scheduler is not None:
            self._job_scheduler.new_scheduled_job(IDLE_CHECK_INTERVAL_SECONDS, self._idle_check_tick)

    @event_handler(ShutdownEvent)
    async def on_shutdown(self, event: ShutdownEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        await self._acquire_lifecycle()
        try:
            await self._stop_locked_async()
        finally:
            self._lifecycle_lock.release()

    # --- public API -------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """Whether the tagger has a model source configured (i.e. can serve requests at all)."""
        return bool(self._configuration.tagger_model_path.strip()) or bool(self._configuration.tagger_repo_id.strip())

    @property
    def is_available(self) -> bool:
        """Whether the tagger subprocess is currently running."""
        return self._tagger_client is not None and self._tagger_client.is_alive

    def _source_label(self) -> str:
        """Describe the configured model source for diagnostics / SSE payloads.

        Returns ``"hf:<repo_id>"`` for HuggingFace Hub downloads,
        ``"local:<path>"`` for local files, ``""`` when unconfigured.
        """
        repo_id = self._configuration.tagger_repo_id.strip()
        if repo_id:
            return f"hf:{repo_id}"
        path = self._configuration.tagger_model_path.strip()
        if path:
            return f"local:{path}"
        return ""

    def _emit_status(self, state: TaggerState, error: str | None = None) -> None:
        """Dispatch a ``TaggerStatusEvent``. Safe to call without a running event loop."""
        self._event_dispatcher.dispatch(TaggerStatusEvent(state=state, source=self._source_label(), error=error))

    def _tag_result_key(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds,
        replace_underscores: bool,
    ) -> TaggerResultKey:
        """Build a :class:`TaggerResultKey` from the current config + effective settings.

        ``thresholds`` and ``replace_underscores`` are the *effective*
        values (per-request override or config default) — the caller's
        responsibility. Two keys with different effective settings hash
        to different buckets and never collide.
        """
        cfg = self._configuration
        model_id = cfg.tagger_repo_id.strip() or cfg.tagger_model_path.strip()
        return TaggerResultKey(
            dataset_name=dataset_name,
            image_id=image_id,
            model_id=model_id,
            rating_threshold=thresholds.rating,
            general_threshold=thresholds.general,
            character_threshold=thresholds.character,
            replace_underscores=replace_underscores,
        )

    async def get_tag_result(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
    ) -> TaggerResult | None:
        """Return the cached tag result for an image, if any.

        ``thresholds`` and ``replace_underscores`` are the effective
        values the caller will display (so it lands in the same cache
        bucket as the entry that was originally written). When omitted
        the config defaults are used — matching what :meth:`tag_image`
        falls back to when no per-request override is supplied.

        Returns ``None`` when no entry matches (either never tagged,
        evicted by LRU, or tagged under different settings).
        """
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        eff_replace = replace_underscores if replace_underscores is not None else self._configuration.tagger_replace_underscores
        key = self._tag_result_key(dataset_name, image_id, eff_thresholds, eff_replace)
        async with self._tag_lock:
            return self._tag_results.get(key)

    async def evict_tag_result(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
    ) -> bool:
        """Drop the cached tag result for an image (best-effort).

        Returns ``True`` when an entry was removed, ``False`` when no
        matching entry was present (no error). The fingerprint rules
        match :meth:`get_tag_result`.
        """
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        eff_replace = replace_underscores if replace_underscores is not None else self._configuration.tagger_replace_underscores
        key = self._tag_result_key(dataset_name, image_id, eff_thresholds, eff_replace)
        async with self._tag_lock:
            try:
                del self._tag_results[key]
            except KeyError:
                return False
            return True

    async def tag_image(
        self,
        dataset_name: str,
        image_info: ImageInfo,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
        source: str | None = None,
    ) -> TaggerResult:
        """Tag a single image.

        The subprocess is started on demand if not already running.
        The idle timer is reset at the start of the request so an
        in-flight request can't be torn down mid-flight.

        Dispatches ``ImageTaggedEvent`` on success and
        ``ImageTagErrorEvent`` on failure (via ``EventDispatcher``)
        so SSE clients can react to per-image results. The HTTP
        response carries the same data for the initiating request.

        The caller is responsible for resolving ``image_info`` (e.g.
        via ``DatasetService.get_image``) and verifying the on-disk
        file is present — this service trusts its inputs and focuses
        on the tag lifecycle.

        Args:
            source: Optional frontend-supplied label for the
                tagging source (e.g. a model name chosen from a
                dropdown). Included in the dispatched events so other
                tabs / clients can display what the requester used.
                Falls back to the server's configured model when not
                provided.

        Raises:
            RuntimeError: If the tagger is not configured, or the
                subprocess fails to start.
        """
        if not self.is_configured:
            raise RuntimeError("Tagger is not configured on this server")

        # The ``source`` label for SSE events. Frontend-supplied
        # value wins; otherwise fall back to the configured model.
        event_source = source if source else self._source_label()

        image_bytes = Path(image_info.path).read_bytes()
        start_t = time.monotonic()

        try:
            # Acquire without blocking the loop (see ``_acquire_lifecycle``)
            # so cancelling an in-flight tag can't deadlock the unwind.
            await self._acquire_lifecycle()
            try:
                await self._ensure_running_locked()
                assert self._tagger_client is not None  # ensured above
                client = self._tagger_client
                # Reset the idle timer while holding the lock so the
                # periodic idle check sees an in-use server.
                self._last_used_t = time.monotonic()
                result = await client.tag(image_bytes)
            finally:
                self._lifecycle_lock.release()
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_t) * 1000)
            self._event_dispatcher.dispatch(
                ImageTagErrorEvent(
                    dataset_name=dataset_name,
                    image_id=image_info.id,
                    error=str(exc),
                    source=event_source,
                    duration_ms=duration_ms,
                )
            )
            raise

        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        thresholded = apply_thresholds(
            result,
            rating_threshold=eff_thresholds.rating,
            general_threshold=eff_thresholds.general,
            character_threshold=eff_thresholds.character,
        )
        eff_replace = replace_underscores if replace_underscores is not None else self._configuration.tagger_replace_underscores
        if eff_replace:
            thresholded = replace_underscores_in(thresholded)

        duration_ms = int((time.monotonic() - start_t) * 1000)
        self._event_dispatcher.dispatch(
            ImageTaggedEvent(
                dataset_name=dataset_name,
                image_id=image_info.id,
                file_name=image_info.file_name,
                path=image_info.path,
                tags=thresholded.tags,
                categories=thresholded.categories,
                source=event_source,
                duration_ms=duration_ms,
            )
        )
        # Cache the thresholded result for cross-session reads (Tags tab
        # on a fresh page load). The key fingerprint is the *effective*
        # model + thresholds + replace_underscores — so a request that
        # uses different per-request overrides lands in a different
        # bucket from a request that used the config defaults.
        async with self._tag_lock:
            self._tag_results[self._tag_result_key(dataset_name, image_info.id, eff_thresholds, eff_replace)] = thresholded
        return thresholded

    async def tag_single_image_async(
        self,
        dataset_name: str,
        image_info: ImageInfo,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
        source: str | None = None,
    ) -> TaggerResult:
        """Tag a single image, gated by the cross-service dataset coordinator.

        Wraps :meth:`tag_image` (which is also called internally by batch
        jobs) with a transient claim so a single-image tag can't overlap a
        captioning run — or vice versa — on the same dataset.

        Raises:
            DatasetBusyError: if another job kind holds the dataset (→ 409).
            RuntimeError: if the tagger is not configured / fails to start.
        """
        claim = await self._dataset_jobs.try_acquire(dataset_name, "tagging", uuid.uuid4().hex[:12])
        try:
            return await self.tag_image(
                dataset_name,
                image_info,
                thresholds=thresholds,
                replace_underscores=replace_underscores,
                source=source,
            )
        finally:
            await self._dataset_jobs.release(claim)

    # --- lifecycle helpers (callers must hold ``_lifecycle_lock``) -------

    async def _ensure_running_locked(self) -> None:
        """Spawn the tagger subprocess if not already running.

        Raises:
            RuntimeError: if the subprocess fails to start.
        """
        if self._tagger_client is not None and self._tagger_client.is_alive:
            return

        # Either no client yet, or the previous one died. Build a
        # fresh client and start it. We do NOT reuse the dead client
        # — its multiprocessing queues are stale.
        model_path = self._configuration.tagger_model_path
        repo_id = self._configuration.tagger_repo_id.strip()
        tagger_kwargs: dict[str, Any] = {}
        label_path: str | None = None

        if repo_id:
            # Worker downloads from HuggingFace; local paths are ignored.
            tagger_kwargs["repo_id"] = repo_id
            tagger_kwargs["repo_model_filename"] = self._configuration.tagger_repo_model_filename
            tagger_kwargs["repo_label_filename"] = self._configuration.tagger_repo_label_filename
        else:
            label_path = self._resolve_label_path(model_path)
            if label_path:
                tagger_kwargs["label_path"] = label_path

        self._emit_status("starting")

        client = TaggerClient(
            OnnxTagger,
            model_path,
            tagger_kwargs,
            heartbeat_interval=self._configuration.tagger_heartbeat_interval_seconds,
            response_timeout=self._configuration.tagger_response_timeout_seconds,
            poll_interval=self._configuration.tagger_liveness_poll_seconds,
        )
        try:
            await client.start()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Failed to start tagger process.")
            self._emit_status("failed", error=str(exc))
            raise RuntimeError("Failed to start tagger process — check server logs") from None

        self._tagger_client = client
        self._logger.info(
            "Tagger process started. [source=%s, labels=%s]",
            f"hf:{repo_id}" if repo_id else model_path,
            label_path or ("<downloaded from HF>" if repo_id else "<none>"),
        )
        self._emit_status("ready")

    async def _stop_locked_async(self) -> None:
        """Stop the tagger subprocess (async). Used from ``on_shutdown``."""
        if self._tagger_client is None:
            return
        self._emit_status("stopping")
        client = self._tagger_client
        # Clear the state first so a new request arriving while
        # ``client.stop()`` is in flight sees "no client" and spawns
        # a fresh subprocess. The lock (held by the caller) prevents
        # a concurrent spawn from racing this stop.
        self._tagger_client = None
        self._last_used_t = None
        try:
            await client.stop()
        except Exception:  # noqa: BLE001
            self._logger.exception("Error stopping tagger process.")
            self._emit_status("failed", error="stop failed")
        else:
            self._logger.info("Tagger process stopped.")
            self._emit_status("stopped")

    def _stop_locked_sync(self) -> None:
        """Stop the tagger subprocess (sync). Used from the idle check thread.

        Blocks the calling thread until the subprocess exits (up to
        ~15s with the default ``TaggerServer.stop`` join/terminate
        timeouts). Fine because the caller is a daemon thread.

        SSE dispatch is still safe from this thread because
        ``EventDispatcher.dispatch`` walks subscribers synchronously;
        the SSE queue push is sync.
        """
        if self._tagger_client is None:
            return
        self._emit_status("stopping")
        client = self._tagger_client
        self._tagger_client = None
        self._last_used_t = None
        try:
            # ``TaggerServer.stop`` is sync; calling it via
            # ``client._server`` skips the ``asyncio.to_thread`` in
            # ``TaggerClient.stop``. This is safe here because we
            # are already in a non-event-loop thread.
            client._server.stop()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error stopping tagger process.")
            self._emit_status("failed", error=str(exc))
        else:
            self._logger.info("Tagger process stopped.")
            self._emit_status("stopped")

    async def _acquire_lifecycle(self) -> None:
        """Acquire the lifecycle lock without blocking the event loop.

        ``threading.Lock.acquire`` is a blocking call; doing it directly on
        the loop thread would freeze the loop and deadlock the unwind of an
        in-flight tag we're cancelling (the loop must run that coroutine so
        it can release the lock). Poll with a non-blocking acquire plus
        ``await asyncio.sleep`` so the loop keeps running while we wait.
        """
        while not self._lifecycle_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)

    def _resolve_label_path(self, model_path: str) -> str | None:
        """Pick the label file, preferring the configured one and falling back to ``<model_dir>/selected_tags.csv``."""
        configured = self._configuration.tagger_label_path.strip()
        if configured:
            return configured
        candidate = Path(model_path).parent / "selected_tags.csv"
        return str(candidate) if candidate.exists() else None

    # --- idle check (called by JobScheduler in a daemon thread) ----------

    def _idle_check_tick(self) -> None:
        """Stop the subprocess if it has been idle longer than the configured timeout.

        In-flight requests and recent activity reset the idle timer,
        so this never tears down a server that's actively serving.
        ``tagger_idle_timeout_seconds <= 0`` disables teardown
        entirely (the subprocess stays up once started).
        """
        timeout = self._configuration.tagger_idle_timeout_seconds
        if timeout <= 0:
            return

        last_used = self._last_used_t
        if last_used is None:
            return  # never used since startup

        if time.monotonic() - last_used < timeout:
            return

        # Non-blocking acquire: if an async request is currently
        # holding the lock, skip this tick — the request will reset
        # the timer and the next tick will see the fresh timestamp.
        if not self._lifecycle_lock.acquire(blocking=False):
            return
        try:
            # Re-check everything under the lock — state may have
            # changed while we were waiting.
            if self._tagger_client is None or self._last_used_t is None:
                return
            if time.monotonic() - self._last_used_t < timeout:
                return
            if not self._tagger_client.is_alive:
                # Subprocess died but the client object lingered.
                # Clean up state; next request will respawn.
                self._tagger_client = None
                self._last_used_t = None
                return
            idle_for = time.monotonic() - self._last_used_t
            self._logger.info("Tagger idle for %.0fs, stopping. [timeout=%ss]", idle_for, timeout)
            self._stop_locked_sync()
        finally:
            self._lifecycle_lock.release()

    # --- batch tagging jobs ----------------------------------------------

    async def start_tag_job_async(self, dataset_name: str, options: TagJobOptions) -> TagJobInfo:
        """Start a batch tagging job for *dataset_name*.

        At most one tagging job per dataset may run at a time. Raises
        :class:`ValueError` (→ 409) if one is already running. The job
        runs as an ``asyncio.Task``; per-image results reach clients via
        ``ImageTaggedEvent`` / ``ImageTagErrorEvent`` and aggregate
        progress via ``TagJobStatusEvent``.
        """
        if not self.is_configured:
            raise RuntimeError("Tagger is not configured on this server")

        job_id = uuid.uuid4().hex[:12]
        # Resolve the image set synchronously so the initial status is
        # accurate and config/lookup errors surface as 4xx before the
        # background task starts (mirrors captioning preflight).
        images = self._resolve_job_images(dataset_name, options.image_ids)

        async with self._tag_jobs_lock:
            existing = self._tag_jobs.get(dataset_name)
            if existing is not None and existing.task is not None and not existing.task.done():
                raise ValueError(f"A tagging job is already running for dataset '{dataset_name}'")

            # Cross-service gate: at most one sidecar-writing job per
            # dataset. Raises DatasetBusyError (a ValueError) if a
            # captioning job — or any future job kind — holds it.
            claim = await self._dataset_jobs.try_acquire(dataset_name, "tagging", job_id)

            # Preflight: partition out images the overwrite=False skip
            # would drop anyway (mirrors captioning preflight).
            to_do, skipped = self._filter_skipped_images(dataset_name, images, options.save)

            # Nothing to tag (e.g. overwrite=False and every image
            # already has saved tags). Return done immediately WITHOUT
            # spawning a task or registering expected changes — the
            # HTTP response carries "done", so there's no SSE event to
            # race with the response. Mirrors captioning's no-op path.
            if not to_do:
                await self._dataset_jobs.release(claim)
                return TagJobInfo(
                    status="done",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    processed=skipped,
                    total=skipped,
                    errors=0,
                    source=options.source or self._source_label(),
                )

            state = _TagJobState(
                job_id=job_id,
                dataset_name=dataset_name,
                source=options.source or self._source_label(),
                total=len(to_do) + skipped,
                thresholds=self._thresholds_from_options(options),
                replace_underscores=options.replace_underscores,
                save=options.save,
                claim=claim,
                # Seed progress with the preflight skip count so the bar
                # starts at the right position (e.g. 40/50 when 40 were
                # already tagged).
                processed=skipped,
            )
            self._tag_jobs[dataset_name] = state
            # Tag filesystem change events with the job_id so the
            # initiating frontend tab can suppress the redundant
            # dataset_changed refresh (mirrors captioning). Cleared by a
            # deferred task scheduled in ``_run_tag_job``'s finally — see
            # ``_deferred_clear_expected_changes`` for why the clear is
            # delayed past the loop end.
            self._dataset_watcher.expect_changes(dataset_name, job_id)
            state.task = asyncio.create_task(self._run_tag_job(dataset_name, state, to_do))

        return await self.get_tag_job_status_async(dataset_name)

    async def stop_tag_job_async(self, dataset_name: str) -> bool:
        """Request graceful cancellation of the running tagging job.

        Returns ``False`` if no job is running.
        """
        async with self._tag_jobs_lock:
            state = self._tag_jobs.get(dataset_name)
            if state is None or state.task is None or state.task.done():
                return False
            state.stop_event.set()
        # Don't await the task — the controller returns immediately and
        # the job transitions to "cancelled" via its own finally block.
        return True

    async def cancel_async(self, job_id: str | None = None) -> CancelResult:
        """Cancel a tagging job and/or interrupt an in-flight tag.

        Graceful-first: when *job_id* matches a running batch job, set its
        ``stop_event`` (the loop breaks at the next image boundary). If an
        inference is still in flight after ``tagger_cancel_grace_seconds``,
        escalate to force-killing the subprocess — ONNX ``session.run``
        can't be interrupted mid-call except by ending the process.

        With *job_id* omitted (the single-image synchronous-tag case),
        there is no job to stop — cancel goes straight to the escalation
        check and kills the process if a request is mid-inference.

        Returns the outcome; see :class:`CancelResult`.
        """
        # 1. Graceful — set the matched job's stop_event.
        stopped_job_id = ""
        if job_id is not None:
            async with self._tag_jobs_lock:
                match: _TagJobState | None = None
                for state in self._tag_jobs.values():
                    if state.job_id == job_id:
                        match = state
                        break
                if match is None:
                    return CancelResult(outcome="stale_job")
                if match.task is None or match.task.done():
                    return CancelResult(outcome="nothing_running", job_id=match.job_id)
                match.stop_event.set()
                stopped_job_id = match.job_id

        # 2. Escalation — is an inference in flight right now? The
        #    lifecycle lock is held across ``await client.tag()``, so
        #    held == mid-inference. Free means the job is between images
        #    (or nothing is running): the stop_event catches it; no kill.
        if self._lifecycle_lock.acquire(blocking=False):
            self._lifecycle_lock.release()
            return CancelResult(outcome="stopped" if stopped_job_id else "nothing_running", job_id=stopped_job_id)

        # Held — give it a grace window to finish on its own (small images
        # may complete), then force-kill.
        await asyncio.sleep(self._configuration.tagger_cancel_grace_seconds)
        if self._lifecycle_lock.acquire(blocking=False):
            self._lifecycle_lock.release()
            return CancelResult(outcome="stopped", job_id=stopped_job_id)

        killed = await self._force_kill_async()
        return CancelResult(outcome="killed" if killed else "stopped", killed=killed, job_id=stopped_job_id)

    async def _force_kill_async(self) -> bool:
        """Terminate the subprocess to interrupt an in-flight inference, then
        finalize state before returning.

        Returns ``True`` if a live subprocess was killed; ``False`` if there
        was nothing live to kill. Emits the stopping/stopped status events
        on a successful kill. After terminating, waits (off the event loop,
        bounded) for the in-flight request to unwind and release the
        lifecycle lock, then clears the dead client ref under it. Waiting
        off-loop is essential: the unwind coroutine runs on the loop, so
        the loop must stay free for it to proceed.
        """
        client = self._tagger_client
        if client is None or not client.is_alive:
            return False
        self._emit_status("stopping")
        # ``client.kill`` is async (it off-threads the sync ``server.kill``);
        # await it directly rather than wrapping in ``to_thread``.
        await client.kill()  # process is dead after this

        def _wait_and_clear() -> bool:
            # Block (in a worker thread) until the killed worker's in-flight
            # ``tag()`` surfaces "worker died" (~poll_interval) and its
            # handler releases the lock. Bounded so a truly wedged worker
            # can't hang cancel forever; the self-healing in
            # ``_ensure_running_locked`` handles a leftover stale ref.
            timeout = max(5.0, self._configuration.tagger_liveness_poll_seconds * 10)
            acquired = self._lifecycle_lock.acquire(timeout=timeout)
            if not acquired:
                return False
            try:
                # Only clear if it's still the client we killed — a request
                # that arrived after the kill may have respawned a fresh one.
                if self._tagger_client is client:
                    self._tagger_client = None
                    self._last_used_t = None
                    return True
                return False
            finally:
                self._lifecycle_lock.release()

        cleared = await asyncio.to_thread(_wait_and_clear)
        if cleared:
            self._emit_status("stopped")
        else:
            # Killed, but the in-flight request didn't unwind within the
            # timeout. The (dead) client ref is left for self-healing.
            self._logger.warning("Tagger killed but in-flight request did not unwind within timeout; leaving stale client ref.")
        return True

    async def get_tag_job_status_async(self, dataset_name: str) -> TagJobInfo:
        """Return a snapshot of the tagging job status for *dataset_name*."""
        async with self._tag_jobs_lock:
            state = self._tag_jobs.get(dataset_name)
            if state is None:
                return TagJobInfo(status="idle", dataset_name=dataset_name)
            return state.snapshot()

    async def list_tag_jobs_async(self) -> list[TagJobInfo]:
        """Return a snapshot of every tracked tagging job (active or finished)."""
        async with self._tag_jobs_lock:
            states = list(self._tag_jobs.values())
        return [s.snapshot() for s in states]

    def is_tagging(self, dataset_name: str) -> bool:
        """Return ``True`` if a tagging job is actively running for *dataset_name*."""
        state = self._tag_jobs.get(dataset_name)
        return state is not None and state.task is not None and not state.task.done()

    def job_id_for_dataset(self, dataset_name: str) -> str | None:
        """Return the job_id of the running job for *dataset_name*, or ``None``."""
        state = self._tag_jobs.get(dataset_name)
        if state is None or state.task is None or state.task.done():
            return None
        return state.job_id

    async def save_tags(self, dataset_name: str, image_info: ImageInfo, result: TaggerResult, save_options: TagSaveOptions, *, source: str = "tagger") -> None:
        """Persist a tag *result* for an image per *save_options*.

        Shared by the batch job (auto-save per image) and the interactive
        save endpoint (user-pruned tags). All writes go through
        :class:`DatasetService` so the file watcher's expected-change
        suppression and history-saving are uniform.
        """
        if save_options.mode == "draft":
            text = format_draft(save_options.draft_format, result)
            self._dataset_service.write_draft(dataset_name, image_info.id, save_options.draft_name, text, source=source)
        elif save_options.mode == "extras":
            self._merge_extras_tags(dataset_name, image_info.id, extras_tags(result), source=source)
        # mode == "none" → write nothing.

    def _merge_extras_tags(self, dataset_name: str, image_id: int, tags: dict[str, Any], *, source: str = SELF_JOB_ID) -> bool:
        """Write the tagger's categorized output as a ``[tags]`` sub-table on the image's TOML extras.

        Overwrites the existing ``[tags]`` sub-table (authoritative
        tagger output) while leaving other top-level keys untouched.
        Goes through :meth:`DatasetService.update_extras` so the
        watcher's expected-change suppression and history-saving are
        uniform. Returns ``False`` when the image is missing.
        """
        if self._dataset_service.get_image(dataset_name, image_id) is None:
            return False
        # Load the existing extras as a tomlkit doc (plain=False) so
        # comments and formatting on other keys survive the round-trip.
        # None means the image has no sidecar (or it's unparseable) —
        # start from an empty doc so the write path repairs it.
        extras = self._dataset_service.load_extras(dataset_name, image_id) or tomlkit.document()
        extras["tags"] = tags
        return self._dataset_service.update_extras(dataset_name, image_id, tomlkit.dumps(extras), source=source)

    def _has_tags_table(self, dataset_name: str, image_id: int) -> bool:
        """Whether the image's TOML extras already carries a ``[tags]`` sub-table.

        Drives the batch ``overwrite=False`` skip for the extras path.
        Returns ``False`` when the image, the sidecar, or the ``[tags]``
        key is missing, or when the sidecar is unparseable.
        """
        extras = self._dataset_service.load_extras(dataset_name, image_id)
        return extras is not None and "tags" in extras

    def preview_save_tags(self, result: TaggerResult, save_options: TagSaveOptions) -> str:
        """Format what :meth:`save_tags` would write, without touching disk.

        ``draft`` → the formatter text; ``extras`` → the ``[tags]`` sub-table
        as TOML (only the tagger's contribution, not the merged sidecar);
        ``none`` → empty string. Used by the interactive save bar's preview.
        """
        if save_options.mode == "draft":
            return format_draft(save_options.draft_format, result)
        if save_options.mode == "extras":
            doc = tomlkit.document()
            doc["tags"] = extras_tags(result)
            return tomlkit.dumps(doc).strip()
        return ""

    def _image_already_has_tags(self, dataset_name: str, image_info: ImageInfo, save_options: TagSaveOptions) -> bool:
        """Whether the image already carries the target save artifact.

        Drives the batch ``overwrite=False`` skip. Draft mode checks the
        DB-backed ``draft_names`` list (cheap); extras mode reads the
        ``[tags]`` sub-table from the TOML sidecar via
        :meth:`_has_tags_table`. ``none`` mode never skips (nothing is
        written).
        """
        if save_options.mode == "draft":
            return save_options.draft_name in image_info.draft_names
        if save_options.mode == "extras":
            return self._has_tags_table(dataset_name, image_info.id)
        return False

    def _filter_skipped_images(self, dataset_name: str, images: list[ImageInfo], save_options: TagSaveOptions) -> tuple[list[ImageInfo], int]:
        """Preflight partition into ``(to_do, skipped)``.

        Mirrors captioning's preflight filter: when ``overwrite=False``
        and the save mode is real (not ``none``), images already
        carrying the target artifact are dropped here rather than
        skipped inside the loop. This lets the start path return
        ``done`` immediately when nothing remains (no background task,
        no SSE event to race the HTTP response) and seeds accurate
        ``total``/``processed`` accounting. Returns the full list with
        ``skipped=0`` when skipping doesn't apply.
        """
        if save_options.mode == "none" or save_options.overwrite:
            return images, 0
        to_do = [img for img in images if not self._image_already_has_tags(dataset_name, img, save_options)]
        return to_do, len(images) - len(to_do)

    async def _run_tag_job(self, dataset_name: str, state: "_TagJobState", images: list[ImageInfo]) -> None:
        """Background task: iterate images, tag each, optionally save, emit status."""
        state.started_at = time.monotonic()
        await self._emit_tag_status(dataset_name, state)
        try:
            for image_info in images:
                if state.stop_event.is_set():
                    break
                self._event_dispatcher.dispatch(
                    ImageTagStartedEvent(
                        dataset_name=dataset_name,
                        job_id=state.job_id,
                        image_id=image_info.id,
                        file_name=image_info.file_name,
                    )
                )
                try:
                    result = await self.tag_image(
                        dataset_name,
                        image_info,
                        thresholds=state.thresholds,
                        replace_underscores=state.replace_underscores,
                        source=state.source,
                    )
                    if state.save.mode != "none":
                        try:
                            await self.save_tags(dataset_name, image_info, result, state.save, source=f"tagger:{state.job_id}")
                        except Exception:  # noqa: BLE001
                            # Saving is best-effort per image — a failed write
                            # shouldn't abort the run. The tag itself already
                            # succeeded and was surfaced via ImageTaggedEvent.
                            self._logger.exception("Failed to save tags for image. [dataset=%s, image_id=%s]", dataset_name, image_info.id)
                except Exception:  # noqa: BLE001
                    # ``tag_image`` already dispatched ImageTagErrorEvent.
                    async with state.lock:
                        state.errors += 1
                        state.error_messages.append(f"image {image_info.id}: tag failed")
                    continue
                async with state.lock:
                    state.processed += 1
                await self._emit_tag_status(dataset_name, state)

            async with state.lock:
                if state.stop_event.is_set():
                    state.status = "cancelled"
                elif state.errors > 0 and state.processed == 0:
                    state.status = "error"
                else:
                    state.status = "done"
        except asyncio.CancelledError:
            async with state.lock:
                state.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            async with state.lock:
                state.status = "error"
                state.error = str(exc)
                state.error_messages.append(str(exc))
        finally:
            await self._dataset_jobs.release(state.claim)
            # Defer clearing the expected-changes tag: residual inotify
            # events from the last writes land after the loop exits, and
            # the watcher would re-tag them with the per-file ``source``
            # ("tagger") instead of the job_id — leaking through as a
            # toast. ``clear_expected_changes_for_job`` only fires if the
            # job_id still matches, so a newer job's tag is safe.
            clear_task = asyncio.create_task(self._deferred_clear_expected_changes(dataset_name, state.job_id))
            self._pending_clears.add(clear_task)
            clear_task.add_done_callback(self._pending_clears.discard)
            await self._emit_tag_status(dataset_name, state)

    async def _deferred_clear_expected_changes(self, dataset_name: str, job_id: str) -> None:
        """Clear the job's expected-changes tag after a grace window.

        Residual inotify events from the last batch writes land slightly
        after the loop exits (kernel buffering + the watcher debounce).
        Sleeping before the clear keeps ``_expected_sources`` populated
        with the real job_id long enough for those events to be captured
        and suppressed by the originating tab. Mirrors captioning's
        ``_cleanup_async``.
        """
        try:
            await asyncio.sleep(self._configuration.tagger_expected_changes_grace_seconds)
            self._dataset_watcher.clear_expected_changes_for_job(dataset_name, job_id)
        finally:
            task = asyncio.current_task()
            if task is not None:
                self._pending_clears.discard(task)

    async def _emit_tag_status(self, dataset_name: str, state: "_TagJobState") -> None:
        snap = state.snapshot()
        self._event_dispatcher.dispatch(
            TagJobStatusEvent(
                status=snap.status,
                dataset_name=dataset_name,
                processed=snap.processed,
                total=snap.total,
                errors=snap.errors,
                job_id=snap.job_id,
                error=snap.error,
                source=snap.source,
                elapsed=snap.elapsed,
            )
        )

    def _thresholds_from_options(self, options: TagJobOptions) -> TaggingThresholds:
        """Resolve per-request threshold overrides against the server config."""
        cfg = self._configuration
        return TaggingThresholds(
            rating=options.rating_threshold if options.rating_threshold is not None else cfg.tagger_rating_threshold,
            general=options.general_threshold if options.general_threshold is not None else cfg.tagger_general_threshold,
            character=options.character_threshold if options.character_threshold is not None else cfg.tagger_character_threshold,
        )

    def _resolve_job_images(self, dataset_name: str, image_ids: list[int] | None) -> list[ImageInfo]:
        """Resolve the image set for a job: explicit ids, or the whole dataset.

        ``image_ids`` omitted/empty → paginate every image in the dataset
        (newest first, matching the browse order). Explicit ids resolve
        one ``get_image`` each; unknown ids are silently dropped.
        """
        if image_ids:
            resolved: list[ImageInfo] = []
            for image_id in image_ids:
                info = self._dataset_service.get_image(dataset_name, image_id)
                if info is not None:
                    resolved.append(info)
            return resolved

        all_images: list[ImageInfo] = []
        next_token: str | None = None
        while True:
            page = self._dataset_service.list_images(dataset_name, limit=500, next=next_token)
            all_images.extend(page.images)
            next_token = page.next_token
            if not next_token:
                break
        return all_images


@dataclass
class _TagJobState:
    """Mutable state for one batch tagging job (guarded by ``lock`` for counts)."""

    job_id: str
    dataset_name: str
    source: str
    total: int
    thresholds: TaggingThresholds
    replace_underscores: bool | None
    save: TagSaveOptions
    claim: JobClaim
    status: TagJobStatus = "running"
    processed: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    started_at: float | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[Any] | None = None

    def snapshot(self) -> TagJobInfo:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = time.monotonic() - self.started_at
        return TagJobInfo(
            status=self.status,
            dataset_name=self.dataset_name,
            job_id=self.job_id,
            processed=self.processed,
            total=self.total,
            errors=self.errors,
            error=self.error,
            error_messages=list(self.error_messages),
            source=self.source,
            elapsed=elapsed,
        )
