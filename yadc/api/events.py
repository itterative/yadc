"""Event types and dataclasses dispatched on the :class:`EventDispatcher`.

Each event is a ``@dataclass`` subclass of :class:`Event` with a
``TYPE: ClassVar[str]`` carrying the wire name (e.g. ``"captioning_status"``).
Events broadcast to SSE clients subclass :class:`SSEEvent` so
:class:`SSEEvents` can fan them all out from a single handler.

- ``SetupAppEvent`` — fired during ``Application.configure_app()``,
  *before* blueprints are registered on the Quart app. Carries the
  ``Quart`` instance so services can register request/response hooks
  (e.g. CORS attaches ``after_request`` to the ``ApiBlueprint``).
  Fired synchronously, while the blueprints are still unwired — at
  this point there is no running event loop, so handlers should be
  sync.
- ``StartupEvent`` / ``ShutdownEvent`` — application lifecycle, used to
  bring services up/down. ``StartupEvent`` is dispatched from Quart's
  ``before_serving`` callback, after the event loop is running, so
  async handlers can use ``asyncio.create_task()`` directly.
- ``PingEvent`` — periodic keepalive (every 5s); not stored in the SSE
  history ring buffer.
- ``CaptioningStatusEvent`` — per-job progress (status, processed, total,
  errors, elapsed, max_concurrent) for the UI's progress bar.
- ``DatasetChangedEvent`` — emitted by the dataset watcher and on manual
  rescan; ``job_id`` carries the originating source id so the
  originating UI tab can suppress its own reload. ``changed_paths`` is
  populated by the watcher with the filesystem paths observed during
  the debounce window so the dataset service can do targeted
  upsert/delete for just the affected image rows instead of walking
  the whole dataset. Empty for events not driven by a watcher burst
  (manual rescan, periodic refresh, dataset upload).
- ``ResumptionFailedEvent`` — returned to a single SSE client when its
  ``Last-Event-ID`` is older than the oldest buffered event.
- ``ImageCaptionedEvent`` / ``ImageRefinedEvent`` /
  ``ImageCaptionStartedEvent`` / ``ImageCaptionErrorEvent`` — per-image
  notifications, consumed by the UI to update the image grid.
- ``EnvironmentsChangedEvent`` / ``TemplatesChangedEvent`` — emitted by
  the env / template watchers so the UI can refresh the env/template
  pickers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Literal

from quart import Quart


@dataclass
class Event:
    TYPE: ClassVar[str]


@dataclass
class SSEEvent(Event):
    """Marker base for events broadcast to SSE clients.

    Subscribing to ``SSEEvent`` (the dispatcher walks the event's MRO) catches
    every broadcast event, so ``SSEEvents`` needs a single handler instead of
    one per type. Lifecycle events (SetupApp/Startup/Shutdown) and
    ``ResumptionFailedEvent`` stay plain ``Event``s and are excluded from the
    SSE stream.
    """


@dataclass
class SetupAppEvent(Event):
    TYPE: ClassVar[str] = "setup_app"
    app: Quart


@dataclass
class StartupEvent(Event):
    TYPE: ClassVar[str] = "startup"


@dataclass
class ShutdownEvent(Event):
    TYPE: ClassVar[str] = "shutdown"


@dataclass
class PingEvent(SSEEvent):
    TYPE: ClassVar[str] = "ping"
    time: str


@dataclass
class CaptioningStatusEvent(SSEEvent):
    TYPE: ClassVar[str] = "captioning_status"
    status: Literal["idle", "running", "stopping", "error", "done", "cancelled"]
    dataset_name: str
    processed: int
    total: int
    errors: int
    job_id: str = ""
    error: str | None = None
    error_messages: list[str] = dataclasses.field(default_factory=list)
    api_url: str = ""
    api_model_name: str = ""
    # Seconds since the job started. 0 before the job has actually
    # started.  Used by the frontend for the "elapsed" display and as
    # a sanity check on per-image timing under concurrency.
    elapsed: float = 0.0
    # Configured concurrency for the job. The frontend divides the
    # per-image ETA by this so the estimate reflects actual wall-clock
    # throughput (max_concurrent requests in flight).
    max_concurrent: int = 1


@dataclass
class DatasetChangedEvent(SSEEvent):
    TYPE: ClassVar[str] = "dataset_changed"
    dataset_name: str
    job_id: str | None = None
    # Filesystem paths that changed during the watcher's debounce
    # window. Used by the dataset service to scope its index update
    # to just the affected images. Empty when the event isn't from
    # the watcher (manual rescan, background refresh, dataset upload)
    # — consumers should fall back to a full scan in that case.
    changed_paths: list[str] = dataclasses.field(default_factory=list)


@dataclass
class ResumptionFailedEvent(Event):
    TYPE: ClassVar[str] = "resumption_failed"
    requested_event_id: int


@dataclass
class ImageCaptionedEvent(SSEEvent):
    TYPE: ClassVar[str] = "image_captioned"
    dataset_name: str
    job_id: str
    id: int
    file_name: str
    path: str
    has_caption: bool = False
    has_toml: bool = False
    width: int = 0
    height: int = 0
    draft_names: list[str] = dataclasses.field(default_factory=list)
    last_modified_t: float | None = None
    caption: str = ""
    duration_ms: int = 0
    api_url: str = ""
    api_model_name: str = ""


@dataclass
class ImageRefinedEvent(SSEEvent):
    TYPE: ClassVar[str] = "image_refined"
    dataset_name: str
    job_id: str
    image_id: int
    caption: str = ""
    source: Literal["caption", "draft"] = "caption"
    draft_name: str = ""


@dataclass
class EnvironmentsChangedEvent(SSEEvent):
    TYPE: ClassVar[str] = "environments_changed"
    envs: list[str] = dataclasses.field(default_factory=list)


@dataclass
class TemplatesChangedEvent(SSEEvent):
    TYPE: ClassVar[str] = "templates_changed"
    templates: list[str] = dataclasses.field(default_factory=list)


@dataclass
class ImageCaptionStartedEvent(SSEEvent):
    TYPE: ClassVar[str] = "image_caption_started"
    dataset_name: str
    job_id: str
    image_id: int
    file_name: str


@dataclass
class ImageCaptionErrorEvent(SSEEvent):
    TYPE: ClassVar[str] = "image_caption_error"
    dataset_name: str
    job_id: str
    image_id: int
    error: str
    duration_ms: int = 0
    api_url: str = ""
    api_model_name: str = ""


@dataclass
class TaggerStatusEvent(SSEEvent):
    """Lifecycle of the tagger subprocess.

    Fired when the subprocess transitions state so the frontend can
    surface a status indicator (e.g. "tagger warming up" during
    lazy spawn, "tagger stopped" after idle teardown).

    ``source`` describes where the model came from: ``"hf:<repo_id>"``
    for HuggingFace Hub downloads, ``"local:<path>"`` for local files,
    ``""`` when no model is configured.
    """

    TYPE: ClassVar[str] = "tagger_status"
    state: Literal["starting", "ready", "stopping", "stopped", "failed"]
    source: str = ""
    error: str | None = None


@dataclass
class ImageTagStartedEvent(SSEEvent):
    """Per-image start signal from the tagger batch runner.

    Mirrors :class:`ImageCaptionStartedEvent` so the frontend can add the
    tile to its in-flight set (and drive the shimmer indicator) before the
    result arrives via :class:`ImageTaggedEvent`.
    """

    TYPE: ClassVar[str] = "image_tag_started"
    dataset_name: str
    job_id: str
    image_id: int
    file_name: str


@dataclass
class ImageTaggedEvent(SSEEvent):
    """Per-image success from the tagger endpoint.

    Carries the *thresholded* tag result so other tabs / clients that
    didn't initiate the request see the same view the requester sees.
    For SmilingWolf-style models, ``tags`` is ``{tag: score}`` and
    ``categories`` groups tags by category (e.g.
    ``{"rating": [...], "general": [...], "character": [...]}``).
    """

    TYPE: ClassVar[str] = "image_tagged"
    dataset_name: str
    image_id: int
    file_name: str
    path: str
    tags: dict[str, float]
    categories: dict[str, list[str]]
    source: str = ""
    duration_ms: int = 0


@dataclass
class ImageTagErrorEvent(SSEEvent):
    """Per-image failure from the tagger endpoint.

    Dispatched when the subprocess is reached but the tag call
    itself fails (model inference error, subprocess crash, etc.).
    404 ``FileNotFoundError`` from the controller is not surfaced
    here — the HTTP response carries that.
    """

    TYPE: ClassVar[str] = "image_tag_error"
    dataset_name: str
    image_id: int
    error: str
    source: str = ""
    duration_ms: int = 0


@dataclass
class TagJobStatusEvent(SSEEvent):
    """Progress snapshot for a batch tagging job.

    Mirrors :class:`CaptioningStatusEvent` for the tagger so the
    frontend can drive a progress bar from a single streamed event.
    Per-image success / failure still arrive via
    :class:`ImageTaggedEvent` / :class:`ImageTagErrorEvent`; this event
    carries the aggregate counts.
    """

    TYPE: ClassVar[str] = "tag_job_status"
    status: Literal["idle", "running", "stopping", "error", "done", "cancelled"]
    dataset_name: str
    processed: int
    total: int
    errors: int
    job_id: str = ""
    error: str | None = None
    source: str = ""
    elapsed: float = 0.0
