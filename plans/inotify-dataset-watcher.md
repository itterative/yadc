# Plan: inotify Dataset File Watcher with SSE + Frontend Integration

## Goal

Automatically detect when new images are added to (or removed from) a dataset's watched directories, and notify the frontend via SSE so the user can refresh the dataset browser without manual rescans.

## Current State

- **Backend**: `DatasetService._refresh_stale_datasets()` rescans all datasets every 60s when any listing endpoint is called. There's also a manual `POST /api/datasets/<name>/rescan` endpoint.
- **SSE**: Global event stream at `GET /api/events` uses `SSEEvents` (Condition-based queue) + `EventDispatcher`. Events are `PingEvent` and `CaptioningStatusEvent`. Captioning also has a per-dataset SSE endpoint.
- **Frontend**: The dataset listing page (`+page.svelte`) connects to the global SSE stream. The dataset browser page (`datasets/[name]/+page.svelte`) does NOT listen to any SSE events currently — it only fetches images on load and infinite-scroll. Captioning progress has its own per-dataset SSE subscription via `CaptionProgress.svelte`.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Backend (Flask)                                          │
│                                                          │
│  DatasetWatcherService ──watch──▶ filesystem paths       │
│  (inotify via watchdog)         (per dataset [[dataset]]) │
│         │                                                │
│         ▼                                                │
│  EventDispatcher.dispatch(DatasetChangedEvent)           │
│         │                                                │
│         ▼                                                │
│  SSEEvents.on_dataset_changed() ──▶ SSE queue            │
│         │                                                │
│         ▼                                                │
│  GET /api/events ──▶ EventSource                         │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Frontend (SvelteKit)                                     │
│                                                          │
│  Global SSE listener (+page.svelte or layout)            │
│         │  DatasetChangedEvent                           │
│         ▼                                                │
│  datasets/[name]/+page.svelte:                           │
│    - Show "New images detected — Refresh" banner         │
│    - User clicks → reloadInitial()                       │
└──────────────────────────────────────────────────────────┘
```

## Steps

### Step 1: Add `watchdog` dependency

Add `watchdog` to `pyproject.toml` dependencies. It's a cross-platform filesystem watching library (uses inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows).

```toml
dependencies = [
    # ... existing ...
    "watchdog>=4.0",
]
```

### Step 2: Create `DatasetChangedEvent` in `events.py`

A new event type emitted when a dataset's watched directories change.

```python
@dataclass
class DatasetChangedEvent(Event):
    TYPE: ClassVar[str] = "dataset_changed"
    dataset_name: str
    change_type: Literal["files_added", "files_removed", "files_modified"]  # coarse-grained
    path: str  # the directory that changed
```

This is coarse-grained — we don't list individual files. The frontend just knows "something changed in this dataset" and can offer a refresh.

### Step 3: Create `DatasetWatcherService` in `yadc/api/modules/dataset_watcher.py`

A new `Service` subclass that:

1. **On startup**: Queries all registered datasets from `DatasetService`, extracts their image directory paths, and starts a `watchdog` observer with per-directory event handlers.
2. **Debouncing**: Uses a per-dataset debounce timer (e.g. 2 seconds). Multiple inotify events within the window for the same dataset are coalesced into a single `DatasetChangedEvent`.
3. **Filtering**: Watches both image files (`IMAGE_EXTENSIONS`) and sidecar files (`.txt`, `.toml`, `.draft~`, `.history~`). External tools may edit sidecars.
4. **Event emission**: Dispatches `DatasetChangedEvent` via `EventDispatcher`.
5. **Lifecycle**: When a dataset is registered/unregistered, adds/removes its paths from the observer. Uses `EventDispatcher` to listen for dataset registration events (or is called directly by `DatasetService`).

```python
class DatasetWatcherService(Service):
    def __init__(
        self,
        dataset_service: DatasetService,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
    ):
        ...

    def start_watching(self, dataset_name: str, paths: list[str]):
        """Add paths for a dataset to the watchdog observer."""

    def stop_watching(self, dataset_name: str):
        """Remove all watched paths for a dataset."""

    def _on_fs_event(self, dataset_name: str, event: FileSystemEvent):
        """Debounced handler — dispatches DatasetChangedEvent."""
```

**Key design decisions:**
- **One `Observer` instance**, shared across all datasets. Each path gets its own `EventHandler` that knows which dataset it belongs to.
- **Debounce via `threading.Timer`**: Reset the timer on each event; when it fires, dispatch the event. Default 1 second, configurable via `Configuration.watcher_debounce_seconds`.
- **Only watch directories, not files**: The `[[dataset]]` entries specify directory paths. We watch those directories for `created`, `deleted`, `moved` events on files.
- **No recursive watching**: Dataset entries point to flat image directories (same as CLI behavior). Can be added later if needed.

### Step 4: Wire `DatasetWatcherService` into SSEEvents

In `SSEEvents`, add an event handler for `DatasetChangedEvent`:

```python
@event_handler(DatasetChangedEvent)
def on_dataset_changed(self, event: DatasetChangedEvent):
    self.push(event)
```

This follows the exact same pattern as `on_captioning_status`.

### Step 5: Integration with `DatasetService`

The watcher needs to know what directories to watch for each dataset. Two approaches:

**Option A (recommended): `DatasetService` calls `DatasetWatcherService` directly.**
- After `_register()` (import/create), call `watcher.start_watching(name, paths)`.
- After `unregister_dataset()`, call `watcher.stop_watching(name)`.
- On startup, the watcher queries existing datasets via `DatasetService.list_datasets()` + loading configs.

**Option B: Event-driven.**
- `DatasetService` dispatches a `DatasetRegisteredEvent` / `DatasetUnregisteredEvent`.
- `DatasetWatcherService` subscribes and reacts.
- More decoupled but adds more event types for an internal-only flow.

Going with **Option A** for simplicity — these are internal services in the same process.

### Step 6: Backend API — add Zod schema and SSE event type

The `DatasetChangedEvent` will flow through the existing `GET /api/events` SSE endpoint — no new endpoint needed. The frontend just needs to listen for the `dataset_changed` event type.

### Step 7: Frontend — Zod schema in `captioning.ts` (or a new `datasetEvents.ts`)

```typescript
export const DatasetChangedEventZ = z.object({
  dataset_name: z.string(),
  change_type: z.enum(["files_added", "files_removed", "files_modified"]),
  path: z.string(),
});

export type DatasetChangedEvent = z.infer<typeof DatasetChangedEventZ>;
```

### Step 8: Frontend — Listen in `datasets/[name]/+page.svelte`

Add a global SSE listener (reuse the connection from `+page.svelte` or create a new one) that watches for `dataset_changed` events matching the current dataset name. When received:

1. Set a `hasPendingChanges` flag.
2. Show a banner: **"📁 New changes detected — Refresh"** with a button.
3. On click: call `loadInitial(datasetName)` + clear the flag.

The banner should be a subtle, non-intrusive notification — not a modal. It could slide in at the top of the dataset browser area.

```svelte
{#if hasPendingChanges}
  <div class="rounded-lg bg-blue-900/50 border border-blue-700/50 px-4 py-2 flex items-center justify-between">
    <span class="text-sm text-blue-200">Dataset has new changes</span>
    <button class="btn-primary text-xs" onclick={handleRefresh}>Refresh</button>
  </div>
{/if}
```

### Step 9: Frontend — Also update the dataset listing page

If the user is on the dataset listing (`+page.svelte`) and a dataset changes, update the dataset's image count (or show a badge). This is optional for v1 — the listing already refreshes on navigation.

## File Change Summary

| File | Change |
|------|--------|
| `pyproject.toml` | Add `watchdog` dependency |
| `yadc/api/events.py` | Add `DatasetChangedEvent` dataclass |
| `yadc/api/modules/dataset_watcher.py` | **New file** — `DatasetWatcherService` |
| `yadc/api/modules/__init__.py` | Re-export `DatasetWatcherService` |
| `yadc/api/modules/sse_events.py` | Add `@event_handler(DatasetChangedEvent)` method |
| `yadc/api/services/datasets.py` | Inject `DatasetWatcherService`, call it on register/unregister, expose method to get dataset paths |
| `webui/src/lib/stores/captioning.ts` | Add `DatasetChangedEventZ` Zod schema |
| `webui/src/routes/datasets/[name]/+page.svelte` | Listen for `dataset_changed` SSE events, show refresh banner |
| `webui/src/routes/+page.svelte` | Forward `dataset_changed` events to store (optional) |

## Resolved Questions

1. **Recursive watching?** No — defer for now. The CLI doesn't do recursive scanning either.

2. **Auto-refresh vs. manual refresh?** Manual "Refresh" button. Auto-refresh would disrupt scroll position / open modals.

3. **Sidecar file changes?** Yes — watch `.txt`, `.toml`, `.draft~`, `.history~` sidecar changes too. External tools may edit these. All changes for a dataset are coalesced into a single debounced event.

4. **Startup race condition?** Acceptable — the 60s stale refresh is still a fallback.

5. **Cross-platform?** `watchdog` abstracts platform differences (inotify/Linux, FSEvents/macOS, ReadDirectoryChangesW/Windows).

6. **Debounce interval?** 1 second — fast enough to feel immediate, slow enough to avoid excessive rescans. Configurable via `Configuration.watcher_debounce_seconds`.
