"""Dataset service — orchestration, image queries, caption/draft/history operations.

A "dataset" is a named yadc config TOML stored in the XDG state directory
(``STATE_PATH/datasets/<name>/config.toml``). The TOML is the
source of truth — it defines API settings, prompts, and ``[[dataset]]``
entries pointing to image directories.

Two registration flows:

- **Import**: register an existing TOML path (resolving relative paths to
  absolute first).
- **Create**: write a fresh TOML with the given image paths.

Both end up as ``STATE_PATH/datasets/<name>/config.toml``.

This service owns:

- The watcher integration (auto-rescan on external changes)
- The file-system side of caption / extras / history / draft operations
- Lifecycle orchestration: delegates disk walking to
  :class:`DatasetScanner` and config reading to :class:`DatasetLoader`.

The :class:`DatasetRepository` owns the SQL, the data model
(``DatasetInfo``, ``ImageInfo``), and connection / transaction
management.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import tomlkit

from yadc.cmd.app import STATE_PATH
from yadc.core.dataset import DatasetImage
from yadc.utils.dict_utils import load_toml, toml_to_plain

from ..configuration import Configuration
from ..events import DatasetChangedEvent
from ..modules.dataset_watcher import SELF_JOB_ID, DatasetWatcherService
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.event_dispatcher import EventDispatcher, event_handler
from ..modules.job_scheduler import JobScheduler
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .dataset_loader import DatasetLoader
from .dataset_scanner import DatasetScanner

# Base directory for all dataset state directories.
DATASETS_DIR: Path = STATE_PATH / "datasets"

# Data models — re-exported from the repository so the public service
# API still references ``DatasetInfo`` / ``ImageInfo`` from this module.
from .dataset_repository import (  # noqa: E402
    DatasetInfo,
    DatasetRepository,  # noqa: E402
    ImageInfo,
)


def _dataset_config_path(name: str) -> Path:
    """Return the config TOML path for a named dataset."""
    return DATASETS_DIR / name / "config.toml"


@dataclass
class HistoryEntry:
    """A single history snapshot for an image.

    Service-level type: the underlying data is read from a file
    (``.history~`` sidecar), not a SQL row.
    """

    index: int
    caption: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        import hashlib
        import json as _json

        content = self.caption + "\0" + _json.dumps(self.extras, sort_keys=True)
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ImagePage:
    """A paginated page of images. Service-level type (the repo returns a flat list)."""

    images: list[ImageInfo]
    next_token: str | None = None


class DatasetService(Service):
    """Scans filesystem for dataset configs, indexes images in SQLite, and serves queries.

    A "dataset" is a directory containing a ``config.toml`` (v1 or v2 format) with
    ``[[dataset]]`` entries pointing to image directories. The service scans the
    yadc config directories and any registered paths, indexes image metadata into
    the ``datasets`` and ``dataset_images`` tables, and provides paginated queries.
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        watcher: DatasetWatcherService,
        configuration: Configuration,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
        repo: DatasetRepository,
        scanner: DatasetScanner,
        loader: DatasetLoader,
        job_scheduler: JobScheduler | None = None,
    ):
        self._db: DBConnectionFactory = db
        self._watcher: DatasetWatcherService = watcher
        self._configuration: Configuration = configuration
        self._repo: DatasetRepository = repo
        self._scanner: DatasetScanner = scanner
        self._loader: DatasetLoader = loader
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        # Serialize the stale-refresh pass so the background thread and a
        # future manual refresh (see dataset-config-ux-plan todo) don't
        # collide on the SQLite write lock. Without this, the two callers
        # would race and the second would fail with "database is locked"
        # once it exceeded ``busy_timeout`` (5s).
        self._refresh_lock: threading.Lock = threading.Lock()
        # Ensure state dir exists
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        # Configure watcher and register existing datasets
        self._watch_existing_datasets()

        # Register for DatasetChangedEvent to auto-rescan
        event_dispatcher.register_service(self)

        # Stale-dataset refresh runs in a background thread so parallel
        # API requests don't all try to scan the disk at once. The watcher
        # handles inotify-driven updates; this is the fallback for
        # changes the watcher can't see (e.g. external edits to a
        # dataset's config.toml that add new image paths, where no
        # inotify event fires).
        if job_scheduler is not None:
            job_scheduler.new_scheduled_job(self._configuration.dataset_refresh_interval_seconds, self._refresh_stale_datasets)

    # --- Dataset listing ---

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all indexed datasets.

        Stale entries are refreshed by a background thread (scheduled in
        ``__init__``); this method reads the current index state.
        """
        return self._repo.list_datasets()

    def get_dataset(self, name: str) -> DatasetInfo | None:
        """Get a single dataset by name.

        Stale entries are refreshed by a background thread (scheduled in
        ``__init__``); this method reads the current index state.
        """
        return self._repo.get_dataset(name)

    # --- Image listing (paginated) ---

    def list_images(self, dataset_name: str, *, limit: int = 50, next: str | None = None) -> ImagePage:
        """List images for a dataset, paginated by auto-increment ID (newest first).

        Args:
            dataset_name: Name of the dataset.
            limit: Max images to return.
            next: Opaque pagination token returned by a previous
                call as ``next_token``. ``None`` (the default) means
                "first page" — returns the newest images. The client
                treats this as a string token; the server decodes it
                to determine the next page.

        Returns:
            An ImagePage with images (newest first) and optional
            ``next_token``. Pass the token as the next call's
            ``next`` argument to get the next (older) page.
        """
        # Decode the opaque ``next`` token into the repo's internal
        # ``before_id`` cursor. Currently the token is the smallest
        # id on the previous page, but this detail is hidden from the
        # client so the encoding can change later without breaking
        # the public API.
        before_id = self._decode_next_token(next)
        # Fetch limit+1 to detect "is there a next page".
        rows = self._repo.list_images(dataset_name, before_id=before_id, limit=limit + 1)
        images = rows[:limit]
        next_token = None
        if len(rows) > limit:
            # Smallest id on the current page is the last image in
            # DESC order. Use it as the next cursor.
            next_token = str(images[-1].id)
        return ImagePage(images=images, next_token=next_token)

    @staticmethod
    def _decode_next_token(token: str | None) -> int:
        """Decode an opaque ``next`` token into the repo's ``before_id`` cursor.

        ``None`` or an empty token means "no upper bound" — use a
        value larger than any real id so all rows pass the
        ``id < ?`` filter. SQLite's INTEGER is 64-bit; ``2**63 - 1``
        is the max signed value and won't be reached by
        AUTOINCREMENT for billions of years.
        """
        if not token:
            return 2**63 - 1
        return int(token)

    def get_image_paths_in_desc_order(self, dataset_name: str) -> list[str]:
        """Return every image path for a dataset, ordered by id DESC.

        Single SQL query — used by the captioning service to
        reorder the filesystem-resolved image list so the runner
        starts with the newest images first.
        """
        return [path for path, _id in self._repo.list_image_paths_desc(dataset_name)]

    def get_image(self, dataset_name: str, image_id: int) -> ImageInfo | None:
        """Get a single image by ID within a dataset."""
        return self._repo.get_image(dataset_name, image_id)

    # --- Image content access ---

    def get_image_path(self, dataset_name: str, image_id: int) -> Path | None:
        """Return the filesystem path for an image, or None if not found."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None
        p = Path(info.path)
        return p if p.exists() else None

    def get_image_by_path(self, dataset_name: str, image_path: str | Path) -> ImageInfo | None:
        """Look up an image by its filesystem path within a dataset."""
        return self._repo.get_image_by_path(dataset_name, str(image_path))

    def get_draft_names(self, dataset_name: str) -> list[str]:
        """Return sorted list of unique draft names across all images in a dataset."""
        csvs = self._repo.list_draft_names_csv(dataset_name)
        names: set[str] = set()
        for csv in csvs:
            for part in csv.split(","):
                part = part.strip()
                if part:
                    names.add(part)
        return sorted(names)

    def get_draft_summary(self, dataset_name: str) -> list[dict[str, Any]]:
        """Return draft names with image counts for a dataset.

        Returns a list of dicts with ``name`` and ``image_count`` keys,
        sorted by name.
        """
        counts = self._repo.get_draft_name_counts(dataset_name)
        return [{"name": name, "image_count": count} for name, count in sorted(counts.items())]

    def delete_draft_all(self, dataset_name: str, draft_name: str, *, source: str = SELF_JOB_ID) -> int:
        """Delete a named draft from all images in a dataset.

        Returns the number of images from which the draft was deleted.
        """
        info = self.get_dataset(dataset_name)
        if info is None:
            return 0

        # Get all images that have this draft name
        image_rows = self._repo.search_by_draft_name(dataset_name, draft_name)
        deleted = 0
        for image_id, image_path_str in image_rows:
            image_path = Path(image_path_str)
            if not image_path.exists():
                continue
            dataset_image = DatasetImage(path=str(image_path))
            draft_path = dataset_image.draft_path(draft_name)
            self._watcher.expect_file_change(dataset_name, str(draft_path), source=source)
            if dataset_image.delete_draft(draft_name):
                deleted += 1
                self.refresh_image_index(dataset_name, image_id)

        if deleted:
            self._logger.info("Deleted draft '%s' from %d image(s) [dataset=%s]", draft_name, deleted, dataset_name)
        return deleted

    def get_caption(self, dataset_name: str, image_id: int) -> dict[str, Any] | None:
        """Read the caption text and TOML extras for an image.

        Returns dict with keys: caption, extras, drafts — or None if image not found.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))
        caption = dataset_image.read_caption()

        extras_raw: str = ""
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras_raw = f.read()
                    f.seek(0)
                    extras = load_toml(extras_raw)
            except Exception:
                pass

        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        return {"caption": caption, "extras": extras, "extras_raw": extras_raw, "drafts": drafts}

    def preview_prompt(self, dataset_name: str, image_id: int, template: str) -> dict[str, Any] | None:
        """Render the system and user prompts for an image using a Jinja2 template.

        Returns dict with keys: system_prompt, user_prompt, template_context — or None if not found.
        """
        from yadc.core.captioner import PromptRenderer

        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))

        # Load extras from TOML sidecar
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = load_toml(f.read())
            except Exception:
                pass

        # Apply extras as additional fields on the DatasetImage
        if extras:
            dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})

        # Read current caption from .txt file
        dataset_image.caption = dataset_image.read_caption()

        # Load drafts
        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        # Build template context (same as what PromptRenderer.render uses)
        template_context = dataset_image.model_dump()
        if drafts:
            template_context["drafts"] = drafts

        renderer = PromptRenderer(prompt_template=template)
        system_prompt, user_prompt = renderer.render(dataset_image, drafts=drafts or None)

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "template_context": template_context,
            "template_context_toml": tomlkit.dumps(template_context),
        }

    def get_history(self, dataset_name: str, image_id: int, limit: int = 3) -> list[HistoryEntry] | None:
        """Read history entries for an image, most recent first.

        Returns None if image not found, empty list if no history.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))
        all_entries = dataset_image.read_history()

        # Take the last `limit` entries and reverse (most recent first)
        recent = all_entries[-limit:] if len(all_entries) > limit else all_entries
        recent.reverse()

        result: list[HistoryEntry] = []
        for i, entry in enumerate(recent):
            extras = toml_to_plain(dict(entry.__pydantic_extra__ or {}))
            he = HistoryEntry(index=len(recent) - 1 - i, caption=entry.caption, extras=extras)
            result.append(he)

        # Re-index using absolute positions from the end of the full list
        total = len(all_entries)
        for i, entry in enumerate(result):
            entry.index = total - 1 - i

        return result

    def restore_history(self, dataset_name: str, image_id: int, history_index: int, *, source: str = SELF_JOB_ID) -> bool:
        """Restore caption + extras from a history entry.

        Saves the current state to history before restoring.
        Returns True on success.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.caption_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current extras into the model so save_history captures them
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = load_toml(f.read())
                if extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})
            except Exception:
                pass

        # Read current caption
        dataset_image.caption = dataset_image.read_caption()

        # Save current state to history before restoring
        dataset_image.save_history()

        # Read all history and find the target entry
        all_entries = dataset_image.read_history()
        if history_index < 0 or history_index >= len(all_entries):
            return False

        target = all_entries[history_index]

        # Restore caption
        dataset_image.update_caption(target.caption)

        # Restore extras
        target_extras = dict(target.__pydantic_extra__ or {})
        if target_extras:
            dataset_image.toml_path.write_text(tomlkit.dumps(target_extras))
            self._update_image_index(image_id, has_toml=True)
        else:
            # Clear extras if the history entry had none
            if dataset_image.toml_path.exists():
                dataset_image.toml_path.write_text("")
            self._update_image_index(image_id, has_toml=False)

        self._update_image_index(image_id, has_caption=True)
        return True

    def delete_history(self, dataset_name: str, image_id: int, entry_hash: str, *, source: str = SELF_JOB_ID) -> bool:
        """Delete a single history entry for an image by content hash.

        Returns True on success, False if image or entry not found.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file change so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)

        # Find the entry matching the hash
        all_entries = dataset_image.read_history()
        for i, entry in enumerate(all_entries):
            extras = toml_to_plain(dict(entry.__pydantic_extra__ or {}))
            he = HistoryEntry(index=i, caption=entry.caption, extras=extras)
            if he.hash == entry_hash:
                return dataset_image.delete_history_entry(i)

        return False

    def write_draft(self, dataset_name: str, image_id: int, draft_name: str, content: str, *, source: str = SELF_JOB_ID) -> bool:
        """Write or overwrite a named draft for an image.

        Returns True on success, False if image not found.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        self._watcher.expect_file_change(dataset_name, str(dataset_image.draft_path(draft_name)), source=source)

        dataset_image.write_draft(draft_name, content)
        return True

    def delete_draft(self, dataset_name: str, image_id: int, draft_name: str, *, source: str = SELF_JOB_ID) -> bool:
        """Delete a named draft for an image.

        Returns True on success, False if image not found.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file change so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.draft_path(draft_name)), source=source)

        deleted = dataset_image.delete_draft(draft_name)
        if deleted:
            # Refresh the index to update draft_names
            self.refresh_image_index(dataset_name, image_id)
        return deleted

    def update_caption(self, dataset_name: str, image_id: int, caption: str, *, source: str = SELF_JOB_ID) -> bool:
        """Update the caption file for an image. Saves history first. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.caption_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current state and save to history before overwriting
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = load_toml(f.read())
                if extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})
            except Exception:
                pass
        dataset_image.caption = dataset_image.read_caption()
        dataset_image.save_history()

        # Now apply the update
        dataset_image = DatasetImage(path=str(image_path))
        dataset_image.update_caption(caption)

        # Update the index
        self._update_image_index(image_id, has_caption=True)
        return True

    def update_extras(self, dataset_name: str, image_id: int, extras_raw: str, *, source: str = SELF_JOB_ID) -> bool:
        """Update the TOML extras sidecar for an image. Saves history first. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        # Validate TOML before writing
        try:
            load_toml(extras_raw)
        except Exception as e:
            raise ValueError(f"Invalid TOML: {e}") from e

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current state and save to history before overwriting
        current_extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    current_extras = load_toml(f.read())
                if current_extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **current_extras})
            except Exception:
                pass
        dataset_image.caption = dataset_image.read_caption()
        dataset_image.save_history()

        # Now write the new extras
        dataset_image.toml_path.write_text(extras_raw)

        self._update_image_index(image_id, has_toml=bool(extras_raw.strip()))
        return True

    # --- Dataset registration ---

    def import_dataset(self, name: str, toml_path: str) -> DatasetInfo:
        """Import an existing TOML config as a named dataset.

        Resolves any relative paths in ``[[dataset]]`` entries to absolute
        and writes them back to the original TOML file. The dataset is
        registered with the original path (no local copy is kept).
        """
        source = Path(toml_path).resolve()
        if not source.is_file():
            raise ValueError(f"Not a file: {source}")

        # Load and resolve relative paths
        raw = self._loader.load_raw_config(source)
        raw = self._loader.resolve_relative_paths(raw, source.parent)

        # Write resolved paths back to the original file
        with open(source, "w") as f:
            tomlkit.dump(raw, f)

        return self.register(name, str(source), source="import")

    def create_dataset(self, name: str, image_paths: list[str]) -> DatasetInfo:
        """Create a new dataset with the given image directories.

        Writes a minimal TOML to ``STATE_PATH/datasets/<name>/config.toml``
        with one ``[[dataset]]`` entry per path.
        """
        if not image_paths:
            raise ValueError("At least one image path is required")

        resolved_paths: list[str] = []
        for p in image_paths:
            rp = Path(p).resolve()
            if not rp.is_dir():
                raise ValueError(f"Not a directory: {rp}")
            resolved_paths.append(str(rp))

        # Build a minimal v2 TOML
        raw: dict[str, Any] = {"dataset": []}
        for ip in resolved_paths:
            raw["dataset"].append({"path": ip})

        dest_dir = DATASETS_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _dataset_config_path(name)
        with open(dest, "w") as f:
            tomlkit.dump(raw, f)

        return self.register(name, str(dest), source="create")

    def rescan_dataset(self, name: str, *, source: str = "") -> bool:
        """Force a rescan of a dataset by name. Returns True if dataset was found."""
        row = self._repo.get_dataset_row(name)
        if row is None:
            return False
        dataset_id, config_path = row
        changed = self._scanner.scan_disk(dataset_id, config_path)

        # Re-register filesystem watcher with current paths (picks up path changes)
        if config_path:
            config_path_obj = Path(config_path)
            config = self._loader.load_config(config_path_obj)
            paths = self._loader.get_dataset_paths(config, config_path_obj)
            self._watcher.watch_dataset(name, paths)

        if changed:
            job_id = source or None
            self._event_dispatcher.dispatch(DatasetChangedEvent(dataset_name=name, job_id=job_id))

        return True

    def unregister_dataset(self, name: str) -> bool:
        """Remove a dataset, its images from the index, and its state dir."""
        self._watcher.unwatch_dataset(name)
        found = self._repo.delete_dataset(name)

        if found:
            state_dir = DATASETS_DIR / name
            if state_dir.is_dir():
                shutil.rmtree(state_dir, ignore_errors=True)

        return found

    def register(self, name: str, config_path: str, *, source: str = "import") -> DatasetInfo:
        """Upsert a dataset record and scan its images."""
        config = self._loader.load_config(Path(config_path))

        # Walk the disk outside the transaction: file I/O is unrelated to
        # the DB and shouldn't hold a transaction open.
        disk_images = self._scanner.read_disk(config_path, config=config)

        with self._db.transaction():
            dataset_id = self._repo.upsert_dataset(name, config_path, source)
            existing_by_path = self._repo.list_image_paths(dataset_id)
            for path, meta in disk_images.items():
                self._repo.upsert_image(
                    dataset_id=dataset_id,
                    path=path,
                    file_name=meta["file_name"],
                    has_caption=meta["has_caption"],
                    has_toml=meta["has_toml"],
                    width=meta["width"],
                    height=meta["height"],
                    draft_names=meta["draft_names"],
                    last_modified_t=meta["last_modified_t"],
                )
            for path, img_id in existing_by_path.items():
                if path not in disk_images:
                    self._repo.delete_image(img_id)
            self._repo.update_dataset_stats(dataset_id, len(disk_images))

        # Start watching the dataset's directories
        paths = self._loader.get_dataset_paths(config, Path(config_path))
        if paths:
            self._watcher.watch_dataset(name, paths)

        result = self.get_dataset(name)
        assert result is not None
        return result

    def _watch_existing_datasets(self) -> None:
        """Watch all existing datasets' directories. Called during startup."""
        for name, config_path in self._repo.list_all_for_watcher():
            if not config_path:
                continue
            config_path_obj = Path(config_path)
            config = self._loader.load_config(config_path_obj)
            paths = self._loader.get_dataset_paths(config, config_path_obj)
            if paths:
                self._watcher.watch_dataset(name, paths)
                self._logger.debug("Watching existing dataset. [dataset=%s, paths=%d]", name, len(paths))

    def _refresh_stale_datasets(self, max_age_seconds: float | None = None) -> None:
        """Rescan datasets that haven't been scanned recently.

        Serialized via ``self._refresh_lock`` so the background thread and
        a future manual refresh (see dataset-config-ux-plan todo) can't
        race for the SQLite write lock. Callers block until the in-flight
        refresh finishes; the cost is one disk walk, not a long wait.

        ``max_age_seconds`` defaults to the configured refresh interval
        when called by the background job, but tests pass a smaller value
        to make freshly-registered datasets immediately eligible.
        """
        if max_age_seconds is None:
            max_age_seconds = self._configuration.dataset_refresh_interval_seconds
        with self._refresh_lock:
            cutoff = time.time() - max_age_seconds
            stale = self._repo.list_stale(cutoff)

            for dataset_id, name, config_path in stale:
                try:
                    changed = self._scanner.scan_disk(dataset_id, config_path)
                except Exception as e:
                    self._logger.warning("Failed to refresh dataset id=%d: %s", dataset_id, e)
                    continue

                if changed:
                    self._event_dispatcher.dispatch(DatasetChangedEvent(dataset_name=name))

                # Re-register filesystem watcher so path changes are tracked.
                # Done outside the scan transaction because watcher IO is unrelated.
                if not config_path:
                    continue
                config_path_obj = Path(config_path)
                config = self._loader.load_config(config_path_obj)
                paths = self._loader.get_dataset_paths(config, config_path_obj)
                if paths:
                    self._watcher.watch_dataset(name, paths)

    def _update_image_index(self, image_id: int, *, has_caption: bool | None = None, has_toml: bool | None = None) -> None:
        """Update specific fields on an indexed image."""
        self._repo.update_image_flags(image_id, has_caption=has_caption, has_toml=has_toml)

    def refresh_image_index(self, dataset_name: str, image_id: int) -> None:
        """Re-read a single image's disk state and update its index row.

        Used by the API endpoints (caption/extras/draft/history
        mutations) to keep the index in sync without triggering a
        watcher-driven full scan. The watcher event for the
        originating change will be tagged with the job_id of the
        caller so the SSE listener suppresses it.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return

        image_path = Path(info.path)
        if not image_path.exists():
            return

        # Reuse the scanner's per-image meta extraction. We only care
        # about has_caption / has_toml / draft_names — the rest is
        # already in the index and doesn't change here.
        meta = self._scanner.scan_image_meta(image_path)
        if meta is None:
            return

        self._repo.refresh_image_disk_state(
            image_id,
            has_caption=meta["has_caption"],
            has_toml=meta["has_toml"],
            draft_names=meta["draft_names"],
        )

    @event_handler(DatasetChangedEvent)
    def _on_dataset_changed(self, event: DatasetChangedEvent) -> None:
        """Auto-update the index when the watcher detects filesystem changes.

        Three branches:

        - **Self-originated** (``event.job_id`` set): the API endpoint or
          captioning job that triggered the change is already keeping the
          index in sync via ``refresh_image_index`` / direct row writes,
          so we skip the update entirely. Frontend suppression is handled
          by the SSE listener on the client side.
        - **Watcher event with paths**: do a targeted update via
          :meth:`DatasetScanner.scan_targeted` — only the rows for the
          affected images are touched, so a single-file external change
          is a constant-time update regardless of dataset size.
        - **Watcher event without paths** (legacy/manual): fall back to
          a full :meth:`DatasetScanner.scan_disk` for backward
          compatibility.
        """
        if event.job_id:
            self._logger.debug(
                "Skipping auto-update for self-originated change. [dataset=%s, job_id=%s]",
                event.dataset_name,
                event.job_id,
            )
            return

        row = self._repo.get_dataset_row(event.dataset_name)
        if row is None:
            self._logger.debug("Dataset not found, skipping auto-update. [dataset=%s]", event.dataset_name)
            return
        dataset_id, config_path = row

        if event.changed_paths:
            self._logger.debug(
                "Targeted auto-update from watcher. [dataset=%s, changed_paths=%d]",
                event.dataset_name,
                len(event.changed_paths),
            )
            changed = self._scanner.scan_targeted(dataset_id, event.changed_paths)
        else:
            self._logger.debug(
                "Full auto-rescan from watcher (no changed_paths). [dataset=%s]",
                event.dataset_name,
            )
            changed = self._scanner.scan_disk(dataset_id, config_path)

        if not changed:
            return

        # Re-register filesystem watcher with current paths (picks up path changes)
        if config_path:
            config_path_obj = Path(config_path)
            config = self._loader.load_config(config_path_obj)
            paths = self._loader.get_dataset_paths(config, config_path_obj)
            self._watcher.watch_dataset(event.dataset_name, paths)
