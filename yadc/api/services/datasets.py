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

import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import tomlkit

from yadc.cmd.app import STATE_PATH
from yadc.core.config import Config, ConfigDatasetEntry
from yadc.core.dataset import DatasetImage
from yadc.utils.dict_utils import load_toml, toml_to_plain

from ..configuration import Configuration
from ..constants import IMAGE_EXTENSIONS
from ..events import DatasetChangedEvent
from ..modules.dataset_watcher import SELF_JOB_ID, DatasetWatcherService
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.event_dispatcher import EventDispatcher, event_handler
from ..modules.job_scheduler import JobScheduler
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .dataset_loader import DatasetLoader
from .dataset_scanner import DatasetScanner
from .managed_paths import (
    MANAGED_FOLDERS_PREFIX,
    MANAGED_IMAGES_PREFIX,
    managed_folders_dir,
    managed_images_dir,
)

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


class HardlinkNotSupportedError(OSError):
    """Raised by :meth:`DatasetService.duplicate_managed_dataset` when
    ``mode="hardlink"`` is requested but the target filesystem doesn't
    support hardlinks.

    Subclass of :class:`OSError` so it integrates with existing
    exception handling. The duplicate endpoint maps this to HTTP
    409 so the frontend can prompt the user to retry with
    ``mode="copy"``.
    """


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

        Returns dict with keys: caption, extras, extras_raw, drafts — or None if image not found.

        ``extras_raw`` is the raw content of the per-image TOML sidecar (used by
        the Extras tab editor). ``extras`` is the parsed version merged with the
        matching ``[[dataset]]`` entry's dataset-level extras — per-image keys
        win — matching the template context the captioning runner sees.
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

        # Merge dataset-level extras (defaults) into the parsed view. Keep
        # ``extras_raw`` as the per-image TOML only — the Extras tab editor
        # edits the per-image file, not the dataset config.
        merged_extras = self._merge_extras(dataset_name, image_path, extras)

        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        return {"caption": caption, "extras": merged_extras, "extras_raw": extras_raw, "drafts": drafts}

    def _find_entry_for_image(
        self,
        config: Config,
        image_path: Path,
        config_path: Path,
    ) -> ConfigDatasetEntry | None:
        """Find the ``[[dataset]]`` entry that contains the given image.

        Path-based entries match when the image's absolute path is inside the
        entry's resolved directory. Inline-image entries (with no ``path``
        set) match when the image's absolute path equals one of the entry's
        image paths.

        When multiple path-based entries contain the image (e.g.
        ``example/`` and ``example/foo/``), the most specific one wins — the
        deepest ancestor — mirroring the resolver's non-recursive ``iterdir``
        walk, which would only have the inner entry pick the image up. An
        inline-image match always wins over a path-based match (inline
        declarations are explicit; path-based are implicit). Returns ``None``
        when no entry matches.
        """
        try:
            image_abs = image_path.resolve()
        except OSError:
            return None

        # Pass 1: inline images always win — an explicit declaration beats
        # any path-based claim on the same image.
        for entry in config.dataset:
            for inline in entry.images:
                try:
                    if Path(inline.path).resolve() == image_abs:
                        return entry
                except (OSError, ValueError):
                    continue

        # Pass 2: path-based — collect all candidates, return the deepest
        # ancestor. Use ``is_dir()`` to skip file paths (the resolver does
        # the same — non-directory paths produce no images).
        best: ConfigDatasetEntry | None = None
        best_depth = -1
        for entry in config.dataset:
            if not entry.path:
                continue
            entry_path = Path(entry.path)
            if not entry_path.is_absolute():
                entry_path = config_path.parent / entry_path
            try:
                entry_path = entry_path.resolve()
            except OSError:
                continue
            if not entry_path.is_dir():
                continue
            try:
                if not image_abs.is_relative_to(entry_path):
                    continue
            except (OSError, ValueError):
                continue
            depth = len(entry_path.parts)
            if depth > best_depth:
                best = entry
                best_depth = depth

        return best

    def _get_dataset_extras_for_image(
        self,
        dataset_name: str,
        image_path: Path,
    ) -> dict[str, Any]:
        """Return the dataset-level extras for the entry that contains ``image_path``.

        Falls back to an empty dict on any failure (missing config, parse
        error, no matching entry) so the caller can merge unconditionally.
        """
        try:
            info = self.get_dataset(dataset_name)
        except Exception:
            return {}
        if info is None or not info.config_path:
            return {}

        config_path = Path(info.config_path)
        if not config_path.exists():
            return {}

        try:
            config = self._loader.load_config(config_path)
        except Exception:
            return {}
        if config is None:
            return {}

        entry = self._find_entry_for_image(config, image_path, config_path)
        if entry is None or not entry.extras:
            return {}
        return dict(entry.extras)

    def _merge_extras(
        self,
        dataset_name: str,
        image_path: Path,
        per_image_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge dataset-level extras (defaults) with per-image extras (overrides).

        Per-image keys win over dataset-level keys, matching the merge
        semantics of :func:`yadc.core.dataset_resolver.resolve_dataset`.
        Returns ``per_image_extras`` unchanged when no dataset-level extras
        apply.
        """
        dataset_extras = self._get_dataset_extras_for_image(dataset_name, image_path)
        if not dataset_extras:
            return per_image_extras
        merged = dict(dataset_extras)
        merged.update(per_image_extras)
        return merged

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

        # Merge in dataset-level extras (per-image wins) so the preview matches
        # what the actual captioning runner sees.
        extras = self._merge_extras(dataset_name, image_path, extras)

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

    def _probe_hardlink_in_dir(self, target_dir: Path) -> bool:
        """Probe whether ``os.link`` succeeds in ``target_dir``.

        Creates a small temp file in ``target_dir``, attempts to
        hardlink it, cleans up. Returns True on success, False on
        any ``OSError`` or ``PermissionError``.

        Used by :meth:`duplicate_managed_dataset` to fail fast on
        filesystems that don't support hardlinks (exFAT, FAT32,
        some network FS). The probe runs against the actual
        destination dir (``STATE_PATH / datasets / <new_name>``)
        so the result reflects the filesystem the real hardlinks
        will run on.
        """
        target_dir = Path(target_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            return False
        tmp_path: Path | None = None
        link_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".yadc-probe-", dir=str(target_dir))
            try:
                os.write(fd, b"probe")
            finally:
                os.close(fd)
            tmp_path = Path(tmp_name)
            link_path = tmp_path.with_suffix(tmp_path.suffix + ".link")
            try:
                os.link(tmp_path, link_path)
                return True
            except (OSError, PermissionError):
                return False
        except (OSError, PermissionError):
            return False
        finally:
            if link_path is not None:
                link_path.unlink(missing_ok=True)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def duplicate_managed_dataset(
        self,
        name: str,
        new_name: str,
        *,
        mode: Literal["copy", "hardlink"] = "hardlink",
    ) -> DatasetInfo:
        """Duplicate a managed dataset to a new name under the state dir.

        Copies ``config.toml`` and every file under ``images/`` and
        ``folders/*`` (recursively) to a new managed dataset at
        ``DATASETS_DIR / new_name /``. ``.staging/`` is not part of
        the walk (it's a sibling of ``images/``/``folders/``, not
        underneath them) and is therefore naturally excluded.

        Per-file copy policy:

        - **Image bytes** (suffix in :data:`IMAGE_EXTENSIONS`):
          ``os.link`` if ``mode == "hardlink"``, else
          ``shutil.copy2``.
        - **Sidecars** (``.txt``, ``.toml``, ``.history~``,
          ``.draft~``) and **config**: always ``shutil.copy2``,
          regardless of mode.
        - **``.staging/``**: skipped (not walked).

        Default ``mode`` is ``"hardlink"``. The service honors the
        requested mode and never silently falls back. If
        ``mode == "hardlink"`` and the target filesystem doesn't
        support hardlinks, the service:

        1. Creates the new dataset dir and copies the config.
        2. Probes the actual destination dir (not the parent)
           with a small hardlink test.
        3. If the probe fails, removes the partial new dir and
           raises :class:`HardlinkNotSupportedError`. The
           controller maps this to HTTP 409 so the frontend can
           prompt the user to retry with ``mode="copy"``.

        On success, the new dataset is registered (indexed,
        watched, DB stats updated) via :meth:`register` and the
        resulting :class:`DatasetInfo` is returned.

        Note: per-tab suppression of ``DatasetChangedEvent`` is not
        implemented here. The new dataset's watcher doesn't exist
        during the copy, and the source's watcher won't see events
        on the destination path, so no events need to be suppressed
        in the common case. If we later add a ``?source=`` query
        param to the duplicate endpoint for originating-tab
        suppression, plumb it through to ``register`` and the
        watcher the same way the upload endpoints do.

        Raises:
            ValueError: if the source dataset doesn't exist, isn't
                managed (``source != "upload"``), has no config
                path, the new name is empty / same as the source /
                collides with an existing dataset, or the new state
                dir already exists on disk.
            HardlinkNotSupportedError: if ``mode == "hardlink"``
                and the target filesystem doesn't support
                hardlinks. The partial new dir is removed before
                raising.
            OSError: for any other I/O failure during the
                duplication. The partial new dir is removed before
                raising.
        """
        info = self.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")
        if not new_name:
            raise ValueError("new_name must not be empty")
        if new_name == name:
            raise ValueError(f"new_name must differ from source name: {new_name!r}")
        if self.get_dataset(new_name) is not None:
            raise ValueError(f"Dataset '{new_name}' already exists")

        src_config_path = Path(info.config_path)
        src_images_dir = managed_images_dir(src_config_path)
        src_folders_dir = managed_folders_dir(src_config_path)

        dest_dir = DATASETS_DIR / new_name
        dest_config_path = dest_dir / "config.toml"

        if dest_dir.exists():
            raise ValueError(f"Target directory already exists: {dest_dir}")

        try:
            # 1. Create the new dataset dir and copy the config TOML
            dest_dir.mkdir(parents=True, exist_ok=False)
            shutil.copy2(src_config_path, dest_config_path)

            # 2. If hardlink mode, probe the actual destination dir
            # so we fail fast on filesystems that don't support
            # hardlinks. The probe runs after the dir exists so it
            # tests the same filesystem the real hardlinks will
            # run on.
            if mode == "hardlink" and not self._probe_hardlink_in_dir(dest_dir):
                raise HardlinkNotSupportedError(f"Hardlinks are not supported on the target filesystem ({dest_dir})")

            # 3. Walk images/ and folders/* and copy or hardlink each file
            for source_dir, prefix in (
                (src_images_dir, MANAGED_IMAGES_PREFIX),
                (src_folders_dir, MANAGED_FOLDERS_PREFIX),
            ):
                if not source_dir.exists():
                    continue
                for src_file in source_dir.rglob("*"):
                    if not src_file.is_file():
                        continue
                    rel = src_file.relative_to(source_dir)
                    dest_file = dest_dir / prefix / rel
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    # Image bytes go through os.link in hardlink mode;
                    # sidecars (and config above) always copy2.
                    if mode == "hardlink" and src_file.suffix.lower() in IMAGE_EXTENSIONS:
                        os.link(src_file, dest_file)
                    else:
                        shutil.copy2(src_file, dest_file)
        except Exception:
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise

        # Register the new dataset (index, watch, db). The
        # originating-tab suppression pattern from upload/delete:
        # pass ``source`` through so the resulting
        # ``DatasetChangedEvent`` can be filtered.
        return self.register(new_name, str(dest_config_path), source="upload")

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
