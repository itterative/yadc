"""Dataset upload service — handles file uploads for dataset creation.

Validates, writes, and indexes uploaded image files (plus sidecars) into
a new dataset under ``STATE_PATH/datasets/<name>/``.

This module is the public orchestration layer. Pure validation helpers
live in :mod:`dataset_upload_validation`, and staging/conflict helpers
live in :mod:`dataset_upload_staging`.
"""

import shutil
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import tomlkit
from tomlkit.toml_document import TOMLDocument

from yadc.api.configuration import Configuration
from yadc.api.modules.dataset_watcher import SIDECAR_EXTENSIONS, DatasetWatcherService
from yadc.api.modules.job_scheduler import JobScheduler
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.api.services.dataset_upload_staging import (
    build_staged_groups,
    cleanup_staging_dirs,
    detect_conflicts,
    filter_orphan_sidecars,
    find_live_image_stems,
)
from yadc.api.services.dataset_upload_validation import (
    unique_path,
    validate_image_stream,
    validate_toml_stream,
)
from yadc.api.services.datasets import DATASETS_DIR, IMAGE_EXTENSIONS, DatasetService, _dataset_config_path
from yadc.api.services.managed_paths import MANAGED_FOLDERS_PREFIX, MANAGED_IMAGES_PREFIX
from yadc.utils.dict_utils import load_toml

# Extensions allowed for uploaded files (images + sidecars).
UPLOAD_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | SIDECAR_EXTENSIONS


@dataclass
class DatasetUploadResult:
    """Result of a dataset upload, including the created dataset and any warnings."""

    dataset: Any  # DatasetInfo — avoiding circular import; typed at usage site
    warnings: list[str] = field(default_factory=list)


@dataclass
class UploadProgressEvent:
    """Streaming progress event for dataset uploads."""

    phase: str  # "validating" | "writing" | "conflicts" | "complete" | "error"
    file: str = ""
    index: int = 0
    total: int = 0
    dataset: Any = None  # DatasetInfo on "complete"
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    staging_id: str = ""
    conflicts: list[dict[str, Any]] = field(default_factory=list)


class DatasetUploadService(Service):
    """Handles validation, writing, and registration of uploaded dataset files."""

    def __init__(
        self,
        datasets: DatasetService,
        watcher: DatasetWatcherService,
        configuration: Configuration,
        logging: LoggingFactory,
        job_scheduler: JobScheduler | None = None,
    ):
        self._datasets: DatasetService = datasets
        self._watcher: DatasetWatcherService = watcher
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)
        if job_scheduler is not None:
            job_scheduler.new_scheduled_job(3600, self._cleanup_staging_dirs)

    # --- Thin wrappers around module-level helpers (kept for test compat) ---

    def _cleanup_staging_dirs(self) -> None:
        """Remove stale staging directories older than 24 hours.

        Thin wrapper around :func:`dataset_upload_staging.cleanup_staging_dirs`
        — kept on the service for direct test invocation.
        """
        cleanup_staging_dirs(DATASETS_DIR, self._logger)

    def _validate_image(self, stream: BinaryIO, filename: str) -> bool:
        """Thin wrapper around :func:`validate_image_stream` for test patching."""
        return validate_image_stream(stream, filename, self._logger)

    def _validate_toml(self, stream: BinaryIO, _filename: str) -> bool:
        """Thin wrapper around :func:`validate_toml_stream` for test patching."""
        return validate_toml_stream(stream)

    async def create_dataset_from_upload(
        self,
        name: str,
        files: list[tuple[str, BinaryIO]],
        *,
        source: str = "",
    ) -> AsyncGenerator[UploadProgressEvent, None]:
        """Create a new dataset from uploaded image files.

        Root files (no directory component) are written flat to
        ``STATE_PATH/datasets/<name>/images/``. Files with exactly one directory
        component (e.g. ``train/cat.jpg``) are written to
        ``STATE_PATH/datasets/<name>/folders/``. Nested files (deeper than one
        directory level) are skipped with a warning. The generated config
        has one ``[[dataset]]`` entry for ``images/`` and one per
        top-level folder inside ``folders/``.

        Uploaded files are validated before writing:
        - Images are verified with PIL.
        - TOML sidecars are parsed to ensure valid syntax.
        - Orphan sidecars (no matching image with the same stem) are skipped.

        *source* is the originating client identifier (typically a frontend
        tab's ``"ui:<uuid>"``). It is registered with the dataset watcher
        before each file write so the resulting ``DatasetChangedEvent`` is
        tagged with the same source and can be suppressed by the originating
        tab. Defaults to an empty string (no client association) for
        callers that don't supply one.
        """
        if not name or not name.strip():
            raise ValueError("Dataset name is required")

        name = name.strip()

        if not files:
            raise ValueError("At least one file is required")

        base_dir = DATASETS_DIR / name
        images_dir = base_dir / MANAGED_IMAGES_PREFIX
        folders_dir = base_dir / MANAGED_FOLDERS_PREFIX
        images_dir.mkdir(parents=True, exist_ok=True)
        folders_dir.mkdir(parents=True, exist_ok=True)

        resolved_images_dir = images_dir.resolve()
        resolved_folders_dir = folders_dir.resolve()

        warnings: list[str] = []
        # (directory, stem) for every valid image — used for orphan sidecar checks
        image_stems: set[tuple[str, str]] = set()
        # Files to write after validation: (filename, stream, pure, dest_path)
        files_to_write: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []

        # --- Phase 1: validate and filter ---
        total_files = len(files)
        for i, (filename, stream) in enumerate(files):
            ext = Path(filename).suffix.lower()
            if ext not in UPLOAD_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {filename}")

            pure = PurePosixPath(filename.replace("\\", "/"))
            if not pure.name or pure.name in (".", ".."):
                raise ValueError(f"Invalid file path: {filename}")

            # Skip nested files (deeper than one directory level)
            if len(pure.parts) > 2:
                warnings.append(f"Skipped nested file (not supported): {filename}")
                yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                continue

            if len(pure.parts) == 1:
                dest_path = (images_dir / str(pure)).resolve()
                try:
                    dest_path.relative_to(resolved_images_dir)
                except ValueError:
                    raise ValueError(f"Invalid file path: {filename}")
            else:
                dest_path = (folders_dir / str(pure)).resolve()
                try:
                    dest_path.relative_to(resolved_folders_dir)
                except ValueError:
                    raise ValueError(f"Invalid file path: {filename}")

            # Validate content by type
            if ext in IMAGE_EXTENSIONS:
                if not self._validate_image(stream, filename):
                    warnings.append(f"Skipped invalid image: {filename}")
                    yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                    continue
                stream.seek(0)
                dir_key = pure.parts[0] if len(pure.parts) > 1 else ""
                image_stems.add((dir_key, pure.stem))
            elif ext in (".toml", ".toml~"):
                if not self._validate_toml(stream, filename):
                    warnings.append(f"Skipped invalid TOML: {filename}")
                    yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                    continue
            # txt, draft~, history~ — no content validation

            files_to_write.append((filename, stream, pure, dest_path))
            yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)

        # --- Phase 2: orphan sidecar check (shared with append) ---
        final_files = filter_orphan_sidecars(files_to_write, image_stems, warnings)

        if not final_files:
            shutil.rmtree(base_dir, ignore_errors=True)
            yield UploadProgressEvent(phase="error", message="No valid files to upload after validation")
            return

        # --- Phase 3: write files ---
        has_root_files = False
        folder_names: set[str] = set()

        total_writing = len(final_files)
        total_written = 0
        max_size = self._configuration.max_upload_size_bytes
        try:
            for i, (filename, stream, pure, dest_path) in enumerate(final_files):
                if len(pure.parts) == 1:
                    has_root_files = True
                else:
                    folder_names.add(pure.parts[0])

                # Register the upcoming write so the watcher tags the
                # resulting DatasetChangedEvent with our source (if any)
                # and the originating frontend tab can suppress it.
                if source:
                    self._watcher.expect_file_change(name, str(dest_path), source=source)

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    chunk = stream.read(8192)
                    while chunk:
                        total_written += len(chunk)
                        if max_size > 0 and total_written > max_size:
                            raise ValueError(f"Total upload size exceeds {max_size} bytes")
                        f.write(chunk)
                        chunk = stream.read(8192)

                yield UploadProgressEvent(phase="writing", file=filename, index=i + 1, total=total_writing)

            dataset_entries: list[dict[str, str]] = []
            if has_root_files:
                dataset_entries.append({"path": MANAGED_IMAGES_PREFIX})
            for folder_name in sorted(folder_names):
                dataset_entries.append({"path": f"{MANAGED_FOLDERS_PREFIX}/{folder_name}"})

            # Managed datasets use relative paths (e.g. "images", "folders/train")
            # for portability. Paths are resolved against config.toml's directory
            # at scan time.
            raw: dict[str, Any] = {"dataset": dataset_entries}

            dest = _dataset_config_path(name)
            with open(dest, "w") as f:
                tomlkit.dump(raw, f)

            dataset = self._datasets.register(name, str(dest), source="upload")
            yield UploadProgressEvent(phase="complete", dataset=dataset, warnings=warnings)
        except Exception as e:
            shutil.rmtree(base_dir, ignore_errors=True)
            yield UploadProgressEvent(phase="error", message=str(e))

    async def append_dataset_from_upload(
        self,
        name: str,
        files: list[tuple[str, BinaryIO]],
        *,
        source: str = "",
    ) -> AsyncGenerator[UploadProgressEvent, None]:
        """Stage uploaded files for appending to an existing managed dataset.

        Files are written to a staging directory first
        (``.staging/<staging_id>/images/`` and ``folders/``).
        After staging, conflicts with existing files are detected.
        If conflicts exist, a ``phase: "conflicts"`` event is emitted
        with the ``staging_id`` and conflict list. The caller must then
        call :meth:`commit_staged_upload` to resolve conflicts and
        complete the append. If no conflicts exist, the upload is
        committed automatically.

        *source* is the originating client identifier (typically a frontend
        tab's ``"ui:<uuid>"``). It is registered with the watcher before
        each staging write and is passed through to
        :meth:`commit_staged_upload` for the eventual move from staging
        to the live dataset directories.
        """
        info = self._datasets.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")

        if not files:
            raise ValueError("At least one file is required")

        staging_id = str(uuid.uuid4())
        base_dir = Path(info.config_path).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        images_dir = base_dir / MANAGED_IMAGES_PREFIX
        folders_dir = base_dir / MANAGED_FOLDERS_PREFIX
        staging_base = base_dir / ".staging" / staging_id
        staging_base.mkdir(parents=True, exist_ok=True)
        staging_images = staging_base / MANAGED_IMAGES_PREFIX
        staging_folders = staging_base / MANAGED_FOLDERS_PREFIX
        staging_images.mkdir(parents=True, exist_ok=True)
        staging_folders.mkdir(parents=True, exist_ok=True)

        resolved_staging_images = staging_images.resolve()
        resolved_staging_folders = staging_folders.resolve()

        warnings: list[str] = []
        image_stems: set[tuple[str, str]] = set()
        files_to_write: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []

        total_written = 0
        max_size = self._configuration.max_upload_size_bytes

        def _abort_staging() -> None:
            shutil.rmtree(staging_base, ignore_errors=True)

        # --- Phase 1: validate and filter ---
        total_files = len(files)
        for i, (filename, stream) in enumerate(files):
            ext = Path(filename).suffix.lower()
            if ext not in UPLOAD_EXTENSIONS:
                _abort_staging()
                raise ValueError(f"Unsupported file type: {filename}")

            pure = PurePosixPath(filename.replace("\\", "/"))
            if not pure.name or pure.name in (".", ".."):
                _abort_staging()
                raise ValueError(f"Invalid file path: {filename}")

            if len(pure.parts) > 2:
                warnings.append(f"Skipped nested file (not supported): {filename}")
                yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                continue

            if len(pure.parts) == 1:
                dest_path = (staging_images / str(pure)).resolve()
                try:
                    dest_path.relative_to(resolved_staging_images)
                except ValueError:
                    _abort_staging()
                    raise ValueError(f"Invalid file path: {filename}")
            else:
                dest_path = (staging_folders / str(pure)).resolve()
                try:
                    dest_path.relative_to(resolved_staging_folders)
                except ValueError:
                    _abort_staging()
                    raise ValueError(f"Invalid file path: {filename}")

            if ext in IMAGE_EXTENSIONS:
                if not self._validate_image(stream, filename):
                    warnings.append(f"Skipped invalid image: {filename}")
                    yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                    continue
                stream.seek(0)
                dir_key = pure.parts[0] if len(pure.parts) > 1 else ""
                image_stems.add((dir_key, pure.stem))
            elif ext in (".toml", ".toml~"):
                if not self._validate_toml(stream, filename):
                    warnings.append(f"Skipped invalid TOML: {filename}")
                    yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)
                    continue

            files_to_write.append((filename, stream, pure, dest_path))
            yield UploadProgressEvent(phase="validating", file=filename, index=i + 1, total=total_files)

        # --- Phase 2: build live image stems + orphan sidecar check (shared with create) ---
        live_image_stems = find_live_image_stems(images_dir, folders_dir, IMAGE_EXTENSIONS)
        new_folder_names: set[str] = set()
        final_files = filter_orphan_sidecars(
            files_to_write,
            image_stems | live_image_stems,
            warnings,
            new_folder_names,
        )

        if not final_files:
            _abort_staging()
            yield UploadProgressEvent(phase="error", message="No valid files to upload after validation")
            return

        # --- Phase 3: write files to staging ---
        total_writing = len(final_files)
        for i, (filename, stream, pure, dest_path) in enumerate(final_files):
            # Register the staging write so the watcher treats subsequent
            # inotify events on this path as expected (and tags them with
            # our source). Note: staging files are inside .staging/ and
            # are not under the dataset's watched images/folders dirs, so
            # this mainly matters when they are later moved into place by
            # commit_staged_upload (which re-registers with the same source).
            if source:
                self._watcher.expect_file_change(name, str(dest_path), source=source)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                chunk = stream.read(8192)
                while chunk:
                    total_written += len(chunk)
                    if max_size > 0 and total_written > max_size:
                        _abort_staging()
                        raise ValueError(f"Total upload size exceeds {max_size} bytes")
                    f.write(chunk)
                    chunk = stream.read(8192)

            yield UploadProgressEvent(phase="writing", file=filename, index=i + 1, total=total_writing)

        # --- Phase 4: detect conflicts (grouped by stem) ---
        staged_groups = build_staged_groups(staging_base, IMAGE_EXTENSIONS)
        conflicts = detect_conflicts(staged_groups, staging_base, images_dir, folders_dir)

        if conflicts:
            yield UploadProgressEvent(
                phase="conflicts",
                staging_id=staging_id,
                conflicts=conflicts,
                warnings=warnings,
            )
            return

        # No conflicts — commit immediately. Pass the source through so
        # the resulting DatasetChangedEvent is tagged for the originating tab.
        async for event in self.commit_staged_upload(name, staging_id, {}, warnings, source=source):
            yield event

    async def commit_staged_upload(
        self,
        name: str,
        staging_id: str,
        resolutions: dict[str, str],
        warnings: list[str] | None = None,
        *,
        source: str = "",
    ) -> AsyncGenerator[UploadProgressEvent, None]:
        """Commit a staged upload to the live dataset directories.

        ``resolutions`` maps relative file paths to actions:
        ``"overwrite"`` (default), ``"skip"``, or ``"keep_both"``.

        Resolutions are applied per-stem group: if an image is present,
        its filename is the resolution key; for sidecar-only groups,
        the first sidecar's filename is the key. The action applies to
        all files in the group (image + all sidecars).

        *source* is the originating client identifier (typically a frontend
        tab's ``"ui:<uuid>"``). It is registered with the watcher before
        each move from staging into the live directories and is passed
        to the final ``rescan_dataset`` call so the resulting
        ``DatasetChangedEvent`` can be suppressed by the originating tab.
        """
        info = self._datasets.get_dataset(name)
        if info is None or not info.config_path:
            raise ValueError(f"Dataset '{name}' not found or has no config path")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")

        base_dir = Path(info.config_path).parent
        staging_base = base_dir / ".staging" / staging_id
        if not staging_base.exists():
            raise ValueError(f"Staging upload '{staging_id}' not found")

        images_dir = base_dir / MANAGED_IMAGES_PREFIX
        folders_dir = base_dir / MANAGED_FOLDERS_PREFIX

        # Build groups from staged files
        staged_groups = build_staged_groups(staging_base, IMAGE_EXTENSIONS)

        # Determine resolution key for each group
        group_resolutions: dict[tuple[str, str], str] = {}
        for group_key, group in staged_groups.items():
            if group["image"] is not None:
                rep_pure = PurePosixPath(group["image"].name)
            elif group["sidecars"]:
                rep_pure = PurePosixPath(group["sidecars"][0].name)
            else:
                continue
            # For folder files, the resolution key includes the folder prefix
            if group_key[0]:
                rel_name = f"{group_key[0]}/{rep_pure.name}"
            else:
                rel_name = str(rep_pure)
            group_resolutions[group_key] = resolutions.get(rel_name, "overwrite")

        has_root_files = False
        new_folder_names: set[str] = set()

        # Move files group by group
        for group_key, group in staged_groups.items():
            resolution = group_resolutions.get(group_key, "overwrite")
            if resolution == "skip":
                continue

            dir_key = group_key[0]

            # Move image
            if group["image"] is not None:
                src_path = group["image"]
                if dir_key:
                    dest_path = folders_dir / dir_key / src_path.name
                    new_folder_names.add(dir_key)
                else:
                    dest_path = images_dir / src_path.name
                    has_root_files = True

                if resolution == "keep_both":
                    dest_path = unique_path(dest_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Register the upcoming move so the watcher tags the
                # resulting DatasetChangedEvent with our source (if any)
                # and the originating frontend tab can suppress it.
                if source:
                    self._watcher.expect_file_change(name, str(dest_path), source=source)
                shutil.move(str(src_path), str(dest_path))

            # Move sidecars
            for src_path in group["sidecars"]:
                if dir_key:
                    dest_path = folders_dir / dir_key / src_path.name
                    new_folder_names.add(dir_key)
                else:
                    dest_path = images_dir / src_path.name
                    has_root_files = True

                if resolution == "keep_both":
                    dest_path = unique_path(dest_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if source:
                    self._watcher.expect_file_change(name, str(dest_path), source=source)
                shutil.move(str(src_path), str(dest_path))

        # Update config TOML with new folder entries
        config_path = Path(info.config_path)
        if new_folder_names or has_root_files:
            try:
                doc = cast(TOMLDocument, load_toml(config_path.read_text(), plain=False))
            except Exception:
                doc = tomlkit.document()

            if "dataset" not in doc or not isinstance(doc["dataset"], list):
                doc["dataset"] = tomlkit.aot()

            existing_paths = {entry.get("path", "") for entry in cast(list[dict[str, Any]], doc["dataset"])}

            root_rel = MANAGED_IMAGES_PREFIX
            root_abs = str(images_dir.resolve())
            if has_root_files and root_rel not in existing_paths and root_abs not in existing_paths:
                entry = tomlkit.table()
                entry["path"] = root_rel
                doc["dataset"].append(entry)

            for folder_name in sorted(new_folder_names):
                folder_rel = f"{MANAGED_FOLDERS_PREFIX}/{folder_name}"
                folder_abs = str((folders_dir / folder_name).resolve())
                if folder_rel not in existing_paths and folder_abs not in existing_paths:
                    entry = tomlkit.table()
                    entry["path"] = folder_rel
                    doc["dataset"].append(entry)

            config_path.write_text(tomlkit.dumps(doc))

        # Clean up staging
        shutil.rmtree(staging_base, ignore_errors=True)

        # Rescan. Pass the source through so the resulting
        # DatasetChangedEvent (if anything changed) is tagged for the
        # originating tab. Other clients still get a refresh notification
        # because the rescan is a real change to the dataset.
        self._datasets.rescan_dataset(name, source=source)
        updated_info = self._datasets.get_dataset(name)

        yield UploadProgressEvent(
            phase="complete",
            dataset=updated_info,
            warnings=warnings or [],
        )
