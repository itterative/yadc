"""Dataset upload service — handles file uploads for dataset creation.

Validates, writes, and indexes uploaded image files (plus sidecars) into
a new dataset under ``STATE_PATH/datasets/<name>/``.
"""

import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import tomlkit
from PIL import Image

from yadc.api.configuration import Configuration
from yadc.api.modules.dataset_watcher import SIDECAR_EXTENSIONS
from yadc.api.modules.job_scheduler import JobScheduler
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.api.services.datasets import DATASETS_DIR, IMAGE_EXTENSIONS, DatasetService, _dataset_config_path

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


def _unique_path(path: Path) -> Path:
    """Return a unique path by appending a number before the extension."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


class DatasetUploadService(Service):
    """Handles validation, writing, and registration of uploaded dataset files."""

    def __init__(
        self,
        datasets: DatasetService,
        configuration: Configuration,
        logging: LoggingFactory,
        job_scheduler: JobScheduler | None = None,
    ):
        self._datasets: DatasetService = datasets
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)
        if job_scheduler is not None:
            job_scheduler.new_scheduled_job(3600, self._cleanup_staging_dirs)

    def _cleanup_staging_dirs(self) -> None:
        """Remove stale staging directories older than 24 hours."""
        now = time.time()
        max_age = 24 * 3600  # 24 hours
        if not DATASETS_DIR.exists():
            return
        for base_dir in DATASETS_DIR.iterdir():
            if not base_dir.is_dir():
                continue
            staging_dir = base_dir / ".staging"
            if not staging_dir.exists():
                continue
            for subdir in staging_dir.iterdir():
                try:
                    mtime = subdir.stat().st_mtime
                    if now - mtime > max_age:
                        shutil.rmtree(subdir, ignore_errors=True)
                        self._logger.debug("Cleaned up stale staging dir: %s", subdir)
                except Exception:
                    pass
            # Remove empty .staging dirs
            try:
                if not any(staging_dir.iterdir()):
                    staging_dir.rmdir()
            except Exception:
                pass

    def _validate_image(self, stream: BinaryIO, filename: str) -> bool:
        """Validate an image stream with PIL. Returns True if valid."""
        try:
            with Image.open(stream) as img:
                img.verify()
        except Exception:
            # verify() is overly strict with some valid images;
            # fall back to the slower but more reliable load() check.
            stream.seek(0)
            try:
                with Image.open(stream) as img:
                    img.load()
            except Exception as e:
                self._logger.debug("Image validation failed: %s — %s", filename, e)
                return False
        return True

    def _validate_toml(self, stream: BinaryIO, filename: str) -> bool:
        """Validate a TOML stream. Returns True if valid."""
        content = stream.read()
        stream.seek(0)
        try:
            tomlkit.loads(content.decode("utf-8"))
            return True
        except Exception:
            return False

    async def create_dataset_from_upload(self, name: str, files: list[tuple[str, BinaryIO]]) -> AsyncGenerator[UploadProgressEvent, None]:
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
        """
        if not name or not name.strip():
            raise ValueError("Dataset name is required")

        name = name.strip()

        if not files:
            raise ValueError("At least one file is required")

        base_dir = DATASETS_DIR / name
        images_dir = base_dir / "images"
        folders_dir = base_dir / "folders"
        images_dir.mkdir(parents=True, exist_ok=True)
        folders_dir.mkdir(parents=True, exist_ok=True)

        resolved_images_dir = images_dir.resolve()
        resolved_folders_dir = folders_dir.resolve()

        warnings: list[str] = []
        # (directory, stem) for every valid image — used for orphan sidecar checks
        image_stems: set[tuple[str, str]] = set()
        # Files to write after validation: (filename, stream, pure, dest_path)
        files_to_write: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []

        total_written = 0
        max_size = self._configuration.max_upload_size_bytes

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

        # --- Phase 2: orphan sidecar check ---
        final_files: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []
        for filename, stream, pure, dest_path in files_to_write:
            ext = pure.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                final_files.append((filename, stream, pure, dest_path))
                continue

            dir_key = pure.parts[0] if len(pure.parts) > 1 else ""

            if ext == ".draft~":
                # Draft format: IMAGE_STEM.DRAFT_NAME.draft~
                # Try each known image stem as a prefix.
                matched = False
                for known_dir, known_stem in image_stems:
                    if known_dir == dir_key and pure.name.startswith(known_stem + ".") and pure.name.endswith(".draft~"):
                        matched = True
                        break
                if not matched:
                    warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
                    continue
            elif (dir_key, pure.stem) not in image_stems:
                warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
                continue

            final_files.append((filename, stream, pure, dest_path))

        if not final_files:
            shutil.rmtree(base_dir, ignore_errors=True)
            yield UploadProgressEvent(phase="error", message="No valid files to upload after validation")
            return

        # --- Phase 3: write files ---
        has_root_files = False
        folder_names: set[str] = set()

        total_writing = len(final_files)
        try:
            for i, (filename, stream, pure, dest_path) in enumerate(final_files):
                if len(pure.parts) == 1:
                    has_root_files = True
                else:
                    folder_names.add(pure.parts[0])

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
                dataset_entries.append({"path": "images"})
            for folder_name in sorted(folder_names):
                dataset_entries.append({"path": f"folders/{folder_name}"})

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

    async def append_dataset_from_upload(self, name: str, files: list[tuple[str, BinaryIO]]) -> AsyncGenerator[UploadProgressEvent, None]:
        """Stage uploaded files for appending to an existing managed dataset.

        Files are written to a staging directory first
        (``.staging/<staging_id>/images/`` and ``folders/``).
        After staging, conflicts with existing files are detected.
        If conflicts exist, a ``phase: "conflicts"`` event is emitted
        with the ``staging_id`` and conflict list. The caller must then
        call :meth:`commit_staged_upload` to resolve conflicts and
        complete the append. If no conflicts exist, the upload is
        committed automatically.
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
        images_dir = base_dir / "images"
        folders_dir = base_dir / "folders"
        staging_base = base_dir / ".staging" / staging_id
        staging_base.mkdir(parents=True, exist_ok=True)
        staging_images = staging_base / "images"
        staging_folders = staging_base / "folders"
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

        # --- Phase 2: build live image stems + orphan sidecar check ---
        live_image_stems: set[tuple[str, str]] = set()
        if images_dir.exists():
            for f in images_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    live_image_stems.add(("", f.stem))
        if folders_dir.exists():
            for folder_path in folders_dir.iterdir():
                if not folder_path.is_dir():
                    continue
                for f in folder_path.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        live_image_stems.add((folder_path.name, f.stem))

        final_files: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []
        new_folder_names: set[str] = set()
        for filename, stream, pure, dest_path in files_to_write:
            ext = pure.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                final_files.append((filename, stream, pure, dest_path))
                if len(pure.parts) > 1:
                    new_folder_names.add(pure.parts[0])
                continue

            dir_key = pure.parts[0] if len(pure.parts) > 1 else ""

            if ext == ".draft~":
                matched = False
                for known_dir, known_stem in image_stems | live_image_stems:
                    if known_dir == dir_key and pure.name.startswith(known_stem + ".") and pure.name.endswith(".draft~"):
                        matched = True
                        break
                if not matched:
                    warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
                    continue
            elif (dir_key, pure.stem) not in image_stems and (dir_key, pure.stem) not in live_image_stems:
                warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
                continue

            final_files.append((filename, stream, pure, dest_path))
            if len(pure.parts) > 1:
                new_folder_names.add(pure.parts[0])

        if not final_files:
            _abort_staging()
            yield UploadProgressEvent(phase="error", message="No valid files to upload after validation")
            return

        # --- Phase 3: write files to staging ---
        total_writing = len(final_files)
        for i, (filename, stream, pure, dest_path) in enumerate(final_files):
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
        # Build groups: group_key -> { "image": Path|None, "sidecars": [Path] }
        from collections import defaultdict

        staged_groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"image": None, "sidecars": []})
        for _filename, _stream, pure, dest_path in final_files:
            dir_key = pure.parts[0] if len(pure.parts) > 1 else ""
            stem = pure.stem
            if pure.suffix.lower() in IMAGE_EXTENSIONS:
                staged_groups[(dir_key, stem)]["image"] = dest_path
            else:
                staged_groups[(dir_key, stem)]["sidecars"].append(dest_path)

        conflicts: list[dict[str, Any]] = []
        for (dir_key, stem), group in staged_groups.items():
            # Find any conflicting file in the group
            conflict_live_path: Path | None = None
            conflict_staged_path: Path | None = None

            if group["image"] is not None:
                pure = PurePosixPath(group["image"].relative_to(staging_base).as_posix().replace("images/", "").replace("folders/", ""))
                if len(pure.parts) == 1:
                    live_path = images_dir / pure.name
                else:
                    live_path = folders_dir / pure.parts[0] / pure.name
                if live_path.exists():
                    conflict_live_path = live_path
                    conflict_staged_path = group["image"]
            else:
                # Sidecar-only group — check if any sidecar conflicts
                for sidecar_path in group["sidecars"]:
                    pure = PurePosixPath(sidecar_path.relative_to(staging_base).as_posix().replace("images/", "").replace("folders/", ""))
                    if len(pure.parts) == 1:
                        live_path = images_dir / pure.name
                    else:
                        live_path = folders_dir / pure.parts[0] / pure.name
                    if live_path.exists():
                        conflict_live_path = live_path
                        conflict_staged_path = sidecar_path
                        break

            if conflict_live_path is not None and conflict_staged_path is not None:
                # Representative file: image if present, else first sidecar
                rep_path = group["image"] if group["image"] is not None else group["sidecars"][0]
                rep_pure = PurePosixPath(rep_path.relative_to(staging_base).as_posix().replace("images/", "").replace("folders/", ""))
                conflicts.append(
                    {
                        "file": str(rep_pure),
                        "existing_size": conflict_live_path.stat().st_size,
                        "new_size": conflict_staged_path.stat().st_size,
                    }
                )

        if conflicts:
            yield UploadProgressEvent(
                phase="conflicts",
                staging_id=staging_id,
                conflicts=conflicts,
                warnings=warnings,
            )
            return

        # No conflicts — commit immediately
        async for event in self.commit_staged_upload(name, staging_id, {}, warnings):
            yield event

    async def commit_staged_upload(
        self,
        name: str,
        staging_id: str,
        resolutions: dict[str, str],
        warnings: list[str] | None = None,
    ) -> AsyncGenerator[UploadProgressEvent, None]:
        """Commit a staged upload to the live dataset directories.

        ``resolutions`` maps relative file paths to actions:
        ``"overwrite"`` (default), ``"skip"``, or ``"keep_both"``.

        Resolutions are applied per-stem group: if an image is present,
        its filename is the resolution key; for sidecar-only groups,
        the first sidecar's filename is the key. The action applies to
        all files in the group (image + all sidecars).
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

        staging_images = staging_base / "images"
        staging_folders = staging_base / "folders"
        images_dir = base_dir / "images"
        folders_dir = base_dir / "folders"

        # Build groups from staged files
        from collections import defaultdict

        staged_groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"image": None, "sidecars": []})
        all_staged_files: list[Path] = []

        if staging_images.exists():
            for src_path in staging_images.iterdir():
                if src_path.is_file():
                    all_staged_files.append(src_path)
                    ext = src_path.suffix.lower()
                    if ext in IMAGE_EXTENSIONS:
                        staged_groups[("", src_path.stem)]["image"] = src_path
                    else:
                        staged_groups[("", src_path.stem)]["sidecars"].append(src_path)

        if staging_folders.exists():
            for folder_path in staging_folders.iterdir():
                if not folder_path.is_dir():
                    continue
                for src_path in folder_path.iterdir():
                    if src_path.is_file():
                        all_staged_files.append(src_path)
                        ext = src_path.suffix.lower()
                        if ext in IMAGE_EXTENSIONS:
                            staged_groups[(folder_path.name, src_path.stem)]["image"] = src_path
                        else:
                            staged_groups[(folder_path.name, src_path.stem)]["sidecars"].append(src_path)

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
                    dest_path = _unique_path(dest_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
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
                    dest_path = _unique_path(dest_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dest_path))

        # Update config TOML with new folder entries
        config_path = Path(info.config_path)
        if new_folder_names or has_root_files:
            try:
                doc = tomlkit.parse(config_path.read_text())
            except Exception:
                doc = tomlkit.document()

            if "dataset" not in doc or not isinstance(doc["dataset"], list):
                doc["dataset"] = tomlkit.aot()

            existing_paths = {entry.get("path", "") for entry in doc["dataset"] if isinstance(entry, dict)}

            root_rel = "images"
            root_abs = str(images_dir.resolve())
            if has_root_files and root_rel not in existing_paths and root_abs not in existing_paths:
                entry = tomlkit.table()
                entry["path"] = root_rel
                doc["dataset"].append(entry)

            for folder_name in sorted(new_folder_names):
                folder_rel = f"folders/{folder_name}"
                folder_abs = str((folders_dir / folder_name).resolve())
                if folder_rel not in existing_paths and folder_abs not in existing_paths:
                    entry = tomlkit.table()
                    entry["path"] = folder_rel
                    doc["dataset"].append(entry)

            config_path.write_text(tomlkit.dumps(doc))

        # Clean up staging
        shutil.rmtree(staging_base, ignore_errors=True)

        # Rescan
        self._datasets.rescan_dataset(name)
        updated_info = self._datasets.get_dataset(name)

        yield UploadProgressEvent(
            phase="complete",
            dataset=updated_info,
            warnings=warnings or [],
        )
