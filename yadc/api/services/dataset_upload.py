"""Dataset upload service — handles file uploads for dataset creation.

Validates, writes, and indexes uploaded image files (plus sidecars) into
a new dataset under ``STATE_PATH/<name>/``.
"""

import shutil
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import toml
from PIL import Image

from yadc.api.configuration import Configuration
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.api.services.datasets import IMAGE_EXTENSIONS, DatasetService, _dataset_config_path, _dataset_state_dir

# Extensions allowed for uploaded files (images + sidecars).
UPLOAD_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | frozenset({".txt", ".toml", ".draft~", ".history~"})


@dataclass
class DatasetUploadResult:
    """Result of a dataset upload, including the created dataset and any warnings."""

    dataset: Any  # DatasetInfo — avoiding circular import; typed at usage site
    warnings: list[str] = field(default_factory=list)


@dataclass
class UploadProgressEvent:
    """Streaming progress event for dataset uploads."""

    phase: str  # "validating" | "writing" | "complete" | "error"
    file: str = ""
    index: int = 0
    total: int = 0
    dataset: Any = None  # DatasetInfo on "complete"
    warnings: list[str] = field(default_factory=list)
    message: str = ""


class DatasetUploadService(Service):
    """Handles validation, writing, and registration of uploaded dataset files."""

    def __init__(
        self,
        datasets: DatasetService,
        configuration: Configuration,
        logging: LoggingFactory,
    ):
        self._datasets: DatasetService = datasets
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)

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
            toml.loads(content.decode("utf-8"))
            return True
        except Exception:
            return False

    async def create_dataset_from_upload(self, name: str, files: list[tuple[str, BinaryIO]]) -> AsyncGenerator[UploadProgressEvent, None]:
        """Create a new dataset from uploaded image files.

        Root files (no directory component) are written flat to
        ``STATE_PATH/<name>/images/``. Files with exactly one directory
        component (e.g. ``train/cat.jpg``) are written to
        ``STATE_PATH/<name>/folders/``. Nested files (deeper than one
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

        base_dir = _dataset_state_dir(name)
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
                dataset_entries.append({"path": str(images_dir)})
            for folder_name in sorted(folder_names):
                dataset_entries.append({"path": str(folders_dir / folder_name)})

            raw: dict[str, Any] = {"dataset": dataset_entries}

            dest = _dataset_config_path(name)
            with open(dest, "w") as f:
                toml.dump(raw, f)

            dataset = self._datasets.register(name, str(dest), source="upload")
            yield UploadProgressEvent(phase="complete", dataset=dataset, warnings=warnings)
        except Exception as e:
            shutil.rmtree(base_dir, ignore_errors=True)
            yield UploadProgressEvent(phase="error", message=str(e))
