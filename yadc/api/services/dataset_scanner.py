"""``DatasetScanner`` — reconcile on-disk dataset state with the SQLite index.

Two update paths:

- **Full walk** (:meth:`scan_disk`): walks every configured image
  directory, diffs against the current index, and writes SQL for
  changed rows. Used by the periodic background refresh and by
  manual rescan / register flows.
- **Targeted** (:meth:`scan_targeted`): given a known set of
  filesystem paths from the watcher, updates only the affected
  image rows. O(|changed|) instead of O(|dataset|) — the long pole
  is gone for the common case of a single file being added, removed,
  or modified externally.

A pure read path (:meth:`read_disk`) returns the on-disk meta dict
without touching the DB, used by :class:`DatasetService.register` for
the initial bulk insert where all rows are new.

The scanner delegates config loading and path extraction to
:class:`DatasetLoader`.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

from PIL import Image

from yadc.core.config import Config

from ..configuration import Configuration
from ..constants import IMAGE_EXTENSIONS
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .dataset_loader import DatasetLoader
from .dataset_repository import DatasetRepository, ImageInfo


class DatasetScanner(Service):
    """Reads dataset image state from disk and reconciles it with the SQLite index.

    Owns the per-image metadata extraction, the disk walker, the
    diff-vs-index reconciliation loop, and the targeted-update path
    triggered by the inotify watcher. Returns ``bool`` from update
    methods indicating whether any rows actually changed (so callers
    can decide whether to re-dispatch events).
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        repo: DatasetRepository,
        loader: DatasetLoader,
        logging: LoggingFactory,
        configuration: Configuration,
    ):
        self._db: DBConnectionFactory = db
        self._repo: DatasetRepository = repo
        self._loader: DatasetLoader = loader
        self._logger: Logger = logging.get_logger(__name__)
        self._configuration: Configuration = configuration

    # --- Public API: update paths ---

    def scan_disk(
        self,
        dataset_id: int,
        config_path: str | None,
        config: Config | None = None,
    ) -> bool:
        """Walk the disk for a dataset and reconcile the index in one transaction.

        The disk walk happens outside the transaction (it's file I/O, not
        SQL). The DB reconciliation runs inside a single transaction so
        the index never reflects a half-applied scan.

        Only writes SQL for rows that actually changed. The watcher
        keeps the index in sync in the common case, so most passes
        (including the periodic background refresh) find no diff and
        skip every per-image ``upsert_image`` / ``delete_image`` call.
        ``update_dataset_stats`` still runs to bump ``last_scanned_t``,
        so the next pass can skip a recently-checked dataset.

        Returns ``True`` if any rows were upserted or deleted (i.e. the
        index changed).
        """
        disk_images = self.read_disk(config_path, config=config)
        self._logger.debug(
            "Disk scan started. [dataset_id=%d, config_path=%s, disk_images=%d]",
            dataset_id,
            config_path,
            len(disk_images),
        )
        with self._db.transaction():
            existing = self._repo.list_image_infos(dataset_id)
            to_upsert: dict[str, dict[str, Any]] = {}
            for path, meta in disk_images.items():
                current = existing.get(path)
                if current is None or self._image_meta_differs(current, meta):
                    to_upsert[path] = meta
            to_delete: list[int] = [img.id for path, img in existing.items() if path not in disk_images]
            if to_upsert or to_delete:
                self._logger.debug(
                    "Disk scan diff. [dataset_id=%d, existing=%d, disk=%d, to_upsert=%d, to_delete=%d]",
                    dataset_id,
                    len(existing),
                    len(disk_images),
                    len(to_upsert),
                    len(to_delete),
                )
            else:
                self._logger.debug(
                    "Disk scan in sync (no diff). [dataset_id=%d, images=%d]",
                    dataset_id,
                    len(existing),
                )
            for path, meta in to_upsert.items():
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
            for image_id in to_delete:
                self._repo.delete_image(image_id)
            self._repo.update_dataset_stats(dataset_id, len(disk_images))

        return bool(to_upsert or to_delete)

    def scan_targeted(
        self,
        dataset_id: int,
        changed_paths: list[str],
    ) -> bool:
        """Apply a targeted index update for a known set of changed paths.

        Used by ``DatasetService._on_dataset_changed`` when the
        watcher dispatches a ``DatasetChangedEvent`` with
        ``changed_paths`` populated. We map each filesystem change to
        its image row(s), stat each image to decide between upsert
        and delete, and only write SQL for the affected rows.
        Returns ``True`` if any rows changed.

        The image walk is O(|changed_paths|) and the SQL is bounded by
        the same, so a single-file external change is a constant-time
        update regardless of dataset size.
        """
        affected = self.resolve_affected_image_paths(changed_paths)
        if not affected:
            return False

        to_upsert: dict[str, dict[str, Any]] = {}
        to_delete: list[int] = []

        with self._db.transaction():
            existing = self._repo.list_image_infos(dataset_id)
            for image_path_str in affected:
                # The resolver emits multiple ``<stem>.<ext>`` candidates
                # for stem-based sidecars (captions, drafts). Most won't
                # correspond to an indexed image — skip them. This also
                # handles the case where a new image file is dropped:
                # the watcher reports its path directly, so it maps to
                # itself and the ``current is None`` branch handles it.
                current = existing.get(image_path_str)
                if current is None and Path(image_path_str).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image_path = Path(image_path_str)

                if image_path.is_file():
                    meta = self.scan_image_meta(image_path)
                    if meta is None:
                        # The watcher only reports watched extensions,
                        # but a sidecar resolve could land on a
                        # non-image path. Skip.
                        continue
                    if current is None or self._image_meta_differs(current, meta):
                        to_upsert[image_path_str] = meta
                else:
                    # File gone — drop the index row only if it was
                    # actually present (a stray sidecar for a never-
                    # indexed image is a no-op).
                    if current is not None:
                        to_delete.append(current.id)

            if to_upsert or to_delete:
                self._logger.debug(
                    "Targeted update diff. [dataset_id=%d, affected=%d, to_upsert=%d, to_delete=%d]",
                    dataset_id,
                    len(affected),
                    len(to_upsert),
                    len(to_delete),
                )
            else:
                self._logger.debug(
                    "Targeted update in sync (no diff). [dataset_id=%d, affected=%d]",
                    dataset_id,
                    len(affected),
                )

            for path_str, meta in to_upsert.items():
                self._repo.upsert_image(
                    dataset_id=dataset_id,
                    path=path_str,
                    file_name=meta["file_name"],
                    has_caption=meta["has_caption"],
                    has_toml=meta["has_toml"],
                    width=meta["width"],
                    height=meta["height"],
                    draft_names=meta["draft_names"],
                    last_modified_t=meta["last_modified_t"],
                )
            for image_id in to_delete:
                self._repo.delete_image(image_id)

            if to_upsert or to_delete:
                # ``to_upsert`` is a mix of inserts (new images) and
                # updates (existing images whose meta changed). Only
                # inserts add to the count. Compute that by checking
                # which ``to_upsert`` paths were already in ``existing``.
                new_rows = sum(1 for path in to_upsert if path not in existing)
                new_count = len(existing) + new_rows - len(to_delete)
                self._repo.update_dataset_stats(dataset_id, new_count)

        return bool(to_upsert or to_delete)

    # --- Public API: pure read paths ---

    def read_disk(self, config_path: str | None, config: Config | None = None) -> dict[str, dict[str, Any]]:
        """Walk the image directories for a dataset and return scanned metadata.

        Returns a dict mapping each on-disk image path to its metadata.
        Empty dict if the config is missing or has no image directories.

        Does not touch the database. Used by ``DatasetService.register``
        for the initial bulk insert where the index starts empty.
        """
        if not config_path:
            return {}

        config_file = Path(config_path)
        if not config_file.exists():
            return {}

        if config is None:
            config = self._loader.load_config(config_file)
        if config is None:
            return {}

        image_dirs: list[Path] = []
        for entry in config.dataset:
            if entry.path:
                p = Path(entry.path)
                if not p.is_absolute():
                    p = config_file.parent / p
                if p.is_dir():
                    image_dirs.append(p.resolve())

        if not image_dirs:
            return {}

        disk_images: dict[str, dict[str, Any]] = {}
        for img_dir in image_dirs:
            for child in sorted(img_dir.iterdir()):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                meta = self.scan_image_meta(child)
                if meta is not None:
                    disk_images[str(child)] = meta
        return disk_images

    def resolve_affected_image_paths(self, changed_paths: list[str]) -> set[str]:
        """Map filesystem change paths to the image rows they affect.

        The watcher reports the *raw* filesystem path (which may be a
        sidecar like ``photo.txt`` or ``photo.gemma.draft~``). The
        index is keyed on the *image* path (``photo.jpg``), so we
        need to translate sidecar events back to the image they
        decorate. Multiple changed paths can map to the same image
        (e.g. a caption and a draft for the same image), and the
        returned set is deduplicated.

        Sidecar naming convention (see :class:`core.dataset.DatasetImage`):
        ``<image_stem>.<sidecar_suffix>`` — the stem is the image
        filename *minus* the image extension. So for image
        ``photo.jpg``, the caption is ``photo.txt`` (NOT
        ``photo.jpg.txt``). The image extension isn't recoverable
        from the sidecar filename, so the resolver emits all
        ``<image_stem>.<ext>`` candidates and the targeted update
        filters to the one that exists in the index.

        Image files map to themselves. Files we can't resolve are
        silently dropped — the targeted update only acts on rows
        that already exist in the index.
        """
        affected: set[str] = set()
        for path_str in changed_paths:
            path = Path(path_str)
            name = path.name
            parent = path.parent

            # Image file — maps to itself.
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                affected.add(str(path))
                continue

            # ``<image_stem>.<draft_name>.draft~`` — emit all
            # ``<image_stem>.<ext>`` candidates.
            if name.endswith(".draft~"):
                prefix = name[: -len(".draft~")]
                if "." in prefix:
                    stem = prefix.rsplit(".", 1)[0]
                else:
                    stem = prefix
                for ext in IMAGE_EXTENSIONS:
                    affected.add(str(parent / (stem + ext)))
                continue

            # ``<image_stem>.txt`` / ``.toml`` / ``.history~`` — the
            # sidecar suffix replaces the image extension, so the
            # stem is just the prefix. Emit all ``<stem>.<ext>``
            # candidates; the targeted update will pick the one in
            # the index.
            for sidecar_suffix in (".txt", ".toml", ".history~"):
                if name.endswith(sidecar_suffix):
                    stem = name[: -len(sidecar_suffix)]
                    for ext in IMAGE_EXTENSIONS:
                        affected.add(str(parent / (stem + ext)))
                    break

        return affected

    def scan_image_meta(self, image_path: Path) -> dict[str, Any] | None:
        """Return the indexable metadata for a single image file.

        Returns ``None`` if *image_path* doesn't have a recognised image
        extension. Errors reading dimensions or mtime are swallowed and
        result in zeroed-out fields (matches the behaviour of
        :meth:`read_disk`).
        """
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None

        caption_path = image_path.with_suffix(".txt")
        toml_path = image_path.with_suffix(".toml")

        # Find drafts
        draft_names: list[str] = []
        stem = image_path.stem
        try:
            for f in image_path.parent.iterdir():
                if f.name.startswith(stem + ".") and f.name.endswith(".draft~"):
                    draft_name = f.name[len(stem) + 1 : -len(".draft~")]
                    if draft_name:
                        draft_names.append(draft_name)
        except OSError:
            pass

        try:
            mod_time = image_path.stat().st_mtime
        except OSError:
            mod_time = None

        # Read image dimensions
        img_width = 0
        img_height = 0
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            pass

        return {
            "file_name": image_path.name,
            "has_caption": caption_path.exists(),
            "has_toml": toml_path.exists(),
            "width": img_width,
            "height": img_height,
            "draft_names": ",".join(sorted(draft_names)),
            "last_modified_t": mod_time,
        }

    @staticmethod
    def _image_meta_differs(info: ImageInfo, meta: dict[str, Any]) -> bool:
        """Return True if the disk-read ``meta`` for an image differs from the stored :class:`ImageInfo`.

        Used by :meth:`scan_disk` and :meth:`scan_targeted` to filter
        out images that haven't actually changed since the last index
        pass. The comparison covers every field the upsert would
        write: caption / toml / draft presence, image dimensions, and
        the file's ``last_modified_t``. ``id``, ``file_name``,
        ``path``, and the derived ``delete_path`` are excluded —
        they're either identifiers or computed from the path.
        """
        if info.has_caption != meta["has_caption"]:
            return True
        if info.has_toml != meta["has_toml"]:
            return True
        if info.width != meta["width"]:
            return True
        if info.height != meta["height"]:
            return True
        if info.last_modified_t != meta["last_modified_t"]:
            return True
        # ``draft_names`` is stored as a sorted CSV; the in-memory
        # ImageInfo parses it into a list. Sort-compare the lists so the
        # two representations are equivalent.
        disk_drafts: list[str] = meta["draft_names"].split(",") if meta["draft_names"] else []
        if sorted(info.draft_names) != sorted(disk_drafts):
            return True
        return False
