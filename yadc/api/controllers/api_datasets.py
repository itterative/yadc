import hashlib
import json
import typing
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, cast

import pydantic
from PIL import Image
from quart import (
    Response,
    jsonify,
    request,
    send_file,
)
from werkzeug.datastructures import FileStorage, MultiDict

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from ..services.captioning import CaptioningService
from ..services.config_history import ConfigHistoryService
from ..services.dataset_upload import DatasetUploadService
from ..services.datasets import DatasetService
from ..services.managed_datasets import ManagedDatasetsService
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import DataclassJSONEncoder, ErrorCode, jsonify_dataclass, jsonify_error, validate_body


class AddDatasetBody(pydantic.BaseModel):
    name: str
    toml_path: str | None = None
    image_paths: list[str] | None = None


class CommitStagedUploadBody(pydantic.BaseModel):
    staging_id: str
    resolutions: dict[str, str] = {}


class DeleteItemsBody(pydantic.BaseModel):
    paths: list[str]


class UpdateImageCaptionBody(pydantic.BaseModel):
    caption: str


class UpdateImageExtrasBody(pydantic.BaseModel):
    extras_raw: str


class PreviewPromptBody(pydantic.BaseModel):
    template: str = ""
    template_name: str = ""


# --- Multipart form helpers ------------------------------------------------


_ModelT = typing.TypeVar("_ModelT", bound=pydantic.BaseModel)


def _read_uploaded_files(files: Any) -> list[tuple[str, BinaryIO]]:
    """Read uploaded ``FileStorage`` files into ``(filename, BytesIO)`` tuples.

    Quart closes the SpooledTemporaryFile handles after the multipart body is
    consumed, so accessing .stream during the async generator would fail with
    "seek of closed file". Reading into BytesIO buffers up front side-steps
    that.

    Entries with no ``filename`` (empty form fields) are skipped.

    Returns the list typed as ``list[tuple[str, BinaryIO]]`` to match the
    upload service signature. The runtime values are ``BytesIO`` instances,
    which are ``BinaryIO`` subclasses.
    """
    uploaded: list[FileStorage] = cast(list[FileStorage], files.getlist("files"))
    return [(f.filename, BytesIO(f.read())) for f in uploaded if f.filename]


def _thumbnail_cache_path(cache_dir: Path, image_path: Path, size: int) -> Path:
    """Derive a deterministic cache path for a thumbnail."""
    key = hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
    return cache_dir / f"{key}_{size}.webp"


def _generate_thumbnail(image_path: Path, cache_path: Path, size: int) -> Path:
    """Generate a thumbnail at the given long-side size and save as WebP."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        img.save(cache_path, "WEBP", quality=80)

    return cache_path


def _snapshot_initial(config_history: ConfigHistoryService, datasets: DatasetService, name: str) -> None:
    """Save an initial config snapshot for a newly created/imported dataset."""
    info = datasets.get_dataset(name)
    if info is None or info.config_path is None:
        return
    try:
        with open(info.config_path) as f:
            content = f.read()
        config_history.save_snapshot(name, content)
    except FileNotFoundError:
        pass


@controller
def api_datasets(
    configuration: Configuration,
    app: ApiBlueprint,
    logging: LoggingFactory,
    datasets: DatasetService,
    managed_datasets: ManagedDatasetsService,
    dataset_upload: DatasetUploadService,
    config_history: ConfigHistoryService,
    captioning: CaptioningService,
):
    _logger = logging.get_logger(__name__)
    _thumb_cache_dir = Path(configuration.cache_path) / "thumbnails"

    @app.post("/datasets")
    async def add_dataset():  # pyright: ignore[reportUnusedFunction]
        """Import an existing TOML or create a new dataset.

        JSON body:
            Import: {"name": "...", "toml_path": "..."}
            Create: {"name": "...", "image_paths": ["...", ...]}
        """
        body = validate_body(AddDatasetBody, await request.get_json(silent=True))

        try:
            if body.toml_path is not None:
                result = datasets.import_dataset(body.name, body.toml_path)
                # Snapshot the initial config
                _snapshot_initial(config_history, datasets, body.name)
            elif body.image_paths is not None:
                result = datasets.create_dataset(body.name, body.image_paths)
                _snapshot_initial(config_history, datasets, body.name)
            else:
                return jsonify_error("Provide 'toml_path' to import or 'image_paths' to create", status=400)

            return jsonify_dataclass(result), 201
        except Exception as e:
            return jsonify_error(str(e), status=400)

    @app.post("/datasets/upload")
    async def upload_dataset():  # pyright: ignore[reportUnusedFunction]
        """Create a dataset from uploaded image files.

        Returns a streaming NDJSON response with progress events:
        - {"phase": "validating", "file": "...", "index": N, "total": M}
        - {"phase": "writing", "file": "...", "index": N, "total": M}
        - {"phase": "complete", "dataset": {...}, "warnings": [...]}
        - {"phase": "error", "message": "..."}
        """
        form = cast("dict[str, str]", await request.form)
        name = form.get("name", "").strip()

        if not name:
            return jsonify_error("Dataset name is required", status=400, code=ErrorCode.BAD_REQUEST)

        files = cast("MultiDict[str, Any]", await request.files)
        if not files.getlist("files"):
            return jsonify_error("At least one file is required", status=400, code=ErrorCode.BAD_REQUEST)

        if request.content_length and request.content_length > configuration.max_upload_size_bytes:
            return jsonify_error(
                f"Total upload size exceeds {configuration.max_upload_size_bytes} bytes limit",
                status=413,
                code=ErrorCode.PAYLOAD_TOO_LARGE,
            )

        if datasets.get_dataset(name) is not None:
            return jsonify_error(f"Dataset '{name}' already exists", status=409, code=ErrorCode.CONFLICT)

        # ``source`` is the originating client identifier (``"ui:<uuid>"`` from
        # a frontend tab). It is propagated to the upload service so the
        # resulting DatasetChangedEvent can be suppressed by the originating
        # tab.
        source = request.args.get("source", "")

        # Read all file contents into BytesIO buffers before streaming the response.
        # Quart closes the SpooledTemporaryFile handles after the multipart body is
        # consumed, so accessing .stream during the async generator would fail with
        # "seek of closed file".
        file_tuples = _read_uploaded_files(files)

        async def _stream_upload():
            try:
                async for event in dataset_upload.create_dataset_from_upload(name, file_tuples, source=source):
                    yield json.dumps(event, cls=DataclassJSONEncoder) + "\n"
                    if event.phase == "complete":
                        _snapshot_initial(config_history, datasets, name)
            except Exception as e:
                yield json.dumps({"phase": "error", "message": str(e)}, cls=DataclassJSONEncoder) + "\n"

        response = Response(_stream_upload(), mimetype="application/x-ndjson")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post("/datasets/<name>/upload")
    async def append_upload_dataset(name: str):  # pyright: ignore[reportUnusedFunction]
        """Stage files for appending to an existing managed dataset.

        Returns a streaming NDJSON response with progress events:
        - {"phase": "validating", "file": "...", "index": N, "total": M}
        - {"phase": "writing", "file": "...", "index": N, "total": M}
        - {"phase": "conflicts", "staging_id": "...", "conflicts": [...]}
        - {"phase": "complete", "dataset": {...}, "warnings": [...]}
        - {"phase": "error", "message": "..."}
        """
        if captioning.is_captioning(name):
            return jsonify_error("Cannot modify dataset while captioning is in progress", status=409, code=ErrorCode.CONFLICT)

        # See upload_dataset() for the rationale behind ``source``.
        source = request.args.get("source", "")

        files = cast("MultiDict[str, Any]", await request.files)
        if not files.getlist("files"):
            return jsonify_error("At least one file is required", status=400, code=ErrorCode.BAD_REQUEST)

        if request.content_length and request.content_length > configuration.max_upload_size_bytes:
            return jsonify_error(
                f"Total upload size exceeds {configuration.max_upload_size_bytes} bytes limit",
                status=413,
                code=ErrorCode.PAYLOAD_TOO_LARGE,
            )

        file_tuples = _read_uploaded_files(files)

        async def _stream_append():
            try:
                async for event in dataset_upload.append_dataset_from_upload(name, file_tuples, source=source):
                    yield json.dumps(event, cls=DataclassJSONEncoder) + "\n"
            except Exception as e:
                yield json.dumps({"phase": "error", "message": str(e)}, cls=DataclassJSONEncoder) + "\n"

        response = Response(_stream_append(), mimetype="application/x-ndjson")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post("/datasets/<name>/staging/commit")
    async def commit_staged_upload(name: str):  # pyright: ignore[reportUnusedFunction]
        """Commit a staged upload after conflict resolution.

        JSON body:
            {"staging_id": "...", "resolutions": {"file.jpg": "overwrite", "folder/file.jpg": "skip"}}

        Returns a streaming NDJSON response:
        - {"phase": "complete", "dataset": {...}, "warnings": [...]}
        - {"phase": "error", "message": "..."}
        """
        if captioning.is_captioning(name):
            return jsonify_error("Cannot modify dataset while captioning is in progress", status=409, code=ErrorCode.CONFLICT)

        # See upload_dataset() for the rationale behind ``source``.
        source = request.args.get("source", "")
        body = validate_body(CommitStagedUploadBody, await request.get_json(silent=True))

        async def _stream_commit():
            try:
                async for event in dataset_upload.commit_staged_upload(name, body.staging_id, body.resolutions, source=source):
                    yield json.dumps(event, cls=DataclassJSONEncoder) + "\n"
            except Exception as e:
                yield json.dumps({"phase": "error", "message": str(e)}, cls=DataclassJSONEncoder) + "\n"

        response = Response(_stream_commit(), mimetype="application/x-ndjson")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.delete("/datasets/<name>/items")
    async def delete_dataset_items(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete files and/or folders from a managed dataset.

        Only datasets with ``source == "upload"`` can be modified.

        JSON body:
            {"paths": ["foo.jpg", "train/img.jpg", "train"]}

        Returns:
            {"deleted": ["foo.jpg", "train"], "warnings": []}
        """
        if captioning.is_captioning(name):
            return jsonify_error("Cannot modify dataset while captioning is in progress", status=409, code=ErrorCode.CONFLICT)

        body = validate_body(DeleteItemsBody, await request.get_json(silent=True))

        try:
            deleted, warnings = managed_datasets.delete_items(name, body.paths, source=request.args.get("source", ""))
            return jsonify({"deleted": deleted, "warnings": warnings})
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)
        except Exception as e:
            _logger.exception("Failed to delete items from dataset '%s'", name)
            return jsonify_error(str(e), status=500)

    @app.get("/datasets/<name>/folders")
    def list_dataset_folders(name: str):  # pyright: ignore[reportUnusedFunction]
        """List folders for a managed dataset with image counts."""
        try:
            folders = managed_datasets.list_folders(name)
            return jsonify(folders)
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)

    @app.get("/datasets")
    def list_datasets():  # pyright: ignore[reportUnusedFunction]
        """List available datasets."""
        result = datasets.list_datasets()
        return jsonify_dataclass(result)

    @app.delete("/datasets/<name>")
    def delete_dataset(name: str):  # pyright: ignore[reportUnusedFunction]
        """Unregister a dataset and delete its state dir."""
        if captioning.is_captioning(name):
            return jsonify_error("Cannot delete dataset while captioning is in progress", status=409, code=ErrorCode.CONFLICT)

        found = datasets.unregister_dataset(name)
        if not found:
            return jsonify_error("Dataset not found", status=404)
        return jsonify({"status": "ok"})

    @app.post("/datasets/<name>/rescan")
    def rescan_dataset(name: str):  # pyright: ignore[reportUnusedFunction]
        """Force a rescan of a dataset's images."""
        source = request.args.get("source", "")
        found = datasets.rescan_dataset(name, source=source)
        if not found:
            return jsonify_error("Dataset not found", status=404)
        result = datasets.get_dataset(name)
        return jsonify_dataclass(result)

    @app.get("/datasets/<name>/drafts")
    def list_dataset_drafts(name: str):  # pyright: ignore[reportUnusedFunction]
        """Return sorted list of unique draft names across all images in a dataset."""
        names = datasets.get_draft_names(name)
        return jsonify(names)

    @app.get("/datasets/<name>/drafts/summary")
    def list_dataset_drafts_summary(name: str):  # pyright: ignore[reportUnusedFunction]
        """Return draft names with image counts for a dataset."""
        summary = datasets.get_draft_summary(name)
        return jsonify(summary)

    @app.delete("/datasets/<name>/drafts/<draft_name>")
    def delete_dataset_draft(name: str, draft_name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a named draft from all images in a dataset."""
        deleted = datasets.delete_draft_all(name, draft_name, source=request.args.get("source", ""))
        if deleted == 0:
            return jsonify_error("Draft not found", status=404)
        return jsonify({"status": "ok", "deleted": deleted})

    @app.get("/datasets/<name>/images")
    def list_images(name: str):  # pyright: ignore[reportUnusedFunction]
        """List images with captions/drafts (paginated)."""
        limit = request.args.get("limit", 50, type=int)
        limit = max(1, min(limit, 200))
        after_id = request.args.get("after_id", 0, type=int)

        page = datasets.list_images(name, limit=limit, after_id=after_id)
        return jsonify_dataclass(page)

    @app.get("/datasets/<name>/images/<int:image_id>/media")
    async def serve_image_media(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Serve the original image file."""
        image_path = datasets.get_image_path(name, image_id)
        if image_path is None:
            return jsonify_error("Image not found", status=404)
        return await send_file(str(image_path))

    @app.get("/datasets/<name>/images/<int:image_id>/thumbnail")
    async def serve_image_thumbnail(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Serve a cached thumbnail, generating it on first request."""
        image_path = datasets.get_image_path(name, image_id)
        if image_path is None:
            return jsonify_error("Image not found", status=404)

        size = request.args.get("size", 256, type=int)
        size = max(32, min(size, 1024))

        cache_path = _thumbnail_cache_path(_thumb_cache_dir, image_path, size)

        if not cache_path.exists():
            try:
                _generate_thumbnail(image_path, cache_path, size)
            except Exception as e:
                _logger.warning("Failed to generate thumbnail for %s: %s", image_path, e)
                return await send_file(str(image_path))

        return await send_file(str(cache_path), mimetype="image/webp")

    @app.get("/datasets/<name>/images/<int:image_id>/caption")
    def get_image_caption(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Get caption text, TOML extras, and drafts for an image."""
        result = datasets.get_caption(name, image_id)
        if result is None:
            return jsonify_error("Image not found", status=404)
        return jsonify(result)

    @app.put("/datasets/<name>/images/<int:image_id>/caption")
    async def update_image_caption(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Update the caption text for an image."""
        body = validate_body(UpdateImageCaptionBody, await request.get_json(silent=True))

        ok = datasets.update_caption(name, image_id, body.caption, source=request.args.get("source", ""))
        if not ok:
            return jsonify_error("Image not found", status=404)
        return jsonify({"status": "ok"})

    @app.get("/datasets/<name>/images/<int:image_id>/history")
    def get_image_history(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Get history entries for an image (most recent first, up to 3)."""
        result = datasets.get_history(name, image_id)
        if result is None:
            return jsonify_error("Image not found", status=404)
        return jsonify_dataclass(result)

    @app.put("/datasets/<name>/images/<int:image_id>/history/<int:history_index>/restore")
    def restore_image_history(name: str, image_id: int, history_index: int):  # pyright: ignore[reportUnusedFunction]
        """Restore caption + extras from a history entry."""
        ok = datasets.restore_history(name, image_id, history_index, source=request.args.get("source", ""))
        if not ok:
            return jsonify_error("Image or history entry not found", status=404)
        return jsonify({"status": "ok"})

    @app.delete("/datasets/<name>/images/<int:image_id>/history/<entry_hash>")
    def delete_image_history(name: str, image_id: int, entry_hash: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a single history entry for an image by content hash."""
        ok = datasets.delete_history(name, image_id, entry_hash, source=request.args.get("source", ""))
        if not ok:
            return jsonify_error("Image or history entry not found", status=404)
        return jsonify({"status": "ok"})

    @app.delete("/datasets/<name>/images/<int:image_id>/drafts/<draft_name>")
    def delete_image_draft(name: str, image_id: int, draft_name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a named draft for an image."""
        ok = datasets.delete_draft(name, image_id, draft_name, source=request.args.get("source", ""))
        if not ok:
            return jsonify_error("Image or draft not found", status=404)
        return jsonify({"status": "ok"})

    @app.put("/datasets/<name>/images/<int:image_id>/extras")
    async def update_image_extras(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Update the TOML extras sidecar for an image."""
        body = validate_body(UpdateImageExtrasBody, await request.get_json(silent=True))

        try:
            ok = datasets.update_extras(name, image_id, body.extras_raw, source=request.args.get("source", ""))
        except ValueError as e:
            return jsonify_error(str(e), status=400)

        if not ok:
            return jsonify_error("Image not found", status=404)
        return jsonify({"status": "ok"})

    @app.post("/datasets/<name>/images/<int:image_id>/preview-prompt")
    async def preview_prompt(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Render the system and user prompts for an image using a given template.

        JSON body:
            {"template": "..."}  — raw Jinja2 template content
            {"template_name": "..."}  — name of a user/builtin template
            {}  — use the default template
        If both are provided, "template" takes priority.
        """
        body = validate_body(PreviewPromptBody, await request.get_json(silent=True))

        template = body.template
        template_name = body.template_name

        # Resolve template name to content if needed
        if not template and template_name:
            from yadc.cmd import templates as cmd_templates
            from yadc.templates import load_builtin_template

            for loader in (cmd_templates.load_user_template, load_builtin_template):
                try:
                    template = loader(template_name)
                    break
                except Exception:
                    continue

            if not template:
                return jsonify_error(f"Template '{template_name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        try:
            result = datasets.preview_prompt(name, image_id, template)
        except Exception as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)

        if result is None:
            return jsonify_error("Image not found", status=404, code=ErrorCode.NOT_FOUND)

        return jsonify(result)
