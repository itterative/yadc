import hashlib
from pathlib import Path

from flask import jsonify, request, send_file
from PIL import Image

from ..configuration import Configuration
from ..json_utils import jsonify_dataclass
from ..modules.logging_factory import LoggingFactory
from ..services.datasets import DatasetService
from . import controller
from .blueprints import ApiBlueprint


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


@controller
def api_datasets(
    configuration: Configuration,
    app: ApiBlueprint,
    logging: LoggingFactory,
    datasets: DatasetService,
):
    _logger = logging.get_logger(__name__)
    _thumb_cache_dir = Path(configuration.cache_path) / "thumbnails"

    @app.post("/datasets")
    def add_dataset():  # pyright: ignore[reportUnusedFunction]
        """Import an existing TOML or create a new dataset.

        JSON body:
            Import: {"name": "...", "toml_path": "..."}
            Create: {"name": "...", "image_paths": ["...", ...]}
        """
        body = request.get_json(silent=True)
        if body is None or "name" not in body:
            return jsonify({"error": "Request body must include 'name'"}), 400

        name = body["name"]

        try:
            if "toml_path" in body:
                result = datasets.import_dataset(name, body["toml_path"])
            elif "image_paths" in body:
                result = datasets.create_dataset(name, body["image_paths"])
            else:
                return jsonify({"error": "Provide 'toml_path' to import or 'image_paths' to create"}), 400

            return jsonify_dataclass(result), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.get("/datasets")
    def list_datasets():  # pyright: ignore[reportUnusedFunction]
        """List available datasets."""
        result = datasets.list_datasets()
        return jsonify_dataclass(result)

    @app.get("/datasets/<name>/images")
    def list_images(name: str):  # pyright: ignore[reportUnusedFunction]
        """List images with captions/drafts (paginated)."""
        limit = request.args.get("limit", 50, type=int)
        limit = max(1, min(limit, 200))
        after_id = request.args.get("after_id", 0, type=int)

        page = datasets.list_images(name, limit=limit, after_id=after_id)
        return jsonify_dataclass(page)

    @app.get("/datasets/<name>/images/<int:image_id>/media")
    def serve_image_media(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Serve the original image file."""
        image_path = datasets.get_image_path(name, image_id)
        if image_path is None:
            return jsonify({"error": "Image not found"}), 404
        return send_file(str(image_path))

    @app.get("/datasets/<name>/images/<int:image_id>/thumbnail")
    def serve_image_thumbnail(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Serve a cached thumbnail, generating it on first request."""
        image_path = datasets.get_image_path(name, image_id)
        if image_path is None:
            return jsonify({"error": "Image not found"}), 404

        size = request.args.get("size", 256, type=int)
        size = max(32, min(size, 1024))

        cache_path = _thumbnail_cache_path(_thumb_cache_dir, image_path, size)

        if not cache_path.exists():
            try:
                _generate_thumbnail(image_path, cache_path, size)
            except Exception as e:
                _logger.warning("Failed to generate thumbnail for %s: %s", image_path, e)
                return send_file(str(image_path))

        return send_file(str(cache_path), mimetype="image/webp")

    @app.get("/datasets/<name>/images/<int:image_id>/caption")
    def get_image_caption(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Get caption text, TOML extras, and drafts for an image."""
        result = datasets.get_caption(name, image_id)
        if result is None:
            return jsonify({"error": "Image not found"}), 404
        return jsonify(result)

    @app.put("/datasets/<name>/images/<int:image_id>/caption")
    def update_image_caption(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Update the caption text for an image."""
        body = request.get_json(silent=True)
        if body is None or "caption" not in body:
            return jsonify({"error": "Request body must include 'caption'"}), 400

        ok = datasets.update_caption(name, image_id, body["caption"])
        if not ok:
            return jsonify({"error": "Image not found"}), 404
        return jsonify({"status": "ok"})
