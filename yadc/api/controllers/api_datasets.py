from flask import jsonify

from .blueprints import ApiBlueprint


@ApiBlueprint.route("/datasets")
def list_datasets():
    """List available datasets (scan for .toml configs)."""
    # TODO: implement dataset scanning
    return jsonify([])


@ApiBlueprint.route("/datasets/<name>/images")
def list_images(name: str):  # noqa: ARG001
    """List images with captions/drafts (paginated)."""
    # TODO: implement image listing
    return jsonify([])
