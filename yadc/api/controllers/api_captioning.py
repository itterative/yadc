from flask import Response, jsonify

from .blueprints import ApiBlueprint


@ApiBlueprint.route("/datasets/<name>/caption", methods=["POST"])
def start_captioning(name: str):  # noqa: ARG001
    """Start a captioning run."""
    # TODO: implement
    return jsonify({"status": "ok"})


@ApiBlueprint.route("/datasets/<name>/caption/status")
def captioning_status(name: str):  # noqa: ARG001
    """SSE stream for captioning progress."""
    # TODO: implement SSE
    return Response("data: {}\n\n", mimetype="text/event-stream")


@ApiBlueprint.route("/datasets/<name>/caption", methods=["DELETE"])
def stop_captioning(name: str):  # noqa: ARG001
    """Stop a captioning run."""
    # TODO: implement
    return jsonify({"status": "ok"})
