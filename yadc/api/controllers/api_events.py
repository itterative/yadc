from flask import jsonify

from .blueprints import ApiBlueprint


@ApiBlueprint.route("/events")
def global_events():
    """Global SSE event stream."""
    # TODO: implement
    return jsonify({})
