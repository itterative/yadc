from flask import Blueprint

# Singletons — shared across all controllers
ApiBlueprint = Blueprint("api", __name__, url_prefix="/api")
AppBlueprint = Blueprint("app", __name__)
