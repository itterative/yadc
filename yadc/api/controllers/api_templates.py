"""Template CRUD endpoints — backed by the ``cmd.templates`` module."""

import re

from flask import jsonify, request

from yadc.cmd import templates as cmd_templates

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error

# Light Jinja2 variable extractor — finds {{ var }} and {% for x in ... %} references.
# Not a full parser, but good enough for editor hints.
_JINJA_VAR_RE = re.compile(r"\{\{-?\s*(\w+)(?:\.[\w.]+)*\s*(?:\|[^}]*)?-?\}\}")
_JINJA_FOR_RE = re.compile(r"\{%[-\s]+for\s+\w+\s+in\s+(\w+)")


def _extract_variables(template: str) -> list[str]:
    """Extract referenced variable names from a Jinja2 template string."""
    names: set[str] = set()

    for m in _JINJA_VAR_RE.finditer(template):
        names.add(m.group(1))

    for m in _JINJA_FOR_RE.finditer(template):
        names.add(m.group(1))

    # Filter out Jinja builtins
    builtins = {"true", "false", "none", "True", "False", "None", "range", "lipsum", "dict", "namespace"}
    names -= builtins

    return sorted(names)


@controller
def api_templates(app: ApiBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/templates")
    def list_templates():  # pyright: ignore[reportUnusedFunction]
        """List all templates (user + built-in) with source type."""
        user_names = cmd_templates.list_user_template()
        # Built-in templates — for now only 'default'
        builtin_names: list[str] = ["default"]

        # Deduplicate: if user created a template named 'default', show it as user
        results: list[dict[str, str]] = []
        seen: set[str] = set()

        for name in user_names:
            results.append({"name": name, "source": "user"})
            seen.add(name)

        for name in builtin_names:
            if name not in seen:
                results.append({"name": name, "source": "builtin"})

        return jsonify(results)

    @app.get("/templates/<name>")
    def get_template(name: str):  # pyright: ignore[reportUnusedFunction]
        """Return template content and metadata."""
        # Try user template first, then built-in
        source = "user"
        try:
            content = cmd_templates.load_user_template(name)
        except FileNotFoundError:
            try:
                content = cmd_templates.load_builtin_template(name)
                source = "builtin"
            except Exception:
                return jsonify_error(f"Template '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)
        except Exception as e:
            return jsonify_error(str(e), status=500, code=ErrorCode.INTERNAL_ERROR)

        variables = _extract_variables(content)

        return jsonify(
            {
                "name": name,
                "source": source,
                "content": content,
                "variables": variables,
            }
        )

    @app.put("/templates/<name>")
    def put_template(name: str):  # pyright: ignore[reportUnusedFunction]
        """Create or update a user template.

        JSON body: {"content": "..."}
        """
        body = request.get_json(silent=True)
        if body is None or "content" not in body:
            return jsonify_error("Request body must include 'content'", status=400, code=ErrorCode.BAD_REQUEST)

        content = body["content"]
        if not isinstance(content, str):
            return jsonify_error("'content' must be a string", status=400, code=ErrorCode.BAD_REQUEST)

        cmd_templates.save_user_template(name, content)
        _logger.info("Template '%s' saved.", name)

        variables = _extract_variables(content)
        return jsonify(
            {
                "name": name,
                "source": "user",
                "content": content,
                "variables": variables,
            }
        )

    @app.delete("/templates/<name>")
    def delete_template(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a user template. Built-in templates cannot be deleted."""
        # Check if it's a built-in
        builtin_names = ["default"]
        if name in builtin_names:
            try:
                cmd_templates.load_user_template(name)
            except FileNotFoundError:
                return jsonify_error("Cannot delete built-in templates", status=400, code=ErrorCode.BAD_REQUEST)

        deleted = cmd_templates.delete_user_template(name)
        if not deleted:
            return jsonify_error(f"Template '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        _logger.info("Template '%s' deleted.", name)
        return jsonify({"status": "ok"})
