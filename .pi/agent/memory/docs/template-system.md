---
name: template-system
description: How the Jinja2 prompt template system works — template resolution, loading, and variable context.
---

# Template System

## Template Resolution Chain

1. If `prompt.template` is set in config → used directly as template string
2. If `prompt.name` is set:
   - Try user templates: `~/.local/state/yadc/templates/{name}.jinja`
   - Try builtin templates: `yadc/templates/jinja/{name}.jinja`
   - Error if not found
3. If neither → use default template

CLI (`cli_caption.py`) resolves via `_resolve_template()` before creating the captioner.

## Template Format

Templates are Jinja2 files that define three blocks as `{% set %}` variables:

```jinja
{% set system_prompt %}
  System instructions...
{% endset %}

{% set user_prompt %}
  Single-round user prompt...
  Available variables: {{ artist }}, {{ tags }}, etc. (from DatasetImage extras)
{% endset %}

{% set user_prompt_multiple_rounds %}
  Multi-round prompt...
  {% for round in caption_rounds %}
  Description #{{ round.iteration }}
  {{ round.caption }}
  {% endfor %}
{% endset %}
```

## Internal Jinja Loading

`Captioner._load_jinja_template()` uses special template names:
- `__system_prompt__` — imports default + user, uses `user_template.system_prompt` with fallback to default
- `__user_prompt__` — same pattern for single-round
- `__user_prompt_multiple_rounds__` — same for multi-round
- `__default_template__` — loads from `yadc/templates/jinja/default.jinja`
- `__user_template__` — user-provided template string or default

## Template Context Variables

`prompts_from_image()` passes `dataset_image.model_dump()` as globals, which includes:
- `path`, `caption`, `caption_suffix`, `toml_suffix`, `history_suffix`
- All extra fields from the image's `__pydantic_extra__` (e.g., artist, tags, characters, etc.)
- `caption_rounds` (list of CaptionerRound) — only in multi-round mode
- `drafts` (dict[str, str]) — if drafts exist for the image

## Template Storage

- **Built-in**: bundled with package in `yadc/templates/jinja/` (loaded via `importlib.resources`)
- **User**: stored in `~/.local/state/yadc/templates/` (managed by `cmd/templates/`)
- **Project**: `templates/` dir in project root (gitignored except `example_*`)

## Built-in Templates

- `default` — generic image captioning assistant
