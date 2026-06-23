---
name: cli-cmd-structure
description: How CLI commands and cmd/ modules are structured — click commands vs pure logic split.
category: architecture
---

# CLI & cmd/ Structure

## Pattern

Each command group has two pieces:

1. **`yadc/cmd/<name>/<name>.py`** — pure logic module (no click)
2. **`yadc/cli_<name>.py`** — click command group that imports and delegates to the cmd module

### cmd/ package

```
yadc/cmd/<name>/
  __init__.py    # re-exports public functions from the .py module
  <name>.py      # implementation (imports from yadc.cmd.app for paths, etc.)
```

`__init__.py` pattern:
```python
from .<name> import func_a, func_b

__all__ = ["func_a", "func_b"]
```

### cli_ module

```python
from yadc.cmd import <name> as cmd_<name>   # import the package, not the submodule
```

Commands use `cmd_<name>.func_a(...)` to call logic.

### Registration

In `yadc/cli.py`:
```python
from . import cli_<name>
cli.add_command(cli_<name>.<name>)
```

## Existing commands

| CLI file | cmd package | Click group name |
|---|---|---|
| `cli_caption.py` | `core/captioning/` (shared with the API) | `caption` |
| `cli_cache.py` | `cmd/cache/` | `cache` |
| `cli_configs.py` | `cmd/configs/` | `configs` |
| `cli_draft.py` | (uses core directly) | `draft` |
| `cli_envs.py` | `cmd/envs/` | `envs` |
| `cli_export.py` | (uses core directly) | `export` |
| `cli_prompts.py` | `cmd/prompts/` | `prompts` |
| `cli_templates.py` | `cmd/templates/` | `templates` |

The `caption` command's pure logic lives in `yadc/core/captioning/` rather than a `cmd/caption/` package because the API also uses it (via `AsyncCaptionJob` and `CaptioningService`). See `captioning-runner` for the shared loop and `captioning-workflow` for the end-to-end flow.
