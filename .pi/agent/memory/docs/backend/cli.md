---
name: backend/cli
description: CLI entry points — click commands at yadc/ root (cli_*.py modules) plus the main entry point and click group registration.
category: architecture
---

# Backend: CLI

The `yadc/` root and click-based CLI commands.

```
yadc/
  __init__.py         # version
  __main__.py         # entry point → yadc.cli:cli
  cli.py              # click group registration, version command

  cli_caption.py      # main captioning command (largest CLI module)
  cli_cache.py        # cache management CLI
  cli_common.py       # shared click options (--log-level, --env)
  cli_configs.py      # user configs management CLI
  cli_draft.py        # draft save/list/show/remove CLI
  cli_envs.py         # environment settings CLI
  cli_export.py       # export captions to training formats
  cli_logging.py      # ClickHandler — routes logs through click.secho
  cli_templates.py    # template management CLI
  cli_webui.py        # web UI CLI (yadc webui serve)
```

CLI commands import pure logic from `yadc/cmd/` (see `cmd.md`). The Click/Pythonic split is documented in `cli-cmd-structure`.
