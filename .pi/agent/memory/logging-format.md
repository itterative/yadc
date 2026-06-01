---
name: logging-format
description: Structured log message format used across the yadc API — sentence-style message followed by optional [key=value, ...] block, with positional %-formatting (never f-strings).
category: convention
priority: 2
---

# Logging format

The yadc API uses a consistent log message format that's easy to grep and
parse. Match this style when adding new log calls.

## Shape

```python
self._logger.debug("Watching existing dataset. [dataset=%s, paths=%d]", name, len(paths))
```

Two parts:

1. **Human-readable sentence** — ends with a period. Tells you what
   happened in plain English. Keep it short and concrete.
2. **Structured `[key=value, ...]` block** — comma-separated, each pair
   uses `key=value`. Provides machine-parseable context for grepping
   and log aggregation. Omit the block entirely if there's no
   structured data to add (e.g. `"Config for dataset '%s' deleted."`).

## Formatting rules

- **Use positional `%`-formatting** (`%s`, `%d`, `%.1f`, etc.), never
  f-strings. The arguments are passed separately so the logger can
  skip formatting entirely when the level is filtered out. This is
  the standard Python logging idiom; the codebase follows it
  uniformly.
- **Keys are identifiers** (`dataset`, `job_id`, `path`, `id`,
  `unexpected`) — short, no spaces, no quotes around them.
- **Values are formatted via `%`-specifiers**, not by Python's
  `repr()` or `str()` on objects. For floats, pick the precision you
  need (e.g. `%.1f` for seconds).
- **One log line per event** — don't emit multiple lines for the
  same thing. Use a single `debug` call with a multi-line message
  only when the second line is genuinely a separate fact.

## Examples

```python
self._logger.info("Dataset watcher started. [debounce=%.1fs]", debounce)
self._logger.debug("Watching existing dataset. [dataset=%s, paths=%d]", name, len(paths))
self._logger.debug("Auto-rescanning dataset due to filesystem change. [dataset=%s]", event.dataset_name)
self._logger.warning("Failed to parse config at %s: %s", config_path, e)
self._logger.warning("Failed to refresh dataset id=%d: %s", dataset_id, e)
self._logger.info("Config for dataset '%s' updated.", name)
```

## Anti-patterns

```python
# ✗ f-string in logger call — forces formatting even when filtered
self._logger.debug(f"Watching dataset {name} with {len(paths)} paths")

# ✗ key="value" with quotes (JSON-ish) — codebase doesn't use this style
self._logger.debug('Watching dataset. {"dataset": "%s"}', name)

# ✗ trailing colon without the structured block
self._logger.debug("Watching dataset: %s", name)
```

If you're adding a logger call, follow the established pattern — the
grep-friendly `[key=value]` block is what makes our logs analyzable
in production.
