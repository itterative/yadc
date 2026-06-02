---
name: paths-and-storage
description: File system paths used by yadc (platformdirs) and file storage conventions for DatasetImage persistence.
---

# Paths and Storage

## Platformdirs Paths (cmd/app.py)

| Constant | Path | Purpose |
|----------|------|---------|
| `CONFIG_PATH` | `~/.config/yadc/` | User config.toml with envs, encryption settings |
| `STATE_PATH` | `~/.local/state/yadc/` | Templates, user configs, public key |
| `CACHE_PATH` | `~/.cache/yadc/` | API response cache, debug logs |

## User Config (CONFIG_PATH)

- `config.toml` — environments with `[env.<name>]` sections containing `api_url`, `api_token` (encrypted), `api_model_name`
- `public_key.pem` — **migrated to STATE_PATH** (old location still checked for migration)

## State Files (STATE_PATH)

- `templates/*.jinja` — user prompt templates
- `configs/*.toml` — user config overlays (deep-merged into dataset configs)
- `public_key.pem` — RSA public key for token encryption

## Cache Files (CACHE_PATH)

- `api_requests/` — HTTP response cache (SHA256 keyed JSON files)
- `api-debug/{date}/{dataset}/` — debug JSONL logs (see debug-api-logging memory)

## DatasetImage File Conventions

For an image `photo.jpg`:

| File | Extension | Purpose |
|------|-----------|---------|
| `photo.txt` | `caption_suffix` (configurable, default `.txt`) | Caption text |
| `photo.toml` | `toml_suffix` (default `.toml`) | Metadata extras (artist, tags, etc.) |
| `photo.toml~` | backup | Previous TOML state before update |
| `photo.history~` | `history_suffix` | Versioned TOML snapshots (separated by `----------` marker) |
| `photo.gemma.draft~` | `.{name}.draft~` | Named draft caption |
| `photo.qwen.draft~` | `.{name}.draft~` | Another named draft |

## Encryption

API tokens are encrypted with RSA-OAEP (SHA-256). Private key stored in system keyring (`yadc_keys` service), public key on disk. Uses `keyring` + `keyring-pass` packages + `cryptography` (transitive dep).
