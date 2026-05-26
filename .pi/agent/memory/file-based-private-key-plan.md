---
name: file-based-private-key-plan
description: Plan for replacing keyring-based RSA private key storage with a file-based alternative protected by an optional user-provided password.
---

# File-Based Private Key Plan

## Motivation

The current RSA encryption for API tokens relies on the `keyring` library (via the `pass` backend on Linux). This requires a working GPG setup with `pinentry`, which frequently fails in non-interactive environments (e.g., tmux sessions, headless servers, WSL without a working pinentry). When GPG decryption fails, captioning silently falls back to an empty token, producing confusing HTTP 401 errors.

## Goal

Provide an alternative private key storage mechanism that does not depend on GPG/keyring. The private key is stored in a file with restrictive permissions (`0o600`), optionally encrypted with a user-provided password (via PBKDF2 + AES). Users choose between keyring and file-based storage in both CLI and WebUI.

## High-Level Design

### 1. Storage Modes

| Mode | Private Key Location | Encryption | PIN Required |
|---|---|---|---|
| `keyring` (current default) | `keyring.get_password(...)` | RSA via GPG | GPG pinentry |
| `file` (new) | `~/.local/state/yadc/private_key.pem` | AES-256-GCM with PBKDF2 | User-provided password |

### 2. New Models / Types

```python
class KeyStorageMode(enum.StrEnum):
    KEYRING = "keyring"
    FILE = "file"
```

### 3. CLI Changes

- Add `yadc envs key-mode` command to switch between `keyring` and `file`
- When switching to `file`, prompt for a password (or allow empty for unencrypted storage — **not recommended**, but useful for dev)
- When switching to `keyring`, migrate the private key back to keyring and delete the file
- `envs set` / `envs show` should work transparently regardless of mode

### 4. WebUI Changes

- **SettingsDialog → Environments tab**: Add a "Key Storage" section with a dropdown (`keyring` / `file`)
- When switching to `file`, show a password prompt (optional: allow empty for unencrypted)
- When an env has a saved token and the key storage mode is `keyring`, show a warning: "Keyring storage requires a working GPG pinentry. If captioning fails with 401, switch to file-based storage."
- Backend: Add `PUT /envs/key-mode` endpoint, `GET /envs/key-mode` endpoint

### 5. Backend Changes

- `yadc/cmd/envs/encryption.py`:
  - Add `KeyStorage` abstract base class with `load_private_key()` / `save_private_key()` methods
  - Implement `KeyringStorage` (current behavior)
  - Implement `FileStorage` with PBKDF2 + AES-256-GCM encryption
  - Add `get_storage()` factory that reads `settings.key_storage_mode` from SQLite settings
  - Update `_get_private_key()` and `_generate_key_pair()` to use the active storage

- `yadc/api/modules/settings.py` (SettingsService):
  - Add `key_storage_mode` getter/setter
  - Default to `keyring` for backward compatibility

- `yadc/api/controllers/api_envs.py`:
  - `GET /envs/key-mode` → returns `{ mode: "keyring" | "file" }`
  - `PUT /envs/key-mode` → accepts `{ mode, password? }`, migrates the key, returns success/error

### 6. Migration Strategy

- On first run after upgrade, mode defaults to `keyring` (backward compatible)
- If `keyring` load fails during env decryption, raise `EnvDecryptionError` with a message that suggests switching to file-based storage
- No automatic migration — user must explicitly switch modes via CLI or WebUI

### 7. Security Considerations

- File-based private key file: `0o600` permissions, stored in `STATE_PATH`
- Password-derived key: PBKDF2-HMAC-SHA256, 600k iterations, random 16-byte salt
- AES-256-GCM with random 12-byte nonce
- If no password is provided, store the PEM plaintext with `0o600` (dev convenience, warn user)
- The password is NEVER stored; user must enter it when switching modes or when the file is first created

### 8. Files to Touch

| File | Change |
|---|---|
| `yadc/cmd/envs/encryption.py` | Add `KeyStorage` abstraction, `FileStorage`, `get_storage()` |
| `yadc/cmd/envs/envs.py` | Update `load_env` error message to suggest file-based storage |
| `yadc/cli_envs.py` | Add `key-mode` command |
| `yadc/api/modules/settings.py` | Add `key_storage_mode` setting |
| `yadc/api/controllers/api_envs.py` | Add `GET/PUT /envs/key-mode` endpoints |
| `yadc/webui/src/lib/stores/envs.ts` | Add key-mode types + API helpers |
| `yadc/webui/src/lib/components/dialogs/SettingsDialog.svelte` | Add Key Storage section |
| `yadc/webui/src/lib/components/settings/EnvSelector.svelte` | Show keyring warning |

## Deferred Details

- Exact PBKDF2 iteration count (600k is a reasonable starting point)
- Whether to allow empty password (yes, with a warning)
- Whether to cache the decrypted private key in memory (yes, via `@functools.cache` like today)
- Whether to support multiple key pairs (no, single key pair per installation)
