---
name: file-based-private-key-plan
description: Complete history of replacing keyring-based RSA private key storage with a password-protected alternative. Tracks original intent, deviations, decisions, and all files touched.
---

# File-Based Private Key Plan — Historical Summary

## Status: Complete

All planned items are done. Remaining work (if any) is tracked in `todo.md`.

---

## Original Intent (d80d15d — plan created)

**Problem:** The existing RSA encryption for API tokens relied on the `keyring` library (via `pass` backend on Linux). This required a working GPG setup with `pinentry`, which frequently failed in non-interactive environments (tmux, headless servers, WSL without pinentry). When GPG decryption failed, captioning silently fell back to an empty token, producing confusing HTTP 401 errors.

**Goal:** Provide an alternative private key storage mechanism that does not depend on GPG/keyring. The private key would be stored in `~/.local/state/yadc/private_key.pem` with `0o600` permissions, optionally encrypted via PBKDF2 + AES-256-GCM. Users choose between `keyring` and `file` storage in both CLI and WebUI.

**Original design highlights:**
- One RSA key pair, moved between keyring and file when switching modes
- `KeyStorage` ABC with `KeyringStorage` and `FileStorage` implementations
- `key_storage_mode` stored in SQLite `SettingsService`
- CLI: `yadc envs key-mode get/set`
- API: `GET/PUT /envs/key-mode`
- WebUI: dropdown in SettingsDialog to switch modes

---

## Pre-Existing Foundation (before this plan)

Before the file-based private key work started, the env/config system had already been refactored once:

- **cd59d31** `refactor: common env` — extracted `yadc/core/env.py` with shared env constants
- **f837114** `refactor: change cli config to envs` — split `yadc/cli_config.py` (471 lines deleted) into:
  - `yadc/cli_envs.py` (new, 220 lines)
  - `yadc/cmd/envs/__init__.py` (new)
  - `yadc/cmd/envs/encryption.py` (new, 145 lines)
  - `yadc/cmd/envs/envs.py` (new, 131 lines)
  - `yadc/cmd/envs/setting.py` (new)
  - `yadc/cmd/envs/user_config.py` (new)
  - `yadc/cmd/app.py` (new, basic `load_config()`)

This gave us the `envs` module structure that the file-based key work would later build on.

---

## Deviation 1: Two Separate Key Pairs

**Original plan:** One RSA key pair, moved between keyring and file when switching modes.

**What we built:** Two **independent** RSA key pairs with **independent public keys**:
- **Keyring pair**: private key in system keyring (`yadc_keys/private_key_pem`), public key on disk (`~/.local/state/yadc/public_key.pem`)
- **Password pair**: private key in config TOML (`[key_storage.password]`), public key also in config TOML (base64)

The public keys are **not shared**. `_get_public_key()` is mode-aware. Decryption fails with a clear error if the wrong storage mode is active.

**Decision rationale:** The user explicitly requested separate keys. This avoids a complex migration path and means `password:`-prefixed values are fundamentally tied to the password key pair.

---

## Deviation 2: Keys Stored in Config TOML

**Original plan:** Private key stored in `~/.local/state/yadc/private_key.pem`.

**What we built:** Password-protected keys live **inside** `~/.config/yadc/config.toml` under `[key_storage.password]`.

**Decision rationale:** The user wanted everything in the config file. The entire config is already `0o600`, so no additional file permission management is needed.

---

## Deviation 3: `AppConfig` Pydantic Model

**Original plan:** Raw dict manipulation via `app.load_config()` / `app.save_config()`.

**What we built:** Full Pydantic model hierarchy in `yadc/cmd/config.py` (`AppConfig` → `AppConfigKeyStorage` → `AppConfigEnv` → `AppConfigEnvValue`). Versioned loading (v0 legacy → v1) with `0o600` on write.

---

## Deviation 4: `AppConfigEnvValue` — Per-Field Encryption State

**Original plan:** Prefix parsing everywhere, bare values assumed legacy keyring.

**Problem discovered:** Non-encrypted fields like `api_url` were mistaken for legacy ciphertext, causing `envs get api_url` to try decryption and `envs show` to redact it.

**What we built:** Each env field is wrapped in an `AppConfigEnvValue` that carries its own encryption state:

```python
class AppConfigEnvValue(pydantic.BaseModel):
    value: str | None = None
    method: Literal["none", "keyring", "password"] = "none"

    @property
    def is_encrypted(self) -> bool:
        return self.method != "none"
```

`AppConfigEnv` fields are **never nullable** — always an `AppConfigEnvValue` object. Prefix parsing is centralized in `load_config()` / `save_config()` only.

---

## Deviation 5: `KeyStorage` Split into Separate Files

**Original plan:** All in `encryption.py`.

**What we built:** Modular files:

| File | Contents |
|---|---|
| `keystorage.py` | `KeyStorage` ABC |
| `keystorage_keyring.py` | `KeyringKeyStorage` |
| `keystorage_password.py` | `PasswordKeyStorage` — PBKDF2 + AES-256-GCM, config TOML |
| `encryption.py` | Active storage management, RSA encrypt/decrypt, key-mode switch |
| `envs.py` | Env CRUD using `AppConfig` |

`PasswordKeyStorage` always stores the private key in encrypted payload format (empty-string PBKDF2 when no password is set). Never writes plaintext PEM.

---

## Later Additions (after initial backend refactor)

### Cleanup Pass

The envs CLI and `envs.py` module underwent a cleanup pass to remove redundant constants (`ENV_KEYS` → `AppConfigEnv.model_fields`, `ENCRYPTED_KEYS` → `AppConfigEnv.ENCRYPTED_FIELDS`), simplify `None` checks (`or AppConfigEnv()` instead of `if env_config is None`), and unify error handling (`STATUS_USER_ERROR` for recoverable failures).

### Environment Value Reveal

Added per-field value reveal in the Environment Settings UI so users can view encrypted tokens (and any other env value) on demand.

- **Backend:** `POST /envs/<name>/reveal` — accepts `{ key: string, password?: string }`, returns `{ value: string }`. Works for any field regardless of encryption state. If password-encrypted and no password provided, returns `403 PASSWORD_REQUIRED`.
- **Frontend:** `envs.ts` — added `revealEnvValue(name, key, password?)` helper. `EnvironmentSettings.svelte` — API Token input has an eye-icon toggle.

**Bug fixes during implementation:**
- Dialog not in DOM: `PasswordPromptDialog` was inside `{#if !editingEnv}` so didn't exist when editing. Moved outside the conditional.
- Close-event race: dialog's native `close` event fired synchronously and called `cancelPassword` before `submitPassword` resolved. Fixed by resolving promise *before* closing.
- Closure type capture: `envName` narrowed to `string` but TS saw it as `string | undefined` inside `doReveal` closure. Fixed by passing `envName` as parameter.
- Cancel cleanup: `cancelEditEnv` resets `showToken`, `showPasswordPrompt`, and rejects pending password promise.

### Global Password Prompt

Consolidated `PasswordPromptDialog` and `withPasswordRetry` into a single global system to avoid duplicated state/logic across pages.

- **`passwordPrompt.ts`** — new global store: `requestPassword()`, `submitPassword()`, `cancelPassword()`, `passwordPromptOpen` store, `withPasswordRetry(action)`, `PasswordPromptCancelled`
- **`+layout.svelte`** — renders `PasswordPromptDialog` once at root level
- **`+page.svelte`** — removed ~40 lines of local prompt state
- **`EnvironmentSettings.svelte`** — `toggleTokenReveal` uses `withPasswordRetry`
- **Concurrent-call deduplication:** `requestPassword` keeps a module-level `pendingPromise`. Multiple concurrent callers share one dialog and one user input.

### `YADC_PASSWORD` Environment Warning

Added warning in Security Settings UI when `YADC_PASSWORD` env var is set on backend. Changing password in UI without updating env var silently has no effect.

- **Backend:** `GET /envs/key-mode` returns `env_password_set: bool`. `YADC_PASSWORD` centralized in `yadc/core/env.py`.
- **Frontend:** `SecuritySettings.svelte` displays yellow warning banner.
- **Tests:** `tests/api/test_envs.py` — added `test_get_key_mode_env_password_not_set` and `test_get_key_mode_env_password_set`.

---

## Files Touched

### Added

| File | Commit | Notes |
|---|---|---|
| `yadc/cmd/config.py` | e7314b7 | `AppConfig`, `AppConfigKeyStorage`, `AppConfigEnv`, `AppConfigEnvValue`, `save_config()`, `load_config()` |
| `yadc/cmd/envs/keystorage.py` | e7314b7 | `KeyStorage` ABC |
| `yadc/cmd/envs/keystorage_keyring.py` | e7314b7 | `KeyringKeyStorage` |
| `yadc/cmd/envs/keystorage_password.py` | e7314b7 | `PasswordKeyStorage` |
| `yadc/webui/src/lib/stores/sessionPassword.ts` | 975315d | Tab-scoped password persistence |
| `yadc/webui/src/lib/stores/storageStore.ts` | 975315d | Generic local/session storage store |
| `yadc/webui/src/lib/stores/passwordPrompt.ts` | 48791e7 | Global password prompt store + `withPasswordRetry()` |
| `yadc/webui/src/lib/components/dialogs/EnvironmentSettings.svelte` | cbec470 | Split from SettingsDialog |
| `yadc/webui/src/lib/components/dialogs/GeneralSettings.svelte` | cbec470 | Split from SettingsDialog |
| `yadc/webui/src/lib/components/dialogs/SecuritySettings.svelte` | cbec470 | Split from SettingsDialog |
| `tests/cli/test_cli_envs.py` | 2aaff5b | CLI integration tests for envs |
| `tests/cmd/envs/test_keystorage_password.py` | (existing area, updated) | Always-encrypted key tests |

### Deleted

| File | Commit | Notes |
|---|---|---|
| `yadc/cli_config.py` | f837114 | Replaced by `yadc/cli_envs.py` (220 lines vs 471 lines) |

### Moved / Refactored

| File | Change | Notes |
|---|---|---|
| `yadc/cmd/app.py` | Cleaned | `save_config()` moved to `yadc/cmd/config.py` |
| `yadc/webui/src/lib/components/dialogs/SettingsDialog.svelte` | Split | Environment/General/Security settings extracted into own components |

### Modified (selected significant ones)

| File | Notes |
|---|---|
| `yadc/cmd/envs/encryption.py` | Active storage, encrypt/decrypt, key-mode switch, `change_password()` |
| `yadc/cmd/envs/envs.py` | Env CRUD using `AppConfig`; defensive `isinstance` checks |
| `yadc/cmd/envs/__init__.py` | Updated exports |
| `yadc/cli_envs.py` | Uses `AppConfig`; unified error codes; `key-mode get/set` commands |
| `yadc/core/env.py` | Centralized `YADC_PASSWORD` |
| `yadc/api/controllers/api_envs.py` | Uses `AppConfig`; `GET/PUT /envs/key-mode`; `POST /envs/<name>/reveal`; `env_password_set` flag |
| `yadc/api/services/captioning.py` | `CaptionJobOptions.password`; `error_messages` |
| `yadc/api/events.py` | `CaptioningStatusEvent.error_messages` |
| `yadc/api/controllers/api_captioning.py` | `ErrorCode` usage; pre-flight `load_env()` |
| `yadc/api/controllers/api_configs.py` | `ErrorCode` usage |
| `yadc/api/controllers/api_datasets.py` | `ErrorCode` usage |
| `yadc/api/controllers/api_export.py` | `ErrorCode` usage |
| `yadc/api/controllers/api_templates.py` | `ErrorCode` usage |
| `yadc/api/controllers/models_errors.py` | `code: str \| None` field |
| `yadc/api/controllers/utils_json.py` | `ErrorCode(StrEnum)` |
| `yadc/webui/src/lib/stores/captionOptions.ts` | Added `password?` |
| `yadc/webui/src/lib/stores/datasetImages.ts` | Auto-includes session password |
| `yadc/webui/src/lib/stores/events.ts` | `error_messages` in schema |
| `yadc/webui/src/lib/stores/toasts.ts` | `details?: string[]` |
| `yadc/webui/src/lib/stores/envs.ts` | `revealEnvValue()` helper; `fetchKeyMode` with `env_password_set` |
| `yadc/webui/src/lib/api.ts` | `PasswordRequiredError`, `code` detection |
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Password retry logic |
| `yadc/webui/src/routes/+layout.svelte` | Global `PasswordPromptDialog` |
| `tests/cli/conftest.py` | Isolated temp dir fixture |
| `tests/api/test_envs.py` | `env_password_set` tests |
| `tests/api/test_captioning.py` | Single-image job tests |
| `pyproject.toml` | `keyrings.alt` test dependency |

---

## Architecture Summary (End State)

```
~/.config/yadc/config.toml          (0o600)
├── version = 1
├── [key_storage]
│   ├── mode = "keyring" | "password"
│   └── [key_storage.password]       (only when mode=password)
│       ├── private_key = "base64(salt+nonce+ciphertext)"
│       └── public_key = "base64(-----BEGIN PUBLIC KEY-----...)"
└── [envs.<name>]
    ├── api_url = "..."
    ├── api_token = "keyring:<ciphertext>" | "password:<ciphertext>"
    └── api_model_name = "..."

Runtime (in-memory):
    AppConfigEnv(
        api_url=AppConfigEnvValue(value="...", method="none"),
        api_token=AppConfigEnvValue(value="<ciphertext>", method="keyring"),
        api_model_name=AppConfigEnvValue(value="...", method="none"),
    )

~/.local/state/yadc/
└── public_key.pem                   (keyring mode public key ONLY)

System keyring:
└── yadc_keys / private_key_pem      (keyring mode private key)
```

---

## Key Decisions Log

1. **Separate key pairs** — Chosen to avoid migration complexity and keep each prefix type strictly bound to its key pair.
2. **Config TOML over separate file** — Simpler file management; config already has correct permissions.
3. **Pydantic models over raw dicts** — Type safety, validation, and clean serialization.
4. **`AppConfigEnvValue` wrapper** — Solved the "bare URL mistaken for ciphertext" bug; eliminated prefix parsing outside config layer.
5. **Always-encrypted `PasswordKeyStorage`** — Even with empty password, stores PEM in AES-GCM payload format (empty-string PBKDF2). No plaintext PEM on disk.
6. **Global password prompt store** — Eliminated duplicate dialog instances and close-event races across pages.
7. **Concurrent prompt deduplication** — Module-level `pendingPromise` ensures one dialog serves all concurrent callers.
8. **`YADC_PASSWORD` env warning** — Prevents silent misconfiguration where UI password changes have no effect because env var takes precedence.
