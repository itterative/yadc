---
name: backend/cmd
description: Pure logic under yadc/cmd/ — no click imports. CLI commands (cli_*.py) import from here.
category: architecture
---

# Backend: Pure Logic (`yadc/cmd/`)

Pure logic modules — no click imports. CLI commands (`cli_*.py` files in `yadc/` root) import from here. The split is documented in `cli-cmd-structure`.

```
yadc/cmd/
  app.py            # paths (CONFIG_PATH, STATE_PATH, CACHE_PATH via platformdirs), load_config()
  config.py         # AppConfig Pydantic model hierarchy (v0→v1 TOML migration), save_config()
  status.py         # exit codes: STATUS_OK=0, STATUS_ERROR=1, STATUS_USER_ERROR=2
  cache/            # cache dir helpers, clean_cache()
  configs/          # user config CRUD, deep merge
  envs/             # env loading/saving, RSA encryption via keyring or password
    encryption.py   # active KeyStorage management, RSA encrypt/decrypt, key-mode switch
    envs.py         # env CRUD using AppConfig
    setting.py      # Setting base class, EncryptionMethod enum
    keystorage.py   # KeyStorage ABC (load/save private key, generate key pair)
    keystorage_keyring.py   # KeyringKeyStorage — system keyring private key
    keystorage_password.py  # PasswordKeyStorage — config-TOML private key, PBKDF2 + AES-256-GCM
    user_config.py  # UserConfig / UserConfigApi models (legacy)
  templates/        # user template CRUD in STATE_PATH/templates/
```
