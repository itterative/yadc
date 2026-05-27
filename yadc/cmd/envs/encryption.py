import base64
import shutil
from typing import Literal

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from yadc.cmd import app
from yadc.cmd.config import AppConfig
from yadc.cmd.config import load_config as _load_app_config
from yadc.core import logging
from yadc.core.env import YADC_PASSWORD

from .keystorage import KeyStorage
from .keystorage_keyring import KeyringKeyStorage
from .keystorage_password import PasswordKeyStorage, PasswordRequiredError
from .setting import EncryptionMethod

PUBLIC_KEY_PATH_OLD = app.CONFIG_PATH / "public_key.pem"
PUBLIC_KEY_PATH = app.STATE_PATH / "public_key.pem"

_logger = logging.get_logger(__name__)


def _resolve_password(password: str | None) -> str | None:
    """Return *password* if given, otherwise fall back to ``YADC_PASSWORD`` env var."""
    if password is not None:
        return password
    return YADC_PASSWORD


def create_storage(config: AppConfig | None = None, password: str | None = None) -> KeyStorage:
    """Create the appropriate ``KeyStorage`` for *config*.

    Defaults to ``KeyringKeyStorage`` when *config* is ``None`` or mode is
    not ``password``.
    """
    if config is None:
        config = _load_app_config()
    if config.key_storage.mode == "password":
        return PasswordKeyStorage(config, password=password)
    return KeyringKeyStorage()


def get_storage(password: str | None = None) -> KeyStorage:
    """Return the active storage backend.

    When *password* is ``None`` and the current mode is ``password``,
    falls back to the ``YADC_PASSWORD`` environment variable.
    """
    config = _load_app_config()
    resolved = _resolve_password(password)
    return create_storage(config, password=resolved)


def _active_storage_method() -> EncryptionMethod:
    storage = get_storage()
    if isinstance(storage, PasswordKeyStorage):
        return EncryptionMethod.PASSWORD
    return EncryptionMethod.KEYRING


def _get_storage_for_method(method: EncryptionMethod, password: str | None = None) -> KeyStorage:
    resolved = _resolve_password(password)
    config = _load_app_config()
    if method == EncryptionMethod.KEYRING:
        return KeyringKeyStorage()
    if method == EncryptionMethod.PASSWORD:
        return PasswordKeyStorage(config, password=resolved)
    raise ValueError(f"Unsupported encryption method: {method}")


def _load_private_key_for_storage(storage: KeyStorage) -> rsa.RSAPrivateKey | None:
    pem_data = storage.load_private_key()
    if pem_data is None:
        return None
    try:
        key = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
        if isinstance(key, rsa.RSAPrivateKey):
            return key
        _logger.error("Error: Loaded private key is not an RSA private key.")
        return None
    except Exception as e:
        _logger.error("Error: Failed to load private key: %s", e)
        return None


def _load_public_key_for_storage(storage: KeyStorage) -> rsa.RSAPublicKey | None:
    if isinstance(storage, PasswordKeyStorage):
        pem_data = storage.load_public_key()
        if pem_data is None:
            _generate_key_pair_for_storage(storage)
            pem_data = storage.load_public_key()
    else:
        if PUBLIC_KEY_PATH.exists():
            with open(PUBLIC_KEY_PATH, "rb") as f:
                pem_data = f.read()
        elif PUBLIC_KEY_PATH_OLD.exists():
            shutil.move(PUBLIC_KEY_PATH_OLD, PUBLIC_KEY_PATH)
            with open(PUBLIC_KEY_PATH, "rb") as f:
                pem_data = f.read()
        else:
            _generate_key_pair_for_storage(storage)
            with open(PUBLIC_KEY_PATH, "rb") as f:
                pem_data = f.read()

    if pem_data is None:
        return None

    try:
        key = serialization.load_pem_public_key(pem_data, backend=default_backend())
        if isinstance(key, rsa.RSAPublicKey):
            return key
        _logger.error("Error: Public key is not an RSA public key.")
        return None
    except Exception as e:
        _logger.error("Error: Failed to load public key: %s", e)
        return None


def change_password(
    config: AppConfig,
    new_password: str | None = None,
    old_password: str | None = None,
) -> None:
    """Change the password for password-mode key storage.

    Re-encrypts the existing private key with *new_password* without
    changing the RSA key pair, so no env token re-encryption is needed.
    """
    from yadc.cmd.config import save_config

    if config.key_storage.mode != "password":
        raise ValueError("Can only change password when key storage mode is 'password'.")

    old_storage = PasswordKeyStorage(config, password=old_password)
    pem_data = old_storage.load_private_key()
    if pem_data is None:
        raise ValueError("No private key found.")

    new_storage = PasswordKeyStorage(config, password=new_password)
    new_storage.save_private_key(pem_data)
    save_config(config)


def set_key_mode(
    config: AppConfig,
    mode: Literal["keyring", "password"],
    password: str | None = None,
    old_password: str | None = None,
) -> None:
    """Switch key storage *mode* and ensure a key pair exists.

    Re-encrypts existing environment tokens so they remain usable.

    *password* is the password for the **new** mode (when switching TO password).
    *old_password* is the password for the **current** mode (when switching FROM password).
    """
    from yadc.cmd.config import AppConfigKeyStoragePassword, save_config

    old_method = _active_storage_method()
    new_method = EncryptionMethod.PASSWORD if mode == "password" else EncryptionMethod.KEYRING

    if old_method == new_method:
        if mode == "password" and password is not None:
            change_password(config, password, old_password)
        return

    # 1. Ensure old key pair is available for decryption
    old_storage = _get_storage_for_method(old_method, password=old_password)
    _generate_key_pair_for_storage(old_storage)

    # 2. Prepare new storage (config not yet updated, so create directly)
    if mode == "password":
        if config.key_storage.password is None:
            config.key_storage.password = AppConfigKeyStoragePassword(private_key="", public_key="")
        new_storage = PasswordKeyStorage(config, password=password)
    else:
        new_storage = KeyringKeyStorage()

    # 3. Generate new key pair for encryption
    _generate_key_pair_for_storage(new_storage)

    # 4. Re-encrypt existing tokens
    _reencrypt_env_tokens(config, new_method, old_password)

    # 5. Update config mode and persist
    config.key_storage.mode = mode  # type: ignore[assignment]
    save_config(config)


def _reencrypt_env_tokens(
    config: AppConfig,
    new_method: EncryptionMethod,
    old_password: str | None = None,
) -> None:
    """Re-encrypt all encrypted env tokens to *new_method*."""
    from yadc.cmd.config import save_config

    changed = False

    for env_data in config.envs.values():
        token = env_data.api_token
        if token.value is None or token.method == "none":
            continue

        method = EncryptionMethod(token.method)
        decrypted = decrypt_setting(token.value, method=method, password=old_password)
        if decrypted is None:
            _logger.warning("Failed to re-encrypt token for env '%s'; leaving as-is", env_data.api_url.value or "?")
            continue

        token.value = encrypt_setting(decrypted, method=new_method)
        token.method = new_method.value
        changed = True

    if changed:
        save_config(config)
        _logger.info("Re-encrypted environment tokens for %s storage.", new_method.value)


def _generate_key_pair_for_storage(storage: KeyStorage) -> None:
    """Generate an RSA key pair for *storage* if one doesn't already exist."""
    if storage.has_private_key():
        return

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    storage.save_private_key(private_pem)
    _logger.debug("Generated new RSA key pair.")

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    if isinstance(storage, PasswordKeyStorage):
        storage.save_public_key(public_pem)
        _logger.debug("Saved public key to config.")
    else:
        PUBLIC_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(public_pem)
        _logger.debug("Saved public key to disk.")


def reset_encryption():
    if PUBLIC_KEY_PATH_OLD.exists():
        PUBLIC_KEY_PATH_OLD.unlink()

    if PUBLIC_KEY_PATH.exists():
        PUBLIC_KEY_PATH.unlink()

    get_storage().delete_private_key()


def encrypt_setting(value: str, method: EncryptionMethod | None = None) -> str:
    if method is None:
        method = _active_storage_method()

    if method == EncryptionMethod.NONE:
        return value

    storage = _get_storage_for_method(method, password=None)
    public_key = _load_public_key_for_storage(storage)
    if public_key is None:
        raise ValueError(f"Public key not available for {method} encryption.")

    encrypted = public_key.encrypt(
        value.encode("utf-8"),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )

    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_setting(encrypted_token: str, method: EncryptionMethod | None = None, password: str | None = None) -> str | None:
    if method is None:
        method = _active_storage_method()

    if method == EncryptionMethod.NONE:
        return encrypted_token

    storage = _get_storage_for_method(method, password=password)
    private_key = _load_private_key_for_storage(storage)
    if private_key is None:
        _logger.error("Error: Private key not available for %s. Cannot decrypt setting.", method)
        return None

    try:
        encrypted_data = base64.b64decode(encrypted_token)
        decrypted = private_key.decrypt(
            encrypted_data,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        return decrypted.decode("utf-8")
    except PasswordRequiredError:
        raise
    except Exception as e:
        _logger.error("Error: Failed to decrypt setting with %s: %s", method, e)
        return None
