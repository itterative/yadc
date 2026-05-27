import base64
import secrets
from typing import override

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from yadc.cmd.config import (
    AppConfig,
    AppConfigKeyStoragePassword,
    save_config,
)
from yadc.core import logging

from .keystorage import KeyStorage

PBKDF2_ITERATIONS = 600_000
SALT_LENGTH = 16
NONCE_LENGTH = 12

_logger = logging.get_logger(__name__)


class PasswordRequiredError(ValueError):
    """Raised when a password-protected private key is accessed without a password."""


class PasswordKeyStorage(KeyStorage):
    """Stores the private key in the yadc config TOML, encrypted with AES-256-GCM.

    The key material lives under ``key_storage.password`` in the config file.
    When a password is given it is used for PBKDF2-HMAC-SHA256 key derivation.
    When *password* is ``None`` an empty string is used as the passphrase,
    so the private key is still encrypted on disk ( defence in depth ).
    """

    def __init__(self, config: AppConfig, password: str | None = None):
        self._config: AppConfig = config
        self._password: str | None = password

    @override
    def load_private_key(self) -> bytes | None:
        password_cfg = self._config.key_storage.password
        if password_cfg is None:
            return None

        stored = password_cfg.private_key
        if not stored:
            return None

        try:
            data = base64.b64decode(stored)
        except Exception:
            _logger.error("Error: private key is not valid base64.")
            return None

        if len(data) < SALT_LENGTH + NONCE_LENGTH:
            _logger.error("Error: password private key data is too short.")
            return None

        try:
            salt = data[:SALT_LENGTH]
            nonce = data[SALT_LENGTH : SALT_LENGTH + NONCE_LENGTH]
            ciphertext = data[SALT_LENGTH + NONCE_LENGTH :]

            key = self._derive_key(salt)
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            _logger.error("Error: failed to decrypt password-protected private key: %s", e)
            raise PasswordRequiredError("Password-protected private key requires a password to decrypt.")

    @override
    def save_private_key(self, private_key_pem: bytes) -> None:
        if self._config.key_storage.password is None:
            self._config.key_storage.password = AppConfigKeyStoragePassword(private_key="", public_key="")

        salt = secrets.token_bytes(SALT_LENGTH)
        nonce = secrets.token_bytes(NONCE_LENGTH)
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        payload = salt + nonce + aesgcm.encrypt(nonce, private_key_pem, None)

        self._config.key_storage.password.private_key = base64.b64encode(payload).decode()
        save_config(self._config)
        _logger.debug("Saved password-protected private key to config.")

    @override
    def delete_private_key(self) -> None:
        self._config.key_storage.password = None
        save_config(self._config)

    @override
    def has_private_key(self) -> bool:
        password_cfg = self._config.key_storage.password
        if password_cfg is None:
            return False
        return bool(password_cfg.private_key)

    def save_public_key(self, public_key_pem: bytes) -> None:
        """Store the public key PEM as base64 in the config."""
        if self._config.key_storage.password is None:
            self._config.key_storage.password = AppConfigKeyStoragePassword(private_key="", public_key="")
        self._config.key_storage.password.public_key = base64.b64encode(public_key_pem).decode()
        save_config(self._config)
        _logger.debug("Saved public key to config.")

    def load_public_key(self) -> bytes | None:
        """Load the public key PEM from the config (base64-encoded)."""
        password_cfg = self._config.key_storage.password
        if password_cfg is None:
            return None
        public_pem = password_cfg.public_key
        if not public_pem:
            return None
        return base64.b64decode(public_pem)

    def _derive_key(self, salt: bytes) -> bytes:
        password = self._password if self._password is not None else ""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        return kdf.derive(password.encode("utf-8"))
