"""``KeyringKeyStorage`` — system keyring backend for the RSA private key."""

import base64
from typing import override

import keyring

from yadc.cmd import app
from yadc.core import logging

from .keystorage import KeyStorage

KEYRING_SERVICE = f"{app.NAME}_keys"
PRIVATE_KEY_KEYRING_KEY = "private_key_pem"

PUBLIC_KEY_PATH_OLD = app.CONFIG_PATH / "public_key.pem"
PUBLIC_KEY_PATH = app.STATE_PATH / "public_key.pem"

_logger = logging.get_logger(__name__)


class KeyringKeyStorage(KeyStorage):
    """Stores the private key in the system keyring (current default)."""

    @override
    def load_private_key(self) -> bytes | None:
        try:
            pem_data_encoded = keyring.get_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY)
            if pem_data_encoded is None:
                return None
            return base64.b64decode(pem_data_encoded)
        except Exception as e:
            _logger.error("Error: Failed to load private key from keyring: %s", e)
            return None

    @override
    def save_private_key(self, private_key_pem: bytes) -> None:
        private_pem_encoded = base64.b64encode(private_key_pem).decode()
        keyring.set_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY, private_pem_encoded)
        _logger.debug("Stored private key in keyring.")

    @override
    def delete_private_key(self) -> None:
        try:
            keyring.delete_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY)
        except Exception:
            pass

    @override
    def has_private_key(self) -> bool:
        return keyring.get_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY) is not None
