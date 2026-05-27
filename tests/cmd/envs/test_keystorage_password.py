import base64
import secrets

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from yadc.cmd.config import AppConfig, AppConfigKeyStorage, AppConfigKeyStoragePassword
from yadc.cmd.envs.keystorage_password import (
    NONCE_LENGTH,
    PBKDF2_ITERATIONS,
    SALT_LENGTH,
    PasswordKeyStorage,
    PasswordRequiredError,
)


def _encrypt_pem(pem: bytes, password: str) -> str:
    """Encrypt a PEM with the same scheme PasswordKeyStorage uses."""
    salt = secrets.token_bytes(SALT_LENGTH)
    nonce = secrets.token_bytes(NONCE_LENGTH)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    key = kdf.derive(password.encode("utf-8"))
    aesgcm = AESGCM(key)
    payload = salt + nonce + aesgcm.encrypt(nonce, pem, None)
    return base64.b64encode(payload).decode()


def test_load_with_correct_password():
    """Loading with the matching password returns the original PEM."""
    pem = b"-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----"
    encrypted = _encrypt_pem(pem, "sekrit")
    config = AppConfig(
        key_storage=AppConfigKeyStorage(
            mode="password",
            password=AppConfigKeyStoragePassword(private_key=encrypted, public_key=""),
        ),
        envs={},
    )
    storage = PasswordKeyStorage(config, password="sekrit")
    result = storage.load_private_key()
    assert result == pem


def test_load_with_empty_password():
    """Loading with empty-string password (no explicit password) returns the original PEM."""
    pem = b"-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----"
    encrypted = _encrypt_pem(pem, "")
    config = AppConfig(
        key_storage=AppConfigKeyStorage(
            mode="password",
            password=AppConfigKeyStoragePassword(private_key=encrypted, public_key=""),
        ),
        envs={},
    )
    storage = PasswordKeyStorage(config, password=None)
    result = storage.load_private_key()
    assert result == pem


def test_load_with_wrong_password_raises():
    """Loading with the wrong password raises PasswordRequiredError."""
    pem = b"-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----"
    encrypted = _encrypt_pem(pem, "correct")
    config = AppConfig(
        key_storage=AppConfigKeyStorage(
            mode="password",
            password=AppConfigKeyStoragePassword(private_key=encrypted, public_key=""),
        ),
        envs={},
    )
    storage = PasswordKeyStorage(config, password="wrong")
    with pytest.raises(PasswordRequiredError):
        storage.load_private_key()


def test_load_with_short_data_returns_none():
    """Loading garbage/short data returns None."""
    bad_data = base64.b64encode(b"\x00" * 4).decode()
    config = AppConfig(
        key_storage=AppConfigKeyStorage(
            mode="password",
            password=AppConfigKeyStoragePassword(private_key=bad_data, public_key=""),
        ),
        envs={},
    )
    storage = PasswordKeyStorage(config, password=None)
    result = storage.load_private_key()
    assert result is None
