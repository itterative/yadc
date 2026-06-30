import base64
import secrets

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from yadc.cmd.config import AppConfig, AppConfigKeyStorage, AppConfigKeyStoragePassword
from yadc.cmd.envs import keystorage_password as ks_password
from yadc.cmd.envs.keystorage_password import (
    NONCE_LENGTH,
    SALT_LENGTH,
    PasswordKeyStorage,
    PasswordRequiredError,
)


@pytest.fixture(autouse=True)
def _low_pbkdf2_iterations(monkeypatch):
    """Run PBKDF2 with 1 iteration instead of the production 600k.

    These tests only verify the encrypt/decrypt round-trip and error
    paths — not the KDF's computational cost — so the full iteration
    count (≈0.1s per derivation) is pure waste. Patched in the
    production module, which both ``PasswordKeyStorage.load_private_key``
    and the local ``_encrypt_pem`` helper read, so the two sides stay
    in sync.
    """
    monkeypatch.setattr(ks_password, "PBKDF2_ITERATIONS", 1)


def _encrypt_pem(pem: bytes, password: str) -> str:
    """Encrypt a PEM with the same scheme PasswordKeyStorage uses."""
    salt = secrets.token_bytes(SALT_LENGTH)
    nonce = secrets.token_bytes(NONCE_LENGTH)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        # Use the (patched-low) module constant so encrypt and decrypt stay
        # in sync — see the ``_low_pbkdf2_iterations`` autouse fixture below.
        iterations=ks_password.PBKDF2_ITERATIONS,
    )
    key = kdf.derive(password.encode("utf-8"))
    aesgcm = AESGCM(key)
    payload = salt + nonce + aesgcm.encrypt(nonce, pem, None)
    return base64.b64encode(payload).decode()


class TestPasswordKeyStorageLoad:
    """``PasswordKeyStorage.load_private_key`` — load and decrypt the PEM
    private key using the supplied password."""

    def test_load_with_correct_password(self):
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

    def test_load_with_empty_password(self):
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

    def test_load_with_wrong_password_raises(self):
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

    def test_load_with_short_data_returns_none(self):
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
