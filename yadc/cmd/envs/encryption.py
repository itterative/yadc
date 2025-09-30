from typing import Optional

import base64
import shutil
import functools

import keyring
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from yadc.core import logging
from yadc.cmd import app

KEYRING_SERVICE = f"{app.NAME}_keys"

PRIVATE_KEY_KEYRING_KEY = "private_key_pem"
PUBLIC_KEY_PATH_OLD = app.CONFIG_PATH / "public_key.pem"
PUBLIC_KEY_PATH = app.STATE_PATH / "public_key.pem"


_logger = logging.get_logger(__name__)

def _generate_key_pair():
    if PUBLIC_KEY_PATH.exists():
        return

    # note: migrate old path
    if PUBLIC_KEY_PATH_OLD.exists():
        shutil.move(PUBLIC_KEY_PATH_OLD, PUBLIC_KEY_PATH)
        return

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    private_pem_encoded = base64.b64encode(private_pem).decode()

    keyring.set_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY, private_pem_encoded)
    _logger.debug("Generated and stored private key in keyring.")

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    PUBLIC_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC_KEY_PATH, 'wb') as f:
        f.write(public_pem)

    _logger.debug("Saved public key to disk.")

@functools.cache
def _get_private_key() -> Optional[rsa.RSAPrivateKey]:
    try:
        pem_data_encoded = keyring.get_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY)
        if pem_data_encoded is None:
            return None

        pem_data = base64.b64decode(pem_data_encoded)

        key = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())

        if isinstance(key, rsa.RSAPrivateKey):
            return key
        else:
            _logger.error("Error: Key in keyring is not an RSA private key.")
            return None
    except Exception as e:
        _logger.error("Error: Failed to load private key from keyring: %s", e)
        return None

@functools.cache
def _get_public_key() -> Optional[rsa.RSAPublicKey]:
    _generate_key_pair()

    try:
        with open(PUBLIC_KEY_PATH, 'rb') as f:
            pem_data = f.read()

        key = serialization.load_pem_public_key(pem_data, backend=default_backend())

        if isinstance(key, rsa.RSAPublicKey):
            return key
        else:
            _logger.error("Error: Public key on disk is not an RSA public key.")
            return None
    except Exception as e:
        _logger.error("Error: Failed to load public key: %s", e)
        return None


def reset_encryption():
    if PUBLIC_KEY_PATH_OLD.exists():
        PUBLIC_KEY_PATH_OLD.unlink()

    if PUBLIC_KEY_PATH.exists():
        PUBLIC_KEY_PATH.unlink()

    keyring.delete_password(KEYRING_SERVICE, PRIVATE_KEY_KEYRING_KEY)


def encrypt_setting(value: str) -> str:
    _generate_key_pair()

    public_key = _get_public_key()
    if public_key is None:
        raise ValueError("Public key not available for encryption.")

    encrypted = public_key.encrypt(
        value.encode('utf-8'),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return base64.b64encode(encrypted).decode('utf-8')

def decrypt_setting(encrypted_token: str) -> Optional[str]:
    private_key = _get_private_key()
    if private_key is None:
        _logger.error("Error: Private key not found in keyring. Cannot decrypt setting.")
        return None

    try:
        encrypted_data = base64.b64decode(encrypted_token)
        decrypted = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode('utf-8')
    except Exception as e:
        _logger.error("Error: Failed to decrypt setting: %s", e)
        return None
