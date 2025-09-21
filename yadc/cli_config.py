from typing import Optional, Dict, Any

import os
import toml
import base64
import pydantic
import platformdirs
import functools

import keyring
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from yadc.core import logging
from yadc.core.user_config import UserConfig, UserConfigApi

# NOTE: storing tokens in plain text is not great, but using keyrings comes with some caveats

APP_NAME = 'yadc'
CONFIG_NAME = 'config.toml'
CONFIG_PATH = platformdirs.user_config_path(APP_NAME) / CONFIG_NAME

KEYRING_SERVICE = f"{APP_NAME}_keys"
PRIVATE_KEY_KEYRING_KEY = "private_key_pem"
PUBLIC_KEY_PATH = platformdirs.user_config_path(APP_NAME) / "public_key.pem"

ENV_KEYS = [ 'api_url', 'api_token', 'api_model_name', ]
ENCRYPTED_KEYS = [ 'api_token_encrypted' ]

_logger = logging.get_logger(__name__)

def _generate_key_pair():
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
    if not PUBLIC_KEY_PATH.exists():
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


def _encrypt_setting(value: str) -> str:
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

def _decrypt_setting(encrypted_token: str) -> Optional[str]:
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


def _load_config_toml() -> dict:
    config_path = platformdirs.user_config_path(APP_NAME) / CONFIG_NAME

    try:
        with open(config_path, 'r') as f:
            return toml.load(f)
    except FileNotFoundError:
        return {}
    except PermissionError:
        raise
    except Exception as e:
        raise ValueError('user config is invalid') from e

def _save_config_toml(config: dict):
    config_path = platformdirs.user_config_path(APP_NAME, ensure_exists=True) / CONFIG_NAME

    with open(config_path, 'w') as f:
        toml.dump(config, f)

    try:
        os.chmod(config_path, 0o600)
    except Exception as e:
        _logger.warning('Warning: failed to restrict permissions on user config: %s', e)


def _get_env_config(config_toml: dict, env: str = "default") -> Dict[str, Any]:
    env_section = config_toml.get("env", {}).get(env, {})

    return {
        "api_url": env_section.get("api_url", ""),
        "api_token_encrypted": env_section.get("api_token", ""),  # encrypted
        "api_model_name": env_section.get("api_model_name", ""),
    }

def _set_env_config(config_toml: dict, env: str, key: str, value: str):
    if "env" not in config_toml:
        config_toml["env"] = {}

    if env not in config_toml["env"]:
        config_toml["env"][env] = {}

    config_toml["env"][env][key] = value

def load_config(env: str = "default") -> UserConfig:
    try:
        config_toml = _load_config_toml()
        default_env_config = _get_env_config(config_toml, "default")
        env_config = _get_env_config(config_toml, env)

        config = {}
        for key in set(default_env_config.keys()) | set(env_config.keys()):
            value = env_config.get(key , '')
            value = value or default_env_config.get(key , '')

            if not value:
                continue

            if key.endswith('_encrypted'):
                key = key.removesuffix('_encrypted')
                value = _decrypt_setting(value)

                if value is None:
                    _logger.warning("Warning: Could not decrypt api_token for env '%s'", env)
                    continue

            config[key] = value

        return UserConfig(
            api=UserConfigApi(
                url=config.get('api_url', ''),
                token=config.get('api_token', ''),
                model_name=config.get('api_model_name', ''),
            ),
        )
    except pydantic.ValidationError|ValueError:
        _logger.warning('Warning: user config is invalid')
    except PermissionError:
        _logger.warning('Warning: failed to read user config: permission denied')
    except Exception as e:
        _logger.warning('Warning: failed to read user config: %s', e)

    return UserConfig(api=UserConfigApi())

def config_get(key: str, env: str = "default"):
    if key not in ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        return 3

    key_encrypted = key + '_encrypted'

    try:
        config_toml = _load_config_toml()
        env_config = _get_env_config(config_toml, env)

        value = env_config.get(key)
        value = value or env_config.get(key_encrypted)

        if not value:
            _logger.error('Error: key not found: %s (env: %s)', key, env)
            return 3

        # Special handling for token
        if key_encrypted in ENCRYPTED_KEYS:
            decrypted = _decrypt_setting(value)
            if decrypted is None:
                _logger.error("Error: failed to decrypted setting.")
                return 1
            value = decrypted

        print(value)
        return 0

    except Exception as e:
        _logger.error('Error: failed to read config: %s', e)
        return 1

def config_set(key: str, value: str, env: str = "default", force: bool = False):
    if key not in ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        return 3

    key_encrypted = key + '_encrypted'

    try:
        config_toml = _load_config_toml()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error('Error: user config is invalid')
            return 1
        config_toml = {}
    except Exception as e:
        _logger.error('Error: failed to read user config: %s', e)
        return 1

    if key_encrypted in ENCRYPTED_KEYS:
        try:
            encrypted_value = _encrypt_setting(value)
            _set_env_config(config_toml, env, key, encrypted_value)
        except Exception as e:
            _logger.error("Error: Failed to encrypt token: %s", e)
            return 1
    else:
        _set_env_config(config_toml, env, key, value)

    try:
        _save_config_toml(config_toml)
        _logger.info("User config %s (env: %s) has been updated.", key, env)
    except Exception as e:
        _logger.error("Error: user config could not be updated: %s", e)
        return 1

    return 0

def config_delete(key: str, env: str = "default", force: bool = False):
    if key not in ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        return 3

    try:
        config_toml = _load_config_toml()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error('Error: user config is invalid')
            return 1
        config_toml = {}
    except Exception as e:
        _logger.error('Error: failed to read user config: %s', e)
        return 1

    if "env" in config_toml and env in config_toml["env"]:
        config_toml["env"][env].pop(key, None)

    try:
        _save_config_toml(config_toml)
        _logger.info("User config %s (env: %s) has been deleted.", key, env)
    except Exception as e:
        _logger.error("Error: user config could not be updated: %s", e)
        return 1

    return 0

def config_clear(env: str = "default", all_envs: bool = False):
    if all_envs:
        try:
            if CONFIG_PATH.exists():
                CONFIG_PATH.unlink()

            if PUBLIC_KEY_PATH.exists():
                PUBLIC_KEY_PATH.unlink()
        except PermissionError:
            _logger.error("Error: user config could not be cleared: permission denied")
            return 1
        except Exception as e:
            _logger.error("Error: user config could not be cleared: %s", e)
            return 1

        _logger.info("User config cleared.")
        return 0
    
    try:
        config_toml = _load_config_toml()

        if env is None:
            config_toml = {}
        else:
            if "env" in config_toml:
                config_toml["env"].pop(env, None)

        _save_config_toml(config_toml)
        _logger.info("User config cleared. (env: %s)", env)
    except PermissionError:
        _logger.error("Error: user config could not be cleared: permission denied")
        return 1
    except Exception as e:
        _logger.error("Error: user config could not be cleared: %s", e)
        return 1
    return 0

def config_list_envs():
    try:
        config_toml = _load_config_toml()
    except Exception as e:
        _logger.error('Error: failed to read user config: %s', e)
        return 1

    envs: list[str] = config_toml.get("env", {}).keys()

    if not envs:
        _logger.warning("No environment found in user config.")
        return 0

    for env in envs:
        print(env)

    return 0

def config_list(env: str = "default"):
    try:
        config_toml = _load_config_toml()
    except Exception as e:
        _logger.error('Error: failed to read user config: %s', e)
        return 1

    found = False

    # buffer so error and warnings show at the beginning
    buffer = ''

    env_config = _get_env_config(config_toml, env)

    for key, value in env_config.items():
        if not value:
            continue

        if not key in ENV_KEYS and not key in ENCRYPTED_KEYS:
            continue

        # Decrypt token for display
        if key.endswith('_encrypted'):
            key = key.removesuffix('_encrypted')
            decrypted = _decrypt_setting(value)
            if decrypted:
                buffer += f"{key} = [REDACTED]\n"
            else:
                buffer += f"{key} = [DECRYPTION FAILED]\n"
        else:
            buffer += f"{key} = {value}\n"

        found = True

    if not found:
        _logger.warning("No user config values found.")
        return 0

    print(buffer.strip())

    return 0
