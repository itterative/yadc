import os
import toml
import pydantic
import platformdirs

from yadc.core.user_config import UserConfig, UserConfigApi

# NOTE: storing tokens in plain text is not great, but using keyrings comes with some caveats

APP_NAME = 'yadc'
CONFIG_NAME = 'config.toml'

ALL_KEYS = [
    'api_url', 'api_token', 'api_model_name',
]

# flag for enabling user config
FLAG_USER_CONFIG = False # disabled, needs further testing

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
        print('Warning: failed to restrict permissions on user config:', e)

def load_config():
    if not FLAG_USER_CONFIG:
        return UserConfig(api=UserConfigApi())

    try:
        config_toml = _load_config_toml()

        return UserConfig(
            api=UserConfigApi(
                url=config_toml.get('api_url', ''),
                token=config_toml.get('api_token', ''),
                model_name=config_toml.get('api_model_name', ''),
            ),
        )
    except pydantic.ValidationError|ValueError:
        print('Warning: user config is invalid')
    except PermissionError:
        print('Warning: failed to read user config: permission denied')
    except Exception as e:
        print('Warning: failed to read user config:', e)

    return UserConfig(api=UserConfigApi())

def config_get(key: str):
    if key not in ALL_KEYS:
        print('Error: invalid setting:', key)
        return 3

    config_toml = _load_config_toml()
    value = config_toml.get(key, None)

    if value is None:
        print('Error: key not found:', value)
        return 3

    print(value)

    return 0

def config_set(key: str, value: str, force: bool = False):
    if key not in ALL_KEYS:
        print('Error: invalid setting:', key)
        return 3

    try:
        config_toml = _load_config_toml()
    except pydantic.ValidationError|ValueError:
        print('Warning: user config is invalid')

        if not force:
            return 1

        config_toml = {}
    except PermissionError:
        print('Error: failed to read user config: permission denied')
        return 1
    except Exception as e:
        print('Error: failed to read user config:', e)
        return 1

    config_toml[key] = value

    try:
        _save_config_toml(config_toml)
        print(f'User config {key} has been updated.')
    except Exception as e:
        print('Error: user config could not be updated:', e)
        return 1

    return 0

def config_delete(key: str, force: bool = False):
    try:
        config_toml = _load_config_toml()
    except pydantic.ValidationError|ValueError:
        if not force:
            print('Error: user config is invalid')
            return 1

        print('Warning: user config is invalid')
        config_toml = {}
    except PermissionError:
        print('Error: failed to read user config: permission denied')
        return 1
    except Exception as e:
        print('Error: failed to read user config:', e)
        return 1

    config_toml.pop(key, None)

    try:
        _save_config_toml(config_toml)
        print(f'User config {key} has been updated.')
    except Exception as e:
        print('Error: user config could not be updated:', e)
        return 1

    return 0

def config_clear():
    try:
        _save_config_toml({})
        print(f'User config has been cleared.')
    except Exception as e:
        print('Error: user config could not be updated:', e)
        return 1

    return 0

def config_list():
    try:
        config_toml = _load_config_toml()
    except pydantic.ValidationError|ValueError:
        print('Error: user config is invalid')
        return 1
    except PermissionError:
        print('Error: failed to read user config: permission denied')
        return 1
    except Exception as e:
        print('Error: failed to read user config:', e)
        return 1

    found_keys = False
    
    for key in ALL_KEYS:
        value = config_toml.get(key, None)

        if value is None:
            continue

        found_keys = True
        print(f'{key} = {value}')

    if not found_keys:
        print('No user config values found.')

    return 0
