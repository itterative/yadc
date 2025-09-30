import toml

from yadc.cmd import app

CONFIG_PATH = app.STATE_PATH / 'configs'

def merge_user_config(name: str, config: dict) -> dict:
    def _deep_merge(config_part, user_config_part):
        if isinstance(config_part, dict) and isinstance(user_config_part, dict):
            for key, value in user_config_part.items():
                if key in config_part:
                    config_part[key] = _deep_merge(config_part[key], value)
                else:
                    config_part[key] = value

        return config_part

    try:
        user_config = toml.loads(load_user_config(name))
    except Exception as e:
        raise ValueError(f'failed to load user config: {name}') from e

    return _deep_merge(config, user_config) # type: ignore

def load_user_config(name: str):
    config = CONFIG_PATH / f'{name}.toml'

    with open(config) as f:
        return f.read()

def save_user_config(name: str, content: str):
    CONFIG_PATH.mkdir(mode=0o750, exist_ok=True)

    config = CONFIG_PATH / f'{name}.toml'

    with open(config, 'w') as f:
        f.write(content)

def list_user_config():
    return [
        config.name.removesuffix('.toml') for config in CONFIG_PATH.glob('*.toml')
    ]

def delete_user_config(name: str):
    config = CONFIG_PATH / f'{name}.toml'

    if not config.exists():
        return False
    
    config.unlink()
    return True
