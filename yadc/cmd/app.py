import toml
import platformdirs

NAME = 'yadc'

CONFIG_PATH = platformdirs.user_config_path(NAME, ensure_exists=True)
STATE_PATH = platformdirs.user_state_path(NAME, ensure_exists=True)

def load_config() -> dict:
    config_path = CONFIG_PATH / 'config.toml'

    try:
        with open(config_path, 'r') as f:
            return toml.load(f)
    except FileNotFoundError:
        return {}
    except PermissionError:
        raise
    except Exception as e:
        raise ValueError('user config is invalid') from e
