import copy
from typing import Any

from yadc.cmd import app
from yadc.utils import deep_merge
from yadc.utils.dict_utils import load_toml

CONFIG_PATH = app.STATE_PATH / "configs"


def merge_user_config(name: str, config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)

    try:
        user_config = load_toml(load_user_config(name))
    except Exception as e:
        raise ValueError(f"failed to load user config: {name}") from e

    return deep_merge(config, user_config, remove_none=True)


def load_user_config(name: str):
    config = CONFIG_PATH / f"{name}.toml"

    with open(config) as f:
        return f.read()


def save_user_config(name: str, content: str):
    CONFIG_PATH.mkdir(mode=0o750, exist_ok=True)

    config = CONFIG_PATH / f"{name}.toml"

    with open(config, "w") as f:
        f.write(content)


def list_user_config():
    return [config.name.removesuffix(".toml") for config in CONFIG_PATH.glob("*.toml")]


def delete_user_config(name: str):
    config = CONFIG_PATH / f"{name}.toml"

    if not config.exists():
        return False

    config.unlink()
    return True
