import platformdirs

APP_NAME = 'yadc'

CONFIG_PATH = platformdirs.user_config_path(APP_NAME, ensure_exists=True)
STATE_PATH = platformdirs.user_state_path(APP_NAME, ensure_exists=True)
