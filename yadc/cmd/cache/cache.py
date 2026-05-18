import shutil
from datetime import datetime
from pathlib import Path

from yadc.cmd import app


def get_cache_dir() -> str:
    return str(app.CACHE_PATH)


def debug_log_dir(toml_path: str) -> Path:
    """Return the run directory for API debug logging."""
    name = _derive_dataset_name(toml_path)
    date_dir = datetime.now().strftime("%Y-%m-%d")
    return app.CACHE_PATH / "api-debug" / date_dir / name


def _derive_dataset_name(toml_path: str) -> str:
    raw = Path(toml_path).stem
    for prefix in ("dataset_",):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    return raw or "unknown"


def clean_cache() -> int:
    if not app.CACHE_PATH.exists():
        return 0

    bytes_freed = sum(f.stat().st_size for f in app.CACHE_PATH.rglob("*") if f.is_file())
    shutil.rmtree(app.CACHE_PATH)
    return bytes_freed
