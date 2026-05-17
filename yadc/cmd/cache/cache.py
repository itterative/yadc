import shutil

from yadc.cmd import app


def get_cache_dir() -> str:
    return str(app.CACHE_PATH)


def clean_cache() -> int:
    if not app.CACHE_PATH.exists():
        return 0

    bytes_freed = sum(f.stat().st_size for f in app.CACHE_PATH.rglob("*") if f.is_file())
    shutil.rmtree(app.CACHE_PATH)
    return bytes_freed
