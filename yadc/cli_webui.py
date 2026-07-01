"""``yadc webui`` — start the web UI server."""

import socket
import sys
import threading
import time
import webbrowser

import click

from yadc.api.application import Application
from yadc.api.configuration import Configuration
from yadc.cmd.app import CACHE_PATH
from yadc.core import logging

# Bounded so a misconfigured ``--port`` doesn't leave a dangling helper thread.
_BROWSER_OPEN_TIMEOUT_SECONDS = 5.0
_BROWSER_OPEN_POLL_INTERVAL_SECONDS = 0.1

# Frozen (desktop .exe) mode falls back to the next free port in this range when
# the requested one is busy. CLI mode keeps loud-failure behavior so an explicit
# ``--port`` is honored literally.
_DESKTOP_PORT_FALLBACK_ATTEMPTS = 20


def _browser_url(host: str, port: int) -> str:
    """Resolve the URL the browser should open.

    Binds to all-interfaces hosts (``0.0.0.0`` / ``::``) are rewritten to
    ``127.0.0.1`` so the URL is reachable from the same machine that
    started the server — without this, the user would have to look up
    their LAN address to open the UI.
    """
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{bind_host}:{port}"


def _find_open_port(preferred: int, attempts: int = _DESKTOP_PORT_FALLBACK_ATTEMPTS) -> int | None:
    """Return a free TCP port on 127.0.0.1 starting from ``preferred``.

    Walks ``[preferred, preferred + attempts)`` and returns the first that
    binds. There's a small TOCTOU window between this check and uvicorn's
    bind — acceptable for the common case (a stale server from a previous
    double-click) without the cost of pre-binding. Returns ``None`` if the
    whole range is busy.
    """
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def _open_browser_when_ready(host: str, port: int, open_browser: bool) -> None:
    """Spawn a daemon thread that opens the webui once the server accepts connections.

    Polls the bind socket so the browser opens *after* uvicorn is ready
    rather than racing the listener. Returns silently when the flag is
    off or the server fails to bind within :data:`_BROWSER_OPEN_TIMEOUT_SECONDS`.
    """
    if not open_browser:
        return

    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = _browser_url(host, port)

    def _run() -> None:
        deadline = time.monotonic() + _BROWSER_OPEN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((bind_host, port), timeout=0.2):
                    break
            except OSError:
                time.sleep(_BROWSER_OPEN_POLL_INTERVAL_SECONDS)
        else:
            return
        webbrowser.open(url)

    threading.Thread(target=_run, daemon=True, name="yadc-webui-browser").start()


@click.group("webui")
def webui():
    """Launch the yadc web UI."""
    pass


@webui.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=7860, type=int, help="Bind port")
@click.option("--cors/--no-cors", default=True, help="Enable CORS (for development)")
@click.option("--banner/--no-banner", default=True, help="Show startup banner")
@click.option(
    "--browser/--no-browser",
    default=False,
    help=("Open the web UI in your default browser once the server is ready. Off by default for the CLI; the desktop .exe enables it via its entry point."),
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Set the logging level",
)
@click.option(
    "--access-log-file",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "File path for the per-request access log. Default: ~/.cache/yadc/webui-access.log "
        "(auto-rotated). When set, the file is watched instead (defer rotation to your "
        "external setup, e.g. logrotate)."
    ),
)
def serve(host: str, port: int, cors: bool, banner: bool, browser: bool, log_level: str, access_log_file: str | None):
    """Start the web UI server."""
    import logging as _logging

    # NOTE: temporary until we can properly merge the two logging systems (cli vs webui)
    logging.set_level("ERROR")

    # Frozen desktop .exe: a busy port usually means a previous double-click
    # is still running, so fall back to the next free port. CLI mode keeps
    # loud-failure behavior so an explicit ``--port`` is honored.
    if getattr(sys, "frozen", False):
        original_port = port
        chosen = _find_open_port(original_port)
        if chosen is None:
            click.echo(
                f"No free port in {original_port}..{original_port + _DESKTOP_PORT_FALLBACK_ATTEMPTS - 1}; trying the requested port anyway.",
                err=True,
            )
        elif chosen != original_port:
            click.echo(
                f"Port {original_port} is in use; falling back to {chosen}.\n",
                err=True,
            )
            port = chosen

    if access_log_file is None:
        access_log_file = str(CACHE_PATH / "webui-access.log")
        access_log_user_specified = False
    else:
        access_log_user_specified = True

    configuration = Configuration(
        http_host=host,
        http_port=port,
        api_cors_enable=cors,
        banner_enable=banner,
        logging_default_level=getattr(_logging, log_level.upper(), _logging.INFO),
        access_log_file=access_log_file,
        access_log_user_specified=access_log_user_specified,
    )
    application = Application(configuration)
    _open_browser_when_ready(host, port, browser)
    application.run()
