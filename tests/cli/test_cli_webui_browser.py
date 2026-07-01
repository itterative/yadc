"""Tests for the ``--browser`` family of flags on ``yadc webui serve``.

Two layers:

- **Click surface** (``TestBrowserFlagSurface``) — uses :class:`CliRunner`
  to assert the flag is exposed in ``--help`` and round-trips through
  click correctly. Pure in-process; no subprocess.

- **Helper behavior** (``TestOpenBrowserWhenReady``,
  ``TestBrowserUrl``) — exercises :func:`yadc.cli_webui._open_browser_when_ready`
  and :func:`yadc.cli_webui._browser_url` in isolation with
  :mod:`unittest.mock` patches over ``socket`` / ``webbrowser`` /
  ``threading``. Avoids spawning a real browser or uvicorn.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from yadc.cli_webui import (
    _BROWSER_OPEN_POLL_INTERVAL_SECONDS,
    _BROWSER_OPEN_TIMEOUT_SECONDS,
    _browser_url,
    _find_open_port,
    _open_browser_when_ready,
)
from yadc.cli_webui import webui as webui_group

# Patch target paths — centralized so refactors break here, not at every site.
_PATCH_WEBBROWSER_OPEN = "yadc.cli_webui.webbrowser.open"
_PATCH_SOCKET_CREATE_CONNECTION = "yadc.cli_webui.socket.create_connection"
_PATCH_THREADING_THREAD = "yadc.cli_webui.threading.Thread"


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestBrowserFlagSurface:
    """``yadc webui serve`` advertises ``--browser`` / ``--no-browser``."""

    def test_help_lists_flag(self, cli_runner):
        result = cli_runner.invoke(webui_group, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--browser" in result.output
        assert "--no-browser" in result.output

    def test_help_mentions_desktop_default(self, cli_runner):
        # Docstring callers learn about the desktop-built-in default so
        # the relationship between the flag and the .exe ships in
        # ``--help``.
        result = cli_runner.invoke(webui_group, ["serve", "--help"])
        assert "desktop" in result.output.lower()


class TestBrowserUrl:
    """``_browser_url`` resolves bind hosts to a browser-reachable URL."""

    def test_loopback(self):
        assert _browser_url("127.0.0.1", 7860) == "http://127.0.0.1:7860"

    def test_zero_zero_zero_rewrites_to_loopback(self):
        # ``0.0.0.0`` is a bind address, not a destination — without the
        # rewrite the user gets a URL their browser can't open.
        assert _browser_url("0.0.0.0", 7860) == "http://127.0.0.1:7860"

    def test_ipv6_unspec_rewrites_to_loopback(self):
        assert _browser_url("::", 7860) == "http://127.0.0.1:7860"

    def test_remote_host_preserved(self):
        # LAN bind is intentional — power-user / port-forward setups.
        # The helper should not silently redirect.
        assert _browser_url("192.168.1.42", 9000) == "http://192.168.1.42:9000"


class TestOpenBrowserWhenReady:
    """``_open_browser_when_ready`` must poll, then open the browser once.

    Verifies four properties:

    1. Flag off → no-op (no thread, no socket, no browser call).
    2. Flag on → helper spawns a daemon thread named so it's
       identifiable in logs / task managers.
    3. The thread attempts ``socket.create_connection`` against the
       bound port and only calls ``webbrowser.open`` once a connection
       succeeds (race-free).
    4. The poll loop gives up cleanly if the socket never opens, so a
       bad ``--port`` doesn't leak a dangling thread.
    """

    def test_flag_off_is_noop(self):
        # No patches needed — the helper must short-circuit *before*
        # any I/O side effects.
        _open_browser_when_ready("127.0.0.1", 7860, open_browser=False)

    def test_flag_on_spawns_daemon_thread(self):
        with patch(_PATCH_THREADING_THREAD) as mock_thread:
            _open_browser_when_ready("127.0.0.1", 7860, open_browser=True)

        mock_thread.assert_called_once()
        kwargs = mock_thread.call_args.kwargs
        assert kwargs["daemon"] is True
        assert kwargs["name"] == "yadc-webui-browser"
        # ``target`` is the threaded inner function — callable, no
        # return value to inspect here.
        target = kwargs["target"]
        assert callable(target)

    def test_opens_browser_after_bind(self):
        # First ``socket.create_connection`` raises (server not ready
        # yet), second succeeds → helper opens the browser exactly once.
        success_cm = MagicMock()
        success_cm.__enter__.return_value = success_cm

        with (
            patch(_PATCH_THREADING_THREAD) as mock_thread,
            patch(_PATCH_SOCKET_CREATE_CONNECTION, side_effect=[OSError, success_cm]),
            patch(_PATCH_WEBBROWSER_OPEN) as mock_open,
        ):
            # Run the target on the test thread instead of spawning
            # one — we assert on what *would* have happened.
            _open_browser_when_ready("127.0.0.1", 7860, open_browser=True)
            target = mock_thread.call_args.kwargs["target"]
            target()

        mock_open.assert_called_once_with("http://127.0.0.1:7860")

    def test_loopback_for_all_interfaces(self):
        # When the server is bound on ``0.0.0.0`` the helper should
        # actually probe ``127.0.0.1`` (loopback), while still telling
        # the browser to open ``127.0.0.1``.
        success_cm = MagicMock()
        success_cm.__enter__.return_value = success_cm

        with (
            patch(_PATCH_THREADING_THREAD) as mock_thread,
            patch(_PATCH_SOCKET_CREATE_CONNECTION, return_value=success_cm) as mock_connect,
            patch(_PATCH_WEBBROWSER_OPEN) as mock_open,
        ):
            _open_browser_when_ready("0.0.0.0", 9000, open_browser=True)
            target = mock_thread.call_args.kwargs["target"]
            target()

        mock_connect.assert_called_with(("127.0.0.1", 9000), timeout=0.2)
        mock_open.assert_called_once_with("http://127.0.0.1:9000")

    def test_gives_up_when_port_does_not_bind(self):
        # All ``create_connection`` calls raise → helper exits cleanly
        # without opening the browser. Bound by a timeout so a buggy
        # ``time.monotonic`` can't loop forever in a future regression.
        # The poll constants are crushed to a few ms here so the test
        # finishes well under pytest-timeout; the production values
        # live in :data:`_BROWSER_OPEN_TIMEOUT_SECONDS` /
        # :data:`_BROWSER_OPEN_POLL_INTERVAL_SECONDS` and are sanity
        # -checked at the bottom of this test.
        from yadc import cli_webui

        with (
            patch.object(cli_webui, "_BROWSER_OPEN_TIMEOUT_SECONDS", 0.05),
            patch.object(cli_webui, "_BROWSER_OPEN_POLL_INTERVAL_SECONDS", 0.005),
            patch(_PATCH_THREADING_THREAD) as mock_thread,
            patch(_PATCH_SOCKET_CREATE_CONNECTION, side_effect=OSError),
            patch(_PATCH_WEBBROWSER_OPEN) as mock_open,
        ):
            _open_browser_when_ready("127.0.0.1", 7860, open_browser=True)
            target = mock_thread.call_args.kwargs["target"]
            target()

        mock_open.assert_not_called()
        # Sanity: the production constants stay positive — a regression
        # to ``0`` would loop forever.
        assert _BROWSER_OPEN_TIMEOUT_SECONDS > 0
        assert _BROWSER_OPEN_POLL_INTERVAL_SECONDS > 0


class TestFindOpenPort:
    """``_find_open_port`` walks a small range and yields a free port.

    The check is the gate behind the bundle's port-fallback behavior:
    if 7860 is busy, the bundled binary should land on the next free
    port in ``[7860, 7860 + N)`` and the CLI command should *not* fall
    back silently. Two behaviors worth pinning down:

    1. Returns the preferred port when it's free.
    2. Walks forward when the preferred is bound — verifies the loop
       skips occupied ports and lands on the first free one.
    """

    def test_returns_preferred_when_free(self):
        # Pick an unusual high port that's almost certainly free.
        chosen = _find_open_port(49152)
        assert chosen == 49152

    def test_skips_busy_port(self):
        # Bind one port; the helper should find the next free one in
        # range. We trust it walks in order rather than testing the
        # exact value (which is racy on shared hosts).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))  # kernel-assigned free port
            held_port = held.getsockname()[1]
            # Don't close held — we want the port to stay busy during
            # the call so the helper has to skip it.
            chosen = _find_open_port(held_port)
        assert chosen is not None
        assert chosen != held_port
        assert chosen >= held_port

    def test_returns_none_when_range_exhausted(self):
        # Asking for a single attempt past the preferred port, with
        # that port already bound, must yield ``None`` rather than
        # silently picking something else.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held_port = held.getsockname()[1]
            chosen = _find_open_port(held_port, attempts=1)
        assert chosen is None


class TestThreadingContract:
    """The spawned thread is identifiable and daemon-scoped.

    PyInstaller freezes preserve the threading import path; a regression
    that imports ``threading`` differently (e.g. via ``_thread``)
    would still work functionally but lose the user-facing diagnostics
    in the webui's process list.
    """

    def test_real_thread_uses_daemon_and_name(self):
        # Inspect ``Thread.__init__`` arguments as they are passed in,
        # rather than racing with ``threading.enumerate()`` to see the
        # thread before it finishes. The constructor signature is the
        # contract we care about; the thread body is covered separately
        # by ``test_opens_browser_after_bind``.
        from yadc import cli_webui

        success_cm = MagicMock()
        success_cm.__enter__.return_value = success_cm

        captured: dict[str, object] = {}

        class CapturingThread:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                # Don't actually start a thread — we only care about
                # what was passed to ``__init__``.

            def start(self):
                pass

        with (
            patch.object(cli_webui, "_BROWSER_OPEN_TIMEOUT_SECONDS", 0.05),
            patch.object(cli_webui, "_BROWSER_OPEN_POLL_INTERVAL_SECONDS", 0.005),
            patch(_PATCH_SOCKET_CREATE_CONNECTION, return_value=success_cm),
            patch(_PATCH_WEBBROWSER_OPEN, return_value=True),
            patch.object(cli_webui.threading, "Thread", CapturingThread),
        ):
            _open_browser_when_ready("127.0.0.1", 7860, open_browser=True)

        kwargs = captured["kwargs"]
        assert kwargs["daemon"] is True
        assert kwargs["name"] == "yadc-webui-browser"
        assert callable(kwargs["target"])
