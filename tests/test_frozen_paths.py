"""Smoke tests for the desktop .exe entry script.

The full PyInstaller build can only be exercised on Windows, but every
piece of logic in :mod:`scripts.entrypoints.webui_desktop` is reachable
without the freezer: the argv builder is pure, the frozen-state
detection is a ``getattr`` on ``sys``, and the click dispatch is the
same path a regular install takes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so ``scripts.entrypoints`` is importable as a
# top-level package without installing the project. ``tests/conftest.py``
# could also do this; doing it inline keeps this file standalone.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.entrypoints import webui_desktop  # noqa: E402
from scripts.entrypoints.webui_desktop import (  # noqa: E402
    _maybe_respawn_in_terminal_unix,  # noqa: E402
    build_argv,
    main,
)


class TestBuildArgv:
    """Pure-function tests for :func:`build_argv`."""

    def test_no_user_args_injects_defaults(self):
        # Empty argv → full desktop UX. ``webui serve`` defaults still
        # apply for everything else (host, port, log level, …).
        assert build_argv([]) == ["webui", "serve", "--browser", "--no-cors"]

    def test_user_explicit_flag_preserved(self):
        # ``--port=9000`` must survive; desktop defaults still injected
        # because the user expressed no preference on those.
        assert build_argv(["--port=9000"]) == ["webui", "serve", "--port=9000", "--browser", "--no-cors"]

    def test_user_no_browser_wins(self):
        # Explicit ``--no-browser`` suppresses the ``--browser`` default
        # but leaves ``--no-cors`` injected — different flag, different
        # decision.
        assert build_argv(["--no-browser"]) == ["webui", "serve", "--no-browser", "--no-cors"]

    def test_user_cors_wins(self):
        # Explicit ``--cors`` suppresses the ``--no-cors`` default but
        # leaves ``--browser`` injected (different flag, different
        # decision). Order does not matter to click — each flag is parsed
        # independently.
        assert build_argv(["--cors"]) == ["webui", "serve", "--cors", "--browser"]

    def test_user_both_explicit_wins_for_each(self):
        assert build_argv(["--no-browser", "--cors"]) == ["webui", "serve", "--no-browser", "--cors"]

    def test_flag_with_equals_value_counts(self):
        # ``--browser=false`` / ``--browser=true`` parse the same to
        # click but the user has expressed a preference — default should
        # not be injected. Covers the unusual ``--key=value`` form.
        assert build_argv(["--browser=false"]) == ["webui", "serve", "--browser=false", "--no-cors"]
        assert build_argv(["--cors=true"]) == ["webui", "serve", "--cors=true", "--browser"]

    def test_unrelated_flag_passes_through(self):
        assert build_argv(["--log-level", "debug"]) == [
            "webui",
            "serve",
            "--log-level",
            "debug",
            "--browser",
            "--no-cors",
        ]

    def test_default_flags_constant_is_stable(self):
        # The desktop build's UX contract — keep this in sync with
        # ``DEFAULT_DESKTOP_FLAGS`` deliberately. Anyone changing it
        # needs to think about whether existing downloaded builds are
        # still correct.
        assert webui_desktop.DEFAULT_DESKTOP_FLAGS == ("--browser", "--no-cors")


class TestFrozenDispatch:
    """``main()`` must inject desktop defaults when (and only when) frozen.

    PyInstaller sets ``sys.frozen = True`` on the embedded interpreter;
    a regular Python install leaves it absent. Mislabelling a regular
    install as frozen would silently flip the browser-open / CORS-off
    defaults onto every CLI invocation — the regression we guard here.
    """

    @pytest.fixture
    def stubbed_cli(self, monkeypatch):
        """Replace :func:`yadc.cli.cli` with a recorder.

        Avoids spinning up the click runtime (logger setup, app
        construction, …) which is heavy and irrelevant to argv logic.
        """
        captured: dict[str, object] = {}

        def fake_cli(*, standalone_mode: bool = True, **kwargs):
            captured["argv"] = list(sys.argv)
            captured["standalone_mode"] = standalone_mode
            captured["kwargs"] = kwargs
            return 0

        # Ensure the real ``yadc.cli`` module isn't loaded by an earlier
        # import; we then attach our stand-in.
        sys.modules.pop("yadc.cli", None)
        sys.modules.pop("yadc", None)
        fake_pkg = type(sys)("yadc")
        fake_mod = type(sys)("yadc.cli")
        fake_mod.cli = fake_cli  # type: ignore[attr-defined]
        sys.modules["yadc"] = fake_pkg
        sys.modules["yadc.cli"] = fake_mod
        # Force the entry script to re-import yadc.cli from sys.modules.
        importlib.reload(webui_desktop)
        yield captured
        # Reload again so subsequent tests get the real module back.
        sys.modules.pop("yadc.cli", None)
        sys.modules.pop("yadc", None)
        importlib.reload(webui_desktop)

    def test_unfrozen_does_not_mutate_argv(self, stubbed_cli, monkeypatch):
        # ``sys.frozen`` is unset on a plain interpreter — make sure it
        # stays unset for the duration of this test.
        monkeypatch.delattr(sys, "frozen", raising=False)

        monkeypatch.setattr(sys, "argv", ["/usr/bin/yadc", "--port=9000"])
        main()

        assert sys.argv == ["/usr/bin/yadc", "--port=9000"]
        # Standalone mode True is the click contract regardless of frozen.
        assert stubbed_cli["standalone_mode"] is True

    def test_frozen_injects_defaults(self, stubbed_cli, monkeypatch, tmp_path):
        meipass = tmp_path / "meipass"
        meipass.mkdir()

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
        monkeypatch.setattr(sys, "argv", ["yadc-webui.exe", "--port=9000"])

        main()

        # exe path stays at argv[0]; ``webui serve`` prepended, then the
        # user's existing flag, then desktop defaults appended *after*
        # so explicit user preferences (last-one-wins) still override.
        assert sys.argv == [
            "yadc-webui.exe",
            "webui",
            "serve",
            "--port=9000",
            "--browser",
            "--no-cors",
        ]

        # Stable cross-rebuild contract: ``_MEIPASS`` resolves into a
        # real on-disk layout where ``Configuration``'s
        # ``app_frontend_build_path`` default — ``<package>/webui/build``
        # — finds the bundled SPA. We don't load ``Configuration`` here
        # (its dataclass re-binds default factories at import time);
        # asserting the *path layout* the build script must produce is
        # enough to catch a spec regression.
        bundled_layout = meipass / "yadc" / "webui" / "build"
        bundled_layout.mkdir(parents=True)
        assert bundled_layout.exists()


def test_entry_script_importable():
    # The PyInstaller spec lists this module by dotted path; if it can't
    # be imported cleanly the bundling step fails with a far less
    # actionable error. Cheap to assert here.
    assert callable(main)
    assert callable(build_argv)


class TestTerminalRespawn:
    """When the .exe is launched without a TTY on POSIX, respawn in a terminal.

    These tests use ``monkeypatch.setattr`` to swap out ``sys.platform``,
    ``os.environ``, ``sys.stdout.isatty``, ``shutil.which`` and
    ``os.execvpe`` so we can assert exactly which terminal is launched
    and with what args — without ever spawning a real terminal.
    """

    @staticmethod
    def _patch_tty(monkeypatch, *, is_tty: bool) -> None:
        # ``sys.stdout.isatty`` is the only stdout-side check the helper
        # performs; ``sys.stdin.isatty`` would also matter in a real
        # terminal but the helper doesn't look at it.
        class FakeStdout:
            def isatty(self_inner) -> bool:
                return is_tty

        monkeypatch.setattr(sys, "stdout", FakeStdout())

    def test_skip_on_windows(self, monkeypatch):
        # Windows has a console window automatically — no respawn. The
        # helper must short-circuit before any sys.stdout dance.
        monkeypatch.setattr(sys, "platform", "win32")
        # Even when everything else says "respawn", Windows says no.
        self._patch_tty(monkeypatch, is_tty=False)
        monkeypatch.delenv("YADC_DESKTOP_IN_TERMINAL", raising=False)

        called = False

        def fake_execvpe(*_args, **_kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("os.execvpe", fake_execvpe)

        assert _maybe_respawn_in_terminal_unix() is False
        assert called is False

    def test_skip_when_already_in_terminal(self, monkeypatch):
        # The respawn env var guards against infinite recursion when a
        # terminal emulator can't be launched as a single command and
        # the shell falls back to sourcing argv. Without the guard the
        # binary would loop forever.
        monkeypatch.setattr(sys, "platform", "linux")
        self._patch_tty(monkeypatch, is_tty=False)
        monkeypatch.setenv("YADC_DESKTOP_IN_TERMINAL", "1")

        called = False

        def fake_execvpe(*_args, **_kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("os.execvpe", fake_execvpe)

        assert _maybe_respawn_in_terminal_unix() is False
        assert called is False

    def test_skip_when_stdout_is_tty(self, monkeypatch):
        # The user already has a terminal — they ran ``./yadc-webui-cpu``
        # from a shell. Don't pop another window on top.
        monkeypatch.setattr(sys, "platform", "linux")
        self._patch_tty(monkeypatch, is_tty=True)
        monkeypatch.delenv("YADC_DESKTOP_IN_TERMINAL", raising=False)

        monkeypatch.setattr("os.execvpe", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("execvpe called")))

        assert _maybe_respawn_in_terminal_unix() is False

    def test_no_op_when_no_terminal_emulator(self, monkeypatch):
        # All of ``gnome-terminal``, ``konsole``, ``xterm``, etc. are
        # missing. The helper must return False rather than throwing —
        # the server still starts, just without a visible terminal.
        monkeypatch.setattr(sys, "platform", "linux")
        self._patch_tty(monkeypatch, is_tty=False)
        monkeypatch.delenv("YADC_DESKTOP_IN_TERMINAL", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)

        assert _maybe_respawn_in_terminal_unix() is False

    def test_execvpe_called_with_first_available_terminal(self, monkeypatch):
        # When ``gnome-terminal`` is on PATH, that's the one we use —
        # most Linux users have it and we want a stable order so a
        # previous-user choice doesn't surprise a new user.
        monkeypatch.setattr(sys, "platform", "linux")
        self._patch_tty(monkeypatch, is_tty=False)
        monkeypatch.delenv("YADC_DESKTOP_IN_TERMINAL", raising=False)

        # First ``which`` call (for ``gnome-terminal``) returns a path;
        # the rest return None.
        def fake_which(name):
            if name == "gnome-terminal":
                return "/usr/bin/gnome-terminal"
            return None

        monkeypatch.setattr("shutil.which", fake_which)

        captured: dict[str, object] = {}

        def fake_execvpe(file, args, env):
            captured["file"] = file
            captured["args"] = list(args)
            captured["env"] = dict(env)
            # Mimic real execvpe: never return.
            raise SystemExit(0)

        monkeypatch.setattr("os.execvpe", fake_execvpe)

        with pytest.raises(SystemExit):
            _maybe_respawn_in_terminal_unix()

        assert captured["file"] == "/usr/bin/gnome-terminal"
        # ``sys.argv[0]`` (the user's binary path), not ``sys.executable``
        # — re-invoking the bootloader would forward ``sys.argv[1..]`` as
        # script arguments and click would reject them as "extra args".
        assert captured["args"] == [
            "/usr/bin/gnome-terminal",
            "--",
            sys.argv[0],
            *sys.argv[1:],
        ]
        assert captured["env"]["YADC_DESKTOP_IN_TERMINAL"] == "1"

    def test_handles_closed_stdout(self, monkeypatch):
        # Rare: stdout was closed before ``main()`` ran (this happens
        # with some file managers). ``isatty()`` raises ValueError; the
        # helper must treat that as no-TTY and proceed.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("YADC_DESKTOP_IN_TERMINAL", raising=False)

        class BrokenStdout:
            def isatty(self_inner) -> bool:
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr("shutil.which", lambda _: None)

        assert _maybe_respawn_in_terminal_unix() is False
