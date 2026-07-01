"""PyInstaller entry point for the desktop ``yadc webui`` build.

The bundled .exe wraps the regular ``yadc webui serve`` CLI. This module
is the single-file launcher: it adjusts ``sys.argv`` so the desktop UX is
the default (``--browser`` to open the SPA, ``--no-cors`` because the SPA
is served same-origin) and dispatches to the click entry point.

Run from a normal install, this is just ``yadc webui serve`` with the
defaults overridden — the desktop-friendly behavior only kicks in when
``sys.frozen`` is set (the sentinel PyInstaller adds to frozen binaries).

The pure argv builder (:func:`build_argv`) is separate so the unit tests
can exercise the logic without spinning up a CLI runner.
"""

from __future__ import annotations

import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import contextmanager

import click
import platformdirs

# Public so tests can assert the defaults the desktop build applies.
DEFAULT_DESKTOP_FLAGS: tuple[str, ...] = ("--browser", "--no-cors")

# Subset of CLI flags the user might pre-set that fully determine whether
# the desktop-default flag should be injected. Splitting this out keeps
# the argv builder testable without spinning up a click parser.
_BROWSER_FLAGS = frozenset({"--browser", "--no-browser"})
_CORS_FLAGS = frozenset({"--cors", "--no-cors"})

# Env var set on the spawned terminal's child so the re-exec'd
# process knows it's already inside our wrapper and won't recurse.
# Picked to be unique enough that no other tool will collide.
_TERMINAL_RESPAWN_ENV = "YADC_DESKTOP_IN_TERMINAL"

# Env var the user can set to a full terminal invocation line
# (e.g. ``"kitty --detach"``) to override auto-detection. Useful when
# the user has a terminal the introspection layer doesn't know about.
_TERMINAL_OVERRIDE_ENV = "YADC_DESKTOP_TERMINAL"

# gsettings introspection of GNOME's preferred terminal. Covers GNOME,
# MATE, Cinnamon, Pop!_OS, and XFCE-with-gnome-settings-daemon. The
# key has gone through schema revisions so we try ``exec-args`` first
# and fall back to the older ``exec-arg``.
_TERMINAL_GSETTINGS_SCHEMA = "org.gnome.desktop.default-applications.terminal"

# Hardcoded fallback list, used only when gsettings returns nothing
# useful. Order matters — the first binary found on PATH wins. ``ptyxis``
# is GNOME's new default on Fedora 39+ and Ubuntu 24.04. ``x-terminal-emulator``
# is the freedesktop.org alias used by Debian/Ubuntu to point at whatever
# the user picked at install time.
_TERMINAL_FALLBACK: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ptyxis", ("--",)),
    ("gnome-terminal", ("--",)),
    ("konsole", ("-e",)),
    ("xfce4-terminal", ("-e",)),
    ("mate-terminal", ("-e",)),
    ("tilix", ("-e",)),
    ("foot", ("-e",)),
    ("alacritty", ("-e",)),
    ("kitty", ("--",)),
    ("wezterm", ("start", "--")),
    ("deepin-terminal", ("-e",)),
    ("xterm", ("-e",)),
    ("x-terminal-emulator", ("-e",)),
)


def _detect_terminal_via_env() -> tuple[str, tuple[str, ...]] | None:
    """Read :data:`_TERMINAL_OVERRIDE_ENV` as a full terminal invocation.

    Returned ``(binary, prefix)`` is suitable for ``[<binary>, *<prefix>, ...]``.
    Raises (returns ``None``) when the variable is unset/empty or contains
    only whitespace.
    """
    raw = os.environ.get(_TERMINAL_OVERRIDE_ENV, "").strip()
    if not raw:
        return None
    try:
        parts = shlex.split(raw)
    except ValueError:
        # Bad quoting — treat the override as opaque tokens.
        parts = raw.split()
    if not parts:
        return None
    return parts[0], tuple(parts[1:])


def _detect_terminal_via_gsettings() -> tuple[str, tuple[str, ...]] | None:
    """Read GNOME's preferred-terminal preferences via ``gsettings``.

    Honors both the modern schema (keys ``exec``, ``exec-args``) and the
    older one (singular ``exec-arg``); some distros lag the rename.

    Returns ``None`` if gsettings isn't on PATH, the schema is missing,
    the user hasn't configured a terminal (empty string), or the resolved
    binary isn't on ``$PATH``.
    """
    gsettings = shutil.which("gsettings")
    if gsettings is None:
        return None

    def _read(key: str) -> str:
        try:
            result = subprocess.run(
                [gsettings, "get", _TERMINAL_GSETTINGS_SCHEMA, key],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode != 0:
            return ""
        # gsettings prints strings quoted with single quotes; ``[]`` for
        # array, ``""`` for empty. Strip both. Truncate backslash-escape
        # sequences inside (e.g. ``\\'``) — exec values are usually
        # plain binary paths so this is rare, but cheap to handle.
        raw = result.stdout.strip()
        return raw

    exec_raw = _read("exec")
    if not exec_raw:
        return None
    # Strip surrounding quotes (single or double) that gsettings uses
    # for string types.
    binary = exec_raw.strip("'\"")
    if not binary or shutil.which(binary) is None:
        return None

    # ``exec-args`` is the modern key; some older schemas use the
    # singular ``exec-arg``. Whichever wins, drop gsettings' ``%s`` /
    # ``$@`` placeholders and split on whitespace. An empty exec-args
    # string collapses to ``()`` which means "launch the bare binary".
    args_raw = _read("exec-args") or _read("exec-arg")
    placeholder_stripped = args_raw.replace("$@", " ").replace("%s", " ")
    # Strip surrounding quotes gsettings adds for string types.
    cleaned = placeholder_stripped.strip().strip("'\"")
    prefix = tuple(shlex.split(cleaned)) if cleaned else ()
    return binary, prefix


def _detect_terminal_via_known_binary() -> tuple[str, tuple[str, ...]] | None:
    """Walk :data:`_TERMINAL_FALLBACK` and pick the first binary on ``$PATH``.

    Returns the *resolved* path (from ``shutil.which``) rather than the
    bare name, so the caller can pass it to ``os.execvpe`` without a
    second ``$PATH`` lookup.
    """
    for binary_name, prefix in _TERMINAL_FALLBACK:
        resolved = shutil.which(binary_name)
        if resolved is not None:
            return resolved, prefix
    return None


def _detect_preferred_terminal() -> tuple[str, tuple[str, ...]] | None:
    """Discover the user's preferred terminal emulator and how to invoke it.

    Detection order — first non-None wins:

    1. ``$YADC_DESKTOP_TERMINAL`` env var (full override).
    2. ``gsettings`` (covers GNOME, MATE, Cinnamon, Pop!_OS, and
       XFCE-with-gnome-settings-daemon; resolves ptyxis correctly on
       Fedora 39+/Ubuntu 24.04 where gnome-terminal is gone).
    3. Hardcoded fallback list of well-known binaries on ``$PATH``.
    """
    return (
        _detect_terminal_via_env()
        or _detect_terminal_via_gsettings()
        or _detect_terminal_via_known_binary()
    )


def _maybe_respawn_in_terminal_unix() -> bool:
    """Re-exec this binary inside a terminal emulator on POSIX, if needed.

    On Windows ``console=True`` already gives the binary a console window,
    so this helper is a no-op there. On Linux/macOS, double-clicking the
    bundled ``.exe`` (or running it from a file manager) gives the process
    no controlling TTY — uvicorn's logs go to ``/dev/null`` and Ctrl+C
    doesn't work to stop it.

    We detect the no-TTY case and replace ourselves with a terminal
    emulator that re-runs the same binary. The terminal is chosen by
    :func:`_detect_preferred_terminal` (gsettings first, then a hardcoded
    fallback list). The re-spawned binary sees a real TTY and proceeds
    normally; an env var tells it not to recurse.

    Returns True if a re-exec was attempted (callers can use this for
    tests). In production the function never returns on success —
    :func:`os.execvpe` replaces the current process. If no terminal
    emulator is on PATH the helper returns False and the calling program
    continues without one (server still starts; logs go to ``/dev/null``,
    which is the user's queue to file an issue if they care).
    """
    if sys.platform == "win32":
        return False
    if os.environ.get(_TERMINAL_RESPAWN_ENV) == "1":
        return False
    try:
        if sys.stdout.isatty():
            return False
    except (ValueError, OSError):
        # Streams closed (rare — happens when stdout is detached).
        # Treat as no-TTY and proceed with respawn.
        pass

    detection = _detect_preferred_terminal()
    if detection is None:
        return False

    binary, prefix = detection

    env = os.environ.copy()
    env[_TERMINAL_RESPAWN_ENV] = "1"
    # Re-exec the *user's* binary path (``sys.argv[0]``) rather than the
    # PyInstaller bootloader (``sys.executable``). In one-file mode
    # ``sys.executable`` points into ``$TMP/_MEI…/`` and re-invoking it
    # would carry ``sys.argv`` as additional arguments — which click
    # reads as "extra positional argument" and aborts with the
    # ``Got unexpected extra argument`` error. Invoking ``sys.argv[0]``
    # directly hands the bootloader the same argv it had on the original
    # double-click, which is what we want.
    cmd = [binary, *prefix, sys.argv[0], *sys.argv[1:]]
    try:
        os.execvpe(binary, cmd, env)
    except OSError:
        # execvpe failed (rare: e.g. SELinux denial). Headless fallback
        # — the server still starts without a visible terminal.
        return False

    # Unreachable in practice — ``os.execvpe`` replaces the process on
    # success and never returns. Some POSIX build configurations of
    # basedpyright still flag the trailing ``return False`` as
    # unreachable, which is fine (the runtime never gets here).
    return True  # pyright: ignore[reportUnreachable]


# Lockfile used as a *detection* mechanism for another yadc-webui
# instance, not a mutex. PID-based: contents are our PID; a subsequent
# launch reads it and treats a live PID as "another instance". Goes in
# the user's platformdirs cache so it survives across rebuilds — the
# stale-lock case (force-kill, crash) is handled by the liveness probe.
_DESKTOP_CACHE_PATH = platformdirs.user_cache_path("yadc", ensure_exists=True)
_DESKTOP_LOCK_PATH = _DESKTOP_CACHE_PATH / "webui-desktop.lock"


def _read_lockfile_pid() -> int | None:
    """Read the PID from the desktop lockfile, or ``None`` on absence/error."""
    try:
        text = _DESKTOP_LOCK_PATH.read_text().strip()
    except (OSError, FileNotFoundError):
        return None
    try:
        return int(text.splitlines()[0])
    except (ValueError, IndexError):
        return None


def _pid_alive(pid: int) -> bool:
    """POSIX liveness probe — signal 0 to a PID succeeds iff the process exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@contextmanager
def _held_lockfile():
    """Take the desktop lockfile for our PID, or skip if another instance holds it.

    On entry: read the existing PID. If it points to a *live* process
    other than us, leave the file alone — owning the lock is a courtesy,
    not a mutex, and clobbering would hide the running sibling from
    subsequent launches (no warning fires because the next launcher
    finds an empty file once the holder exits). The current process
    just continues without taking the lock; cleanup is a no-op too.

    If the file is absent, points to a dead PID (stale lockfile from a
    SIGKILL'd predecessor), or points to our own PID, we overwrite
    with our PID.

    On exit: only remove the file if it still contains our PID. A
    stale lockfile from a SIGKILL'd predecessor that we tried to clean
    up but lost a race against another launcher falls under this same
    guard.
    """
    pid = os.getpid()
    existing_pid = _read_lockfile_pid()
    if existing_pid is not None and existing_pid != pid and _pid_alive(existing_pid):
        # Another live instance holds the lock — stay out of its way.
        yield pid
        return

    try:
        _DESKTOP_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DESKTOP_LOCK_PATH.write_text(f"{pid}\n")
        yield pid
    except OSError:
        yield pid
    finally:
        try:
            if _read_lockfile_pid() == pid:
                _DESKTOP_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def _check_for_other_instance() -> int | None:
    """Return the PID of another live yadc-webui instance, or ``None``.

    ``None`` means "no lockfile" or "lockfile points to a dead process"
    — the latter is benign (the next ``_held_lockfile`` entry will overwrite).
    """
    pid = _read_lockfile_pid()
    if pid is None or pid == os.getpid():
        return None
    if _pid_alive(pid):
        return pid
    return None


def _pause_for_error() -> None:
    """Wait for user input before the spawned terminal closes.

    Frozen-bundle-only. When the server exits with an error (port still
    busy even after the fallback, multiprocessing crash, etc.) the
    spawned terminal would otherwise exit immediately and the user
    would see nothing. :func:`input` blocks the process long enough
    for the user to read the traceback above and decide to copy logs
    / close the window. We catch the cases where there's no connected
    stdin so a non-interactive launch doesn't hang.
    """
    sys.stdout.write(
        "\n"
        "yadc webui exited with an error. See the message above.\n"
        "Press Enter (or Ctrl+C) to close this window.\n"
    )
    sys.stdout.flush()
    try:
        input()
    except (EOFError, KeyboardInterrupt, OSError):
        # EOFError: stdin closed (rare — happens when redirected).
        # KeyboardInterrupt: user pressed Ctrl+C.
        # OSError: readline() failed on a closed pipe (Windows).
        pass


def _has_flag(args: Sequence[str], names: frozenset[str]) -> bool:
    """Return True if any ``name`` appears in ``args`` as a whole token.

    Click accepts ``--browser=true`` / ``--browser true`` / ``--browser``
    interchangeably. We only need to detect the user's *intent* — exact
    spelling of the value is irrelevant for "should we inject the default".
    Matching whole tokens is therefore sufficient and avoids regex.
    """
    for arg in args:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def build_argv(raw_argv: Sequence[str]) -> list[str]:
    """Return the argv to dispatch into the click CLI.

    The function expects ``raw_argv`` to be everything after the executable
    path (i.e. ``sys.argv[1:]``). It prepends ``["webui", "serve"]`` so
    ``raw_argv = ["--port=9000"]`` becomes ``["webui", "serve", "--port=9000"]``.

    Desktop-default flags (``--browser``, ``--no-cors``) are appended when
    the user hasn't already expressed a preference, so an explicit
    ``--no-browser`` from a power user still wins.
    """
    user_args = list(raw_argv)
    argv = ["webui", "serve", *user_args]

    for default in DEFAULT_DESKTOP_FLAGS:
        if default == "--browser" and not _has_flag(user_args, _BROWSER_FLAGS):
            argv.append(default)
        elif default == "--no-cors" and not _has_flag(user_args, _CORS_FLAGS):
            argv.append(default)

    return argv


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point.

    When invoked from a frozen build, ``sys.argv[0]`` is the exe path and
    ``sys.argv[1:]`` carries the user's arguments. We splice in the
    desktop defaults, then run the click CLI exactly as a regular
    ``yadc`` install would.

    In standalone mode click calls :func:`sys.exit` itself, so this
    function only returns when an unexpected error escapes — the type
    signature is kept for clarity / future tests.
    """
    if argv is None:
        argv = sys.argv[1:]

    if getattr(sys, "frozen", False):
        # The tagger subprocess uses ``multiprocessing.get_context('forkserver')``
        # which spawns its helpers via ``sys.executable -B -c ...`` — the
        # ``-B`` flag disables writing ``.pyc`` files and the ``-c``
        # invocation re-runs our entry script. In a frozen build that
        # re-entry path is intercepted by PyInstaller's
        # ``pyi_rth_multiprocessing`` runtime hook, but the hook only
        # fires if ``multiprocessing.freeze_support()`` has been called
        # at startup on at least one of the two sides. Calling it here
        # installs the hook's trampoline and lets the helper subprocess
        # resolve to the bundled ``forkserver.main`` cleanly rather than
        # failing with ``No such option '-B'`` (or equivalent).
        multiprocessing.freeze_support()

        # On POSIX, double-clicking a PyInstaller binary gives no TTY —
        # uvicorn logs disappear and Ctrl+C doesn't stop it. Spawn a
        # terminal first; on success this re-execs and never returns.
        # On Windows the OS already gave us a console window.
        _maybe_respawn_in_terminal_unix()

    # Lazy import so importing this module during tests doesn't trigger
    # yadc's full CLI initialization (logging handlers, click group, …).
    from yadc.cli import cli

    if getattr(sys, "frozen", False):
        sys.argv = [sys.argv[0], *build_argv(argv)]

        # Bundle-mode UX: detect an already-running sibling instance and
        # hold a lockfile so subsequent launches can tell. The check is
        # informational only — we don't block.
        other_pid = _check_for_other_instance()
        if other_pid is not None:
            warning_label = click.style("WARNING", fg="yellow", bold=True)
            click.echo(
                f"{warning_label}: another yadc webui instance is running "
                f"(PID {other_pid}). If that is unexpected, close it before "
                f"starting another.",
                err=True,
            )

        # Install an excepthook so unhandled exceptions (e.g. the DB
        # failing to initialize) also trigger the terminal-pause. Click
        # catches its own errors and translates them to SystemExit,
        # which the explicit ``except SystemExit`` below handles — this
        # hook covers everything *else* that bubbles up unhandled.
        _original_excepthook = sys.excepthook

        def _bundle_excepthook(exc_type, exc_value, exc_tb):
            _original_excepthook(exc_type, exc_value, exc_tb)
            if exc_type is None:
                return
            # Filter: pause only for genuine Python errors. Anything
            # rooted at ``BaseException`` rather than ``Exception`` is a
            # control-flow signal — Ctrl+C (``KeyboardInterrupt``),
            # uvicorn's task cancellation on shutdown
            # (``asyncio.CancelledError``), ``sys.exit`` (``SystemExit``),
            # generator cleanup (``GeneratorExit``). None of those
            # represent bugs the user should action; the spawned
            # terminal should close promptly so the process really
            # does shut down. Without this filter, an
            # ``asyncio.CancelledError`` that escapes uvicorn's
            # shutdown handler would print the cancellation traceback
            # and pause the terminal until the user hits Enter.
            if not issubclass(exc_type, Exception):
                return
            _pause_for_error()

        sys.excepthook = _bundle_excepthook

        # Pause-on-error: click converts Ctrl+C into
        # ``click.exceptions.Abort`` and then ``sys.exit(1)`` — the
        # exit code that arrives here is the user's graceful shutdown
        # signal, not a crash. ``SystemExit`` propagates to Python's
        # normal exit handling without triggering our excepthook, so
        # the spawned terminal closes promptly. Genuine ``Exception``
        # subclasses (RuntimeError, KeyError, sqlite3.OperationalError,
        # etc.) bypass this clause and reach the excepthook above,
        # which pauses the terminal so the user can read the traceback.
        with _held_lockfile():
            return cli(standalone_mode=True)

    return cli(standalone_mode=True)


if __name__ == "__main__":
    raise SystemExit(main())
