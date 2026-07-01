---
date: 2026-07-01
---
# desktop-exe-plan implemented (with two deviations from plan)

**Context:** The plan shipped in commit `daad70b` (`feat(webui): build portable assets and publish on release`; subject typo "protable" is being amended out post-hoc). Two things landed that the plan either ruled out or didn't anticipate.

**Decision 1 — CI workflow shipped anyway.**
**Decision:** Added `.github/workflows/release-assets.yml` (167 lines) that builds the portable assets and publishes them on release.
**Rationale:** The plan listed "A CI workflow that builds the .exe" as an explicit non-goal ("CI is a future iteration"). That was overturned — the release workflow now exists.
**Files touched:** `.github/workflows/release-assets.yml`, `.gitignore`, `README.md`, `docs/README.md`, `pyproject.toml`, `uv.lock`.

**Decision 2 — Port-fallback for the frozen .exe.**
**Decision:** `yadc webui serve` now does a `_find_open_port()` walk when `sys.frozen` is set — if the requested port is busy it falls forward to the next free one in a 20-port range. CLI (non-frozen) mode keeps loud-failure behavior.
**Rationale:** Not in the plan. Motivated by the double-click UX: a previous run still holding 7860 would otherwise make the new double-click immediately fail. Scope-limited to frozen mode so server admins passing an explicit `--port` aren't surprised.
**Files touched:** `yadc/cli_webui.py`, `tests/cli/test_cli_webui_browser.py`.

**Related correctness note (not yet fixed):** `JobScheduler.on_shutdown` joins all jobs (including its own `_cleanup` thread) with `graceful_shutdown_threads_timeout` (1.0s), but `_cleanup` only checks `_shutdown` every 10s — so it always appears in the "did not stop within timeout" warning on a clean shutdown. The scheduled-job loops got chunked sleeps for this reason; `_cleanup` did not.
