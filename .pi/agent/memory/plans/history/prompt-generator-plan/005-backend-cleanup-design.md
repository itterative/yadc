---
date: 2026-06-20
---
# Phase 5b: backend cleanup design
**Context:** While planning Phase 6 (refine mode), the user
proposed a small backend cleanup to set the stage: convert the
single `prompt_generation.py` file into a package (same name, so
the import path stays the same → minimal blast radius) and move
the system prompts out of Python into `.txt` files. While
drafting the design, I noticed that the existing `_SYSTEM_PROMPT`
constant is **dead code** — defined at the bottom of the file
but never referenced by `_build_messages`. The user confirmed
this is "not a huge issue" but wants the system prompt wired
back in as part of the cleanup. This entry captures the design.

**Decision:** Add **Phase 5b (Backend cleanup)** to this plan,
lands **before** Phase 6a so the refine-mode work has a clean
foundation.

Three pieces, all local to `yadc/api/services/prompt_generation/`:

1. **Refactor file → package** (no behaviour change). Convert
   the single `prompt_generation.py` module into a
   `prompt_generation/` package. The import path
   `yadc.api.services.prompt_generation` is unchanged, so:
   - `yadc/api/services/__init__.py` (the DI re-export) keeps
     working.
   - `yadc/api/controllers/api_prompts.py` (the controller
     import) keeps working.
   - `tests/api/test_prompt_generation.py` (the test imports
     and the symbol-level patches for `cmd_envs` and
     `create_client`) keep working — the `__init__.py`
     re-exports those symbols at the package level so the
     patches at `yadc.api.services.prompt_generation.cmd_envs`
     and `yadc.api.services.prompt_generation.create_client`
     keep resolving.

2. **System prompts as `.txt` files, loaded as package
   resources in `service.py`** (no behaviour change for now —
   just data → file). The current `_SYSTEM_PROMPT` string
   moves to `prompts/generate.txt`, loaded once at import time
   **inside `service.py`** (the file that uses the constant —
   no separate loader module, no `__init__.py` involvement, no
   import cycle):

   ```python
   # yadc/api/services/prompt_generation/service.py (top of file)
   from importlib.resources import files

   _GENERATE_SYSTEM_PROMPT: str = (
       files("yadc.api.services.prompt_generation")
       .joinpath("prompts")
       .joinpath("generate.txt")
       .read_text(encoding="utf-8")
       .strip()
   )
   ```

   `service.py` is the right home for the constant because it's
   the only consumer — putting the constant in `__init__.py`
   (or a separate `_prompts.py`) would force `service.py` to
   import the package back, which creates a circular import
   (`__init__.py` imports from `service.py` to re-export
   `PromptGenerationService`; `service.py` importing from the
   package re-enters `__init__.py` mid-execution).

   `prompts/refine.txt` is created as part of this cleanup (the
   structure is set up) but the constant + loading line are
   **deferred to Phase 6a** — the file is only meaningful once
   the refine-mode branching in `_build_messages` exists, and
   loading a file that nothing references would be dead code at
   import time. Reasons for `.txt` over a Python constant:
   - No multi-line string quoting / indentation hazards.
   - Easy to diff in code review when we tune wording.
   - Easy to add new modes later (just add a new file + a new
     constant in `service.py`).
   - `importlib.resources` makes the prompts part of the
     package distribution — works the same in dev (uv
     editable install), wheel installs, and zip-imports. The
     `__file__`-relative approach would break in the latter
     two.

3. **Wire the system prompt into the message list** (bug fix).
   `_build_messages` currently returns a list of
   user/assistant turns only; the system prompt never reaches
   the LLM. Phase 5b fixes this by prepending
   `Message(role="system", content=system_prompt)` to the list.
   The function gains a `system_prompt: str` keyword-only
   parameter (so the constant in `service.py` can be injected
   without a global lookup) and `PromptGenerationService.generate`
   passes `_GENERATE_SYSTEM_PROMPT` (and, after Phase 6a,
   `_REFINE_SYSTEM_PROMPT` when `request.template_content` is
   set). The LLM client layer already maps `role="system"` to
   the native shape (OpenAI's `system` message, Gemini's
   `system_instruction`), so no client changes are needed.

**Test impact:** the 5 existing `TestBuildMessages` tests that
assert message indices (`messages[0]`, `messages[1]`, etc.) all
shift by 1 because the system message now occupies index 0.
`test_final_user_turn_says_go` uses `messages[-1]` so it's
index-agnostic and needs no change. One new test
(`test_system_message_is_first`) verifies the system message is
at index 0 and its content matches the `_GENERATE_SYSTEM_PROMPT`
constant loaded by `service.py` from `prompts/generate.txt` (the
test exercises the same loading path as production by reading
the constant directly from `service.py` — keeps the test honest
about the txt-file indirection so a future refactor can't
silently drop the loading code).

**Files touched** (listed in the plan's "Phase 5b (backend
cleanup, planned)" section under "Key Files Touched"):

- `yadc/api/services/prompt_generation.py` (delete)
- `yadc/api/services/prompt_generation/__init__.py` (new)
- `yadc/api/services/prompt_generation/service.py` (new)
- `yadc/api/services/prompt_generation/prompts/generate.txt` (new)
- `yadc/api/services/prompt_generation/prompts/refine.txt` (new)
- `tests/api/test_prompt_generation.py` (update existing
  index assertions + add `test_system_message_is_first`)
- Memory docs (after impl)

**Out of scope for Phase 5b** (deferred to Phase 6a proper):

- The `template_content` field on `PromptGenerationRequest`.
- Branching in `_build_messages` on
  `request.template_content is not None`.
- Picking the refine system prompt vs the generate system
  prompt based on mode.
- New tests for the refine-mode message structure.

Phase 5b is purely structural + the system-prompt wire-up bug
fix. Phase 6a adds the generate-vs-refine branching on top.

**Rationale:**
- **Same name → reduce blast radius.** Converting to a package
  with the same import path means the controller, the DI
  re-export, and the test patches all keep working unchanged.
  Only the on-disk shape changes.
- **System prompts as data, not code.** The prompts are
  user-tunable English text, not Python logic. Putting them in
  `.txt` files makes them first-class artifacts that can be
  reviewed, diffed, and version-controlled cleanly. It also
  eliminates the multi-line string quoting / indentation
  hazard that comes with `"""`-delimited Python constants.
  **Loading via `importlib.resources`** (rather than a
  `__file__`-relative path) keeps the prompts part of the
  package distribution and avoids the silent break that
  happens when a relative-path loader is run from a wheel
  install or a zip-import where `__file__` no longer points
  at the on-disk source.
- **Constant lives in `service.py`, not `__init__.py`.** The
  constant has one consumer (`service.py`), so it lives next
  to that consumer. Putting the constant (or the loading
  code) in `__init__.py` would force `service.py` to import
  the package back, which creates a circular import:
  `__init__.py` imports from `service.py` to re-export
  `PromptGenerationService`; `service.py` importing the
  package re-enters `__init__.py` mid-execution. The same
  argument rules out a separate `_prompts.py` loader module
  that re-exports the constants through `__init__.py` — it
  adds an extra hop for no benefit. Keep the constant
  private to `service.py` (leading underscore).
- **Wire the system prompt now, not later.** The dead-code
  issue is a real behaviour bug — the LLM has been getting
  prompts without meta-instructions. Fixing it as part of the
  cleanup means Phase 6a doesn't have to thread the system
  prompt through two different code paths (old dead constant +
  new refine constant). One wire-up, used by both modes.
- **Bug-fix tests over production-code relaxation.** The user
  explicitly preferred relaxing tests over production code in
  the past. But in this case, the system prompt is a real
  behaviour change that the LLM needs to see, so the tests
  must reflect the new shape (system message at index 0). The
  new test also pins the txt-file indirection so we don't
  accidentally drop the loading code in a future refactor.
- **`refine.txt` is staged, not loaded.** The file is created
  in Phase 5b so the on-disk structure is in place, but the
  constant + loading line are deferred to Phase 6a when
  `_build_messages` actually branches on it. Loading a file
  that nothing references would be dead code at import time.
