---
date: 2026-06-20
---
# Phase 5b: implementation
**Context:** Phase 5b is the backend cleanup (file → package,
system prompts as `.txt` files, wire the system prompt into the
message list as a bug fix). The design was captured in
`005-backend-cleanup-design.md`. This entry captures what
actually happened in the implementation, including the
corrections the user made along the way.

## What landed
1. `yadc/api/services/prompt_generation.py` → `yadc/api/services/prompt_generation/`
   package. `mkdir -p` for the package + `prompts/` subdir,
   then `mv` for the file (per the user's "don't rewrite the
   service and just use mkdir and mv" instruction).
2. `yadc/api/services/prompt_generation/service.py` — extracted
   from the old `.py` file. Added `from importlib.resources
   import files` + a module-level `_GENERATE_SYSTEM_PROMPT`
   constant (loaded from `prompts/generate.txt` at import
   time). Removed the old dead-code `_SYSTEM_PROMPT` string.
3. `yadc/api/services/prompt_generation/prompts/generate.txt` —
   the current system prompt text (byte-identical to the old
   dead-code constant, verified with a `difflib.unified_diff`
   script).
4. `yadc/api/services/prompt_generation/prompts/refine.txt` —
   new refine system prompt text. Drafted to match the design
   bullets in Phase 6a step 4: apply changes to existing
   template, preserve what works, output FULL refined template
   (not a diff), same Jinja2 structure rules. **File created
   but NOT loaded** — the constant + loading line are deferred
   to Phase 6a when `_build_messages` actually branches on it.
5. `yadc/api/services/prompt_generation/__init__.py` — pure
   re-export of the public surface (`ExamplePair`,
   `PromptGenerationFocus`, `PromptGenerationRequest`,
   `PromptGenerationService`) + `cmd_envs` and `create_client`
   at the package level (for the public API consumers that
   import them through the service module path).
6. `_build_messages` gains a `system_prompt: str` keyword-only
   parameter. Prepends `Message(role="system", content=system_prompt)`
   to the message list. `PromptGenerationService.generate`
   passes `_GENERATE_SYSTEM_PROMPT`. This is the bug fix — the
   old `_SYSTEM_PROMPT` constant was defined but never sent.
7. Tests: shifted `messages[i]` index assertions by 1 in 5
   `TestBuildMessages` tests. Added `test_system_message_is_first`
   that verifies the system message is at index 0 and its
   content matches the `_GENERATE_SYSTEM_PROMPT` constant
   (imported directly from `service.py`).
8. Memory doc `docs/backend/api.md` updated: the
   `prompt_generation.py` one-liner is now a multi-line
   description of the package structure (the new `__init__.py`,
   `service.py`, and `prompts/` subdir are all documented
   inline).

## Corrections during implementation

**Constant lives in `service.py`, not `__init__.py` and not a
separate `_prompts.py` module.** My initial design draft put
the loading code in `__init__.py` (or a dedicated loader
module). The user pointed out this would create a circular
import:

  - `__init__.py` imports from `service.py` to re-export
    `PromptGenerationService`
  - `service.py` would need to import the package back to
    read the constant
  - That re-enters `__init__.py` mid-execution → `ImportError`

Fix: the loading code (`from importlib.resources import files`
+ the `_GENERATE_SYSTEM_PROMPT` constant definition) lives at
the top of `service.py`. `__init__.py` is a pure re-export
(nothing else). The plan + design history entry
`005-backend-cleanup-design.md` were updated to match, with
the circular-import argument spelled out in the rationale
section so future-me doesn't re-suggest the wrong placement.

**Test patches target `service.py`, not the package.** When
the prompt generation code was a single `.py` file, the
test patches `yadc.api.services.prompt_generation.cmd_envs`
and `...create_client` worked because they targeted the
module's own `cmd_envs` / `create_client` names. With the
package split, `service.py` has its own `cmd_envs` /
`create_client` names (imported at the top of the file), and
the `__init__.py` re-exports are separate bindings. Patching
the package re-exports doesn't affect what `service.py` sees.

Fix: the test constants are now
`yadc.api.services.prompt_generation.service.cmd_envs` and
`...service.create_client`. The "Test patches target
**package** path" guidance in the original plan was wrong
about the import cycle but right about the surface — the
test docstring explains why the `.service.` segment matters.

**Use absolute imports for `modules` and `service` modules.**
The original `prompt_generation.py` used relative imports
(`from ..modules.logging_factory import ...`,
`from ..modules.service import ...`). When the file was at
`yadc/api/services/prompt_generation.py`, `..modules`
resolved to `yadc.api.services.modules` (which... actually
doesn't exist — `modules` is at `yadc.api.modules`). I'm
not sure how the old file worked, but the user (who fixed
this manually) converted both imports to absolute paths:
`from yadc.api.modules.logging_factory import LoggingFactory`
and `from yadc.api.modules.service import Service`. This
matches the convention in the other `services/*.py` files
(checked `captioning/service.py` and `datasets.py` — they
all use `from yadc.api.modules import ...`).

## Test results
- 20 service tests pass (was 19, +1 for
  `test_system_message_is_first`)
- 727 backend tests pass total (was 726, +1)
- `ruff check`, `ruff format`, `basedpyright` all clean
  on the new package + the updated test file

## Out of scope (deferred)
- `_REFINE_SYSTEM_PROMPT` constant + the loading line for
  `refine.txt` — Phase 6a
- `template_content` field on `PromptGenerationRequest` —
  Phase 6a
- `_build_messages` branching on `template_content is not None`
  — Phase 6a
