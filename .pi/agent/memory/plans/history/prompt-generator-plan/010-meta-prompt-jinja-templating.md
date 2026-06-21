---
plan: prompt-generator-plan
phase: 8
status: proposed — awaiting go/no-go
---

# Phase 8 — Meta-prompt Jinja templating (design, not yet implemented)

## Motivation

`yadc/api/services/prompt_generation/service.py` builds the
meta-conversation (the multi-turn message list the template-generating
LLM sees) with inline f-strings / string concatenation inside
`_build_messages`, plus the `_format_hints` helper and the
`_EXAMPLE_ACK` / `_REFINE_TEMPLATE_ACK` module constants. The two
**system** prompts are already externalized as `generate.txt` /
`refine.txt` (loaded via `importlib.resources`), but everything else
is Python string literals.

Two reasons to move the rest into Jinja2 templates:

1. **Maintainability** — the wording lives in data, not in Python
   control flow. Easier to read, diff, and tune.
2. **Future user-editable instructions** — once the scaffolding is in
   external files (alongside the already-external system prompts), a
   later feature can let users override these instructions with their
   own, mirroring the caption-template resolution chain
   (`~/.local/state/yadc/templates/` overriding the built-in
   `yadc/templates/jinja/`). That override mechanism is a **separate
   later feature** — Phase 8 only does the externalization; it does
   not build the override UX.

## Decisions (from the design pass)

### 1. Macros throughout, not `{% set %}` blocks

The captioner (`yadc/core/captioner.py`, `PromptRenderer`) uses
`{% set system_prompt %}…{% endset %}` blocks read off
`template.module`. That pattern was considered here and rejected in
favour of **`{% macro %}` blocks** for one concrete reason:

The per-example label (`"Example {i}/{n}: {subject}\n{caption}"`)
varies **inside the example loop** — different `i`/`subject`/`caption`
each iteration. `{% set %}` blocks evaluate **once** with the call's
globals, so they cannot vary per iteration. A macro is therefore
forced for the example label; uniformity then favours macros for the
rest of the bodies too.

(The captioner's `{% set %}` blocks fit there because its templates
are **user-authored output blocks** rendered once per image with a
globals context — not per-call-arg scaffolding that varies in a loop.)

### 2. Loading: cached `get_template().module`, explicit macro args

```python
import jinja2

_ENV = jinja2.Environment(
    loader=jinja2.PackageLoader("yadc.api.services.prompt_generation", "prompts"),
    lstrip_blocks=True,
    trim_blocks=True,
    keep_trailing_newline=False,
)
_MSGS = _ENV.get_template("messages.jinja").module  # cached once at import
```

Each message body is then `_MSGS.<macro>(...)` with explicit args.

**Gotcha verified during design** (decisive for the macro choice):
`env.get_template(name, globals=ctx)` returns the **same cached
`Template`** object, and the `globals` from the **first** call stick —
a later call with different globals silently inherits the earlier
values. So a `{% set %}`-blocks + per-call-`globals` approach would
need `env.from_string(src, globals=ctx)` per call (which *does* isolate
per call, at the cost of recompiling). Macros sidestep this entirely:
the template is compiled once and macros take explicit arguments, so
there is no per-call global state to leak.

### 3. Focus hints live in `messages.jinja`, wrapped in `{% raw %}`

The focus format hints contain literal
`{% set system_prompt %}…{% endset %}` **example** text (showing the
model the output shape). In a `.jinja` file that literal syntax would
be parsed as Jinja, so those passages are wrapped in
`{% raw %}…{% endraw %}`. This keeps all scaffolding in one override
file rather than splitting hints into separate `.txt` blobs.

(While moving the text, fix the two existing `"block.Use"` typos in
`_format_hints` — missing space after "block".)

### 4. System prompts stay as `.txt`

`generate.txt` / `refine.txt` are unchanged. They have no variables, so
Jinja-fying them buys nothing. The future override feature can offer
overrides for **both** the `.txt` system prompts and the `.jinja`
scaffolding by swapping the loader for a `ChoiceLoader`:

```python
loader=jinja2.ChoiceLoader([
    jinja2.FileSystemLoader(user_prompts_dir),  # ~/.local/state/yadc/…/prompts (override)
    jinja2.PackageLoader("yadc.api.services.prompt_generation", "prompts"),  # builtin
])
```

…and applying the same resolution to the `.txt` loads. That is the
later feature, not Phase 8.

## Proposed `messages.jinja`

```jinja
{# Shared focus format hint — embedded in several bodies.
   {% raw %} because the example format literally contains {% set %}. #}
{% macro focus_hints(focus) -%}
{%- if focus == "both" -%}
Output both the system_prompt and user_prompt top-level blocks.

Use the following format:
{% raw %}{% set system_prompt %}
...
{% endset %}

{% set user_prompt %}
...
{% endset %}{% endraw %}
{%- elif focus == "system" -%}
Output only the system_prompt top-level block. Use the following format:
{% raw %}{% set system_prompt %}
...
{% endset %}{% endraw %}
{%- else -%}
Output only the user_prompt top-level block. Use the following format:
{% raw %}{% set user_prompt %}
...
{% endset %}{% endraw %}
{%- endif -%}
{%- endmacro %}

{# ── Generate ── #}
{% macro generate_zero_user(intent, focus) -%}
Generate a Jinja2 prompt template for image captioning. Output only the template,
without any markdown or instructions on how to use the template.{{ focus_hints(focus) }}.

Intent:
{{ intent }}.
{%- endmacro %}

{% macro generate_intro(intent, n) -%}
I need a Jinja2 prompt template for image captioning.

Intent:
{{ intent }}

I'll send you {{ n }} example(s) — each as a subject + caption + image. Briefly
acknowledge each one. When I say "now generate", emit the Jinja2 template.
{%- endmacro %}

{% macro generate_priming_ack() -%}Understood. Send your examples.{%- endmacro %}
{% macro example_ack() -%}Got it. Send the next example.{%- endmacro %}
{% macro example_label(i, n, subject, caption) -%}Example {{ i }}/{{ n }}: {{ subject }}
{{ caption }}{%- endmacro %}
{% macro generate_go(focus) -%}
Now generate the Jinja2 template. Output only the template, without any markdown
or instructions on how to use the template. {{ focus_hints(focus) }}
{%- endmacro %}

{# ── Refine ── #}
{% macro refine_zero_user(intent, template_content, focus) -%}
Refine my existing Jinja2 prompt template for image captioning.

My current template:
{{ template_content }}

Intent for refinement:
{{ intent }}

{{ focus_hints(focus) }}.

Output only the template, without any markdown or instructions on how to use the template.
{%- endmacro %}
{% macro refine_intro(intent, n) -%}
I need to refine my existing Jinja2 prompt template for image captioning. Intent: {{ intent }}.
I'll send you {{ n }} example(s) — each as a subject + caption + image — plus my current
template, then ask you to generate. Briefly acknowledge each item. When I say "now generate",
emit the refined Jinja2 template.
{%- endmacro %}
{% macro refine_priming_ack() -%}Understood. Send your items.{%- endmacro %}
{% macro refine_template_user(template_content) -%}
My current template (please refine it according to the intent):

{{ template_content }}
{%- endmacro %}
{% macro refine_template_ack() -%}Got it. Send the next item.{%- endmacro %}
{% macro refine_go(focus) -%}
Now generate the refined Jinja2 template. Output only the template, without any markdown
or instructions on how to use the template. {{ focus_hints(focus) }}
{%- endmacro %}
```

## Proposed slimmed `_build_messages`

Pure orchestration — every body is a macro call; the example turn wraps
the label macro in a `TextPart` + `ImageUrlPart`.

```python
_MSGS = _ENV.get_template("messages.jinja").module


def _build_messages(request, *, system_prompt, refine_system_prompt):
    is_refine = request.template_content is not None
    messages: list[Message] = [
        Message(role="system", content=refine_system_prompt if is_refine else system_prompt),
    ]

    intent = request.intent or "(none provided)"
    m = _MSGS

    if not request.examples:
        body = (
            m.refine_zero_user(intent, request.template_content, request.focus)
            if is_refine
            else m.generate_zero_user(intent, request.focus)
        )
        messages.append(Message(role="user", content=body))
        return messages

    n = len(request.examples)
    if is_refine:
        messages.append(Message(role="user", content=m.refine_intro(intent, n)))
        messages.append(Message(role="assistant", content=m.refine_priming_ack()))
        messages.append(Message(role="user", content=m.refine_template_user(request.template_content)))
        messages.append(Message(role="assistant", content=m.refine_template_ack()))
    else:
        messages.append(Message(role="user", content=m.generate_intro(intent, n)))
        messages.append(Message(role="assistant", content=m.generate_priming_ack()))

    for i, ex in enumerate(request.examples):
        messages.append(
            Message(
                role="user",
                content=[
                    TextPart(text=m.example_label(i + 1, n, ex.subject, ex.caption)),
                    ImageUrlPart(image_url=ImageUrl(url=ex.image_data_url, detail="auto")),
                ],
            )
        )
        if i != n - 1:
            messages.append(Message(role="assistant", content=m.example_ack()))

    go = m.refine_go if is_refine else m.generate_go
    messages.append(Message(role="user", content=go(request.focus)))
    return messages
```

`_format_hints`, `_EXAMPLE_ACK`, and `_REFINE_TEMPLATE_ACK` are deleted
(their text now lives in the macros).

## Test impact

`tests/api/test_prompt_generation.py` deliberately does **not** pin
scaffolding wording — it only pins the two system-prompt constants
(loaded from the unchanged `.txt` files) and the multimodal example
structure. So the refactor is low-risk: existing tests keep passing
unchanged. Add one smoke test asserting `_build_messages` still yields
the right **roles / turn-count** per scenario (generate 0/N, refine
0/N) so a future template edit can't silently drop or reorder a turn.

## Open question

Whether to implement at all (the user is deciding). If yes, the work is
small and local to `yadc/api/services/prompt_generation/`: one new
`prompts/messages.jinja`, a ~10-line `_ENV`/`_MSGS` block at the top of
`service.py`, the rewritten `_build_messages`, deletion of three
helpers, and the one smoke test.
