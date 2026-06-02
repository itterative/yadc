---
name: dataset-config-settings-plan
description: Plan for integrating the caption settings panel with the dataset's TOML config via a structured read/write API.
---

# Dataset Config Settings Plan

## Overview

The caption settings panel currently uses hardcoded defaults + localStorage, completely ignoring the dataset's TOML config. The dataset config (TOML) already has all the caption-relevant fields (`api.url`, `api.model_name`, `settings.max_tokens`, `settings.image_quality`, `reasoning.*`, `rounds`, `overwrite_captions`, `prompt.name/template`). The goal is to:

1. **Read** these fields from the existing `GET /configs/<name>` (`parsed` key) and surface them as defaults in the caption settings panel
2. **Write** them back through `PATCH /configs/<name>` (JSON body, deep-merged into TOML) — this replaces the old "simplified config editing" TODO

No new read endpoint needed — `GET /configs/<name>` already returns the parsed TOML dict.

---

## Phase 1: Structured Config API ✅

### Backend: `GET /configs/<name>` (existing)

- Already returns `{name, config_path, content, parsed}` where `parsed` is the full TOML dict
- Frontend extracts caption-relevant fields from `parsed` (e.g. `parsed.settings?.max_tokens`, `parsed.reasoning?.enable`)
- No changes needed

### Backend: `PATCH /configs/<name>` ✅

- Accepts a partial JSON dict matching the TOML structure (e.g. `{"settings": {"max_tokens": 1024}, "rounds": 2}`)
- Deep-merges into the existing parsed config, re-serializes to TOML, writes back
- Triggers a rescan (same as `PUT /configs/<name>`)
- **Done** — implemented in `api_configs.py`, uses shared `deep_merge()` from `yadc.utils`

### Backend: `rounds` field ✅

- Added `rounds: int = 1` to `CaptionJobOptions` + `apply_config_overrides()`
- Added `rounds?: number` to frontend `CaptionOptions`
- Wired through in `CaptionSettings.svelte`

### Status

- [x] `PATCH /configs/<name>` endpoint
- [x] Add `rounds` to `CaptionJobOptions` + `apply_config_overrides()`
- [x] Frontend `patchConfig()` in `configs.ts`
- [x] Frontend `rounds` in `CaptionOptions` + `CaptionSettings.svelte`
- [ ] TOML multiline string serialization for templates (template values with newlines serialize as `\n` literals instead of `"""..."""`)

---

## Phase 2: Frontend Integration

### Load dataset config defaults

- `CaptionSettings.svelte` fetches `GET /configs/<name>` on mount, reads caption fields from `parsed`
- **Priority chain**: `dataset config` → `localStorage override` → `env`
  - Fields with a localStorage override use the override
  - Fields without use the dataset config default
  - API URL/token/model still come from `EnvSelector` (env system)

### Visual diff + overrides section

- Small blue dot (`.diff-dot`) next to field labels when current value differs from dataset config default
- Collapsible "Overrides" section at bottom of caption settings panel (only visible when overrides exist)
  - Header: chevron toggle + "Overrides" label + badge count
  - Expanded: list of overridden fields with readable labels and "Reset" text buttons (proper tap targets for mobile)
  - Footer: "Reset all overrides" button
- No inline reset buttons — too small for mobile; all reset actions live in the overrides section

### Write back: dataset config mode

- A toggle or action in the settings panel: "Save as dataset default"
- Calls `PATCH /configs/<name>` with the current field values
- Changes the TOML config — affects all future sessions and CLI users
- Requires confirmation prompt (modifies a shared config file)

### Clean up: remove `ConfigEditor.svelte` from Settings dialog ✅

- Removed "Configs" tab from `SettingsDialog.svelte` (now just General + Environments)
- Deleted `ConfigEditor.svelte` entirely — dataset config management is on the dataset listing page + caption settings panel

### Status

- [x] Frontend fetches config + pre-fill logic
- [x] Visual diff indicator (overridden fields)
- [x] Per-field reset (collapsible overrides section)
- [x] "Save as dataset default" action (PATCH integration) — new "Config" tab in the side panel with structured form fields, saves via `patchConfig()`
- [x] Remove `ConfigEditor.svelte` from `SettingsDialog` — file deleted, tab removed from dialog

---

## Phase 3: Nice-to-haves

Lower priority, incremental.

### Preset profiles
- Save/restore named setting profiles (e.g. "Quick draft", "High quality")
- Named snapshots of localStorage settings

### Config diff banner
- "3 settings differ from dataset defaults" with one-click reset-all

### Status

- [ ] Preset profiles
- [ ] Config diff banner with reset-all

---

## Phase 4: Type-safe Config API

The `GET` and `PATCH /configs/<name>` endpoints currently pass around raw dicts
(`parsed` as `Record<string, unknown>`, PATCH body as untyped JSON). The Pydantic
models in `yadc/core/config.py` already define the full config schema (`Config`,
`ConfigApi`, `ConfigSettings`, `ConfigReasoning`, `ConfigPrompt`, etc.). Use them.

### Backend

- `GET /configs/<name>`: parse TOML → `Config.model_validate(raw, context={"strict": False})`
  → `model_dump()` for the `parsed` response field. This gives a validated, typed
  shape with defaults filled in for missing fields, instead of a raw partial dict.
- `PATCH /configs/<name>`: validate the incoming patch against the config structure
  (at minimum, validate the merged result before writing). Catch `ValidationError`
  and return 400 with field-level error details.
- The `parsed` response shape becomes predictable and self-documenting.

### Frontend

- Generate or hand-write a TypeScript type matching `Config` (the Pydantic dump shape).
  Replace `Record<string, unknown>` on `DatasetConfigDetail.parsed` with this type.
- Extract caption defaults from the typed `parsed` instead of using `?.` chains on `any`.
- PATCH call body gets typed too — only valid config keys accepted.

### Status

- [ ] Backend: validate GET response through `Config` model
- [ ] Backend: validate PATCH input through `Config` model
- [ ] Frontend: typed `Config` interface replacing `Record<string, unknown>`

---

## Notes

- Backend `apply_config_overrides()` already correctly merges: `opts → env → TOML`. The frontend just needs visibility into what the TOML says.
- The caption settings panel (per-session overrides via localStorage) and the dataset config (persistent TOML) serve different purposes but share the same field shape.
- The existing `EditDatasetDialog` (raw TOML editor) stays for power-user full-config editing. The PATCH endpoint provides structured editing for the common caption fields.
