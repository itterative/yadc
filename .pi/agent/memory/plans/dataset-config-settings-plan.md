---
name: dataset-config-settings-plan
description: Caption settings ↔ dataset TOML config integration.
status: Mostly implemented
category: meta
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

### 1. Backend: Validate on reads (`GET /configs/<name>`)

In `yadc/api/controllers/api_configs.py`, after `toml.loads(content)`:

```python
from yadc.core.config import parse_config
from pydantic import ValidationError

parsed = toml.loads(content)
validation_error = None
try:
    parse_config(parsed, strict=False)
except ValidationError as e:
    validation_error = [
        {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
        for err in e.errors()
    ]
```

Return `validation_error` in the JSON response alongside `parsed` and `content`:

```python
return jsonify({
    "name": name,
    "config_path": info.config_path,
    "content": content,
    "parsed": parsed,
    "validation_error": validation_error,
})
```

Why keep going on error? The raw TOML editor must still open a malformed file so the user can fix it. The win is twofold: for **well-formed** configs `parsed` is guaranteed to match the `Config` schema, and for **broken** configs the frontend gets structured error details to display inline (e.g. a banner or badge in the config editor).

### 2. Backend: Strict validate on writes (`PATCH /configs/<name>`)

After `deep_merge(parsed, body)`, validate the merged dict **before** serializing:

```python
merged = deep_merge(parsed, body)
try:
    parse_config(merged, strict=False)
except ValidationError as e:
    return jsonify({
        "error": "Validation failed",
        "details": [
            {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
    }), 400
```

This catches type/range errors (e.g. `max_tokens = 50`, `rounds = 0`, `image_quality = "best"`) and returns field-level details the frontend can display.

### 3. Frontend: Hand-written `Config` TypeScript interface

Add a typed `Config` tree in `yadc/webui/src/lib/stores/configs.ts` (or a new `configTypes.ts`) that mirrors the Pydantic dump shape. All nested objects optional to match TOML semantics:

```ts
export interface Config {
  api?: ConfigApi;
  prompt?: ConfigPrompt;
  settings?: ConfigSettings;
  reasoning?: ConfigReasoning;
  env?: string;
  interactive?: boolean;
  rounds?: number;
  caption_suffix?: string;
  overwrite_captions?: boolean;
}

export interface ConfigApi {
  url?: string;
  token?: string;
  model_name?: string;
}

export interface ConfigPrompt {
  name?: string;
  template?: string;
}

export interface ConfigSettings {
  max_tokens?: number;
  store_conversation?: boolean;
  image_quality?: 'auto' | 'high' | 'low';
  advanced?: ConfigSettingsAdvanced;
}

export interface ConfigSettingsAdvanced {
  system_role?: string;
  user_role?: string;
  assistant_role?: string;
  assistant_prefill?: string;
  [key: string]: unknown; // extra="allow"
}

export interface ConfigReasoning {
  enable?: boolean;
  thinking_effort?: 'low' | 'medium' | 'high';
  exclude_from_output?: boolean;
  advanced?: ConfigReasoningAdvanced;
}

export interface ConfigReasoningAdvanced {
  thinking_start?: string;
  thinking_end?: string;
}
```

Update `DatasetConfigDetail`:

```ts
export interface DatasetConfigDetail {
  name: string;
  config_path: string;
  content: string;
  parsed: Config;
  validation_error?: Array<{ loc: string[]; msg: string; type: string }>;
}
```

Also type the patch body:

```ts
export async function patchConfig(name: string, patch: Partial<Config>): Promise<DatasetConfigDetail> { ... }
```

> **Sync maintenance:** Add a comment on both the Pydantic `Config` class and the TS `Config` interface pointing to each other. The schema is small and stable, so hand-written types are pragmatic. If it grows much larger, we can later add `Config.model_json_schema()` → codegen.

### 4. Frontend: Remove `as Record<string, unknown>` casts

**`CaptionSettings.svelte`** — replace the deeply nested `as` chain:

```ts
// before
const p = config.parsed as Record<string, unknown>;
const settings = (p.settings as Record<string, unknown>) ?? {};
datasetDefaults = {
    maxTokens: (settings.max_tokens as number) ?? HARDCODED_DEFAULTS.maxTokens,
    ...
};

// after
datasetDefaults = {
    maxTokens: config.parsed.settings?.max_tokens ?? HARDCODED_DEFAULTS.maxTokens,
    imageQuality: config.parsed.settings?.image_quality ?? HARDCODED_DEFAULTS.imageQuality,
    draftName: config.parsed.draft ?? HARDCODED_DEFAULTS.draftName,
    ...
};
```

**`DatasetConfig.svelte`** — same pattern:

```ts
// before
const p = config.parsed as Record<string, unknown>;
const settings = (p.settings as Record<string, unknown>) ?? {};
maxTokens = loadedMaxTokens = ('max_tokens' in settings ? settings.max_tokens : null) as number | null;

// after
maxTokens = loadedMaxTokens = config.parsed.settings?.max_tokens ?? null;
```

### 5. Frontend: Surface validation errors from PATCH

Update `apiErrorMessage` in `$lib/api.ts` to check for `body.details` and format field-level messages:

```ts
if (body.details && Array.isArray(body.details)) {
    errorFromBody = body.details
        .map((d: { loc?: string[]; msg: string }) =>
            d.loc ? `${d.loc.join('.')}: ${d.msg}` : d.msg
        )
        .join('; ');
}
```

This lets `DatasetConfig.svelte` show specific errors like  
`settings.max_tokens: must be between 100 and 16384` in the existing `saveError` banner.

Similarly, surface `validation_error` from `GET` in `DatasetConfig.svelte` with a persistent warning banner when the config has issues.

### Key design choice: why not `model_dump()` defaults into `parsed`?

Returning `config.model_dump(mode="json")` would break `DatasetConfig.svelte`'s nullable fields (e.g. "clear max tokens to remove it from TOML") because the frontend couldn't tell whether `512` was explicitly set or just the default. Keeping `parsed` as the **raw TOML dict** (validated but not dumped from the model) preserves that exact semantic while still giving us type safety through the TS interface.

### Status

- [x] Backend: validate GET response + return `validation_error`
- [x] Backend: validate PATCH input through `Config` model
- [x] Frontend: typed `Config` interface replacing `Record<string, unknown>`
- [x] Frontend: surface GET validation errors in DatasetConfig UI
- [x] Frontend: surface PATCH validation errors in save error banner

> **Note:** During implementation, discovered that `draft` was being read from `parsed.draft` in `CaptionSettings.svelte`, but `draft` is not a field on the `Config` model. The original `as Record<string, unknown>` cast masked this. It now falls back to the hardcoded default (`''`) since the field does not exist in the schema.

---

## Notes

- Backend `apply_config_overrides()` already correctly merges: `opts → env → TOML`. The frontend just needs visibility into what the TOML says.
- The caption settings panel (per-session overrides via localStorage) and the dataset config (persistent TOML) serve different purposes but share the same field shape.
- The existing `EditDatasetDialog` (raw TOML editor) stays for power-user full-config editing. The PATCH endpoint provides structured editing for the common caption fields.
