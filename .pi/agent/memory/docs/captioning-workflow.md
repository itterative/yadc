---
name: captioning-workflow
description: End-to-end captioning workflow — dataset loading, filtering, prediction loop, saving.
category: architecture
---

# Captioning Workflow

## Dataset Loading (`_load_dataset()` in cli_caption.py)

1. Parse TOML file → raw dict
2. Optionally merge with user config (`cmd_configs.merge_user_config()`)
3. Load user environment → merge `api.url`, `api.token`, `api.model_name` (CLI > env > config)
4. `parse_config()` → v1/v2 Config
5. `resolve_dataset()` → list of `DatasetImage`

## Image Filtering

- Skip images with existing caption files (unless `--overwrite`)
- Skip images with existing draft files (when `--draft` is set, unless `--overwrite`)
- Log count of skipped images

## Interactive Actions

In interactive mode, for each image the user can choose:

| Key | Action | Description |
|-----|--------|-------------|
| `c` | continue | Generate caption (or accept existing) |
| `q` | quit | Stop captioning entirely |
| `s` | skip | Skip this image |
| `r` | retry | Re-generate caption |
| `e` | edit | Edit image TOML metadata in `$EDITOR` |
| `p` | prompts | Show system/user prompts without generating |
| `y` | reply | Add a user reply to continue conversation |
| `x` | clear replies | Clear reply history |

## Captioning Modes

### Single-round (`rounds <= 1` or has extra_messages/reply)
- One call to `predict()` or `predict_stream()`
- Supports reply history (multi-turn conversation)

### Multi-round (`rounds > 1`)
- N-1 intermediate rounds: generate caption, prompt for acceptance
- Final round: generate improved caption using all accepted rounds as context
- Uses `caption_rounds` template variable

## Saving

- **Normal**: `update_caption()` writes `.txt` + `.toml`, saves `.history~`
- **Draft** (`--draft NAME`): writes to `.{name}.draft~` instead
- History saved only once per image (`when_not_exists=True`) for the original caption

## Model Setup

```
APICaptioner(
    api_url, api_token, prompt_template,
    store_conversation, image_quality,
    reasoning, reasoning_effort, reasoning_exclude_output,
    reasoning_start_token, reasoning_end_token,
    cache, response_logger,
)
model.load_model(model_name)
```

## Usage Logging

After captioning completes, `model.log_usage()` prints total prompt/response/reasoning tokens consumed.
