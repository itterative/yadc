## Table of Contents

- [Caption a Dataset](#caption-a-dataset)
- [Manage Configurations](#manage-configurations)
- [Manage Environments](#manage-environments)
- [Manage Templates](#manage-templates)
- [Version](#version)

---

## Caption a Dataset

Start captioning a dataset using a dataset config file.

```bash
yadc caption dataset_config.toml
```

### Examples

Override API settings and enable streaming:

```bash
yadc caption dataset.toml \
  --api-url "https://openrouter.ai/api/v1" \
  --api-token "$API_TOKEN" \
  --api-model-name "google/gemini-2.5-flash" \
  --stream
```

Use a saved environment, config, and template:

```bash
yadc caption dataset.toml \
  --env openrouter \
  --user-config qwen3_vl_235b \
  --user-template high_quality_caption_template
```

Run in interactive mode to review captions before saving:

```bash
yadc caption dataset.toml --interactive
```

Force overwrite of existing captions:

```bash
yadc caption dataset.toml --overwrite
```

---

## Manage Configurations

User configs let you reuse settings (e.g., model parameters, prompt options) across datasets.

### List Configs

```bash
yadc configs list
```

### Add a Config

```bash
yadc configs add my_config
```

*Opens an editor to define the config in TOML format.*

Example config:
```toml
[settings.advanced]
temperature = 0.8

[reasoning]
enable = true
thinking_effort = "high"
```

### Edit a Config

```bash
yadc configs edit my_config
```

### Delete a Config

```bash
yadc configs delete my_config
```

---

## Manage Environments

Environments securely store API credentials and model URLs.

### List Environments

```bash
yadc envs list
```

### Set API Credentials

```bash
yadc envs set api_url "https://openrouter.ai/api/v1" --env openrouter_qwen3_vl
yadc envs set api_token "$API_TOKEN" --env openrouter_qwen3_vl
yadc envs set api_model_name "google/gemini-2.5-flash" --env openrouter_qwen3_vl
```

*If the value is not specified, it will be taken from stdin (useful for secrets such as API tokens)*

### Retrieve a Setting

```bash
yadc envs get api_model_name --env openrouter_qwen3_vl
```

### Show All Settings (Redacted)

```bash
yadc envs show --env openrouter_qwen3_vl
```

### Delete a Setting

```bash
yadc envs delete api_token --env openrouter_qwen3_vl
```

### Clear Environment

Clear one environment:

```bash
yadc envs clear --env temp
```

Clear all environments:

```bash
yadc envs clear --all
```

---

## Manage Templates

Templates define reusable caption prompts.

### List Templates

```bash
yadc templates list
```

### Add a Template

```bash
yadc templates add detailed_description
```

*Opens an editor to define the prompt template in Jinja format.*

Example template content:
```jinja
{% set user_prompt %}
Describe this image in detail, focusing on objects, colors, and scene context.
{% endset %}
```

### Edit a Template

```bash
yadc templates edit detailed_description
```

### Delete a Template

```bash
yadc templates delete detailed_description
```

---

## Version

Print the current version:

```bash
yadc version
```

