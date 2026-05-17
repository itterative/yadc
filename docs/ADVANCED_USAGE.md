## Table of Contents

* [Shared configuration](#shared-configuration)
* [Thinking models](#thinking-models)
* [Drafts](#drafts)

## Shared configuration
If you have access to several APIs, it might be a good idea to set up different user environments for each, as well as different user configuration.

Example:
```toml
# OpenAI

# $ yadc envs set api_url "https://api.openai.com/v1" --env openai
# $ yadc envs set api_token "$OPENAI_TOKEN" --env openai

# $ yadc configs edit openai_gpt5_mini
env = "openai"

[api]
model_name = "gpt-5-mini"

# Openrouter

# $ yadc envs set api_url "https://openrouter.ai/api/v1" --env openrouter
# $ yadc envs set api_token "$OPENROUTER_TOKEN" --env openrouter

# $ yadc configs edit openrouter_o5_mini
env = "openrouter"

[api]
model_name = "openai/gpt-5-mini"
```

After setting this up, you can simply switch the models like this:
```bash
$ yadc caption dataset.toml --user-config openai_gpt5_mini
$ yadc caption dataset.toml --user-config openrouter_o5_mini
```

## Thinking models
When using official APIs (e.g. Gemini, OpenAI, Openrouter, etc), reasoning/thinking tokens are provided in a structured way, so it's not hard to figure out what is the caption and what are the model's thoughts.

However, local APIs might not always provide the correct structure; this depends mainly on whether the model weights contain the correct metadata and the local server (e.g., Koboldcpp, llama.cpp, etc) can handle the metadata.

In cases where this is not handled correctly, you might want to add configuration specifically for this.

Example:
```toml
[settings.advanced]
# optional, but recommended if the API allows you to prefill the assistant's response
assistant_prefill = "<think>"

[reasoning]
enable = true

[reasoning.advanced]
thinking_start = "<think>"
thinking_end = "</think>"
```

In this example, we tell the captioner which tokens should be used for handling the thinking tokens, as well as prefilling the assistant's response.

Several (GGUF) models might exhibit this issue. For example, Kimi-VL uses a custom token for its thought process.

Kimi-VL configuration:
```toml
[api]
model_name = "Kimi-VL-A3B-Thinking-2506-Q8_0.gguf"

[settings.advanced]
assistant_prefill = "◁think▷"

[reasoning]
enable = true

[reasoning.advanced]
thinking_start = "◁think▷"
thinking_end = "◁/think▷"
```

## Drafts

Drafts let you generate intermediate captions with different models, then use those results in a final captioning round to produce a more refined output.

Each draft is saved as a separate file alongside the image, using the naming convention `image_name.draft_name.draft~`. For example, generating a draft named `gemma` for `photo.png` creates `photo.gemma.draft~`. This keeps drafts separate from the final caption file and makes them easy to identify, copy, or clean up.

### Generating drafts

Use the `--draft` option to save the caption output as a named draft instead of writing the final caption:

```bash
# Generate a draft with Gemma
yadc caption dataset.toml --draft gemma --api-model-name "gemma-3-27b-it"

# Generate a draft with Qwen
yadc caption dataset.toml --draft qwen --api-model-name "qwen2.5vl-32b"
```

Images that already have a draft with the given name are skipped. Use `--overwrite` to regenerate them.

### Using drafts in the final caption

When running `yadc caption` without `--draft`, all existing draft files for each image are read and made available as a `drafts` variable in the prompt template. This is a dictionary mapping draft names to their content. Note that `drafts` is only defined when at least one draft file exists, so your template should check for it.

Example template:
```jinja
{% set user_prompt %}
Provide a detailed description of the image within 1-2 paragraphs.
{% if drafts is defined %}

Use the following AI-generated drafts to refine your description:
{% for name, text in drafts.items() %}
Draft ({{ name }}):
{{ text }}

{% endfor %}
{% endif %}
{% endset %}
```

You can also reference individual drafts by name:
```jinja
{% set user_prompt %}
Describe the image.
{% if drafts is defined %}
{% if drafts.gemma %}
Gemma's description: {{ drafts.gemma }}
{% endif %}
{% if drafts.qwen %}
Qwen's description: {{ drafts.qwen }}
{% endif %}
{% endif %}
{% endset %}
```

Then run the final captioning:
```bash
yadc caption dataset.toml --user-template refined
```

### Cleaning up drafts

Use the `yadc draft remove` command to delete a named draft across all images in a dataset:

```bash
# Remove all 'gemma' drafts
yadc draft remove dataset.toml --name gemma

# Remove all 'v1' drafts
yadc draft remove dataset.toml --name v1
```

You can also list all drafts to see what's available:

```bash
yadc draft list dataset.toml
```

Or manually, since draft files use the `.draft~` suffix:

```bash
rm path_to_images/*.draft~
```

### Backing up captions as drafts

You can save existing captions as drafts before re-captioning:

```bash
# Save current captions as 'before' draft
yadc draft save dataset.toml --name before

# Re-caption...
yadc caption dataset.toml --overwrite

# View the old captions anytime
yadc draft show dataset.toml --name before
```
