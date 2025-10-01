## Table of Contents

* [Shared configuration](#shared-configuration)
* [Thinking models](#thinking-models)

## Shared configuration
If you have access to several APIs, it might be a good idea to set up different user environments for each, as well as different user configuration.

Example:
```toml
# OpenAI

# $ yadc envs set api_url "https://api.openai.com/v1" --env openai
# $ yadc envs set api_token "$OPENAI_TOKEN" --env openai

# $ yadc configs add openai_gpt5_mini
env = "openai"

[api]
model_name = "gpt-5-mini"

# Openrouter

# $ yadc envs set api_url "https://openrouter.ai/api/v1" --env openrouter
# $ yadc envs set api_token "$OPENROUTER_TOKEN" --env openrouter

# $ yadc configs add openrouter_o5_mini
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
