## Table of Contents

* [General](#general)
  * [What is yadc?](#what-is-yadc)
  * [Which APIs are supported?](#which-apis-are-supported)
* [Configurations](#configurations)
  * [What is a user config?](#what-is-a-user-config)
  * [How do I create a user config?](#how-do-i-create-a-user-config)
  * [Can I use multiple configs?](#can-i-use-multiple-configs)
* [Environments](#environments)
  * [What is a user environment?](#what-is-a-user-environment)
  * [How do I set up an environment?](#how-do-i-set-up-an-environment)
  * [Are API tokens stored securely?](#are-api-tokens-stored-securely)
  * [How do I delete an environment setting?](#how-do-i-delete-an-environment-setting)
* [Templates](#templates)
  * [What is a user template?](#what-is-a-user-template)
  * [How do I create a template?](#how-do-i-create-a-template)
  * [How do I use a template?](#how-do-i-use-a-template)
* [Captioning](#captioning)
  * [What is required to start captioning?](#what-is-required-to-start-captioning)
  * [How do I enable streaming captions?](#how-do-i-enable-streaming-captions)
  * [How can I review captions before saving?](#how-can-i-review-captions-before-saving)
  * [Can I regenerate captions for already-processed images?](#can-i-regenerate-captions-for-already-processed-images)
  * [How do I run multiple captioning rounds?](#how-do-i-run-multiple-captioning-rounds)
* [Troubleshooting](#troubleshooting)
  * [I get an error: "Error loading dataset: invalid configuration: ..."](#i-get-an-error-error-loading-dataset-invalid-configuration-)
  * [My API token isn't being recognized](#my-api-token-isnt-being-recognized)
  * [How do I reset everything?](#how-do-i-reset-everything)
* [Advanced](#advanced)
  * [Can I override settings from the command line?](#can-i-override-settings-from-the-command-line)
  * [What logging levels are available?](#what-logging-levels-are-available)



## General

### What is yadc?

yadc (Yet Another Dataset Captioner) is a CLI tool for generating captions for image datasets using vision-capable language models. It supports reusable configurations, environment management, and customizable prompt templates.

### Which APIs are supported?
Both the [Gemini API](https://ai.google.dev/api/generate-content) and [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create) endpoints are supported. 

The list of tested APIs:
* Gemini API
* OpenAI API
* Openrouter API
* Koboldcpp
* Llamacpp



## Configurations

### What is a user config?

A **user config** is a reusable TOML file that stores captioning settings such as prompt templates, generation parameters, or formatting rules. You can apply it across multiple datasets.

### How do I create a user config?

Use the `configs add` command:

```bash
yadc configs add high_quality
```

This opens an editor where you can define your config.

### Can I use multiple configs?

No, only one user config can be used at a time via `--user-config`. However, you can include shared values by referencing environments or templates within the config.



## Environments

### What is a user environment?

A **user environment** securely stores sensitive or repeated API details such as:
- `api_url`
- `api_token`
- `api_model_name`

These can be reused across captioning datasets without exposing secrets in command lines or config files.

### How do I set up an environment?

Set values using `envs set`. Use `--env` to name the environment:

```bash
yadc envs set api_url https://api.example.com/v1 --env myapi
yadc envs set api_token abc123xyz --env myapi
yadc envs set api_model_name llava-13b --env myapi
```

Then use it during captioning:

```bash
yadc caption dataset.toml --env myapi
```

### Are API tokens stored securely?

Yes. API tokens are encrypted using a RSA key. The public key is stored alongside the env definitions, whereas the private key is stored in the os-specific keyring. Use `yadc envs show` to see redacted values, or `yadc envs get api_token` to view the actual value (if needed).

### How do I delete an environment setting?

Delete a specific key:

```bash
yadc envs delete api_token --env myapi
```

To remove all environments:

```bash
yadc envs clear --all
```



## Templates

### What is a user template?

A **user template** is a reusable prompt used during captioning (e.g., "Describe the scene and objects in detail"). Templates help standardize output across datasets.

### How do I create a template?

Run:

```bash
yadc templates add detailed
```

This opens an editor to write your prompt text.

### How do I use a template?

Pass it via `--user-template`:

```bash
yadc caption dataset.toml --user-template detailed
```

You can also reference templates inside dataset or user config files.



## Captioning

### What is required to start captioning?

You need:
1. A **dataset config file** (e.g., `dataset.toml`) defining the image source and output path.
2. Access to a compatible **vision-language API** (via direct override or environment).
3. (Optional) A user config or template.

Example minimal command:

```bash
yadc caption dataset.toml --api-url http://localhost:5001/v1 --api-model-name gemma-3-27b
```

### How do I enable streaming captions?

Use `--stream` to see captions generated in real time:

```bash
yadc caption dataset.toml --stream
```

### How can I review captions before saving?

Use interactive mode:

```bash
yadc caption dataset.toml --interactive
```

You'll be prompted to accept, edit, or skip each caption.

### Can I regenerate captions for already-processed images?

Yes, but you must use `--overwrite`:

```bash
yadc caption dataset.toml --overwrite
```

Otherwise, existing captions are skipped.

### How do I run multiple captioning rounds?

Use `--rounds` to specify how many times to caption each image:

```bash
yadc caption dataset.toml --rounds 3
```

Useful for generating diverse descriptions. After performing the captioning rounds, a final round is performed in order to select/refine the final caption.

*This option will incur extra costs. Additionally, model captioning errors might also accumulate, so extra attention is required when using multiple rounds.*


## Troubleshooting

### I get an error: "Error loading dataset: invalid configuration: ..."

Your dataset configuration is invalid. The error should contain an explication as to why the configuration failed.

Example:
```
Error loading dataset: invalid configuration: 1 validation error for Config
api
  Value error, api url must be provided [type=value_error, input_value={'model_name': '... 'url': '', 'token': ''}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error
```

This means that the API url has not been set in either the dataset TOML file, user environment (if provided), or user configuration (if provided). To fix the issue, you should create an new environment.

Make sure the environment exists:

```bash
yadc envs list
```

If missing, recreate it using `yadc envs set`.

Finally, you can use it by passing `--env` option in the caption command.

### My API token isn't being recognized

Check:
- The environment name matches (`--env`)
- The token was set correctly: `yadc envs get api_token --env your_env`
- No typos in the environment or key name

### How do I reset everything?

To clear all environments and start fresh:

```bash
yadc envs clear --all
```



## Advanced

### Can I override settings from the command line?

Yes. Command-line options take precedence:
- `--api-url`, `--api-token`, `--api-model-name` override dataset values
- `--user-template` overrides the dataset prompt template
- `--user-config` merges the dataset and user configuration, overriding the values from the dataset

Example:

```bash
yadc caption dataset.toml \
  --env myapi \
  --api-model-name debug-model \
  --user-template simple \
  --user-config myconfig
```

Here, only `api_model_name` is overridden; `api_url` and `api_token` come from `myapi`. The prompt template from `simple` will be used.

User config `myconfig` is merged with `dataset.toml`. Values which are provided in `myconfig` will take precedence over the ones in `dataset.toml`

```toml
# dataset "dataset.toml"
[api]
model_name = "dataset_model"

[settings]
max_tokens = 1024

# user config "myconfig"
[api]
model_name = "myconfig_model"

[settings.advanced]
temperature = 0.8

# final result
[api]
model_name = "myconfig_model"

[settings]
max_tokens = 1024

[settings.advanced]
temperature = 0.8
```

### What logging levels are available?

Use `--log-level` with any command:
- `info` (default)
- `warning`
- `error`
- `debug` (most verbose)

Example:

```bash
yadc caption dataset.toml --log-level debug
```

Useful for diagnosing connection or parsing issues.


