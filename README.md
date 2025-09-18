# yadc
Yet Another Dataset Captioner

## Getting started

The easiest way to get started is using either [uv](https://github.com/astral-sh/uv) or python virtual environments.

* clone the repo: `git clone --branch 0.1.0 https://github.com/itterative/yadc`
* install venv: `uv venv` or `python3 -m venv .venv && . .venv/bin/activate`
* install requirements: `uv sync` or `pip install -r requirements.txt`
* run the captioner: `uv run --module yadc DATASET_TOML` or `python3 -m yadc DATASET_TOML`

If you want to use it as an executable, you can also install it directly using you preferred method. Some examples:
* with pipx, `pipx install git+https://github.com/itterative/yadc@0.2.1` then `yadc DATASET_TOML`
* with uvx, `uv tool install git+https://github.com/itterative/yadc@0.2.1` then `yadc DATASET_TOML`

### Requirements
* Python 3.11 (or later)
* OpenAI compatible API (e.g. koboldcpp)

## Config

The captioner takes in a toml file as config. Some examples are available in `configs/`.

Minimal example:

```toml
[api]
url = "http://localhost:5001"
model_name = "gemma-3n-E4B-it-Q8_0"

[dataset]
paths = [ "path_to_your_images" ]
```

### Dataset

In your config, you can either give certain paths or specificy images directory. When using them both at the same time, this allows you to override the toml settings for each image.

The toml files are optional, however they are recommended since you can improve the prompts sent to the model.

Example:

```toml
[settings]
prompt_template = """
{% set user_prompt %}
Describe the image. Use the following additional context when describing the image: {{ context }}
{% endset %}
"""

[[dataset.images]]
path = "path_to_image"
context = "this is an image of my labrador Nessie."
```

### Prompt templates

You can use [Jinja](https://jinja.palletsprojects.com/en/stable/) templates for prompting the models. Some examples can be found in `templates/` folder.

If you wish to create your own template, you can either put it in the above folter or on other parts of you local system. Any templates found in the current working directory will take precedence over any in the `templates/` folder.

You can use variables as provided in your `.toml` files associated with your images, or as part of the overrides in the databset toml file.

The template should set the following variable (as seen in example below):
* **system_prompt**: this sets the system prompt of the model
* **user_prompt**: this sets the user prompt of the chat
* **user_prompt_multiple_rounds**: when running multiple rounds of captioning, this sets the user prompt of the chat in the final round

Example template:
```jinja
{% set system_prompt %}
You are a useful assistant. You help the user caption the images they are sending. You shall provide only the caption, without any other additions.
{% endset %}

{% set user_prompt %}
Provide a detailed description of the image within 1-2 paragraphs.
{% endset %}

{% set user_prompt_multiple_rounds %}
Provide a detailed description of the image within 1-2 paragraphs.

Use the following set of descriptions to generate the best description of the image. Keep the common elements of each description. Remove any elements that are not valid.

{% for round in caption_rounds %}
Description #{{ round.iteration }}
{{ round.caption }}

{% endfor %}
{% endset %}
```

*When doing multiple round captioning, an extra variable `caption_rounds` is available. Use it to build your prompt.*

## Models
If you plan on using paid APIs (e.g. OpenAI), you should not send any images that may contain illegal content, as they are [scanned upon submission](https://platform.openai.com/docs/guides/your-data#image-and-file-inputs).

### Local model recommendations

The following are recommended:
  * Qwen2.5-VL - supports up to 1MP images, however the model doesn't always follow instructions (e.g. formatting in markdown including headers and list, use of quotes for names, etc)
  * Gemma-3 - better at following instruction than Qwen2.5-VL, but can hallucinate more
  * Gemma-3n - fast and fairly high quality for the sizes it is available in

### Supported backends
The following backends are supported: OpenAI (or compatible APIs), Koboldcpp.

Future support will be improved for vLLM and Ollama.

*Note: Certain quantized models might not contain all the files needed for image vision. These are usually accompanied in the Huggingface repository (usually named as **mmproj**).*

#### Koboldcpp

If you only have one model to use as your captioner, you can set the dataset configuration to that model. The captioner will check that the right model is loaded.

If you have a few models you might want to use, you should set up Koboldcpp to be able to load different `.kcpps` files. You can make use the the Koboldcpp cli to [export](https://github.com/LostRuins/koboldcpp/wiki#what-is---config-what-are-kcpps-files) the config and use then set the directory from where the [settings are loaded](https://github.com/LostRuins/koboldcpp/wiki#what-is-admin-mode-can-i-switch-models-at-runtime). It's recommended to export the settings with the same name as the model (e.g. if the name is name 'Qwen2.5-VL-32B-Instruct-UD-Q4_K_XL', then the setting file name should be 'Qwen2.5-VL-32B-Instruct-UD-Q4_K_XL.kcpps'). By doing it this way, you can prevent the model from being reloaded constantly whenever you start the captioning process.

If you enable the admin mode (needed for loading multiple models), you might want to set up an admin token, especially if you will expose the endpoints outside of your local network. Additionally, SSL is also recommended (either through the Koboldcpp cli directly, or through a proxy) if the endpoint is available on the internet. Using self-signed certificates is not recommended unless you are an advanced user.
