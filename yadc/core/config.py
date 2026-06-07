"""Dataset config Pydantic models (v1/v2) and ``parse_config``.

``Config`` is the top-level model used by both the CLI and the web UI
to drive a captioning run. It groups API connection settings
(``ConfigApi``), prompt / template settings (``ConfigPrompt``),
runtime knobs like image quality and token limits (``ConfigSettings``),
reasoning config (``ConfigReasoning``), and a list of dataset entries
(``ConfigDatasetEntry``, each with a path, optional inline images, and
extras). Top-level flags cover env selection, interactive mode, round
count, caption suffix, and overwrite behaviour.

``parse_config`` reads a TOML file and returns a validated
``Config`` — the dataset resolver (``yadc.core.dataset_resolver``) and
the captioning runner (``yadc.core.captioning``) consume it to
enumerate images and run predictions.
"""

from typing import Any, ClassVar

import pydantic

from yadc.core.constants import DEFAULT_THINKING_END, DEFAULT_THINKING_START

from .dataset import DatasetImage


class Config(pydantic.BaseModel):
    """
    Main configuration model (v2) used for configuring the CLI.

    Attributes:
        api: Configuration for connecting to the external API.
        settings: Runtime settings for prompt handling, token limits, and output behavior.
        dataset: List of dataset entries, each with a path, inline images, and extras.

        env: Which user environment to use.
        interactive: If True, enables interactive mode (e.g., manual confirmation between steps).
        rounds: Number of captioning rounds to perform per image (must be >= 1).
        caption_suffix: File extension for generated caption files (must start with '.').
        overwrite_captions: If True, existing caption files will be overwritten.
    """

    api: "ConfigApi" = pydantic.Field(default_factory=lambda: ConfigApi())
    prompt: "ConfigPrompt" = pydantic.Field(default_factory=lambda: ConfigPrompt())
    settings: "ConfigSettings" = pydantic.Field(default_factory=lambda: ConfigSettings())
    reasoning: "ConfigReasoning" = pydantic.Field(default_factory=lambda: ConfigReasoning())
    dataset: list["ConfigDatasetEntry"] = pydantic.Field(default_factory=list)

    env: str = ""
    interactive: bool = False
    rounds: int = 1
    caption_suffix: str = ".txt"
    overwrite_captions: bool = False

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert self.caption_suffix.startswith("."), f"invalid caption_suffix: {self.caption_suffix}"

            assert self.rounds > 0, "rounds must be a positive number"
        except AssertionError as e:
            raise ValueError(e)

        return self


class ConfigApi(pydantic.BaseModel):
    """
    Configuration for connecting to a remote inference API.

    Attributes:
        url: Base URL of the API endpoint (must start with http:// or https:// when provided).
        token: Authorization token for API access (optional depending on server requirements).
        model_name: Identifier of the model to use on the API server (e.g., 'gpt-5-mini', 'gemini-2.5-flash').
    """

    url: str = ""
    token: str = ""
    model_name: str = ""

    @pydantic.model_validator(mode="after")
    def validate_(self, info: pydantic.ValidationInfo):
        strict = info.context.get("strict", True) if info.context else True

        if strict:
            if not self.url:
                raise ValueError("api url must be provided")
            if not self.model_name:
                raise ValueError("api model_name must be provided")

        if self.url:
            if not self.url.startswith(("http://", "https://")):
                raise ValueError("api url must be an http link")

        return self


class ConfigPrompt(pydantic.BaseModel):
    """
    Configuration for the prompt template.

    Attributes:
        name: The name of the user/built-in template
        template: The prompt template itself

    In strict mode (default, used by CLI), at least one must be provided.
    In non-strict mode, both may be empty — the default template is used at
    caption time.
    """

    name: str = ""
    template: str = ""

    @pydantic.model_validator(mode="after")
    def validate_(self, info: pydantic.ValidationInfo):
        strict = info.context.get("strict", True) if info.context else True

        if strict and not self.name and not self.template:
            raise ValueError("either prompt name or prompt template must be provided in the config")

        return self


class ConfigSettings(pydantic.BaseModel):
    """
    Configuration for runtime behavior and prompt generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate (between 100 and 2048).

        store_conversation: If True, retains full conversation history (depends on API implementation).
        image_quality: Image upload quality setting; one of 'auto', 'high', or 'low'.
    """

    max_tokens: int = 512

    store_conversation: bool = False
    image_quality: str = "auto"

    advanced: "ConfigSettingsAdvanced" = pydantic.Field(default_factory=lambda: ConfigSettingsAdvanced())

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert 100 <= self.max_tokens <= 16384, "config max_tokens must be between 100 and 16384"

            assert self.image_quality in ("auto", "high", "low"), "config image_quality must be one of: auto, high, low"
        except AssertionError as e:
            raise ValueError(e)

        return self


class ConfigSettingsAdvanced(pydantic.BaseModel):
    """
    Configuration for runtime behavior and prompt generation.

    Attributes:
        system_role (str): The role to use in the system prompt (either developer or system). This is useful for newer OpenAI models.
        user_role (str): The role to use in the user prompt
        assistant_role (str): The role to use in the assistant prompt
        assistant_prefill: (str): Used to prefill the assistant's responses

    Extra Fields:
        Any additional fields will be passed in the requests to the API.
    """

    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = ""

    assistant_prefill: str = ""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert self.system_role in ("system", "developer"), "advanced settings system role must be one of: developer, system"
            assert self.user_role in ("user"), "advanced settings user role must be one of: user"
            assert self.assistant_role in ("", "assistant", "model"), "advanced settings assistant role must be one of: (empty), assistant, model"
        except AssertionError as e:
            raise ValueError(e)

        return self


class ConfigReasoning(pydantic.BaseModel):
    enable: bool = False
    thinking_effort: str = "low"
    exclude_from_output: bool = True

    advanced: "ConfigReasoningAdvanced" = pydantic.Field(default_factory=lambda: ConfigReasoningAdvanced())

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert self.thinking_effort in ("high", "medium", "low"), "reasoning thinking_effor must be one of: high, medium, low"
        except AssertionError as e:
            raise ValueError(e)

        return self


class ConfigReasoningAdvanced(pydantic.BaseModel):
    thinking_start: str = DEFAULT_THINKING_START
    thinking_end: str = DEFAULT_THINKING_END


class ConfigDatasetEntry(pydantic.BaseModel):
    """
    A single dataset entry (v2 config).

    Attributes:
        path: Directory path to load images from (searched at top-level only).
        images: Pre-loaded list of image configuration.
                Configuration defined in this list will be merged with matching ones from the path.
                Extra arguments will take priority when given here.
        extras: Dataset-level template variables. These are merged into every image's template context
                as defaults — per-image extras (from TOML files or inline images) override these.
    """

    path: str = ""
    images: list[DatasetImage] = []
    extras: dict[str, object] = {}


# --- v1 config (legacy, used for deserialization then converted) ---


class ConfigDataset(pydantic.BaseModel):
    """
    Legacy v1 dataset configuration.

    Attributes:
        paths: List of file or directory paths to load images and their configuration from.
               Images are searched at top-level only.
        images: Pre-loaded list of image configuration.
                Configuration defined in this list will be merged with matching ones from the paths.
                Extra arguments will take priority when given here.
    """

    paths: list[str] = []
    images: list[DatasetImage] = []


class ConfigV1(pydantic.BaseModel):
    """Legacy v1 config model. Deserialized then converted to v2 Config."""

    api: "ConfigApi" = pydantic.Field(default_factory=lambda: ConfigApi())
    prompt: "ConfigPrompt" = pydantic.Field(default_factory=lambda: ConfigPrompt())
    settings: "ConfigSettings" = pydantic.Field(default_factory=lambda: ConfigSettings())
    reasoning: "ConfigReasoning" = pydantic.Field(default_factory=lambda: ConfigReasoning())
    dataset: "ConfigDataset" = pydantic.Field(default_factory=lambda: ConfigDataset())

    env: str = ""
    interactive: bool = False
    rounds: int = 1
    caption_suffix: str = ".txt"
    overwrite_captions: bool = False

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert self.caption_suffix.startswith("."), f"invalid caption_suffix: {self.caption_suffix}"

            assert self.rounds > 0, "rounds must be a positive number"
        except AssertionError as e:
            raise ValueError(e)

        return self

    def to_v2(self) -> Config:
        """Convert this v1 config to a v2 Config."""
        entries: list[ConfigDatasetEntry] = []

        for path in self.dataset.paths:
            entries.append(ConfigDatasetEntry(path=path))

        if self.dataset.images:
            entries.append(ConfigDatasetEntry(images=self.dataset.images))

        # FIXME: model_construct bypasses all validation — could we instead pass the
        # validation context through to_v2 so the v2 Config is properly validated
        # with the same strict flag?
        return Config.model_construct(
            api=self.api,
            prompt=self.prompt,
            settings=self.settings,
            reasoning=self.reasoning,
            dataset=entries,
            env=self.env,
            interactive=self.interactive,
            rounds=self.rounds,
            caption_suffix=self.caption_suffix,
            overwrite_captions=self.overwrite_captions,
        )


def parse_config(raw: dict[str, Any], *, strict: bool = True) -> Config:
    """
    Parse a raw TOML dict into a Config.

    Tries v1 first (``[dataset]`` single table with ``paths`` and ``images``).
    If that fails, falls back to v2 (``[[dataset]]`` array of tables with
    ``path``, ``images``, ``extras``).

    Args:
        raw: Raw TOML dict.
        strict: If True (default), enforce CLI-level validation (api url/model
            and prompt must be provided). If False, allow partial configs
            (used by the webui which provides these at caption time).
    """
    ctx = {"strict": strict}

    # Ensure api/prompt sections exist so default factories don't fire
    # without validation context (which would enforce strict checks on empty defaults).
    raw = {**raw}
    raw.setdefault("api", {})
    raw.setdefault("prompt", {})

    try:
        return ConfigV1.model_validate(raw, context=ctx).to_v2()
    except pydantic.ValidationError:
        pass

    return Config.model_validate(raw, context=ctx)
