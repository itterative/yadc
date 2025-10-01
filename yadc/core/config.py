import pydantic

from .dataset import DatasetImage

class Config(pydantic.BaseModel):
    """
    Main configuration model used for configuring the CLI.

    Attributes:
        api: Configuration for connecting to the external API.
        settings: Runtime settings for prompt handling, token limits, and output behavior.
        dataset: Configuration for dataset paths and pre-loaded images.

        env: Which user environment to use.
        interactive: If True, enables interactive mode (e.g., manual confirmation between steps).
        rounds: Number of captioning rounds to perform per image (must be >= 1).
        caption_suffix: File extension for generated caption files (must start with '.').
        overwrite_captions: If True, existing caption files will be overwritten.
    """

    api: 'ConfigApi' = pydantic.Field(default_factory=lambda: ConfigApi())
    prompt: 'ConfigPrompt' = pydantic.Field(default_factory=lambda: ConfigPrompt())
    settings: 'ConfigSettings' = pydantic.Field(default_factory=lambda: ConfigSettings())
    reasoning: 'ConfigReasoning' = pydantic.Field(default_factory=lambda: ConfigReasoning())
    dataset: 'ConfigDataset' = pydantic.Field(default_factory=lambda: ConfigDataset())

    env: str = ''
    interactive: bool = False
    rounds: int = 1
    caption_suffix: str = '.txt'
    overwrite_captions: bool = False

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.caption_suffix.startswith('.'), f"invalid caption_suffix: {self.caption_suffix}"

            assert self.rounds > 0, 'rounds must be a positive number'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigApi(pydantic.BaseModel):
    """
    Configuration for connecting to a remote inference API.

    Attributes:
        url: Base URL of the API endpoint (must start with http:// or https://).
        token: Authorization token for API access (optional depending on server requirements).
        model_name: Identifier of the model to use on the API server (e.g., 'gpt-5-mini', 'gemini-2.5-flash').
    """

    url: str = ''
    token: str = ''
    model_name: str = ''

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.url, 'api url must be provided'
            assert self.url.startswith('http://') or self.url.startswith('https://'), 'api url must be an http link'

            assert self.model_name, 'api model_name must be provided'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigPrompt(pydantic.BaseModel):
    """
    Configuration for connecting to a remote inference API.

    Attributes:
        name: The name of the user/built-in template
        template: The prompt template itself
    """

    name: str = ''
    template: str = ''

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.name or self.template, 'either prompt name or prompt template must be provided in the config'
        except AssertionError as e:
            raise ValueError(e)

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
    image_quality: str = 'auto'

    advanced: 'ConfigSettingsAdvanced' = pydantic.Field(default_factory=lambda: ConfigSettingsAdvanced())

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert 100 <= self.max_tokens <= 16384, 'config max_tokens must be between 100 and 16384'

            assert self.image_quality in ('auto', 'high', 'low'), 'config image_quality must be one of: auto, high, low'
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

    system_role: str = 'system'
    user_role: str = 'user'
    assistant_role: str = ''

    assistant_prefill: str = ''

    model_config = pydantic.ConfigDict(extra='allow')

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.system_role in ('system', 'developer'), 'advanced settings system role must be one of: developer, system'
            assert self.user_role in ('user'), 'advanced settings user role must be one of: user'
            assert self.assistant_role in ('', 'assistant', 'model'), 'advanced settings assistant role must be one of: (empty), assistant, model'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigReasoning(pydantic.BaseModel):
    enable: bool = False
    thinking_effort: str = 'low'
    exclude_from_output: bool = True

    advanced: 'ConfigReasoningAdvanced' = pydantic.Field(default_factory=lambda: ConfigReasoningAdvanced())

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.thinking_effort in ('high', 'medium', 'low'), 'reasoning thinking_effor must be one of: high, medium, low'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigReasoningAdvanced(pydantic.BaseModel):
    thinking_start: str = '<think>'
    thinking_end: str = '</think>'

class ConfigDataset(pydantic.BaseModel):
    """
    Configuration for dataset input sources.

    Attributes:
        paths: List of file or directory paths to load images and their configuration from.
               Images are searched at top-level only.
        images: Pre-loaded list of image configuration.
                Configuration defined in this list will be merged with matching ones from the paths.
                Extra arguments will take priority when given here.
    """

    paths: list[str] = []
    images: list[DatasetImage] = []
