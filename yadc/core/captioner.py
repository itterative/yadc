from typing import Generator

import io
import abc
import base64
import pathlib

import jinja2
import pydantic

from PIL import Image

from yadc.core import logging
from yadc.core.dataset import DatasetImage
from yadc import templates

_logger = logging.get_logger(__name__)

class CaptionerRound(pydantic.BaseModel):
    """
    Represents a single round of captioning in a multi-round captioning.

    Attributes:
        iteration (int): The sequence number of this caption round (e.g., 1st, 2nd).
        caption (str): The caption generated in this round.
    """

    iteration: int
    caption: str

class Captioner(abc.ABC):
    """
    Abstract base class for image captioning models.

    This class provides a standardized interface for loading models, encoding images,
    generating prompts via Jinja2 templates, and producing captions from images.
    Subclasses must implement model-specific logic for prediction and model management.

    The captioning process supports:
    - Customizable prompt templates using Jinja2
    - Multi-round captioning (e.g., for iterative refinement)
    - Image resizing and base64 encoding with size constraints
    - Logging and debugging of generated prompts

    Template Loading:
    - Looks for templates first in the current working directory
    - Falls back to built-in templates in `yadc.templates`
    - Supports overriding templates via direct string input

    Example template variables (from `DatasetImage` fields):
    - `image_id`, `file_path`, `width`, `height`, etc.

    Usage:
    ```
        class MyCaptioner(Captioner):
            def load_model(self, model_repo, **kwargs):
                ...
            def predict_stream(self, image:, **kwargs)
                ...
            def predict(self, image, **kwargs):
                ...

        captioner = MyCaptioner(prompt_template_name="custom.jinja")
        captioner.load_model("my-model-id")
        caption = captioner.predict(dataset_image)
    ```
    """

    def __init__(self, **kwargs):
        """
        Initializes the Captioner with optional template configuration.

        Args:
            **kwargs: Optional keyword arguments:
                - `prompt_template_name` (str): Filename of the Jinja2 template to use (default: 'default').
                - `prompt_template` (str): Direct template string to override file-based templates.
        """

        self._prompt_template_name: str = kwargs.pop('prompt_template_name', 'default')
        self._prompt_template: str = kwargs.pop('prompt_template', '').strip()

        self._jinja = jinja2.Environment(
            loader=jinja2.FunctionLoader(self._load_jinja_template),
            lstrip_blocks=True,
            trim_blocks=True,
            keep_trailing_newline=False,
        )

    def _unindent_template(self, template: str):
        template = template.strip()
        return '\n'.join([ line.lstrip() for line in template.splitlines() ])

    def _load_jinja_template(self, template: str):
        """
        Loads a Jinja2 template by name using a custom loading mechanism.

        Resolves special template names:
        - `__system_prompt__`: Loads the system prompt combining default and user templates.
        - `__user_prompt__`: Loads the user prompt for single-round captioning.
        - `__user_prompt_multiple_rounds__`: Loads the prompt for multi-round interactions.
        - `__default_template__`: Refers to the built-in default template.
        - `__user_template__`: Refers to the user-provided template (file or string).

        Template resolution order:
        1. Direct string via `prompt_template`
        2. File in current working directory
        3. Built-in template from `yadc.templates`

        Args:
            template (str): The logical template name to load.

        Returns:
            str: The loaded template content.

        Raises:
            ValueError: If an invalid template name is requested.
        """

        if template == '__system_prompt__':
            return self._unindent_template('''
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.system_prompt|default(default_template.system_prompt, true) }}
            ''')

        if template == '__user_prompt__':
            return self._unindent_template('''
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.user_prompt|default(default_template.user_prompt, true) }}
            ''')
        
        if template == '__user_prompt_multiple_rounds__':
            return self._unindent_template('''
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.user_prompt_multiple_rounds|default(default_template.user_prompt_multiple_rounds, true) }}
            ''')

        if template == '__default_template__':
            return templates.default_template()
        elif template == '__user_template__':
            # early exit if prompt template is given directly
            if self._prompt_template:
                return self._prompt_template

            pass
        else:
            raise ValueError(f'bad jinja template: {template}')

        prompt_template: pathlib.Path = pathlib.Path(self._prompt_template_name)

        # if the prompt template exists in the current working directory, use that
        if prompt_template.exists():
            with open(prompt_template, 'r') as f:
                return f.read()

        # if the prompt template is from the user templates, use that
        try:
            return templates.load_user_template(self._prompt_template_name)
        except:
            pass

        # if the prompt template is from the template folder, use that
        try:
            return templates.load_builtin_template(self._prompt_template_name)
        except:
            pass

        raise ValueError(f'bad jinja template: {self._prompt_template_name}')


    def _prompts_from_image(self, dataset_image: DatasetImage, **kwargs):
        """
        Generates system and user prompts for a given image using Jinja2 templating.

        Supports both single-round and multi-round captioning based on provided history.

        Args:
            dataset_image (DatasetImage): The image to generate prompts for.
            **kwargs: Optional arguments:
                - `caption_rounds` (list[CaptionerRound]): Previous caption rounds for context.
                - `system_prompt_override` (str): Override for the system prompt.
                - `user_prompt_override` (str): Override for the user prompt.
                - `debug_prompt` (bool): If True, logs prompts to debug output.

        Returns:
            tuple[str, str]: A tuple containing (system_prompt, user_prompt).

        Raises:
            ValueError: If `caption_rounds` is not a list of `CaptionerRound` instances.
        """

        try:
            caption_rounds: list[CaptionerRound] = kwargs.pop('caption_rounds', [])
            assert isinstance(caption_rounds, list)
            assert all(map(lambda r: isinstance(r, CaptionerRound), caption_rounds))
        except:
            raise ValueError("bad argument for caption_rounds")

        system_prompt_override: str = kwargs.pop('system_prompt_override', '')
        user_prompt_override: str = kwargs.pop('user_prompt_override', '')

        template_context = dataset_image.model_dump()

        system_prompt: str = system_prompt_override or self._jinja.get_template('__system_prompt__', globals=template_context).render()

        if caption_rounds:
            template_context['caption_rounds'] = caption_rounds

            user_prompt: str = user_prompt_override or self._jinja.get_template('__user_prompt_multiple_rounds__', globals=template_context).render()
        else:
            user_prompt: str = user_prompt_override or self._jinja.get_template('__user_prompt__', globals=template_context).render()

        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        if kwargs.get('debug_prompt', False):
            _logger.debug('')
            _logger.debug('SYSTEM PROMPT ---------')
            _logger.debug(system_prompt)
            _logger.debug('USER PROMPT -----------')
            _logger.debug(user_prompt)
            _logger.debug('------------------------')
            _logger.debug('')

        return system_prompt, user_prompt
    
    def _encode_image(self, image: DatasetImage, max_image_size: tuple[int, int], max_image_encoded_size: int, **kwargs):
        """
        Encodes an image as a base64 string suitable for API transmission.

        Automatically resizes images and adjusts quality to meet size limits.

        Process:
        - Resizes image to fit within `max_image_size` using LANCZOS resampling.
        - Converts RGBA/LA images to RGB when saving as JPEG.
        - Attempts lossless encoding first, then reduces JPEG quality iteratively if needed.

        Args:
            image (DatasetImage): The image to encode.
            max_image_size (tuple[int, int]): Maximum allowed dimensions (width, height).
            max_image_encoded_size (int): Maximum allowed size of base64 string in bytes.
            **kwargs: Optional arguments:
                - `call_depth` (int): Internal recursion counter (max 5).
                - `image_format` (str): Output format ('JPEG' or 'PNG', default: 'PNG').
                - `image_quality` (int): JPEG quality level (10-100, default: 100 on retry).

        Returns:
            tuple[str, str]: A tuple of (media_type, base64_encoded_image), e.g., ('image/jpeg', '...').

        Raises:
            ValueError: On invalid argument types.
            AssertionError: On excessive recursion.
        """

        call_depth = kwargs.pop('call_depth', 0)
        assert isinstance(call_depth, int), f"encode_image called with bad call_depth type: {type(call_depth)}"
        assert call_depth < 5, f"encode_image reached maximum call depth"

        try:
            image_format = kwargs.pop('image_format', 'PNG')
            assert isinstance(image_format, str), f"encode_image called with bad image_format type: {type(image_format)}"

            image_format = image_format.upper()
            assert image_format in ('JPEG', 'PNG'), f"encode_image called with bad image_format: only JPEG or PNG is allowed"

            image_quality = kwargs.pop('image_quality', None)
            assert image_quality is None or isinstance(image_quality, int), f"encode_image called with bad image_quality type: {type(image_quality)}"
            assert image_quality is None or (image_quality > 10 and image_quality <= 100), f"encode_image called with bad image_quality: {image_quality}"
        except AssertionError as e:
            raise ValueError(e)

        buffer = io.BytesIO()
        image_obj = image.read_image()

        # resize image if too large
        image_obj.thumbnail(max_image_size, Image.Resampling.LANCZOS)

        if image_format == 'JPEG' and image_obj.mode in ('RGBA', 'LA'):
            image_composite = Image.new('RGB', image_obj.size, (255, 255, 255))
            image_composite.paste(image_obj, mask=image_obj.split()[-1])
            image_obj = image_composite

        image_obj.save(buffer, format=image_format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if len(encoded_image) > max_image_encoded_size:
            # start at lossless, then degrade with each iteration
            image_quality = 100 if image_quality is None else image_quality - 10

            return self._encode_image(image, image_format='JPEG', call_depth=call_depth+1, image_quality=image_quality, **kwargs)

        return f'image/{image_format.lower()}', base64.b64encode(buffer.getvalue()).decode('utf-8')

    @abc.abstractmethod
    def load_model(self, model_repo: str, **kwargs) -> None:
        """
        Loads the captioning model through the API or from a repository or local path.

        Args:
            model_repo (str): Identifier or path to the model (e.g., Hugging Face repo ID).
            **kwargs: Model-specific loading options (e.g., device, dtype, cache_dir).

        Example:
            captioner.load_model("nlpconnect/vit-gpt2-image-captioning", device="cuda")
        """

        raise NotImplemented

    @abc.abstractmethod
    def unload_model(self) -> None:
        """
        Unloads the model from memory.

        Should free all resources associated with the loaded model.
        Called when switching models or shutting down.
        """

        raise NotImplemented

    @abc.abstractmethod
    def offload_model(self) -> None:
        """
        Offloads the model to CPU to free GPU memory.

        Useful when the model is not actively in use.
        """

        raise NotImplemented

    @abc.abstractmethod
    def predict_stream(self, image: DatasetImage, **kwargs) -> 'Generator[str]':
        """
        Generates a caption incrementally and yields partial results.

        Ideal for real-time interfaces where streaming output is desired.

        Args:
            image (DatasetImage): The input image to caption.
            **kwargs: Model-specific inference parameters.

        Yields:
            str: Caption text chunk as it becomes available.

        Raises:
            ValueError: If the API returns an error.

        Example:
        ```
            for token in captioner.predict_stream(img):
                print(token, end='', flush=True)
        ```
        """

        raise NotImplemented

    @abc.abstractmethod
    def predict(self, image: DatasetImage, **kwargs) -> str:
        """
        Generates a complete caption for the given image.

        Blocks until the full caption is generated.

        Args:
            image (DatasetImage): The input image to caption.
            **kwargs: Model-specific inference parameters.

        Raises:
            ValueError: If the API returns an error.

        Returns:
            str: The final generated caption.
        """

        raise NotImplemented
