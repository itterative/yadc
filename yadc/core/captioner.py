"""Abstract ``Captioner`` base class.

Jinja2 prompt rendering, base64 image encoding (with auto-resize/quality
degradation), and the ``predict`` / ``predict_stream`` API used by both
the CLI and the web UI.
"""

import abc
import base64
import io
from collections.abc import AsyncGenerator
from typing import Any

import jinja2
import pydantic
from PIL import Image

from yadc.templates import default_template

from .dataset import DatasetImage
from .logging import get_logger

_logger = get_logger(__name__)


class PromptRenderer:
    """Jinja2 template renderer for caption prompts.

    Handles the template loading and rendering logic used by both the
    captioner (for real captioning) and the preview endpoint (for showing
    what prompts will look like without calling an API).

    Templates use a virtual filesystem with special names:
    - ``__default_template__`` — built-in default Jinja2 template
    - ``__user_template__`` — the user-provided template string (or default if none)
    - ``__system_prompt__``, ``__user_prompt__``, ``__user_prompt_multiple_rounds__`` —
      composed templates that merge default + user template blocks
    """

    def __init__(self, prompt_template: str = ""):
        self._prompt_template: str = prompt_template.strip()
        self._jinja: jinja2.Environment = jinja2.Environment(
            loader=jinja2.FunctionLoader(self._load_template),
            lstrip_blocks=True,
            trim_blocks=True,
            keep_trailing_newline=False,
        )

    @staticmethod
    def _unindent(template: str) -> str:
        return "\n".join(line.lstrip() for line in template.strip().splitlines())

    def _load_template(self, name: str) -> str:
        if name == "__system_prompt__":
            return self._unindent("""
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.system_prompt|default(default_template.system_prompt, true) }}
            """)

        if name == "__user_prompt__":
            return self._unindent("""
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.user_prompt|default(default_template.user_prompt, true) }}
            """)

        if name == "__user_prompt_multiple_rounds__":
            return self._unindent("""
                {% import "__default_template__" as default_template %}
                {% import "__user_template__" as user_template %}
                {{ user_template.user_prompt_multiple_rounds|default(default_template.user_prompt_multiple_rounds, true) }}
            """)

        if name == "__default_template__":
            return default_template()

        if name == "__user_template__":
            return self._prompt_template if self._prompt_template else default_template()

        raise ValueError(f"bad jinja template: {name}")

    def render(
        self,
        dataset_image: DatasetImage,
        *,
        caption_rounds: "list[CaptionerRound] | None" = None,
        drafts: dict[str, str] | None = None,
        system_prompt_override: str = "",
        user_prompt_override: str = "",
    ) -> tuple[str, str]:
        """Render system and user prompts for an image.

        Returns:
            (system_prompt, user_prompt) — both stripped of leading/trailing whitespace.
        """
        if caption_rounds is None:
            caption_rounds = []
        else:
            assert isinstance(caption_rounds, list)
            assert all(isinstance(r, CaptionerRound) for r in caption_rounds)

        template_context = dataset_image.model_dump()
        if drafts:
            template_context["drafts"] = drafts

        system_prompt = system_prompt_override or self._jinja.get_template("__system_prompt__", globals=template_context).render()

        if caption_rounds:
            template_context["caption_rounds"] = caption_rounds
            user_prompt = user_prompt_override or self._jinja.get_template("__user_prompt_multiple_rounds__", globals=template_context).render()
        else:
            user_prompt = user_prompt_override or self._jinja.get_template("__user_prompt__", globals=template_context).render()

        return system_prompt.strip(), user_prompt.strip()


class CaptionerRound(pydantic.BaseModel):
    """
    Represents a single round of captioning in a multi-round captioning.

    Attributes:
        iteration (int): The sequence number of this caption round (e.g., 1st, 2nd).
        caption (str): The caption generated in this round.
    """

    iteration: int
    caption: str


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class ReplyRound(pydantic.BaseModel):
    """
    Represents a single turn in a reply-based conversation for caption refinement.

    Each round contains either an assistant response (with optional reasoning)
    followed by a user reply, or just an assistant response as the latest turn.

    Attributes:
        role (str): The role of the message author (ROLE_USER or ROLE_ASSISTANT).
        content (str): The text content of the message.
        reasoning (str | None): Optional reasoning/thinking content from the assistant.
        reasoning_encrypted (list[dict[str, Any]] | None): Optional encrypted reasoning data to pass back.
    """

    role: str
    content: str
    reasoning: str | None = None
    reasoning_encrypted: list[dict[str, Any]] | None = None


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

    Example template variables (from `DatasetImage` fields):
    - `image_id`, `file_path`, `width`, `height`, etc.

    Usage:
    ```
        class MyCaptioner(Captioner):
            async def load_model(self, model_repo, **kwargs):
                ...
            async def predict_stream(self, image, **kwargs):
                ...
            async def predict(self, image, **kwargs):
                ...

        captioner = MyCaptioner(prompt_template_name="custom.jinja")
        await captioner.load_model("my-model-id")
        caption = await captioner.predict(dataset_image)
    ```
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the Captioner with optional template configuration.

        Args:
            **kwargs: Optional keyword arguments:
                - `prompt_template` (str): The prompt template used for captioning. If none is provided, the default will be used.
        """

        self._renderer: PromptRenderer = PromptRenderer(kwargs.pop("prompt_template", ""))

    def prompts_from_image(self, dataset_image: DatasetImage, **kwargs: Any) -> tuple[str, str]:
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
                - `drafts` (dict[str, str]): The drafts of the image.

        Returns:
            tuple[str, str]: A tuple containing (system_prompt, user_prompt).

        Raises:
            ValueError: If `caption_rounds` is not a list of `CaptionerRound` instances.
        """

        caption_rounds = kwargs.get("caption_rounds", [])
        drafts = kwargs.get("drafts", None)

        return self._renderer.render(
            dataset_image,
            caption_rounds=caption_rounds,
            drafts=drafts,
            system_prompt_override=kwargs.get("system_prompt_override", ""),
            user_prompt_override=kwargs.get("user_prompt_override", ""),
        )

    def _encode_image(self, image: DatasetImage, max_image_size: tuple[int, int], max_image_encoded_size: int, **kwargs: Any) -> tuple[str, str]:
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

        call_depth = kwargs.pop("call_depth", 0)
        assert isinstance(call_depth, int), f"encode_image called with bad call_depth type: {type(call_depth)}"
        assert call_depth < 5, "encode_image reached maximum call depth"

        try:
            image_format = kwargs.pop("image_format", "PNG")
            assert isinstance(image_format, str), f"encode_image called with bad image_format type: {type(image_format)}"

            image_format = image_format.upper()
            assert image_format in ("JPEG", "PNG"), "encode_image called with bad image_format: only JPEG or PNG is allowed"

            image_quality = kwargs.pop("image_quality", None)
            assert image_quality is None or isinstance(image_quality, int), f"encode_image called with bad image_quality type: {type(image_quality)}"
            assert image_quality is None or (image_quality > 10 and image_quality <= 100), f"encode_image called with bad image_quality: {image_quality}"
        except AssertionError as e:
            raise ValueError(e)

        buffer = io.BytesIO()
        image_obj = image.read_image()

        # resize image if too large
        image_obj.thumbnail(max_image_size, Image.Resampling.LANCZOS)

        if image_format == "JPEG" and image_obj.mode in ("RGBA", "LA"):
            image_composite = Image.new("RGB", image_obj.size, (255, 255, 255))
            image_composite.paste(image_obj, mask=image_obj.split()[-1])
            image_obj = image_composite

        image_obj.save(buffer, format=image_format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if len(encoded_image) > max_image_encoded_size:
            # start at lossless, then degrade with each iteration
            image_quality = 100 if image_quality is None else image_quality - 10

            return self._encode_image(image, image_format="JPEG", call_depth=call_depth + 1, image_quality=image_quality, **kwargs)

        return f"image/{image_format.lower()}", base64.b64encode(buffer.getvalue()).decode("utf-8")

    @abc.abstractmethod
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        """Async variant of :meth:`load_model`.

        Backends that natively support asyncio should override this for true
        non-blocking model loading.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def unload_model(self) -> None:
        """
        Unloads the model from memory.

        Should free all resources associated with the loaded model.
        Called when switching models or shutting down.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def offload_model(self) -> None:
        """
        Offloads the model to CPU to free GPU memory.

        Useful when the model is not actively in use.
        """

        raise NotImplementedError

    @abc.abstractmethod
    async def predict(self, image: DatasetImage, **kwargs: Any) -> str:
        """Generate a complete caption asynchronously."""

        raise NotImplementedError

    @abc.abstractmethod
    def predict_stream(self, image: DatasetImage, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate a caption stream asynchronously."""

        raise NotImplementedError
