import abc
from typing import Generator

import io
import base64
import jinja2
import pathlib
import pydantic

from importlib import resources
from PIL import Image

from yadc.core import logging
from yadc.core.dataset import DatasetImage
import yadc.templates

_logger = logging.get_logger(__name__)

class CaptionerRound(pydantic.BaseModel):
    iteration: int
    caption: str

class Captioner(abc.ABC):
    def __init__(self, **kwargs):
        self._prompt_template_name: str = kwargs.pop('prompt_template_name', 'default.jinja')
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
            template = 'default.jinja'
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

        # if the prompt template is from the template folder, use that
        with resources.path(yadc.templates, self._prompt_template_name) as prompt_template_resource:
            with open(prompt_template_resource, 'r') as f:
                return f.read()

        raise ValueError(f'bad jinja template: {self._prompt_template_name}')


    def _prompts_from_image(self, dataset_image: DatasetImage, **kwargs):
        caption_rounds: list[CaptionerRound] = kwargs.pop('caption_rounds', [])
        assert isinstance(caption_rounds, list)
        assert all(map(lambda r: isinstance(r, CaptionerRound), caption_rounds))

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
        call_depth = kwargs.pop('call_depth', 0)
        assert isinstance(call_depth, int), f"encode_image called with bad call_depth type: {type(call_depth)}"
        assert call_depth < 5, f"encode_image reached maximum call depth"

        image_format = kwargs.pop('image_format', 'PNG')
        assert isinstance(image_format, str), f"encode_image called with bad image_format type: {type(image_format)}"

        image_format = image_format.upper()
        assert image_format in ('JPEG', 'PNG'), f"encode_image called with bad image_format: only JPEG or PNG is allowed"

        image_quality = kwargs.pop('image_quality', None)
        assert image_quality is None or isinstance(image_quality, int), f"encode_image called with bad image_quality type: {type(image_quality)}"
        assert image_quality is None or (image_quality > 10 and image_quality <= 100), f"encode_image called with bad image_quality: {image_quality}"

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
        raise NotImplemented

    @abc.abstractmethod
    def unload_model(self) -> None:
        raise NotImplemented

    @abc.abstractmethod
    def offload_model(self) -> None:
        raise NotImplemented

    @abc.abstractmethod
    def predict_stream(self, image: DatasetImage, **kwargs) -> 'Generator[str]':
        raise NotImplemented

    @abc.abstractmethod
    def predict(self, image: DatasetImage, **kwargs) -> str:
        raise NotImplemented
