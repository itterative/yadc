import jinja2
import pathlib
import pydantic
from importlib import resources

from typing import Generator

from yadc.core.dataset import DatasetImage
import yadc.templates

class CaptionerRound(pydantic.BaseModel):
    iteration: int
    caption: str

class Captioner:
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
            print('')
            print('SYSTEM PROMPT ---------')
            print(system_prompt)
            print('USER PROMPT -----------')
            print(user_prompt)
            print('------------------------')
            print('')

        return system_prompt, user_prompt

    def load_model(self, model_repo: str, **kwargs) -> None:
        raise NotImplemented
    
    def unload_model(self) -> None:
        raise NotImplemented

    def offload_model(self) -> None:
        raise NotImplemented

    def predict_stream(self, image: DatasetImage, **kwargs) -> 'Generator[str]':
        raise NotImplemented

    def predict(self, image: DatasetImage, **kwargs) -> str:
        raise NotImplemented
