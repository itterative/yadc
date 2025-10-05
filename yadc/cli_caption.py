from typing import Optional, TextIO

import sys
import toml
import pathlib
import pydantic

import click

from .core import utils
from . import cli_common

from yadc.core import logging
from yadc.core.config import Config, ConfigSettings
from yadc.core.dataset import DatasetImage
from yadc.core.captioner import CaptionerRound

from yadc.captioners.api import APICaptioner, APITypes

from yadc.cmd import status as cmd_status, envs as cmd_envs, configs as cmd_configs, templates as cmd_templates

_logger = logging.get_logger(__name__)

@click.command(
    short_help='Caption a dataset',
    help='Caption a dataset. A dataset config is necessary in order to start captioning. See documentation for details: https://github.com/itterative/yadc'
)
@click.argument('dataset', type=click.File('r'))
@click.option('--env', type=str, default=None, help='Configuration environment')
@click.option('--api-url', type=str, default=None, help='Override API url')
@click.option('--api-token', type=str, default=None, help='Override API auth token')
@click.option('--api-model-name', type=str, default=None, help='Override API model')
@click.option('--user-config', type=str, default=None, help='Base user config')
@click.option('--user-template', type=str, default=None, help='Override user template')
@click.option('--stream/--no-stream', is_flag=True, default=None, help='Enable the streaming of captions')
@click.option('--interactive/--non-interactive', 'interactive', is_flag=True, default=None, help='Enable interactive mode')
@click.option('--overwrite/--no-overwrite', 'overwrite', is_flag=True, default=None, help='Overwrite existing caption')
@click.option('--rounds', type=click.IntRange(min=1, max_open=True), default=None, required=False, help='How many captioning rounds to do')
@cli_common.log_level
def caption(dataset: TextIO, **kwargs):
    def cli_option(option: str, default):
        value = kwargs.get(option, None)
        if value is not None:
            return value
        return default

    _logger.info('Using python %d.%d.%d.', sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    try:
        dataset_toml = _load_dataset(
            dataset,
            env=cli_option('env', None),
            user_config=cli_option('user_config', None),
            user_template=cli_option('user_template', default=None),
            api_url = cli_option('api_url', default=None),
            api_token = cli_option('api_token', default=None),
            api_model_name = cli_option('api_model_name', default=None),
        )
    except (AssertionError, ValueError) as e:
        _logger.error('Error loading dataset: %s', e)
        sys.exit(cmd_status.STATUS_OK)

    # cli arguments

    do_stream = bool(cli_option('stream', default=False))
    interactive = bool(cli_option('interactive', default=dataset_toml.interactive))
    rounds = int(cli_option('rounds', default=dataset_toml.rounds))
    overwrite_captions = bool(cli_option('overwrite', default=dataset_toml.overwrite_captions))


    if not dataset_toml.prompt.template:
        try:
            dataset_toml.prompt.template = cmd_templates.load_user_template(dataset_toml.prompt.name)
        except:
            _logger.debug('No user template found: %s', dataset_toml.prompt.name)

    if not dataset_toml.prompt.template:
        try:
            dataset_toml.prompt.template = cmd_templates.load_builtin_template(dataset_toml.prompt.name)
        except:
            _logger.debug('No built-in template found: %s', dataset_toml.prompt.name)

    if not dataset_toml.prompt.template:
        if dataset_toml.prompt.name:
            if user_templates := cmd_templates.list_user_template():
                _logger.error('Error: prompt template could be loaded: %s; available templates: %s', dataset_toml.prompt.name, ', '.join(user_templates))
                sys.exit(cmd_status.STATUS_USER_ERROR)

            _logger.error('Error: prompt template could be loaded: %s', dataset_toml.prompt.name)
            sys.exit(cmd_status.STATUS_USER_ERROR)

        _logger.warning('Warning: no prompt template defined. Will use the default.')

        try:
            dataset_toml.prompt.template = cmd_templates.default_template()
        except:
            _logger.error('Error: default prompt template could be loaded')
            sys.exit(cmd_status.STATUS_ERROR)

    if dataset_toml.prompt.name:
        _logger.info('Using prompt template: %s', dataset_toml.prompt.name)


    skipped_from_dataset = 0
    dataset_to_do: list[DatasetImage] = []

    for dataset_image in dataset_toml.dataset.images:
        should_skip = not overwrite_captions and dataset_image.caption_path.exists()

        if should_skip:
            skipped_from_dataset += 1
            continue

        dataset_to_do.append(dataset_image)

    _logger.info('Found %d images.', len(dataset_toml.dataset.images))

    if skipped_from_dataset:
        _logger.info('Skipped %d images.', skipped_from_dataset)

    if len(dataset_to_do) == 0:
        _logger.info('Nothing to do.')
        sys.exit(cmd_status.STATUS_OK)

    _logger.info('Loading model...')

    try:
        model = APICaptioner(
            api_url=dataset_toml.api.url,
            api_token=dataset_toml.api.token,
            prompt_template=dataset_toml.prompt.template,
            store_conversation=dataset_toml.settings.store_conversation,
            image_quality=dataset_toml.settings.image_quality,
            reasoning=dataset_toml.reasoning.enable,
            reasoning_effort=dataset_toml.reasoning.thinking_effort,
            reasoning_exclude_output=dataset_toml.reasoning.exclude_from_output,
            reasoning_start_token=dataset_toml.reasoning.advanced.thinking_start,
            reasoning_end_token=dataset_toml.reasoning.advanced.thinking_end,
        )

        model.load_model(dataset_toml.api.model_name)
    except ValueError as e:
        _logger.error('Error: failed to load model: %s', e)
        sys.exit(cmd_status.STATUS_OK)

    _logger.info('')
    _logger.info('Captioning...')


    with utils.Timer() as timer:
        return_code = _caption(
            dataset=dataset_to_do,
            model=model,
            settings=dataset_toml.settings,
            do_stream=do_stream,
            interactive=interactive,
            rounds=rounds,
        )

    model.log_usage()
    _logger.info('Done. (%.1f sec)', timer.elapsed)

    sys.exit(return_code)

def _load_dataset(
    dataset_stream: TextIO,
    env: Optional[str],
    user_config: Optional[str],
    user_template: Optional[str],
    api_url: Optional[str],
    api_token: Optional[str],
    api_model_name: Optional[str],
):
    dataset: list[DatasetImage] = []
    i_dataset: dict[pathlib.Path, DatasetImage] = {}

    dataset_toml_raw = toml.load(dataset_stream)

    try:
        # merge with user config
        if user_config is not None:
            _logger.info('Using %s user config.', user_config)
            dataset_toml_raw = cmd_configs.merge_user_config(user_config, 
            dataset_toml_raw)
    except ValueError:
        if user_configs := cmd_configs.list_user_config():
            _logger.error('Error: failed to load user config: %s; available templates: %s', user_config, ', '.join(user_configs))
            sys.exit(cmd_status.STATUS_USER_ERROR)

        _logger.error('Error: failed to load user config: %s', user_config)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    # merge with user env
    env = env or dataset_toml_raw.get('env', 'default')

    assert isinstance(env, str), "invalid dataset toml env"

    _logger.info('Using %s user environment.', env)
    user_env = cmd_envs.load_env(env=env)

    dataset_toml_raw.setdefault('api', {})
    dataset_toml_raw_api = dataset_toml_raw['api']

    assert isinstance(dataset_toml_raw_api, dict), "invalid dataset toml api section"
    dataset_toml_raw_api['url'] = api_url or user_env.api.url or dataset_toml_raw_api.get('url', '')
    dataset_toml_raw_api['token'] = api_token or user_env.api.token or dataset_toml_raw_api.get('token', '')
    dataset_toml_raw_api['model_name'] = api_model_name or user_env.api.model_name or dataset_toml_raw_api.get('model_name', '')

    dataset_toml_raw.setdefault('prompt', {})
    dataset_toml_raw_prompt = dataset_toml_raw['prompt']

    if user_template is not None:
        dataset_toml_raw_prompt['name'] = user_template
        dataset_toml_raw_prompt.pop('template', None)

    try:
        dataset_toml = Config(**dataset_toml_raw)
    except pydantic.ValidationError as e:
        raise ValueError(f'invalid configuration: {e}')

    for index, path in enumerate(dataset_toml.dataset.paths):
        assert isinstance(path, str), f'path at index {index} is not a string'

        path = pathlib.Path(path)

        if not path.is_dir():
            _logger.warning('Warning: path %s is not a directory', path)
            continue

        for file_path in path.iterdir():
            try:
                dataset_image = DatasetImage(path=str(file_path))
                dataset_image.read_image()
            except:
                continue

            if not dataset_image.toml_path.exists():
                _logger.warning('Warning: path %s has no toml', file_path)
                dataset_image_toml = {}
            else:
                try:
                    with open(dataset_image.toml_path, 'r') as f:
                        dataset_image_toml = toml.load(f)
                except:
                    _logger.warning('Warning: path %s contains an invalid toml', file_path)
                    continue

            dataset_image_toml['path'] = str(dataset_image.absolute_path)
            dataset_image_toml['caption_suffix'] = dataset_toml.caption_suffix

            dataset_image = DatasetImage(**dataset_image_toml)
            dataset_image.caption = dataset_image.read_caption()

            dataset.append(dataset_image)
            i_dataset[dataset_image.absolute_path] = dataset_image


    for index, dataset_image in enumerate(dataset_toml.dataset.images):
        existing_dataset_image = i_dataset.get(dataset_image.absolute_path, None)

        if existing_dataset_image is None:
            dataset.append(dataset_image)
            i_dataset[dataset_image.absolute_path] = dataset_image

            continue

        # overwrites
        assert existing_dataset_image.__pydantic_extra__
        assert dataset_image.__pydantic_extra__

        for k, v in dataset_image.__pydantic_extra__.items():
            if not v:
                continue

            existing_dataset_image.__pydantic_extra__[k] = v

        existing_dataset_image.caption = dataset_image.caption or existing_dataset_image.caption

    dataset_toml.dataset.paths = []
    dataset_toml.dataset.images = dataset

    return dataset_toml

def _caption(
    dataset: list[DatasetImage],
    model: APICaptioner,
    settings: ConfigSettings,
    do_stream: bool,
    interactive: bool,
    rounds: int,
):
    def prompt_for_yes(prompt: str, default: bool = False) -> bool:
        if not interactive:
            return default

        return click.confirm(prompt, default=default)

    def prompt_for_override(value: str, default: str) -> str:
        if not interactive:
            return default

        response: str = click.prompt(f'Override {value}? ({default}) ' if default else f'Override {value}? ', show_default=False, default=default)

        if not response:
            return default
        else:
            return response

    def prompt_for_action(prompt: str, actions: dict[str, str], default_action: str):
        assert default_action in actions

        if not interactive:
            return actions[default_action]

        prompt = f'{prompt} ({" ".join(map(lambda kv: kv[0] + "=" + kv[1], actions.items()))}) [{actions[default_action]}] '
        response = None

        click.echo(prompt, nl=False)
        while response not in actions:
            response = click.getchar() or default_action
        click.echo(response)

        return actions[response]

    do_quit = False
    do_print_separator = False
    return_code = cmd_status.STATUS_OK

    def print_dataset_image_meta(dataset_image: DatasetImage):
        click.echo(f'Path: {dataset_image.path}')

        for key, value in (dataset_image.__pydantic_extra__ or {}).items():
            _logger.info('%s: %s', key.capitalize(), value)

        if caption := dataset_image.read_caption():
            _logger.info('Caption:')
            _logger.info(caption)
            _logger.info('------------')
            _logger.info('')

    if settings.advanced.assistant_prefill and model.api_type in (APITypes.GEMINI, APITypes.OPENAI, APITypes.OPENROUTER):
        _logger.warning("Warning: assistant prefill is set, but the API might not support it")

    conversation_overrides = settings.advanced.model_dump()


    for dataset_image in dataset:
        if do_quit:
            break

        if do_print_separator:
            _logger.info('')
            _logger.info('------------')
            _logger.info('')
        else:
            do_print_separator = True

        print_dataset_image_meta(dataset_image)

        dataset_image_current = DatasetImage(**dataset_image.model_dump())

        caption = ''
        caption_rounds = []
        caption_rounds_debug = True

        do_prompt = True

        while do_prompt:
            return_code = cmd_status.STATUS_OK

            try:
                action = prompt_for_action('Next action', dict(
                    q='quit',
                    s='skip',
                    c='continue',
                    r='retry',
                    e='edit',
                    p='print',
                ), default_action='c')
            except (KeyboardInterrupt, click.Abort):
                click.echo('')
                action = 'quit'

            match action:
                case 'quit':
                    caption = ''
                    do_prompt = False
                    do_quit = True
                    break

                case 'skip':
                    caption = ''
                    do_prompt = False
                    break

                case 'continue':
                    do_prompt = not caption

                case 'retry':
                    pass

                case 'edit':
                    while True:
                        dataset_image_tmp_edited = click.edit(dataset_image_current.dump_toml(), extension='.toml', require_save=True)

                        if dataset_image_tmp_edited is None:
                            _logger.info('Dataset image toml editing was cancelled.')
                            break

                        try:
                            dataset_image_toml = toml.loads(dataset_image_tmp_edited)

                            dataset_image_current = DatasetImage(
                                path=dataset_image_current.path,
                                caption=dataset_image_current.caption,
                                caption_suffix=dataset_image_current.caption_suffix,
                                toml_suffix=dataset_image_current.toml_suffix,
                                history_suffix=dataset_image_current.history_suffix,
                                **dataset_image_toml,
                            )
                        except:
                            _logger.warning('Warning: toml is not valid')

                            if not prompt_for_yes('Retry?', default=True):
                                break

                            continue

                        with open(dataset_image_current.toml_path, 'w') as f:
                            f.write(dataset_image_current.dump_toml())

                        _logger.info('Dataset image toml was updated.')
                        break

                    caption_rounds = []
                    continue

                case 'print':
                    print_dataset_image_meta(dataset_image_current)
                    continue

                case _:
                    raise AssertionError(f'bad action: {action}')

            if not do_prompt and caption:
                break

            try:
                if rounds <= 1:
                    caption = []
                    _logger.info('')

                    try:
                        with utils.Timer() as timer_prediction:
                            if do_stream:
                                tokens = model.predict_stream(
                                    dataset_image_current,
                                    max_new_tokens=settings.max_tokens,
                                    conversation_overrides=conversation_overrides,
                                    prefill=settings.advanced.assistant_prefill,
                                )
                            else:
                                tokens = [model.predict(
                                    dataset_image_current,
                                    max_new_tokens=settings.max_tokens,
                                    conversation_overrides=conversation_overrides,
                                )]

                            for token in tokens:
                                caption.append(token)
                                click.echo(token, nl=False)
                    except ValueError as e:
                        _logger.error('Error: %s', e)
                        raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        if do_stream: click.echo('')
                        raise

                    click.echo('')

                    _logger.info('Captioning done (%.3f sec)', timer_prediction.elapsed)
                    _logger.info('')

                    caption = ''.join(caption).strip()
                else:
                    j = 0

                    if caption_rounds:
                        j = rounds
                    else:
                        _logger.info('Doing %d rounds...', rounds)

                    try:
                        while j < rounds:
                            j +=1

                            new_caption = prompt_for_override(f'round #{j}', default='')

                            if not new_caption:
                                with utils.Timer() as timer_round:
                                    new_caption = model.predict(
                                        dataset_image_current,
                                        max_new_tokens=settings.max_tokens,
                                        use_cache=True,
                                        debug_prompt=caption_rounds_debug,
                                        conversation_overrides=conversation_overrides,
                                        prefill=settings.advanced.assistant_prefill,
                                    ).strip()

                                caption_rounds_debug = False

                                if interactive:
                                    _logger.info(new_caption)

                                _logger.info('Round #%d done. (%.3f ssec)', j, timer_round.elapsed)

                                if not prompt_for_yes('Accept caption?', default=True):
                                    j -= 1
                                    caption_rounds.pop()
                                    continue

                            caption_rounds.append(CaptionerRound(iteration=j, caption=new_caption))

                        caption = []

                        _logger.info('')

                        with utils.Timer() as timer_end_round:
                            if do_stream:
                                tokens = model.predict_stream(
                                    DatasetImage(path=dataset_image.path),
                                    caption_rounds=caption_rounds,
                                    max_new_tokens=settings.max_tokens,
                                    conversation_overrides=conversation_overrides,
                                )
                            else:
                                tokens = [model.predict(
                                    DatasetImage(path=dataset_image.path),
                                    caption_rounds=caption_rounds,
                                    max_new_tokens=settings.max_tokens,
                                    conversation_overrides=conversation_overrides,
                                )]

                            for token in tokens:
                                caption.append(token)
                                click.echo(token, nl=False)
                    except ValueError as e:
                        _logger.error('Error: %s', e)
                        raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        if do_stream: click.echo('')
                        raise

                    click.echo('')

                    _logger.info('End round done. (%.3f sec)', timer_end_round.elapsed)
                    _logger.info('')

                    caption = ''.join(caption).strip()
            except (KeyboardInterrupt, click.Abort):
                if not interactive:
                    caption = '' # don't write to history if cancelled
                    caption_rounds = []
                    do_quit = True
                    do_prompt = False
                    return_code = cmd_status.STATUS_ERROR

                    break

                _logger.info('Cancelled captioning.')
                caption = ''
                pass

        if not caption:
            continue

        # edge case: save current toml history if it hasn't been saved before
        if dataset_image.caption:
            dataset_image.save_history(when_not_exists=True)

        dataset_image_current.update_caption(caption)
        dataset_image_current.save_history(when_not_exists=False)

        caption = ''

        _logger.info('')

    return return_code
