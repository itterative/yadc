from typing import TextIO

import sys
import toml
import pathlib
import pydantic

import click
from . import cli_common, cli_utils

from yadc.core import logging
from yadc.core.config import Config, ConfigSettings
from yadc.core.dataset import DatasetImage
from yadc.core.captioner import CaptionerRound

from yadc.captioners.api import APICaptioner

from yadc.cli_config import load_config

_logger = logging.get_logger(__name__)

@click.command(
    short_help='Caption a dataset',
    help='Caption a dataset. A dataset config is necessary in order to start captioning. See documentation for details: https://github.com/itterative/yadc'
)
@click.argument('dataset', type=click.File('r'))
@click.option('--stream/--no-stream', is_flag=True, default=None, help='Enable the streaming of captions')
@click.option('--interactive/--non-interactive', 'interactive', is_flag=True, default=None, help='Enable interactive mode')
@click.option('--overwrite/--no-overwrite', 'overwrite', is_flag=True, default=None, help='Overwrite existing caption')
@click.option('--rounds', type=click.IntRange(min=1, max_open=True), default=None, required=False, help='How many captioning rounds to do')
@cli_common.log_level
@cli_common.env
def caption(dataset: TextIO, env: str = 'default', **kwargs):
    def cli_option(option: str, default):
        value = kwargs.get(option, None)
        if value is not None:
            return value
        return default

    _logger.info('Using python %d.%d.%d.', sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    try:
        dataset_toml = _load_dataset(dataset, env=env)
    except ValueError as e:
        _logger.error('Error loading dataset: %s', e)
        sys.exit(1)

    # cli arguments
    do_stream = bool(cli_option('stream', default=False))
    interactive = bool(cli_option('interactive', default=dataset_toml.interactive))
    rounds = int(cli_option('rounds', default=dataset_toml.rounds))
    overwrite_captions = bool(cli_option('overwrite', default=dataset_toml.overwrite_captions))

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
        sys.exit(0)

    _logger.info('Loading model...')

    model = APICaptioner(
        api_url=dataset_toml.api.url,
        api_token=dataset_toml.api.token,
        prompt_template=dataset_toml.settings.prompt_template,
        prompt_template_name=dataset_toml.settings.prompt_template_path,
        store_conversation=dataset_toml.settings.store_conversation,
        image_quality=dataset_toml.settings.image_quality,
        reasoning=dataset_toml.reasoning.enable,
        reasoning_effort=dataset_toml.reasoning.thinking_effort,
        reasoning_exclude_output=dataset_toml.reasoning.exclude_from_output,
    )

    try:
        model.load_model(dataset_toml.api.model_name)
    except ValueError as e:
        _logger.error('Error: failed to load model: %s', e)
        sys.exit(1)

    _logger.info('')
    _logger.info('Captioning...')

    
    with cli_utils.Timer() as timer:
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

def _load_dataset(dataset_stream: TextIO, env: str):
    dataset_toml_raw = toml.load(dataset_stream)

    user_config = load_config(env=env)

    dataset: list[DatasetImage] = []
    i_dataset: dict[pathlib.Path, DatasetImage] = {}

    # merge with use config
    dataset_toml_raw.setdefault('api', {})
    dataset_toml_raw_api = dataset_toml_raw['api']

    assert isinstance(dataset_toml_raw_api, dict)
    dataset_toml_raw_api.setdefault('url', user_config.api.url)
    dataset_toml_raw_api.setdefault('token', user_config.api.token)
    dataset_toml_raw_api.setdefault('model_name', user_config.api.model_name)

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
    return_code = 0

    def print_dataset_image_meta(dataset_image: DatasetImage):
        click.echo(f'Path: {dataset_image.path}')

        for key, value in (dataset_image.__pydantic_extra__ or {}).items():
            _logger.info('%s: %s', key.capitalize(), value)

        if caption := dataset_image.read_caption():
            _logger.info('Caption:')
            _logger.info(caption)
            _logger.info('------------')
            _logger.info('')

    conversation_overrides = settings.advanced.model_dump()
    conversation_overrides.pop('debug_prompt', None) # fix: don't want to pass this to the API

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
            return_code = 0

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
                        with cli_utils.Timer() as timer_prediction:
                            if do_stream:
                                tokens = model.predict_stream(
                                    dataset_image_current,
                                    max_new_tokens=settings.max_tokens,
                                    debug_prompt=settings.advanced.debug_prompt,
                                    conversation_overrides=conversation_overrides,
                                )
                            else:
                                tokens = [model.predict(
                                    dataset_image_current,
                                    max_new_tokens=settings.max_tokens,
                                    debug_prompt=settings.advanced.debug_prompt,
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
                                with cli_utils.Timer() as timer_round:
                                    new_caption = model.predict(
                                        dataset_image_current,
                                        max_new_tokens=settings.max_tokens,
                                        use_cache=True,
                                        debug_prompt=settings.advanced.debug_prompt and caption_rounds_debug,
                                        conversation_overrides=conversation_overrides,
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

                        with cli_utils.Timer() as timer_end_round:
                            if do_stream:
                                tokens = model.predict_stream(
                                    DatasetImage(path=dataset_image.path),
                                    caption_rounds=caption_rounds,
                                    max_new_tokens=settings.max_tokens,
                                    debug_prompt=settings.advanced.debug_prompt,
                                    conversation_overrides=conversation_overrides,
                                )
                            else:
                                tokens = [model.predict(
                                    DatasetImage(path=dataset_image.path),
                                    caption_rounds=caption_rounds,
                                    max_new_tokens=settings.max_tokens,
                                    debug_prompt=settings.advanced.debug_prompt,
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
                    return_code = 1

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
