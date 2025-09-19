import os
import sys
import toml
import pathlib
import tempfile
import subprocess
import pydantic

from time import time

import readline # enables better input on unix

from yadc.core import logging
from yadc.core.config import Config
from yadc.core.dataset import DatasetImage
from yadc.core.captioner import CaptionerRound

from yadc.cli_config import load_config

_logger = logging.get_logger(__name__)

def caption(dataset_path: str):
    _logger.info('Using python %d.%d.%d.', sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    user_config = load_config()

    dataset: list[DatasetImage] = []
    i_dataset: dict[pathlib.Path, DatasetImage] = {}

    try:
        with open(dataset_path, 'r') as f:
            dataset_toml_raw = toml.load(f)

            # merge with use config
            dataset_toml_raw.setdefault('api', {})
            dataset_toml_raw_api = dataset_toml_raw['api']

            assert isinstance(dataset_toml_raw_api, dict)
            dataset_toml_raw_api.setdefault('url', user_config.api.url)
            dataset_toml_raw_api.setdefault('token', user_config.api.token)
            dataset_toml_raw_api.setdefault('model_name', user_config.api.model_name)

            dataset_toml = Config(**dataset_toml_raw)

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

    except FileNotFoundError:
        _logger.error('Error loading %s: file not found', dataset_path)
        return 1
    except ValueError as e:
        _logger.error('Error loading %s: %s', dataset_path, e)
        return 1

    if dataset_toml.settings.hf_token:
        os.environ['HF_TOKEN'] = dataset_toml.settings.hf_token

    skipped_from_dataset = 0
    dataset_to_do: list[DatasetImage] = []

    for dataset_image in dataset:
        should_skip = not dataset_toml.overwrite_captions and dataset_image.caption_path.exists()

        if should_skip:
            skipped_from_dataset += 1
            continue

        dataset_to_do.append(dataset_image)

    _logger.info('Found %d images.', len(dataset))

    if skipped_from_dataset:
        _logger.info('Skipped %d images.', skipped_from_dataset)

    if len(dataset_to_do) == 0:
        _logger.info('Nothing to do.')
        sys.exit(0)

    _logger.info('Loading model...')

    from yadc.captioners.api import APICaptioner

    model = APICaptioner(
        api_url=dataset_toml.api.url,
        api_token=dataset_toml.api.token,
        prompt_template=dataset_toml.settings.prompt_template,
        prompt_template_name=dataset_toml.settings.prompt_template_path,
        store_conversation=dataset_toml.settings.store_conversation,
        image_quality=dataset_toml.settings.image_quality,
        reasoning=dataset_toml.reasoning.enable,
        reasoning_effort=dataset_toml.reasoning.thinking_effort,
        reasoning_exclude_output=dataset_toml.reasoning.exclude_from_response,
    )

    try:
        model.load_model(dataset_toml.api.model_name)
    except ValueError as e:
        _logger.error('Error: failed to load model: %s', e)
        return 1

    _logger.info('')
    _logger.info('Captioning...')

    def prompt_for_yes(prompt: str, default: bool = False) -> bool:
        if not dataset_toml.interactive:
            return default

        if default:
            prompt += " [Y/n] "
        else:
            prompt += " [y/N] "
        
        response = input(prompt).strip()

        if not response:
            return default
        else:
            return response.lower()[:1] == 'y'
        
    def prompt_for_override(value: str, default: str) -> str:
        if not dataset_toml.interactive:
            return default

        prompt = f'Override {value}? ({default}) ' if default else f'Override {value}? '

        response = input(prompt).strip()

        if not response:
            return default
        else:
            return response

    def prompt_for_action(prompt: str, actions: dict[str, str], default_action: str):
        assert default_action in actions

        if not dataset_toml.interactive:
            return actions[default_action]

        prompt = f'{prompt} ({" ".join(map(lambda kv: kv[0] + "=" + kv[1], actions.items()))}) [{actions[default_action]}] '
        response = None

        while response not in actions:
            response = input(prompt).strip() or default_action
        
        return actions[response]

    do_quit = False
    do_print_separator = False

    def print_dataset_image_meta(dataset_image: DatasetImage):
        print('Path:', dataset_image.path)

        for key, value in (dataset_image.__pydantic_extra__ or {}).items():
            _logger.info('%s: %s', key.capitalize(), value)

        if caption := dataset_image.read_caption():
            _logger.info('Caption:')
            _logger.info(caption)
            _logger.info('------------')
            _logger.info('')

    caption_loop_start = time()

    for i_dataset_image, dataset_image in enumerate(dataset_to_do):
        if do_quit:
            break

        if do_print_separator:
            _logger.info('')
            _logger.info('------------')
            _logger.info('')
        else:
            do_print_separator = True

        print_dataset_image_meta(dataset_image)

        dataset_image_tmp = DatasetImage(**dataset_image.model_dump())

        caption = ''
        caption_rounds = []
        caption_rounds_debug = True

        do_prompt = True

        while do_prompt:
            action = prompt_for_action('Next action', dict(
                q='quit',
                s='skip',
                c='continue',
                r='retry',
                e='edit',
                p='print',
            ), default_action='c')

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
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_file = pathlib.Path(tmp) / dataset_image.absolute_path.name

                        with open(tmp_file, 'w') as f:
                            f.write(dataset_image_tmp.dump_toml())

                        if 'EDITOR' not in os.environ:
                            _logger.warning('Warning: no EDITOR environment variable is set. Edit the file %s before proceeding.', f.name)

                        do_edit = True
                        while do_edit:
                            do_edit = False

                            if 'EDITOR' not in os.environ:
                                while not prompt_for_yes('Continue?', default=True):
                                    pass
                            else:
                                editor = os.environ['EDITOR']
                                write_status = subprocess.call(f'{editor} {tmp_file}', shell=True)

                                if write_status != 0:
                                    _logger.warning('Warning: editor exitted with non-zero status.')

                                    if prompt_for_yes('Abort?', default=False):
                                        break

                            with open(tmp_file, 'r') as f:
                                try:
                                    dataset_image_toml = toml.load(f)
                                except:
                                    _logger.warning('Warning: toml is not valid')

                                    if not prompt_for_yes('Retry?', default=True):
                                        break

                                    continue

                                try:
                                    dataset_image_toml['path'] = dataset_image.path
                                    dataset_image_toml['caption'] = dataset_image.caption

                                    dataset_image_tmp = DatasetImage(**dataset_image_toml)
                                except Exception as e:
                                    _logger.warning('Warning: toml is not valid:', e)

                                    if not prompt_for_yes('Retry?', default=True):
                                        break

                                    continue
                            
                            

                            do_edit = False
                            break

                    caption_rounds = []
                    _logger.info('Dataset image toml was updated.')
                    continue

                case 'print':
                    print_dataset_image_meta(dataset_image_tmp)
                    continue

                case _:
                    raise AssertionError(f'bad action: {action}')

            if not do_prompt and caption:
                break

            try:
                if dataset_toml.rounds <= 1:
                    caption = []
                    _logger.info('')
                    start_t = time()
                    for token in model.predict_stream(dataset_image_tmp, max_new_tokens=dataset_toml.settings.max_tokens, debug_prompt=dataset_toml.settings.debug_prompt):
                        caption.append(token)
                        print(token, end='', flush=True)
                    end_t = time()
                    print('')

                    _logger.info('Captioning done (%.3f sec)', end_t - start_t)
                    _logger.info('')

                    caption = ''.join(caption).strip()
                else:
                    j = 0

                    if caption_rounds:
                        j = dataset_toml.rounds
                    else:
                        _logger.info('Doing %d rounds...', dataset_toml.rounds)

                    while j < dataset_toml.rounds:
                        j +=1

                        new_caption = ''
                        new_caption = prompt_for_override(f'round #{j}', default='')

                        if not new_caption:
                            start_t = time()
                            new_caption = model.predict(dataset_image_tmp, max_new_tokens=dataset_toml.settings.max_tokens, use_cache=True, debug_prompt=dataset_toml.settings.debug_prompt and caption_rounds_debug).strip()
                            end_t = time()

                            caption_rounds_debug = False

                            if dataset_toml.interactive:
                                _logger.info(new_caption)

                            _logger.info('Round #%d done. (%.3f ssec)', j, end_t - start_t)

                            if not prompt_for_yes('Accept caption?', default=True):
                                j -= 1
                                caption_rounds.pop()
                                continue

                        caption_rounds.append(CaptionerRound(iteration=j, caption=new_caption))

                    caption = []

                    _logger.info('')
                    start_t = time()
                    for token in model.predict_stream(DatasetImage(path=dataset_image.path), caption_rounds=caption_rounds, max_new_tokens=dataset_toml.settings.max_tokens, debug_prompt=dataset_toml.settings.debug_prompt):
                        caption.append(token)
                        print(token, end='', flush=True)
                    end_t = time()
                    print('')

                    _logger.info('End round done. (%.3f sec)', end_t - start_t)
                    _logger.info('')

                    caption = ''.join(caption).strip()
            except KeyboardInterrupt:
                if not dataset_toml.interactive:
                    raise

                print('')
                print('Cancelled captioning.')
                caption = ''
                pass

        if not caption:
            continue

        # edge case: save current toml history if it hasn't been saved before
        if dataset_image.caption:
            dataset_image.save_history(when_not_exists=True)

        dataset_image_tmp.update_caption(caption)
        dataset_image_tmp.save_history(when_not_exists=False)

        caption = ''

        _logger.info('')

    caption_loop_end = time()

    _logger.info('Done. (%.1f sec)', caption_loop_end - caption_loop_start)

    return 0
