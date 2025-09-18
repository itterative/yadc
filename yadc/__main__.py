import argparse
import functools

def main():
    from yadc.core import logging
    from yadc import cli, cli_config

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--log-level', help='Set the logging level', type=str, default='info', choices=['info', 'warning', 'error', 'debug'])

    parser = argparse.ArgumentParser(prog='yadc', description='Yet Another Dataset Captioner')
    parser.set_defaults(_action='help', _subaction='default')

    subparser = parser.add_subparsers(dest='command')

    caption_parser = subparser.add_parser('caption', help='Caption a dataset', description='Caption a dataset. A dataset config is necessary in order to start captioning. See documentation for details: https://github.com/itterative/yadc', parents=[common_parser])
    caption_parser.add_argument('dataset_toml', type=str)
    caption_parser.set_defaults(_action='caption', _subaction='default')

    if cli_config.FLAG_USER_CONFIG:
        config_parser = subparser.add_parser('config', help='Manage the user config', description='Manage the user config. The user config is stored as plain-text currently, so any tokens are visible to programs running under your user.', parents=[common_parser])
        config_parser.set_defaults(_action='config')

        config_subparser = config_parser.add_subparsers(dest='subcommand', description='subcommands to run')

        config_list_parser = config_subparser.add_parser('list', help='List all settings', description='List all settings in user config')
        config_list_parser.set_defaults(_subaction='list')

        config_get_parser = config_subparser.add_parser('get', help='Retrieve a setting', description='Retrieve a setting value from user config')
        config_get_parser.add_argument('key', help='The user config key to retrieve. Options: api_url, api_token, api_model_name', type=str)
        config_get_parser.set_defaults(_subaction='get')

        config_set_parser = config_subparser.add_parser('set', help='Update a setting', description='Update a setting value in user config')
        config_set_parser.add_argument('key', help='The user config key to set. Options: api_url, api_token, api_model_name', type=str)
        config_set_parser.add_argument('value', help='The value of the setting', type=str)
        config_set_parser.add_argument('--force', default=False, action=argparse.BooleanOptionalAction, type=bool, help='Overwrites the config if invalid')
        config_set_parser.set_defaults(_subaction='set')

        config_delete_parser = config_subparser.add_parser('delete', help='Delete a setting', description='Delete a setting from user config')
        config_delete_parser.add_argument('key', help='The user config key to delete. Options: api_url, api_token, api_model_name', type=str)
        config_delete_parser.add_argument('--force', default=False, action=argparse.BooleanOptionalAction, type=bool, help='Overwrites the config if invalid')
        config_delete_parser.set_defaults(_subaction='delete')

        config_clear_parser = config_subparser.add_parser('clear', help='Clear the user config', description='Clear the user config')
        config_clear_parser.set_defaults(_subaction='clear')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        return 2

    logging.set_level(args.log_level)

    if not hasattr(args, '_action') or not hasattr(args, '_subaction'):
        parser.print_help()
        return 0

    try:
        match (args._action, args._subaction):
            case ('caption', _): action = functools.partial(cli.caption, args.dataset_toml)

            case ('config', 'list'): action = functools.partial(cli_config.config_list)
            case ('config', 'get'): action = functools.partial(cli_config.config_get, args.key)
            case ('config', 'set'): action = functools.partial(cli_config.config_set, args.key, args.value, force=args.force)
            case ('config', 'delete'): action = functools.partial(cli_config.config_delete, args.key, force=args.force)
            case ('config', 'clear'): action = functools.partial(cli_config.config_clear)
            case ('config', _): action = functools.partial(config_parser.print_help)

            case _: action = functools.partial(parser.print_help)

        ret = action()
    except KeyboardInterrupt:
        ret = 130

    if isinstance(ret, int):
        return ret
    
    return 0


if __name__ == "__main__":
    main()
