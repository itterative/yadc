import argparse
import functools

def main():
    from yadc import cli

    parser = argparse.ArgumentParser(prog='yadc', description='Yet Another Dataset Captioner')
    parser.set_defaults(_action='help', _subaction='default')

    subparser = parser.add_subparsers(dest='command', description='commands to run')

    caption_parser = subparser.add_parser('caption')
    caption_parser.add_argument('dataset_toml', type=str)
    caption_parser.set_defaults(_action='caption', _subaction='default')

    if cli.FLAG_USER_CONFIG:
        config_parser = subparser.add_parser('config', description='Manage the user config')
        config_parser.set_defaults(_action='config')

        config_subparser = config_parser.add_subparsers(dest='subcommand', description='subcommands to run')

        config_list_parser = config_subparser.add_parser('list')
        config_list_parser.set_defaults(_subaction='list')

        config_get_parser = config_subparser.add_parser('get')
        config_get_parser.add_argument('key', help='The user config key to retrieve. Options: api_url, api_token, api_model_name', type=str)
        config_get_parser.set_defaults(_subaction='get')

        config_set_parser = config_subparser.add_parser('set')
        config_set_parser.add_argument('key', help='The user config key to set. Options: api_url, api_token, api_model_name', type=str)
        config_set_parser.add_argument('value', type=str)
        config_set_parser.set_defaults(_subaction='set')

        config_delete_parser = config_subparser.add_parser('delete')
        config_delete_parser.add_argument('key', help='The user config key to delete. Options: api_url, api_token, api_model_name', type=str)
        config_delete_parser.set_defaults(_subaction='delete')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        return 1

    if not hasattr(args, '_action') or not hasattr(args, '_subaction'):
        parser.print_help()
        return 0

    match (args._action, args._subaction):
        case ('caption', _): action = functools.partial(cli.caption, args.dataset_toml)

        case ('config', 'list'): action = functools.partial(cli.config_list)
        case ('config', 'get'): action = functools.partial(cli.config_get, args.key)
        case ('config', 'set'): action = functools.partial(cli.config_set, args.key, args.value)
        case ('config', 'delete'): action = functools.partial(cli.config_delete, args.key)
        case ('config', _): action = functools.partial(config_parser.print_help)

        case _: action = functools.partial(parser.print_help)

    ret = action()
    if isinstance(ret, int):
        return ret
    
    return 0


if __name__ == "__main__":
    main()
