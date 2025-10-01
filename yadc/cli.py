import click

from . import cli_logging, cli_caption, cli_envs, cli_configs, cli_templates

@click.group(
    'yadc',
    help='Yet Another Dataset Captioner',
)
def cli():
    from yadc.core import logging
    logging.set_handler(cli_logging.ClickHandler())

cli.add_command(cli_caption.caption)
cli.add_command(cli_envs.envs)
cli.add_command(cli_configs.configs)
cli.add_command(cli_templates.templates)

@cli.command(
    short_help='Print version',
    help='Prints the current version of the program',
)
def version():
    from . import __version__
    print(__version__)
