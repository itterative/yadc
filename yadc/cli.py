import click

from . import cli_caption, cli_envs, cli_template

@click.group(
    'yadc',
    help='Yet Another Dataset Captioner',
)
def cli():
    pass

cli.add_command(cli_caption.caption)
cli.add_command(cli_envs.envs)
cli.add_command(cli_template.templates)

@cli.command(
    short_help='Print version',
    help='Prints the current version of the program',
)
def version():
    from . import __version__
    print(__version__)
