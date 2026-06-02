import click

from yadc.api.application import Application
from yadc.api.configuration import Configuration


@click.group("webui")
def webui():
    """Launch the yadc web UI."""
    pass


@webui.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=7860, type=int, help="Bind port")
@click.option("--threads", default=16, type=int, help="Number of server threads")
@click.option("--cors/--no-cors", default=True, help="Enable CORS (for development)")
@click.option("--banner/--no-banner", default=True, help="Show startup banner")
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Set the logging level",
)
def serve(host: str, port: int, threads: int, cors: bool, banner: bool, log_level: str):
    """Start the web UI server."""
    import logging as _logging

    configuration = Configuration(
        http_host=host,
        http_port=port,
        http_threads=threads,
        api_cors_enable=cors,
        banner_enable=banner,
        logging_default_level=getattr(_logging, log_level.upper(), _logging.INFO),
    )
    application = Application(configuration)
    application.run()
