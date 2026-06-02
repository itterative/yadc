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
def serve(host: str, port: int, threads: int, cors: bool):
    """Start the web UI server."""
    configuration = Configuration(
        http_host=host,
        http_port=port,
        http_threads=threads,
        api_cors_enable=cors,
    )
    application = Application(configuration)
    application.run()
