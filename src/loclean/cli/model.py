"""Model management CLI commands.

This module provides subcommands for model operations: download, list, and status.
"""

import typer
from rich.console import Console

from loclean.cli.model_commands import check_status, download_model, list_models

app = typer.Typer(
    name="model",
    help="Model management commands",
    add_completion=False,
)
console = Console()


@app.command()
def download(
    name: str = typer.Option(..., "--name", "-n", help="Model name to download"),
    cache_dir: str = typer.Option(None, "--cache-dir", help="Custom cache directory"),
    force: bool = typer.Option(False, "--force", help="Force re-download"),
) -> None:
    """Download a model from HuggingFace Hub."""
    download_model(name=name, cache_dir=cache_dir, force=force, console=console)


@app.command()
def list() -> None:
    """List all available models in the registry."""
    list_models(console=console)


@app.command()
def status() -> None:
    """Check download status of all models."""
    check_status(console=console)
