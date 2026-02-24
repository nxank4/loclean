"""Model management CLI commands.

This module provides subcommands for Ollama model operations.
"""

import typer
from rich.console import Console

from loclean.cli.model_commands import check_connection, pull_model

app = typer.Typer(
    name="model",
    help="Model management commands",
    add_completion=False,
)
console = Console()


@app.command()
def status(
    host: str = typer.Option(
        "http://localhost:11434", "--host", help="Ollama server URL"
    ),
) -> None:
    """Check Ollama connection and list available models."""
    check_connection(host=host, console=console)


@app.command()
def pull(
    name: str = typer.Argument(..., help="Model tag to pull (e.g. phi3, llama3)"),
    host: str = typer.Option(
        "http://localhost:11434", "--host", help="Ollama server URL"
    ),
) -> None:
    """Pull a model from the Ollama registry."""
    pull_model(model_name=name, host=host, console=console)
