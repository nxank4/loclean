"""CLI module for Loclean.

This module provides command-line interface for model management,
interactive shell, and other operations.
"""

import typer
from rich.console import Console

from loclean.cli.model import app as model_app
from loclean.cli.shell import MODE_CLEAN, run_shell

app = typer.Typer(
    name="loclean",
    help="Loclean - Local AI Data Cleaner",
    add_completion=False,
)
console = Console()

app.add_typer(
    model_app,
    name="model",
    help="Model management commands",
)


@app.command()
def shell(
    model: str = typer.Option(None, "--model", "-m", help="Ollama model tag"),
    host: str = typer.Option(None, "--host", help="Ollama server URL"),
    mode: str = typer.Option(
        MODE_CLEAN, "--mode", help="Initial mode: clean, extract, or scrub"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Start the interactive loclean shell."""
    run_shell(model=model, host=host, mode=mode, verbose=verbose)


if __name__ == "__main__":
    app()
