"""CLI module for Loclean.

This module provides command-line interface for model management and other operations.
"""

import typer
from rich.console import Console

from loclean.cli.model import app as model_app

app = typer.Typer(
    name="loclean",
    help="Loclean - Local AI Data Cleaner",
    add_completion=False,
)
console = Console()

# Add model subcommand group
app.add_typer(
    model_app,
    name="model",
    help="Model management commands",
)

if __name__ == "__main__":
    app()
