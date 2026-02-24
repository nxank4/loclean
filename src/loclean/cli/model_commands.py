"""Implementation of model CLI commands.

This module provides commands for checking the Ollama connection,
listing available models, and pulling new models.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from loclean.inference.daemon import ensure_daemon
from loclean.inference.model_manager import ensure_model

console = Console()


def check_connection(
    host: str = "http://localhost:11434",
    console: Optional[Console] = None,
) -> None:
    """Check connection to Ollama and list available models.

    Automatically starts the daemon if the host is local and
    the server is not yet running.

    Args:
        host: Ollama server URL.
        console: Rich console instance for output.
    """
    if console is None:
        console = Console()

    console.print(f"[bold]Checking Ollama at:[/bold] {host}")

    try:
        ensure_daemon(host)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e
    except ConnectionError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e

    try:
        import ollama as _ollama  # type: ignore[import-untyped]

        client = _ollama.Client(host=host)
        models_response = client.list()
    except Exception as e:
        console.print(f"[red]✗[/red] Could not query Ollama: {e}")
        raise typer.Exit(code=1) from e

    console.print("[green]✓[/green] Connected to Ollama successfully!\n")

    models = models_response.get("models", [])
    if not models:
        console.print("[dim]No models found. Pull one with:[/dim]")
        console.print("  loclean model pull phi3")
        return

    table = Table(
        title="Available Ollama Models",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", style="magenta")
    table.add_column("Modified", style="green")

    for model in models:
        name = model.get("name", "unknown")
        size_bytes = model.get("size", 0)
        size_str = f"{size_bytes / (1024 * 1024):.0f} MB" if size_bytes else "Unknown"
        modified = model.get("modified_at", "Unknown")
        if isinstance(modified, str) and len(modified) > 19:
            modified = modified[:19]
        table.add_row(name, size_str, str(modified))

    console.print(table)


def pull_model(
    model_name: str,
    host: str = "http://localhost:11434",
    console: Optional[Console] = None,
) -> None:
    """Pull a model from the Ollama registry with progress feedback.

    Automatically starts the daemon if the host is local and
    the server is not yet running.

    Args:
        model_name: Ollama model tag to pull (e.g. "phi3", "llama3").
        host: Ollama server URL.
        console: Rich console instance for output.
    """
    if console is None:
        console = Console()

    try:
        ensure_daemon(host)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e
    except ConnectionError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e

    try:
        import ollama as _ollama  # type: ignore[import-untyped]

        client = _ollama.Client(host=host)
        ensure_model(client, model_name, console=console)
    except RuntimeError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to pull model: {e}")
        raise typer.Exit(code=1) from e
