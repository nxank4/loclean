"""Implementation of model CLI commands.

This module contains the actual implementation of download, list, and status commands
with rich progress bars, tables, and error handling.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from loclean.inference.local.downloader import download_model as download_model_func
from loclean.inference.local.llama_cpp import get_model_registry

console = Console()


def download_model(
    name: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
    console: Optional[Console] = None,
) -> None:
    """
    Download a model from HuggingFace Hub.

    Args:
        name: Model name to download.
        cache_dir: Optional custom cache directory.
        force: Force re-download even if exists.
        console: Rich console instance for output.
    """
    if console is None:
        console = Console()

    registry = get_model_registry()

    if name not in registry:
        console.print(f"[red]Error:[/red] Model '{name}' not found in registry.")
        console.print("\nAvailable models:")
        list_models(console=console)
        raise typer.Exit(code=1)

    model_info = registry[name]
    repo_id = model_info["repo"]
    filename = model_info["filename"]
    size_mb = model_info.get("size_mb", 0)
    description = model_info.get("description", "")

    cache_path = Path(cache_dir) if cache_dir else None

    console.print(f"[bold]Downloading model:[/bold] {name}")
    if description:
        console.print(f"[dim]Description:[/dim] {description}")
    if size_mb:
        console.print(f"[dim]Size:[/dim] ~{size_mb} MB")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)
            path = download_model_func(
                model_name=name,
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_path,
                force=force,
                show_progress=True,
            )
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Successfully downloaded to: {path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Download failed: {e}")
        raise typer.Exit(code=1) from e


def list_models(console: Optional[Console] = None) -> None:
    """
    List all available models in the registry.

    Args:
        console: Rich console instance for output.
    """
    if console is None:
        console = Console()

    registry = get_model_registry()

    table = Table(title="Available Models", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", style="magenta")
    table.add_column("Description", style="green")

    for name, info in sorted(registry.items()):
        size_mb = info.get("size_mb", 0)
        size_str = f"~{size_mb} MB" if size_mb else "Unknown"
        description = info.get("description", "No description")
        table.add_row(name, size_str, description)

    console.print(table)


def check_status(console: Optional[Console] = None) -> None:
    """
    Check download status of all models.

    Args:
        console: Rich console instance for output.
    """
    if console is None:
        console = Console()

    registry = get_model_registry()
    cache_dir = Path.home() / ".cache" / "loclean"

    table = Table(title="Model Download Status", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Path", style="green")

    for name, info in sorted(registry.items()):
        filename = info["filename"]
        local_path = cache_dir / filename

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            status = f"[green]✓ Downloaded[/green] ({size_mb:.1f} MB)"
            path_str = str(local_path)
        else:
            status = "[red]✗ Not downloaded[/red]"
            path_str = "N/A"

        table.add_row(name, status, path_str)

    console.print(table)
    console.print(f"\n[dim]Cache directory:[/dim] {cache_dir}")
