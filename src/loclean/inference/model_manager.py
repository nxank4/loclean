"""Ollama model availability and automatic pulling.

Ensures that the requested model is present in the local Ollama registry
before generation begins. If missing, pulls it with a Rich progress bar.
"""

import logging
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


def model_exists(client: Any, model: str) -> bool:
    """Check whether *model* is already available in the local Ollama registry.

    Args:
        client: An ``ollama.Client`` instance.
        model: Model tag to look up (e.g. ``"phi3"``).

    Returns:
        ``True`` if the model is present locally.
    """
    try:
        response = client.list()
    except Exception:
        return False

    models = response.get("models", [])
    for entry in models:
        name: str = entry.get("name", "")
        if name == model or name.startswith(f"{model}:"):
            return True
    return False


def ensure_model(
    client: Any,
    model: str,
    console: Console | None = None,
) -> None:
    """Pull *model* from the Ollama registry if it is not already local.

    Wraps the streaming pull in a Rich progress bar that displays
    byte transfer status, speed, and estimated time remaining.

    Args:
        client: An ``ollama.Client`` instance.
        model: Model tag to pull (e.g. ``"phi3"``).
        console: Optional Rich console; a default one is created if omitted.

    Raises:
        RuntimeError: If the pull fails or encounters an error status.
    """
    if model_exists(client, model):
        logger.info(
            f"[green]✓[/green] Model [bold cyan]{model}[/bold cyan] "
            "is already available."
        )
        return

    if console is None:
        console = Console()

    console.print(
        f"[yellow]⏳[/yellow] Model [bold cyan]{model}[/bold cyan] "
        "not found locally. Pulling …"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task(f"Pulling {model}", total=None)
        current_status = ""

        for chunk in client.pull(model, stream=True):
            status = chunk.get("status", "")

            if status != current_status:
                current_status = status
                progress.update(task_id, description=f"{model}: {status}")

            total = chunk.get("total")
            completed = chunk.get("completed")

            if total is not None:
                progress.update(task_id, total=total)
            if completed is not None:
                progress.update(task_id, completed=completed)

            if "error" in chunk:
                raise RuntimeError(f"Failed to pull model '{model}': {chunk['error']}")

    console.print(
        f"[green]✓[/green] Model [bold cyan]{model}[/bold cyan] pulled successfully."
    )
