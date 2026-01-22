"""Rich output utilities for enhanced logging and display.

This module provides helper functions to display structured information
using Rich components (Table, Panel, Progress) in logging context.
"""

import logging
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)


def get_rich_console() -> Optional[Console]:
    """
    Get Rich Console instance from current logger's RichHandler if available.

    Returns:
        Console instance if RichHandler is found, None otherwise.
    """
    # Try to get console from root logger's RichHandler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if hasattr(handler, "console"):
            return handler.console

    # Try to get from current logger
    for handler in logger.handlers:
        if hasattr(handler, "console"):
            return handler.console

    # Fallback: create new console if TTY
    if sys.stderr.isatty():
        return Console(stderr=True, force_terminal=True)
    return None


def log_batch_stats(
    total_patterns: int,
    num_batches: int,
    batch_size: int,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    col_name: str = "",
) -> None:
    """
    Log batch processing statistics using Rich Panel.

    Args:
        total_patterns: Total number of unique patterns to process.
        num_batches: Number of batches.
        batch_size: Size of each batch.
        parallel: Whether parallel processing is enabled.
        max_workers: Number of workers (if parallel).
        col_name: Column name being processed.
    """
    console = get_rich_console()
    if not console:
        # Fallback to simple logging
        logger.info(
            f"Semantic Cleaning: Processing {total_patterns} unique patterns "
            f"in column '{col_name}'."
        )
        if parallel and max_workers:
            logger.info(
                f"Processing {num_batches} batches in parallel "
                f"with {max_workers} workers"
            )
        return

    # Create info table
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")

    table.add_row("Column:", f"[bold]{col_name}[/bold]")
    table.add_row("Unique Patterns:", f"[yellow]{total_patterns}[/yellow]")
    table.add_row("Batches:", f"[yellow]{num_batches}[/yellow]")
    table.add_row("Batch Size:", f"[yellow]{batch_size}[/yellow]")
    table.add_row(
        "Mode:",
        f"[green]Parallel ({max_workers} workers)[/green]"
        if parallel and max_workers
        else "[dim]Sequential[/dim]",
    )

    panel = Panel(
        table,
        title="[bold cyan]Batch Processing Statistics[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def log_cache_stats(
    total_items: int,
    cache_hits: int,
    cache_misses: int,
    context: str = "Processing",
) -> None:
    """
    Log cache statistics using Rich Table.

    Args:
        total_items: Total number of items.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses.
        context: Context description (e.g., "Extraction", "Cleaning").
    """
    console = get_rich_console()
    if not console:
        # Fallback to simple logging
        hit_ratio = (cache_hits / total_items * 100) if total_items > 0 else 0.0
        logger.info(
            f"Cache: {cache_hits} hits, {cache_misses} misses "
            f"({hit_ratio:.1f}% hit ratio)"
        )
        if cache_misses > 0:
            logger.info(f"Cache miss for {cache_misses} items. Running inference...")
        return

    hit_ratio = (cache_hits / total_items * 100) if total_items > 0 else 0.0

    table = Table(title=f"[bold]{context} Cache Statistics[/bold]", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    table.add_row("Total Items", str(total_items), "100.0%")
    table.add_row(
        "[green]Cache Hits[/green]",
        f"[green]{cache_hits}[/green]",
        f"[green]{hit_ratio:.1f}%[/green]",
    )
    table.add_row(
        "[yellow]Cache Misses[/yellow]",
        f"[yellow]{cache_misses}[/yellow]",
        f"[yellow]{100.0 - hit_ratio:.1f}%[/yellow]",
    )

    console.print(table)

    if cache_misses > 0:
        logger.info(
            f"[yellow]📥[/yellow] Processing [yellow]{cache_misses}[/yellow] "
            f"items from cache miss..."
        )


def log_processing_summary(
    total_processed: int,
    successful: int,
    failed: int,
    time_taken: Optional[float] = None,
    context: str = "Processing",
) -> None:
    """
    Log processing summary using Rich Table.

    Args:
        total_processed: Total number of items processed.
        successful: Number of successful items.
        failed: Number of failed items.
        time_taken: Time taken in seconds (optional).
        context: Context description (e.g., "Extraction", "Cleaning").
    """
    console = get_rich_console()
    if not console:
        # Fallback to simple logging
        success_rate = (
            (successful / total_processed * 100) if total_processed > 0 else 0.0
        )
        logger.info(
            f"{context} Summary: {successful}/{total_processed} successful "
            f"({success_rate:.1f}%)"
        )
        if time_taken:
            logger.info(f"Time taken: {time_taken:.2f}s")
        return

    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0.0

    table = Table(
        title=f"[bold green]✓ {context} Summary[/bold green]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Status", style="green", justify="right")

    table.add_row("Total Processed", str(total_processed), "100.0%")
    table.add_row(
        "[green]Successful[/green]",
        f"[green]{successful}[/green]",
        f"[green]{success_rate:.1f}%[/green]",
    )
    if failed > 0:
        table.add_row(
            "[red]Failed[/red]",
            f"[red]{failed}[/red]",
            f"[red]{100.0 - success_rate:.1f}%[/red]",
        )

    if time_taken:
        table.add_row(
            "[dim]Time Taken[/dim]",
            f"[dim]{time_taken:.2f}s[/dim]",
            "",
        )

    console.print(table)


def log_error_summary(
    errors: List[Dict[str, Any]],
    max_display: int = 5,
    context: str = "Processing",
) -> None:
    """
    Log error summary using Rich Table when there are many errors.

    Args:
        errors: List of error dicts with keys: 'type', 'count', 'sample_items'.
        max_display: Maximum number of error types to display in detail.
        context: Context description (e.g., "Extraction", "Cleaning").
    """
    if not errors:
        return

    console = get_rich_console()
    if not console:
        # Fallback to simple logging
        total_errors = sum(e.get("count", 0) for e in errors)
        logger.warning(f"{context} Errors: {total_errors} total errors")
        for error in errors[:max_display]:
            logger.warning(
                f"  - {error.get('type', 'Unknown')}: "
                f"{error.get('count', 0)} occurrences"
            )
        return

    total_errors = sum(e.get("count", 0) for e in errors)

    table = Table(
        title=f"[bold red]⚠ {context} Error Summary[/bold red]",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Error Type", style="red", no_wrap=True)
    table.add_column("Count", style="yellow", justify="right")
    table.add_column("Sample Items", style="dim")

    for error in errors[:max_display]:
        error_type = error.get("type", "Unknown")
        count = error.get("count", 0)
        samples = error.get("sample_items", [])
        sample_str = ", ".join(
            [f"[dim]'{s[:30]}...'[/dim]" if len(s) > 30 else f"[dim]'{s}'[/dim]"]
            for s in samples[:2]
        )
        if len(samples) > 2:
            sample_str += f" [dim](+{len(samples) - 2} more)[/dim]"

        table.add_row(error_type, str(count), sample_str)

    if len(errors) > max_display:
        remaining = sum(e.get("count", 0) for e in errors[max_display:])
        table.add_row(
            "[dim]... and more[/dim]",
            f"[dim]{remaining}[/dim]",
            "[dim]See logs above[/dim]",
        )

    console.print(table)


def create_progress(
    total: Optional[int] = None,
    description: str = "Processing",
    console: Optional[Console] = None,
) -> Optional[Progress]:
    """
    Create a Rich Progress instance for batch processing.

    Args:
        total: Total number of items (None for indeterminate).
        description: Progress description.
        console: Optional Console instance (uses get_rich_console() if None).

    Returns:
        Progress instance configured for batch processing, or None if Rich unavailable.
    """
    if console is None:
        console = get_rich_console()

    if console is None:
        # Return None - caller should use simple logging fallback
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
