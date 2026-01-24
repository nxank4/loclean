"""Enhanced logging configuration with Rich formatting.

This module provides colored, structured logging using Rich library
for better readability and debugging experience.
"""

import logging
import sys
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


def setup_logging(
    level: int = logging.INFO,
    show_path: bool = False,
    show_time: bool = True,
    rich_tracebacks: bool = True,
    console: Optional[Console] = None,
) -> None:
    """
    Configure logging with Rich formatting.

    Args:
        level: Logging level (default: INFO).
        show_path: Show file path in log messages (default: False).
        show_time: Show timestamp in log messages (default: True).
        rich_tracebacks: Use Rich for traceback formatting (default: True).
        console: Optional Rich Console instance (default: creates new one).
    """
    if console is None:
        console = Console(stderr=True)

    # Create Rich handler with custom formatting
    handler = RichHandler(
        console=console,
        show_path=show_path,
        show_time=show_time,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_level=True,
        level=level,
        omit_repeated_times=False,
    )

    # Custom formatter for cleaner output
    handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with Rich formatting.

    Args:
        name: Logger name (typically __name__).
        level: Optional logging level override.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    # Ensure Rich handler is set up if not already
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        # Check root logger
        root_logger = logging.getLogger()
        if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
            setup_logging()

    return logger


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""

    # Color mapping for log levels
    COLORS = {
        "DEBUG": "dim white",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    # Emoji/icons for log levels
    ICONS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💥",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and icons."""
        level_name = record.levelname
        color = self.COLORS.get(level_name, "white")
        icon = self.ICONS.get(level_name, "•")

        # Create formatted message
        message = super().format(record)

        # Build colored text
        text = Text()
        text.append(f"{icon} ", style="dim")
        text.append(f"[{level_name:8s}] ", style=color)
        text.append(message)

        return text.plain if not hasattr(self, "_console") else str(text)


def configure_module_logger(
    module_name: str,
    level: int = logging.INFO,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure a module-specific logger with enhanced formatting.

    Args:
        module_name: Module name (typically __name__).
        level: Logging level.
        use_colors: Enable colored output (default: True).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Always try to use Rich for better formatting (force_terminal=True enables colors)
    # Fallback to standard handler only if explicitly disabled
    handler: Union[RichHandler, logging.Handler]
    if use_colors:
        # Use Rich handler for colored output with enhanced formatting
        # force_terminal=True enables colors even in non-TTY environments
        console = Console(stderr=True, force_terminal=True, legacy_windows=False)
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            markup=True,
            show_level=True,
            level=logging.NOTSET,  # Allow logger to control filtering
            omit_repeated_times=False,
            keywords=["model", "cache", "extract", "clean", "scrub"],
        )
        handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
    else:
        # Fallback to standard handler when colors are disabled
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)
    logger.propagate = False

    return logger
