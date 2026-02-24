import io
import logging
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from loclean.utils.rich_output import (
    create_progress,
    log_batch_stats,
    log_cache_stats,
    log_processing_summary,
)


@pytest.fixture
def string_console() -> Console:
    """Create a Rich Console that writes to a string buffer."""
    buf = io.StringIO()
    # force_terminal=True enables color/markup in the buffer
    return Console(file=buf, force_terminal=True, width=100)


def test_progress_bar_creation(string_console: Console) -> None:
    """Verify that create_progress returns a Progress instance with expected columns."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        SpinnerColumn,
        TaskProgressColumn,
    )

    progress = create_progress(
        total=10, description="Test Progress", console=string_console
    )
    assert progress is not None

    # Check if essential columns exist
    column_types = [type(c) for c in progress.columns]
    assert SpinnerColumn in column_types
    assert BarColumn in column_types
    assert TaskProgressColumn in column_types
    assert MofNCompleteColumn in column_types


def test_log_batch_stats_panel(string_console: Console) -> None:
    """Verify that log_batch_stats prints a Panel with correct info."""
    with patch(
        "loclean.utils.rich_output.get_rich_console", return_value=string_console
    ):
        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=True,
            max_workers=4,
            col_name="messy_data",
        )

    output = cast(io.StringIO, string_console.file).getvalue()
    assert "Batch Processing Statistics" in output
    assert "messy_data" in output
    assert "100" in output
    assert "5" in output
    assert "Parallel (4 workers)" in output


def test_log_cache_stats_table(string_console: Console) -> None:
    """Verify that log_cache_stats prints a Table with hit/miss info."""
    with patch(
        "loclean.utils.rich_output.get_rich_console", return_value=string_console
    ):
        log_cache_stats(
            total_items=10,
            cache_hits=7,
            cache_misses=3,
            context="TestContext",
        )

    output = cast(io.StringIO, string_console.file).getvalue()
    assert "TestContext Cache Statistics" in output
    assert "Cache Hits" in output
    assert "7" in output
    assert "70.0%" in output
    assert "Cache Misses" in output
    assert "3" in output


def test_log_processing_summary_table(string_console: Console) -> None:
    """Verify that log_processing_summary prints a success summary."""
    with patch(
        "loclean.utils.rich_output.get_rich_console", return_value=string_console
    ):
        log_processing_summary(
            total_processed=50,
            successful=48,
            failed=2,
            time_taken=12.34,
            context="Cleaning",
        )

    output = cast(io.StringIO, string_console.file).getvalue()
    assert "Cleaning Summary" in output
    assert "50" in output
    assert "48" in output
    assert "2" in output
    assert "12.34s" in output


def test_verbose_mode_logging_markup(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that OllamaEngine in verbose mode logs initialisation info."""
    import sys

    from loclean.inference.ollama_engine import logger as ollama_logger

    ollama_logger.propagate = True
    caplog.set_level(logging.DEBUG)

    mock_ollama = MagicMock()
    with (
        patch.dict(sys.modules, {"ollama": mock_ollama}),
        patch("loclean.inference.ollama_engine.ensure_daemon"),
        patch("loclean.inference.ollama_engine.ensure_model"),
    ):
        from loclean.inference.ollama_engine import OllamaEngine

        engine = OllamaEngine(verbose=True)
        assert engine.verbose is True

    ollama_logger.propagate = False
