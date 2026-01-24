import io
import logging
from typing import Any, cast
from unittest.mock import patch

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
    """Verify that LlamaCppEngine in verbose mode logs with Rich markup."""
    from loclean.inference.local.llama_cpp import LlamaCppEngine
    from loclean.inference.local.llama_cpp import logger as llama_logger

    # Ensure the logger propagates so caplog can catch it
    llama_logger.propagate = True
    caplog.set_level(logging.DEBUG)

    with (
        patch("loclean.inference.local.llama_cpp.Llama"),
        patch("loclean.inference.local.llama_cpp.LlamaGrammar"),
        patch("loclean.inference.local.llama_cpp.download_model"),
        patch("loclean.cache.LocleanCache") as mock_cache_class,
    ):
        # Mock cache to return empty results (force cache miss)
        mock_cache = mock_cache_class.return_value
        mock_cache.get_batch.return_value = {}
        mock_cache.set_batch.return_value = None

        engine = LlamaCppEngine(verbose=True)
        # Mock create_completion to return valid JSON
        llm_any: Any = engine.llm
        llm_any.create_completion.return_value = {
            "choices": [{"text": '{"reasoning": "ok", "value": 1, "unit": "kg"}'}]
        }

        engine.clean_batch(["unique_item_for_ux_test"], "instruction")

        # Check captured logs
        log_messages = [r.message for record in [caplog.records] for r in record]
        assert any("[bold blue]PROMPT:[/bold blue]" in msg for msg in log_messages)
        assert any(
            "[bold green]RAW OUTPUT:[/bold green]" in msg for msg in log_messages
        )

    # Restore propagation
    llama_logger.propagate = False
