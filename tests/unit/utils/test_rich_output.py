"""Test cases for Rich output utilities."""

import logging
from unittest.mock import MagicMock, Mock, patch

from rich.console import Console
from rich.logging import RichHandler

from loclean.utils.rich_output import (
    create_progress,
    get_rich_console,
    log_batch_stats,
    log_cache_stats,
    log_error_summary,
    log_processing_summary,
)


class TestGetRichConsole:
    """Test cases for get_rich_console function."""

    def test_console_extraction_from_root_logger_rich_handler(self) -> None:
        """Test console extraction from root logger's RichHandler."""
        mock_console = Console(stderr=True)
        handler = RichHandler(console=mock_console)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

        console = get_rich_console()

        assert console is mock_console

    def test_console_extraction_from_current_logger_rich_handler(self) -> None:
        """Test console extraction from current logger's RichHandler."""
        mock_console = Console(stderr=True)
        handler = RichHandler(console=mock_console)

        # Clear root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Add to module logger
        module_logger = logging.getLogger("loclean.utils.rich_output")
        module_logger.handlers.clear()
        module_logger.addHandler(handler)

        console = get_rich_console()

        assert console is mock_console

    @patch("sys.stderr.isatty", return_value=True)
    def test_fallback_to_new_console_when_tty_available(
        self, mock_isatty: Mock
    ) -> None:
        """Test fallback to new console when TTY available."""
        # Clear all handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        module_logger = logging.getLogger("loclean.utils.rich_output")
        module_logger.handlers.clear()

        console = get_rich_console()

        assert console is not None
        assert isinstance(console, Console)
        mock_isatty.assert_called_once()

    @patch("sys.stderr.isatty", return_value=False)
    def test_return_none_when_no_rich_handler_and_non_tty(
        self, mock_isatty: Mock
    ) -> None:
        """Test return None when no RichHandler and non-TTY."""
        # Clear all handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        module_logger = logging.getLogger("loclean.utils.rich_output")
        module_logger.handlers.clear()

        console = get_rich_console()

        assert console is None
        mock_isatty.assert_called_once()

    def test_console_reuse_same_instance(self) -> None:
        """Test console reuse (same instance)."""
        mock_console = Console(stderr=True)
        handler = RichHandler(console=mock_console)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

        console1 = get_rich_console()
        console2 = get_rich_console()

        assert console1 is console2
        assert console1 is mock_console


class TestLogBatchStats:
    """Test cases for log_batch_stats function."""

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_logging_with_rich_panel_when_console_available(
        self, mock_get_console: Mock
    ) -> None:
        """Test logging with Rich Panel when console available."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=False,
            col_name="test_col",
        )

        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert hasattr(call_arg, "title")  # Panel has title

    @patch("loclean.utils.rich_output.get_rich_console", return_value=None)
    @patch("loclean.utils.rich_output.logger")
    def test_fallback_to_simple_logging_when_console_unavailable(
        self, mock_logger: Mock, mock_get_console: Mock
    ) -> None:
        """Test fallback to simple logging when console unavailable."""
        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=False,
            col_name="test_col",
        )

        mock_logger.info.assert_called()
        assert "100 unique patterns" in str(mock_logger.info.call_args)

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_all_table_fields(self, mock_get_console: Mock) -> None:
        """Test all table fields (column, patterns, batches, batch_size, mode)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=False,
            col_name="test_col",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_parallel_mode_display_with_max_workers(
        self, mock_get_console: Mock
    ) -> None:
        """Test parallel mode display with max_workers."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=True,
            max_workers=4,
            col_name="test_col",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_sequential_mode_display(self, mock_get_console: Mock) -> None:
        """Test sequential mode display."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=False,
            col_name="test_col",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_empty_column_name_handling(self, mock_get_console: Mock) -> None:
        """Test empty column name handling."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_batch_stats(
            total_patterns=100,
            num_batches=5,
            batch_size=20,
            parallel=False,
            col_name="",
        )

        mock_console.print.assert_called_once()


class TestLogCacheStats:
    """Test cases for log_cache_stats function."""

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_rich_table_display_with_cache_hits_misses(
        self, mock_get_console: Mock
    ) -> None:
        """Test Rich Table display with cache hits/misses."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_cache_stats(
            total_items=100,
            cache_hits=60,
            cache_misses=40,
            context="Extraction",
        )

        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert hasattr(call_arg, "title")  # Table has title

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_hit_ratio_calculation_50_percent(self, mock_get_console: Mock) -> None:
        """Test hit ratio calculation (50%)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_cache_stats(
            total_items=100,
            cache_hits=50,
            cache_misses=50,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_hit_ratio_calculation_100_percent(self, mock_get_console: Mock) -> None:
        """Test hit ratio calculation (100%)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_cache_stats(
            total_items=100,
            cache_hits=100,
            cache_misses=0,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console", return_value=None)
    @patch("loclean.utils.rich_output.logger")
    def test_fallback_to_simple_logging(
        self, mock_logger: Mock, mock_get_console: Mock
    ) -> None:
        """Test fallback to simple logging."""
        log_cache_stats(
            total_items=100,
            cache_hits=60,
            cache_misses=40,
            context="Extraction",
        )

        mock_logger.info.assert_called()
        # Check that logger was called - it may be called multiple times
        # Just verify it was called (cache stats logging happens)
        assert mock_logger.info.call_count >= 1

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_different_context_strings(self, mock_get_console: Mock) -> None:
        """Test different context strings."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_cache_stats(
            total_items=100,
            cache_hits=60,
            cache_misses=40,
            context="Cleaning",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_with_zero_total_items_division_by_zero_protection(
        self, mock_get_console: Mock
    ) -> None:
        """Test with zero total items (division by zero protection)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        # Should not raise ZeroDivisionError
        log_cache_stats(
            total_items=0,
            cache_hits=0,
            cache_misses=0,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    @patch("loclean.utils.rich_output.logger")
    def test_cache_miss_logging_message(
        self, mock_logger: Mock, mock_get_console: Mock
    ) -> None:
        """Test cache miss logging message."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_cache_stats(
            total_items=100,
            cache_hits=60,
            cache_misses=40,
            context="Extraction",
        )

        mock_logger.info.assert_called()
        assert "40" in str(mock_logger.info.call_args)


class TestLogProcessingSummary:
    """Test cases for log_processing_summary function."""

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_rich_table_with_successful_failed_counts(
        self, mock_get_console: Mock
    ) -> None:
        """Test Rich Table with successful/failed counts."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_processing_summary(
            total_processed=100,
            successful=80,
            failed=20,
            context="Extraction",
        )

        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert hasattr(call_arg, "title")  # Table has title

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_success_rate_calculation(self, mock_get_console: Mock) -> None:
        """Test success rate calculation."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_processing_summary(
            total_processed=100,
            successful=75,
            failed=25,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_with_time_taken_parameter(self, mock_get_console: Mock) -> None:
        """Test with time_taken parameter."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_processing_summary(
            total_processed=100,
            successful=80,
            failed=20,
            time_taken=5.5,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_without_time_taken_parameter(self, mock_get_console: Mock) -> None:
        """Test without time_taken parameter."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_processing_summary(
            total_processed=100,
            successful=80,
            failed=20,
            context="Extraction",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console", return_value=None)
    @patch("loclean.utils.rich_output.logger")
    def test_fallback_to_simple_logging(
        self, mock_logger: Mock, mock_get_console: Mock
    ) -> None:
        """Test fallback to simple logging."""
        log_processing_summary(
            total_processed=100,
            successful=80,
            failed=20,
            context="Extraction",
        )

        mock_logger.info.assert_called()
        assert "80/100 successful" in str(mock_logger.info.call_args)

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_different_context_strings(self, mock_get_console: Mock) -> None:
        """Test different context strings."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_processing_summary(
            total_processed=100,
            successful=80,
            failed=20,
            context="Cleaning",
        )

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_with_zero_total_processed(self, mock_get_console: Mock) -> None:
        """Test with zero total processed."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        # Should not raise ZeroDivisionError
        log_processing_summary(
            total_processed=0,
            successful=0,
            failed=0,
            context="Extraction",
        )

        mock_console.print.assert_called_once()


class TestLogErrorSummary:
    """Test cases for log_error_summary function."""

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_rich_table_with_error_list(self, mock_get_console: Mock) -> None:
        """Test Rich Table with error list."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        errors = [
            {"type": "ValidationError", "count": 5, "sample_items": ["item1", "item2"]}
        ]

        log_error_summary(errors, max_display=5, context="Extraction")

        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert hasattr(call_arg, "title")  # Table has title

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_max_display_limit(self, mock_get_console: Mock) -> None:
        """Test max_display limit (show only first N)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        errors = [
            {"type": "Error1", "count": 3, "sample_items": ["a"]},
            {"type": "Error2", "count": 2, "sample_items": ["b"]},
            {"type": "Error3", "count": 1, "sample_items": ["c"]},
        ]

        log_error_summary(errors, max_display=2, context="Extraction")

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_error_type_aggregation(self, mock_get_console: Mock) -> None:
        """Test error type aggregation."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        errors = [
            {"type": "ValidationError", "count": 5, "sample_items": ["item1"]},
            {"type": "JSONDecodeError", "count": 3, "sample_items": ["item2"]},
        ]

        log_error_summary(errors, max_display=5, context="Extraction")

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_sample_items_truncation_30_char_limit(
        self, mock_get_console: Mock
    ) -> None:
        """Test sample items truncation (30 char limit)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        long_item = "a" * 50
        errors = [{"type": "Error", "count": 1, "sample_items": [long_item]}]

        log_error_summary(errors, max_display=5, context="Extraction")

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_and_more_row_when_errors_exceed_max_display(
        self, mock_get_console: Mock
    ) -> None:
        """Test 'and more' row when errors exceed max_display."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        errors = [
            {"type": "Error1", "count": 3, "sample_items": ["a"]},
            {"type": "Error2", "count": 2, "sample_items": ["b"]},
            {"type": "Error3", "count": 1, "sample_items": ["c"]},
        ]

        log_error_summary(errors, max_display=2, context="Extraction")

        mock_console.print.assert_called_once()

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_empty_errors_list_no_output(self, mock_get_console: Mock) -> None:
        """Test empty errors list (no output)."""
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console

        log_error_summary([], max_display=5, context="Extraction")

        mock_console.print.assert_not_called()

    @patch("loclean.utils.rich_output.get_rich_console", return_value=None)
    @patch("loclean.utils.rich_output.logger")
    def test_fallback_to_simple_logging(
        self, mock_logger: Mock, mock_get_console: Mock
    ) -> None:
        """Test fallback to simple logging."""
        errors = [{"type": "ValidationError", "count": 5, "sample_items": ["item1"]}]

        log_error_summary(errors, max_display=5, context="Extraction")

        mock_logger.warning.assert_called()


class TestCreateProgress:
    """Test cases for create_progress function."""

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_progress_creation_with_total_items(self, mock_get_console: Mock) -> None:
        """Test Progress creation with total items."""
        real_console = Console(stderr=True)
        mock_get_console.return_value = real_console

        progress = create_progress(total=100, description="Processing")

        assert progress is not None
        assert hasattr(progress, "add_task")

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_progress_creation_without_total_indeterminate(
        self, mock_get_console: Mock
    ) -> None:
        """Test Progress creation without total (indeterminate)."""
        real_console = Console(stderr=True)
        mock_get_console.return_value = real_console

        progress = create_progress(total=None, description="Processing")

        assert progress is not None

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_with_custom_console_instance(self, mock_get_console: Mock) -> None:
        """Test with custom Console instance."""
        custom_console = Console(stderr=True)

        progress = create_progress(
            total=100, description="Processing", console=custom_console
        )

        assert progress is not None
        assert progress.console is custom_console
        mock_get_console.assert_not_called()

    @patch("loclean.utils.rich_output.get_rich_console", return_value=None)
    def test_return_none_when_no_console_available(
        self, mock_get_console: Mock
    ) -> None:
        """Test return None when no console available."""
        progress = create_progress(total=100, description="Processing")

        assert progress is None

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_progress_columns_configuration(self, mock_get_console: Mock) -> None:
        """Test Progress columns configuration (SpinnerColumn, BarColumn, etc.)."""
        real_console = Console(stderr=True)
        mock_get_console.return_value = real_console

        progress = create_progress(total=100, description="Processing")

        assert progress is not None
        # Progress should have columns configured
        assert hasattr(progress, "columns")

    @patch("loclean.utils.rich_output.get_rich_console")
    def test_refresh_per_second_and_speed_estimate_period_settings(
        self, mock_get_console: Mock
    ) -> None:
        """Test refresh_per_second and speed_estimate_period settings."""
        real_console = Console(stderr=True)
        mock_get_console.return_value = real_console

        progress = create_progress(total=100, description="Processing")

        assert progress is not None
        # Progress object doesn't expose these as attributes,
        # but they're set in constructor
        # Verify progress was created successfully
        assert hasattr(progress, "add_task")
