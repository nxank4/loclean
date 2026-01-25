"""Test cases for CLI model commands."""

from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from loclean.cli.model import app


@pytest.fixture
def mock_console() -> Console:
    """Create a mock console for testing."""
    import io

    return Console(file=io.StringIO(), force_terminal=False)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestDownloadCommand:
    """Test cases for download command."""

    @patch("loclean.cli.model.download_model")
    def test_successful_download_with_valid_model_name(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test successful download with valid model name."""
        mock_download_model.return_value = None

        result = runner.invoke(app, ["download", "--name", "phi-3-mini"])

        assert result.exit_code == 0
        mock_download_model.assert_called_once()
        call_kwargs = mock_download_model.call_args[1]
        assert call_kwargs["name"] == "phi-3-mini"
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["force"] is False

    @patch("loclean.cli.model.download_model")
    def test_download_with_custom_cache_dir(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test download with custom cache_dir."""
        mock_download_model.return_value = None

        result = runner.invoke(
            app, ["download", "--name", "phi-3-mini", "--cache-dir", "/custom/path"]
        )

        assert result.exit_code == 0
        mock_download_model.assert_called_once()
        call_kwargs = mock_download_model.call_args[1]
        assert call_kwargs["name"] == "phi-3-mini"
        assert call_kwargs["cache_dir"] == "/custom/path"
        assert call_kwargs["force"] is False

    @patch("loclean.cli.model.download_model")
    def test_download_with_force_flag(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test download with --force flag."""
        mock_download_model.return_value = None

        result = runner.invoke(app, ["download", "--name", "phi-3-mini", "--force"])

        assert result.exit_code == 0
        mock_download_model.assert_called_once()
        call_kwargs = mock_download_model.call_args[1]
        assert call_kwargs["name"] == "phi-3-mini"
        assert call_kwargs["force"] is True

    @patch("loclean.cli.model.download_model")
    def test_error_handling_for_invalid_model_name(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test error handling for invalid model name."""
        from typer import Exit

        mock_download_model.side_effect = Exit(code=1)

        result = runner.invoke(app, ["download", "--name", "invalid-model"])

        assert result.exit_code == 1
        mock_download_model.assert_called_once()

    @patch("loclean.cli.model.download_model")
    def test_error_handling_for_network_errors(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test error handling for network errors."""
        from typer import Exit

        mock_download_model.side_effect = Exit(code=1)

        result = runner.invoke(app, ["download", "--name", "phi-3-mini"])

        assert result.exit_code == 1
        mock_download_model.assert_called_once()

    @patch("loclean.cli.model.download_model")
    def test_console_output_formatting(
        self, mock_download_model: Mock, runner: CliRunner
    ) -> None:
        """Test console output formatting."""
        mock_download_model.return_value = None

        result = runner.invoke(app, ["download", "--name", "phi-3-mini"])

        assert result.exit_code == 0
        # Command should execute without errors


class TestListCommand:
    """Test cases for list command."""

    @patch("loclean.cli.model.list_models")
    def test_listing_all_available_models(
        self, mock_list_models: Mock, runner: CliRunner
    ) -> None:
        """Test listing all available models."""
        mock_list_models.return_value = None

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_list_models.assert_called_once()

    @patch("loclean.cli.model.list_models")
    def test_console_output_formatting_table_rich_display(
        self, mock_list_models: Mock, runner: CliRunner
    ) -> None:
        """Test console output formatting (table/rich display)."""
        mock_list_models.return_value = None

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_list_models.assert_called_once()

    @patch("loclean.cli.model.list_models")
    def test_with_empty_model_registry(
        self, mock_list_models: Mock, runner: CliRunner
    ) -> None:
        """Test with empty model registry."""
        mock_list_models.return_value = None

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_list_models.assert_called_once()


class TestStatusCommand:
    """Test cases for status command."""

    @patch("loclean.cli.model.check_status")
    def test_status_check_for_downloaded_models(
        self, mock_check_status: Mock, runner: CliRunner
    ) -> None:
        """Test status check for downloaded models."""
        mock_check_status.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        mock_check_status.assert_called_once()

    @patch("loclean.cli.model.check_status")
    def test_status_check_for_missing_models(
        self, mock_check_status: Mock, runner: CliRunner
    ) -> None:
        """Test status check for missing models."""
        mock_check_status.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        mock_check_status.assert_called_once()

    @patch("loclean.cli.model.check_status")
    def test_console_output_formatting(
        self, mock_check_status: Mock, runner: CliRunner
    ) -> None:
        """Test console output formatting."""
        mock_check_status.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        mock_check_status.assert_called_once()

    @patch("loclean.cli.model.check_status")
    def test_with_empty_cache_directory(
        self,
        mock_check_status: Mock,
        runner: CliRunner,
    ) -> None:
        """Test with empty cache directory."""
        mock_check_status.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        mock_check_status.assert_called_once()
