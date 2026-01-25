"""Test cases for main CLI app structure."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from loclean.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestAppInitialization:
    """Test cases for app initialization."""

    def test_app_initialization(self) -> None:
        """Test app initialization."""
        assert app is not None
        assert app.info.name == "loclean"
        assert app.info.help is not None
        assert "Local AI Data Cleaner" in app.info.help

    def test_model_subcommand_registration(self) -> None:
        """Test model subcommand registration."""
        # Check that model subcommand group is registered
        groups = [group.name for group in app.registered_groups]
        assert "model" in groups

    def test_help_output(self, runner: CliRunner) -> None:
        """Test help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Loclean" in result.stdout
        assert "model" in result.stdout

    def test_command_routing(self, runner: CliRunner) -> None:
        """Test command routing."""
        # Test that model subcommand is accessible
        result = runner.invoke(app, ["model", "--help"])

        assert result.exit_code == 0
        assert "Model management commands" in result.stdout

    def test_error_handling_for_unknown_commands(self, runner: CliRunner) -> None:
        """Test error handling for unknown commands."""
        result = runner.invoke(app, ["unknown-command"])

        assert result.exit_code != 0
        # Typer outputs errors, check that command failed
        assert result.exit_code == 2  # Typer exit code for unknown command

    def test_model_list_command_routing(self, runner: CliRunner) -> None:
        """Test that model list command is accessible through main app."""
        with patch("loclean.cli.model.list_models") as mock_list:
            mock_list.return_value = None
            result = runner.invoke(app, ["model", "list"])

            assert result.exit_code == 0
            mock_list.assert_called_once()

    def test_model_download_command_routing(self, runner: CliRunner) -> None:
        """Test that model download command is accessible through main app."""
        with patch("loclean.cli.model.download_model") as mock_download:
            from typer import Exit

            mock_download.side_effect = Exit(code=1)
            result = runner.invoke(app, ["model", "download", "--name", "test"])

            # Should exit with error code when model not found
            assert result.exit_code == 1
            mock_download.assert_called_once()

    def test_model_status_command_routing(self, runner: CliRunner) -> None:
        """Test that model status command is accessible through main app."""
        with patch("loclean.cli.model.check_status") as mock_status:
            mock_status.return_value = None
            result = runner.invoke(app, ["model", "status"])

            assert result.exit_code == 0
            mock_status.assert_called_once()
