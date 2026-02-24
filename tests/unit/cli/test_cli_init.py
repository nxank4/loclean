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

    def test_model_status_command_routing(self, runner: CliRunner) -> None:
        """Test that model status command is accessible through main app."""
        with patch("loclean.cli.model.check_connection") as mock_check:
            mock_check.return_value = None
            result = runner.invoke(app, ["model", "status"])

            assert result.exit_code == 0
            mock_check.assert_called_once()
