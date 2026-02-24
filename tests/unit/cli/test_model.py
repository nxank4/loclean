"""Test cases for CLI model subcommand."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from loclean.cli.model import app


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestStatusCommand:
    """Test cases for status command."""

    @patch("loclean.cli.model.check_connection")
    def test_status_calls_check_connection(
        self, mock_check: Mock, runner: CliRunner
    ) -> None:
        mock_check.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        mock_check.assert_called_once()

    @patch("loclean.cli.model.check_connection")
    def test_status_with_custom_host(self, mock_check: Mock, runner: CliRunner) -> None:
        mock_check.return_value = None

        result = runner.invoke(app, ["status", "--host", "http://custom:8080"])

        assert result.exit_code == 0
        mock_check.assert_called_once()
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs["host"] == "http://custom:8080"

    @patch("loclean.cli.model.check_connection")
    def test_status_default_host(self, mock_check: Mock, runner: CliRunner) -> None:
        mock_check.return_value = None

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs["host"] == "http://localhost:11434"


class TestPullCommand:
    """Test cases for pull command."""

    @patch("loclean.cli.model.pull_model")
    def test_pull_calls_pull_model(self, mock_pull: Mock, runner: CliRunner) -> None:
        mock_pull.return_value = None

        result = runner.invoke(app, ["pull", "phi3"])

        assert result.exit_code == 0
        mock_pull.assert_called_once()
        call_kwargs = mock_pull.call_args[1]
        assert call_kwargs["model_name"] == "phi3"

    @patch("loclean.cli.model.pull_model")
    def test_pull_with_custom_host(self, mock_pull: Mock, runner: CliRunner) -> None:
        mock_pull.return_value = None

        result = runner.invoke(app, ["pull", "llama3", "--host", "http://gpu:8080"])

        assert result.exit_code == 0
        call_kwargs = mock_pull.call_args[1]
        assert call_kwargs["model_name"] == "llama3"
        assert call_kwargs["host"] == "http://gpu:8080"

    @patch("loclean.cli.model.pull_model")
    def test_pull_default_host(self, mock_pull: Mock, runner: CliRunner) -> None:
        mock_pull.return_value = None

        result = runner.invoke(app, ["pull", "phi3"])

        assert result.exit_code == 0
        call_kwargs = mock_pull.call_args[1]
        assert call_kwargs["host"] == "http://localhost:11434"

    def test_pull_missing_name_exits(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["pull"])
        assert result.exit_code != 0
