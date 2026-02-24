"""Test cases for CLI model commands."""

import io
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import typer
from rich.console import Console

from loclean.cli.model_commands import check_connection, pull_model


@pytest.fixture
def mock_console() -> Console:
    """Create a mock console for testing."""
    return Console(file=io.StringIO(), force_terminal=False)


@pytest.fixture
def mock_ollama() -> Generator[MagicMock, None, None]:
    """Create and inject a mock ollama module."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"ollama": mock}):
        yield mock


class TestCheckConnection:
    """Test cases for check_connection command."""

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_successful_connection(
        self, mock_daemon: MagicMock, mock_ollama: MagicMock, mock_console: Console
    ) -> None:
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [
                {
                    "name": "phi3:latest",
                    "size": 2_400_000_000,
                    "modified_at": "2024-01-01T00:00:00Z",
                },
            ]
        }
        mock_ollama.Client.return_value = mock_client

        check_connection(console=mock_console)

        mock_daemon.assert_called_once()
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")
        mock_client.list.assert_called_once()

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_daemon_not_found_exits(
        self, mock_daemon: MagicMock, mock_console: Console
    ) -> None:
        mock_daemon.side_effect = FileNotFoundError("ollama not found")

        with pytest.raises(typer.Exit):
            check_connection(console=mock_console)

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_daemon_connection_error_exits(
        self, mock_daemon: MagicMock, mock_console: Console
    ) -> None:
        mock_daemon.side_effect = ConnectionError("timeout")

        with pytest.raises(typer.Exit):
            check_connection(console=mock_console)

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_custom_host(
        self, mock_daemon: MagicMock, mock_ollama: MagicMock, mock_console: Console
    ) -> None:
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_ollama.Client.return_value = mock_client

        check_connection(host="http://custom:8080", console=mock_console)

        mock_daemon.assert_called_once_with("http://custom:8080")
        mock_ollama.Client.assert_called_once_with(host="http://custom:8080")

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_no_models_available(
        self, mock_daemon: MagicMock, mock_ollama: MagicMock, mock_console: Console
    ) -> None:
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_ollama.Client.return_value = mock_client

        check_connection(console=mock_console)

        mock_client.list.assert_called_once()


class TestPullModel:
    """Test cases for pull_model command."""

    @patch("loclean.cli.model_commands.ensure_model")
    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_successful_pull(
        self,
        mock_daemon: MagicMock,
        mock_ensure: MagicMock,
        mock_ollama: MagicMock,
        mock_console: Console,
    ) -> None:
        mock_ollama.Client.return_value = MagicMock()

        pull_model("phi3", console=mock_console)

        mock_daemon.assert_called_once()
        mock_ensure.assert_called_once()

    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_pull_daemon_missing_exits(
        self, mock_daemon: MagicMock, mock_console: Console
    ) -> None:
        mock_daemon.side_effect = FileNotFoundError("not found")

        with pytest.raises(typer.Exit):
            pull_model("phi3", console=mock_console)

    @patch("loclean.cli.model_commands.ensure_model")
    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_pull_runtime_error_exits(
        self,
        mock_daemon: MagicMock,
        mock_ensure: MagicMock,
        mock_ollama: MagicMock,
        mock_console: Console,
    ) -> None:
        mock_ollama.Client.return_value = MagicMock()
        mock_ensure.side_effect = RuntimeError("pull failed")

        with pytest.raises(typer.Exit):
            pull_model("bad-model", console=mock_console)

    @patch("loclean.cli.model_commands.ensure_model")
    @patch("loclean.cli.model_commands.ensure_daemon")
    def test_pull_custom_host(
        self,
        mock_daemon: MagicMock,
        mock_ensure: MagicMock,
        mock_ollama: MagicMock,
        mock_console: Console,
    ) -> None:
        mock_ollama.Client.return_value = MagicMock()

        pull_model("phi3", host="http://custom:8080", console=mock_console)

        mock_daemon.assert_called_once_with("http://custom:8080")
