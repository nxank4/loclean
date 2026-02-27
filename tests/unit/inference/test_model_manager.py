"""Tests for loclean.inference.model_manager module."""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from loclean.inference import model_manager
from loclean.inference.model_manager import ensure_model, model_exists


class TestModelExists:
    """Tests for model_exists."""

    def test_model_present_exact_match(self) -> None:
        client = MagicMock()
        client.list.return_value = {
            "models": [{"name": "phi3:latest"}, {"name": "llama3:latest"}]
        }
        assert model_exists(client, "phi3") is True

    def test_model_present_full_tag_match(self) -> None:
        client = MagicMock()
        client.list.return_value = {"models": [{"name": "phi3:latest"}]}
        assert model_exists(client, "phi3:latest") is True

    def test_model_absent(self) -> None:
        client = MagicMock()
        client.list.return_value = {"models": [{"name": "llama3:latest"}]}
        assert model_exists(client, "phi3") is False

    def test_empty_model_list(self) -> None:
        client = MagicMock()
        client.list.return_value = {"models": []}
        assert model_exists(client, "phi3") is False

    def test_list_raises_returns_false(self) -> None:
        client = MagicMock()
        client.list.side_effect = Exception("connection error")
        assert model_exists(client, "phi3") is False


def _make_test_console() -> Console:
    """Create a non-interactive console writing to a buffer."""
    return Console(file=io.StringIO(), force_terminal=False)


class TestEnsureModel:
    """Tests for ensure_model."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        model_manager._verified_models.clear()

    @patch("loclean.inference.model_manager.model_exists", return_value=True)
    def test_model_already_exists_skips_pull(self, _mock_exists: MagicMock) -> None:
        client = MagicMock()
        ensure_model(client, "phi3")
        client.pull.assert_not_called()

    @patch("loclean.inference.model_manager.model_exists", return_value=False)
    def test_model_not_found_triggers_pull(self, _mock_exists: MagicMock) -> None:
        client = MagicMock()
        client.pull.return_value = iter(
            [
                {"status": "pulling manifest"},
                {"status": "downloading", "total": 1000, "completed": 500},
                {"status": "downloading", "total": 1000, "completed": 1000},
                {"status": "success"},
            ]
        )

        ensure_model(client, "phi3", console=_make_test_console())

        client.pull.assert_called_once_with("phi3", stream=True)

    @patch("loclean.inference.model_manager.model_exists", return_value=False)
    def test_pull_error_raises_runtime_error(self, _mock_exists: MagicMock) -> None:
        client = MagicMock()
        client.pull.return_value = iter(
            [
                {"status": "pulling manifest"},
                {"error": "model not found in registry"},
            ]
        )

        with pytest.raises(RuntimeError, match="model not found"):
            ensure_model(client, "nonexistent-model", console=_make_test_console())

    @patch("loclean.inference.model_manager.model_exists", return_value=False)
    def test_pull_with_default_console(self, _mock_exists: MagicMock) -> None:
        client = MagicMock()
        client.pull.return_value = iter(
            [
                {"status": "success"},
            ]
        )

        ensure_model(client, "phi3")
        client.pull.assert_called_once()

    @patch("loclean.inference.model_manager.model_exists", return_value=False)
    def test_pull_progress_receives_total_and_completed(
        self, _mock_exists: MagicMock
    ) -> None:
        client = MagicMock()
        chunks = [
            {"status": "downloading", "total": 2000, "completed": 0},
            {"status": "downloading", "total": 2000, "completed": 1000},
            {"status": "downloading", "total": 2000, "completed": 2000},
            {"status": "verifying"},
            {"status": "success"},
        ]
        client.pull.return_value = iter(chunks)

        ensure_model(client, "phi3", console=_make_test_console())

        client.pull.assert_called_once_with("phi3", stream=True)
