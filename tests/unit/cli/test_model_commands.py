"""Test cases for CLI model commands."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.exceptions import Exit
from rich.console import Console

from loclean.cli.model_commands import check_status, download_model, list_models


@pytest.fixture
def mock_console() -> Console:
    """Create a mock console for testing."""
    import io

    return Console(file=io.StringIO(), force_terminal=False)


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@patch("loclean.cli.model_commands.get_model_registry")
def test_list_models(mock_registry: Mock, mock_console: Console) -> None:
    """Test list_models command."""
    mock_registry.return_value = {
        "phi-3-mini": {
            "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
            "size_mb": 2400,
            "description": "Microsoft Phi-3 Mini",
        },
        "tinyllama": {
            "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_mb": 800,
            "description": "TinyLlama 1.1B",
        },
    }
    list_models(console=mock_console)
    mock_registry.assert_called_once()


@patch("loclean.cli.model_commands.get_model_registry")
@patch("loclean.cli.model_commands.Path.home")
def test_check_status(
    mock_home: Mock,
    mock_registry: Mock,
    mock_console: Console,
    temp_cache_dir: Path,
) -> None:
    """Test check_status command."""
    mock_home.return_value = temp_cache_dir.parent
    cache_dir = temp_cache_dir / ".cache" / "loclean"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a mock downloaded model
    model_file = cache_dir / "Phi-3-mini-4k-instruct-q4.gguf"
    model_file.write_bytes(b"fake model")

    mock_registry.return_value = {
        "phi-3-mini": {
            "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
            "size_mb": 2400,
            "description": "Microsoft Phi-3 Mini",
        },
        "tinyllama": {
            "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_mb": 800,
            "description": "TinyLlama 1.1B",
        },
    }

    check_status(console=mock_console)
    mock_registry.assert_called_once()


@patch("loclean.cli.model_commands.download_model_func")
@patch("loclean.cli.model_commands.get_model_registry")
def test_download_model_success(
    mock_registry: Mock,
    mock_download: Mock,
    mock_console: Console,
    temp_cache_dir: Path,
) -> None:
    """Test successful model download."""
    mock_registry.return_value = {
        "phi-3-mini": {
            "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
            "size_mb": 2400,
            "description": "Microsoft Phi-3 Mini",
        },
    }
    mock_download.return_value = temp_cache_dir / "Phi-3-mini-4k-instruct-q4.gguf"

    download_model(
        name="phi-3-mini",
        cache_dir=None,
        force=False,
        console=mock_console,
    )

    mock_registry.assert_called_once()
    mock_download.assert_called_once_with(
        model_name="phi-3-mini",
        repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        filename="Phi-3-mini-4k-instruct-q4.gguf",
        cache_dir=None,
        force=False,
        show_progress=True,
    )


@patch("loclean.cli.model_commands.get_model_registry")
def test_download_model_not_found(mock_registry: Mock, mock_console: Console) -> None:
    """Test download_model with invalid model name."""
    mock_registry.return_value = {
        "phi-3-mini": {
            "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        },
    }

    with pytest.raises(Exit):
        download_model(
            name="invalid-model",
            cache_dir=None,
            force=False,
            console=mock_console,
        )


@patch("loclean.cli.model_commands.download_model_func")
@patch("loclean.cli.model_commands.get_model_registry")
def test_download_model_error(
    mock_registry: Mock,
    mock_download: Mock,
    mock_console: Console,
) -> None:
    """Test download_model with download error."""
    mock_registry.return_value = {
        "phi-3-mini": {
            "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        },
    }
    mock_download.side_effect = Exception("Download failed")

    with pytest.raises(Exit):
        download_model(
            name="phi-3-mini",
            cache_dir=None,
            force=False,
            console=mock_console,
        )
