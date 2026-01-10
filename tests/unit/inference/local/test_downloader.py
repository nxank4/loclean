"""Test cases for model downloader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from loclean.inference.local.downloader import download_model
from loclean.inference.local.exceptions import (
    CachePermissionError,
    InsufficientSpaceError,
    ModelNotFoundError,
    NetworkError,
)


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_model_path(temp_cache_dir: Path) -> Path:
    """Create a mock model file."""
    model_path = temp_cache_dir / "test_model.gguf"
    model_path.write_bytes(b"fake model data")
    return model_path


def test_download_model_existing_file(
    temp_cache_dir: Path, mock_model_path: Path
) -> None:
    """Test that download_model returns existing file path."""
    path = download_model(
        model_name="test-model",
        repo_id="test/repo",
        filename="test_model.gguf",
        cache_dir=temp_cache_dir,
        force=False,
        show_progress=False,
    )
    assert path == mock_model_path
    assert path.exists()


def test_download_model_force_redownload(
    temp_cache_dir: Path, mock_model_path: Path
) -> None:
    """Test that force flag triggers re-download."""
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        mock_download.return_value = str(mock_model_path)
        path = download_model(
            model_name="test-model",
            repo_id="test/repo",
            filename="test_model.gguf",
            cache_dir=temp_cache_dir,
            force=True,
            show_progress=False,
        )
        mock_download.assert_called_once()
        assert path == mock_model_path


def test_download_model_success(temp_cache_dir: Path) -> None:
    """Test successful model download."""
    model_path = temp_cache_dir / "new_model.gguf"
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        mock_download.return_value = str(model_path)
        path = download_model(
            model_name="test-model",
            repo_id="test/repo",
            filename="new_model.gguf",
            cache_dir=temp_cache_dir,
            force=False,
            show_progress=False,
        )
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="new_model.gguf",
            local_dir=temp_cache_dir,
        )
        assert path == model_path


def test_download_model_not_found(temp_cache_dir: Path) -> None:
    """Test ModelNotFoundError when model file doesn't exist."""
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        mock_download.side_effect = FileNotFoundError("Model not found")
        with pytest.raises(ModelNotFoundError) as exc_info:
            download_model(
                model_name="test-model",
                repo_id="test/repo",
                filename="missing.gguf",
                cache_dir=temp_cache_dir,
                force=False,
                show_progress=False,
            )
        assert "not found" in str(exc_info.value).lower()


def test_download_model_network_error(temp_cache_dir: Path) -> None:
    """Test NetworkError when download fails due to network issues."""
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        mock_download.side_effect = ConnectionError("Network error")
        with pytest.raises(NetworkError) as exc_info:
            download_model(
                model_name="test-model",
                repo_id="test/repo",
                filename="test.gguf",
                cache_dir=temp_cache_dir,
                force=False,
                show_progress=False,
            )
        assert "network" in str(exc_info.value).lower()


def test_download_model_permission_error(temp_cache_dir: Path) -> None:
    """Test CachePermissionError when cache directory is not writable."""
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        error = OSError("Permission denied")
        error.errno = 13  # EACCES
        mock_download.side_effect = error
        with pytest.raises(CachePermissionError) as exc_info:
            download_model(
                model_name="test-model",
                repo_id="test/repo",
                filename="test.gguf",
                cache_dir=temp_cache_dir,
                force=False,
                show_progress=False,
            )
        assert "permission" in str(exc_info.value).lower()


def test_download_model_insufficient_space(temp_cache_dir: Path) -> None:
    """Test InsufficientSpaceError when disk space is low."""
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        error = OSError("No space left")
        error.errno = 28  # ENOSPC
        mock_download.side_effect = error
        with pytest.raises(InsufficientSpaceError) as exc_info:
            download_model(
                model_name="test-model",
                repo_id="test/repo",
                filename="test.gguf",
                cache_dir=temp_cache_dir,
                force=False,
                show_progress=False,
            )
        assert "space" in str(exc_info.value).lower()


def test_download_model_default_cache_dir() -> None:
    """Test that default cache directory is used when not specified."""
    default_cache = Path.home() / ".cache" / "loclean"
    with patch("loclean.inference.local.downloader.hf_hub_download") as mock_download:
        mock_path = default_cache / "test.gguf"
        mock_download.return_value = str(mock_path)
        path = download_model(
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
            cache_dir=None,
            force=False,
            show_progress=False,
        )
        assert mock_download.called
        assert path == mock_path
