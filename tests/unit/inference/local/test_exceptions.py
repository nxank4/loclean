"""Test cases for custom exception classes."""

import pytest

from loclean.inference.local.exceptions import (
    CachePermissionError,
    InsufficientSpaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NetworkError,
)


class TestModelDownloadError:
    """Test cases for ModelDownloadError class."""

    def test_exception_initialization_with_all_parameters(self) -> None:
        """Test exception initialization with all parameters."""
        error = ModelDownloadError(
            message="Download failed",
            model_name="phi-3-mini",
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
        )

        assert str(error) == "Download failed"
        assert error.model_name == "phi-3-mini"
        assert error.repo_id == "microsoft/Phi-3-mini-4k-instruct-gguf"
        assert error.filename == "Phi-3-mini-4k-instruct-q4.gguf"

    def test_exception_message_formatting(self) -> None:
        """Test exception message formatting."""
        error = ModelDownloadError(
            message="Custom error message",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert str(error) == "Custom error message"

    def test_attribute_access_model_name(self) -> None:
        """Test attribute access (model_name)."""
        error = ModelDownloadError(
            message="Error",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert error.model_name == "phi-3-mini"

    def test_attribute_access_repo_id(self) -> None:
        """Test attribute access (repo_id)."""
        error = ModelDownloadError(
            message="Error",
            model_name="test-model",
            repo_id="microsoft/Phi-3-mini",
            filename="test.gguf",
        )

        assert error.repo_id == "microsoft/Phi-3-mini"

    def test_attribute_access_filename(self) -> None:
        """Test attribute access (filename)."""
        error = ModelDownloadError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="model.gguf",
        )

        assert error.filename == "model.gguf"

    def test_exception_inheritance_from_exception(self) -> None:
        """Test exception inheritance from Exception."""
        error = ModelDownloadError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert isinstance(error, Exception)


class TestModelNotFoundError:
    """Test cases for ModelNotFoundError class."""

    def test_exception_initialization(self) -> None:
        """Test exception initialization."""
        error = ModelNotFoundError(
            message="Model not found",
            model_name="invalid-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert str(error) == "Model not found"
        assert error.model_name == "invalid-model"

    def test_exception_inheritance_from_model_download_error(self) -> None:
        """Test exception inheritance from ModelDownloadError."""
        error = ModelNotFoundError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert isinstance(error, ModelDownloadError)
        assert isinstance(error, Exception)

    def test_usage_in_download_scenarios(self) -> None:
        """Test usage in download scenarios."""
        error = ModelNotFoundError(
            message="Repository not found",
            model_name="phi-3-mini",
            repo_id="nonexistent/repo",
            filename="model.gguf",
        )

        # Should be raiseable
        with pytest.raises(ModelNotFoundError) as exc_info:
            raise error

        assert exc_info.value.model_name == "phi-3-mini"
        assert exc_info.value.repo_id == "nonexistent/repo"


class TestNetworkError:
    """Test cases for NetworkError class."""

    def test_exception_initialization(self) -> None:
        """Test exception initialization."""
        error = NetworkError(
            message="Network connection failed",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert str(error) == "Network connection failed"
        assert error.model_name == "phi-3-mini"

    def test_exception_inheritance_from_model_download_error(self) -> None:
        """Test exception inheritance from ModelDownloadError."""
        error = NetworkError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert isinstance(error, ModelDownloadError)
        assert isinstance(error, Exception)

    def test_usage_in_network_failure_scenarios(self) -> None:
        """Test usage in network failure scenarios."""
        error = NetworkError(
            message="Connection timeout",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="model.gguf",
        )

        with pytest.raises(NetworkError) as exc_info:
            raise error

        assert exc_info.value.model_name == "phi-3-mini"
        assert "timeout" in str(exc_info.value).lower()


class TestCachePermissionError:
    """Test cases for CachePermissionError class."""

    def test_exception_initialization(self) -> None:
        """Test exception initialization."""
        error = CachePermissionError(
            message="Permission denied",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert str(error) == "Permission denied"
        assert error.model_name == "phi-3-mini"

    def test_exception_inheritance_from_model_download_error(self) -> None:
        """Test exception inheritance from ModelDownloadError."""
        error = CachePermissionError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert isinstance(error, ModelDownloadError)
        assert isinstance(error, Exception)

    def test_usage_in_permission_error_scenarios(self) -> None:
        """Test usage in permission error scenarios."""
        error = CachePermissionError(
            message="Cannot write to cache directory",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="model.gguf",
        )

        with pytest.raises(CachePermissionError) as exc_info:
            raise error

        assert exc_info.value.model_name == "phi-3-mini"
        assert (
            "permission" in str(exc_info.value).lower()
            or "write" in str(exc_info.value).lower()
        )


class TestInsufficientSpaceError:
    """Test cases for InsufficientSpaceError class."""

    def test_exception_initialization(self) -> None:
        """Test exception initialization."""
        error = InsufficientSpaceError(
            message="Insufficient disk space",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert str(error) == "Insufficient disk space"
        assert error.model_name == "phi-3-mini"

    def test_exception_inheritance_from_model_download_error(self) -> None:
        """Test exception inheritance from ModelDownloadError."""
        error = InsufficientSpaceError(
            message="Error",
            model_name="test-model",
            repo_id="test/repo",
            filename="test.gguf",
        )

        assert isinstance(error, ModelDownloadError)
        assert isinstance(error, Exception)

    def test_usage_in_disk_space_error_scenarios(self) -> None:
        """Test usage in disk space error scenarios."""
        error = InsufficientSpaceError(
            message="Not enough space on disk",
            model_name="phi-3-mini",
            repo_id="test/repo",
            filename="model.gguf",
        )

        with pytest.raises(InsufficientSpaceError) as exc_info:
            raise error

        assert exc_info.value.model_name == "phi-3-mini"
        assert "space" in str(exc_info.value).lower()
