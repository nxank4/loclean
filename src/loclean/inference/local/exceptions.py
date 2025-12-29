"""Custom exceptions for local inference engine errors."""


class ModelDownloadError(Exception):
    """
    Base exception for model download failures.

    Raised when downloading a model from Hugging Face Hub fails.
    """

    def __init__(
        self, message: str, model_name: str, repo_id: str, filename: str
    ) -> None:
        """
        Initialize ModelDownloadError.

        Args:
            message: Human-readable error message.
            model_name: Name of the model that failed to download.
            repo_id: Hugging Face repository ID.
            filename: Name of the model file.
        """
        self.model_name = model_name
        self.repo_id = repo_id
        self.filename = filename
        super().__init__(message)


class ModelNotFoundError(ModelDownloadError):
    """
    Raised when the model repository or file cannot be found on Hugging Face Hub.
    """

    pass


class NetworkError(ModelDownloadError):
    """
    Raised when network connectivity issues prevent model download.
    """

    pass


class CachePermissionError(ModelDownloadError):
    """
    Raised when insufficient permissions prevent writing to cache directory.
    """

    pass


class InsufficientSpaceError(ModelDownloadError):
    """
    Raised when insufficient disk space prevents model download.
    """

    pass
