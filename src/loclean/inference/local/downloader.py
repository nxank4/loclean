"""Model downloader for HuggingFace Hub models.

This module provides reusable download functionality for GGUF models,
used by both the CLI and the LlamaCppEngine.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from loclean.inference.local.exceptions import (
    CachePermissionError,
    InsufficientSpaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NetworkError,
)
from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


def download_model(
    model_name: str,
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_name: Model name from registry (for error messages).
        repo_id: HuggingFace repository ID.
        filename: Model filename in the repository.
        cache_dir: Optional cache directory (default: ~/.cache/loclean).
        force: Force re-download even if exists.
        show_progress: Show download progress bar.

    Returns:
        Path to downloaded model file.

    Raises:
        ValueError: If model_name not in registry.
        ModelNotFoundError: If the model repository or file doesn't exist.
        NetworkError: If network connectivity issues prevent download.
        CachePermissionError: If insufficient permissions prevent writing to cache.
        InsufficientSpaceError: If insufficient disk space prevents download.
        ModelDownloadError: For other download-related errors.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "loclean"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / filename
    if local_path.exists() and not force:
        logger.info(f"[green]✓[/green] Model found at [dim]{local_path}[/dim]")
        return local_path

    if force and local_path.exists():
        logger.info(
            f"[yellow]⚠[/yellow] Force flag set. "
            f"Re-downloading [cyan]{filename}[/cyan]..."
        )

    logger.info(
        f"[cyan]📥[/cyan] Model not found locally. Downloading [bold]{filename}[/bold] "
        f"from [dim]{repo_id}[/dim]..."
    )

    # Check cache directory permissions before attempting download
    if cache_dir.exists():
        try:
            if not os.access(cache_dir, os.W_OK):
                raise CachePermissionError(
                    f"Cannot write to cache directory: {cache_dir}. "
                    "Please check directory permissions or specify a different "
                    "cache_dir.",
                    model_name=model_name,
                    repo_id=repo_id,
                    filename=filename,
                )
        except CachePermissionError:
            raise
        except Exception as e:
            logger.warning(
                f"[yellow]⚠[/yellow] Could not verify cache directory permissions: "
                f"[dim]{e}[/dim]. Proceeding with download..."
            )

    # Check available disk space (rough estimate - model files are typically 2-8GB)
    try:
        stat = shutil.disk_usage(cache_dir)
        free_gb = stat.free / (1024**3)
        if free_gb < 1.0:  # Require at least 1GB free space
            raise InsufficientSpaceError(
                f"Insufficient disk space. Available: {free_gb:.2f}GB, "
                f"required: ~1GB minimum. Please free up space in {cache_dir}.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            )
    except InsufficientSpaceError:
        raise
    except Exception as e:
        logger.warning(
            f"[yellow]⚠[/yellow] Could not check disk space: [dim]{e}[/dim]. "
            f"Proceeding with download..."
        )

    # Attempt to download the model
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=cache_dir,
        )
        logger.info(
            f"[green]✓[/green] Successfully downloaded model to [dim]{path}[/dim]"
        )
        return Path(path)
    except FileNotFoundError as e:
        raise ModelNotFoundError(
            f"Model file '{filename}' not found in repository '{repo_id}'. "
            "Please verify the model name is correct.",
            model_name=model_name,
            repo_id=repo_id,
            filename=filename,
        ) from e
    except (ConnectionError, TimeoutError, OSError) as e:
        # Check if it's a network-related OSError
        # errno 28=ENOSPC (no space), 13=EACCES (permission denied)
        if isinstance(e, OSError) and e.errno not in (28, 13):
            # Not a disk space or permission error, treat as network error
            raise NetworkError(
                f"Network error while downloading model: {e}. "
                "Please check your internet connection and try again.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        elif isinstance(e, OSError) and e.errno == 13:  # EACCES - Permission denied
            raise CachePermissionError(
                f"Permission denied while downloading model: {e}. "
                f"Please check write permissions for {cache_dir}.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        elif isinstance(e, OSError) and e.errno == 28:  # ENOSPC - No space left
            raise InsufficientSpaceError(
                f"Insufficient disk space while downloading model: {e}. "
                f"Please free up space in {cache_dir}.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        else:
            raise NetworkError(
                f"Network error while downloading model: {e}. "
                "Please check your internet connection and try again.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
    except Exception as e:
        # Catch Hugging Face Hub specific exceptions
        error_msg = str(e).lower()
        if "not found" in error_msg or "repository" in error_msg:
            raise ModelNotFoundError(
                f"Model repository or file not found: {e}. "
                f"Repository: {repo_id}, File: {filename}",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        elif (
            "network" in error_msg
            or "connection" in error_msg
            or "timeout" in error_msg
        ):
            raise NetworkError(
                f"Network error while downloading model: {e}. "
                "Please check your internet connection and try again.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        elif "permission" in error_msg or "access" in error_msg:
            raise CachePermissionError(
                f"Permission error while downloading model: {e}. "
                f"Please check write permissions for {cache_dir}.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        elif "space" in error_msg or "disk" in error_msg or "full" in error_msg:
            raise InsufficientSpaceError(
                f"Insufficient disk space while downloading model: {e}. "
                f"Please free up space in {cache_dir}.",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
        else:
            raise ModelDownloadError(
                f"Failed to download model '{model_name}': {e}. "
                f"Repository: {repo_id}, File: {filename}",
                model_name=model_name,
                repo_id=repo_id,
                filename=filename,
            ) from e
