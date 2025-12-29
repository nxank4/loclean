"""Llama.cpp inference engine implementation.

This module provides the LlamaCppEngine class for local inference using
Llama.cpp with GBNF grammar constraints and prompt adapters.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar  # type: ignore[attr-defined]

from loclean.inference.adapters import PromptAdapter, get_adapter
from loclean.inference.base import InferenceEngine
from loclean.inference.local.exceptions import (
    CachePermissionError,
    InsufficientSpaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NetworkError,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "phi-3-mini": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "qwen3-4b": {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "filename": "Qwen3-4B-Instruct-2507-GGUF.q4_k_m.gguf",
    },
    "gemma-3-4b": {
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-GGUF.q4_k_m.gguf",
    },
    "deepseek-r1": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-GGUF.q4_k_m.gguf",
    },
}


class LlamaCppEngine(InferenceEngine):
    """
    Local inference engine using Llama.cpp with GBNF grammar constraints.

    Supports multiple GGUF models (Phi-3, Qwen, Gemma, DeepSeek) with automatic
    prompt adapter selection based on model name.
    """

    def __init__(
        self,
        model_name: str = "phi-3-mini",
        cache_dir: Optional[Path] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ):
        """
        Initialize the LlamaCppEngine.

        Args:
            model_name: Name of the model to use (e.g., "phi-3-mini", "qwen3-4b").
                       Must be in MODEL_REGISTRY. Defaults to "phi-3-mini".
            cache_dir: Optional custom directory for caching models.
                       Defaults to ~/.cache/loclean.
            n_ctx: Context window size. Defaults to 4096.
            n_gpu_layers: Number of GPU layers to use (0 = CPU only). Defaults to 0.
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "loclean"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        if model_name not in _MODEL_REGISTRY:
            logger.warning(
                f"Model '{model_name}' not in registry. Falling back to 'phi-3-mini'."
            )
            model_name = "phi-3-mini"
            self.model_name = model_name

        model_info = _MODEL_REGISTRY[model_name]
        self.model_repo = model_info["repo"]
        self.model_filename = model_info["filename"]

        self.adapter: PromptAdapter = get_adapter(model_name)
        logger.info(
            f"Using adapter: {type(self.adapter).__name__} for model: {model_name}"
        )

        self.model_path = self._get_model_path()

        logger.info(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.grammar = self._get_json_grammar()

        from loclean.cache import LocleanCache

        self.cache = LocleanCache(cache_dir=self.cache_dir)

        logger.info(f"LlamaCppEngine initialized successfully with model: {model_name}")

    def _get_model_path(self) -> Path:
        """
        Ensure the model exists locally. Download if necessary.

        Returns:
            Path to the model file.

        Raises:
            ModelNotFoundError: If the model repository or file doesn't exist.
            NetworkError: If network connectivity issues prevent download.
            CachePermissionError: If insufficient permissions prevent writing to cache.
            InsufficientSpaceError: If insufficient disk space prevents download.
            ModelDownloadError: For other download-related errors.
        """
        local_path = self.cache_dir / self.model_filename
        if local_path.exists():
            logger.info(f"Model found at {local_path}")
            return local_path

        logger.info(
            f"Model not found locally. Downloading {self.model_filename} "
            f"from {self.model_repo}..."
        )

        # Check cache directory permissions before attempting download
        # Only check if directory already exists (mkdir succeeded, so we have write access)
        # If directory was just created, mkdir() already verified we have write access
        if self.cache_dir.exists():
            try:
                if not os.access(self.cache_dir, os.W_OK):
                    raise CachePermissionError(
                        f"Cannot write to cache directory: {self.cache_dir}. "
                        "Please check directory permissions or specify a different "
                        "cache_dir.",
                        model_name=self.model_name,
                        repo_id=self.model_repo,
                        filename=self.model_filename,
                    )
            except CachePermissionError:
                # Re-raise our custom CachePermissionError
                raise
            except Exception as e:
                # If permission check fails for other reasons, log warning but continue
                # The actual download will fail with a clearer error if permissions are wrong
                logger.warning(
                    f"Could not verify cache directory permissions: {e}. "
                    "Proceeding with download..."
                )

        # Check available disk space (rough estimate - model files are typically 2-8GB)
        try:
            stat = shutil.disk_usage(self.cache_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 1.0:  # Require at least 1GB free space
                raise InsufficientSpaceError(
                    f"Insufficient disk space. Available: {free_gb:.2f}GB, "
                    f"required: ~1GB minimum. Please free up space in "
                    f"{self.cache_dir}.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                )
        except InsufficientSpaceError:
            raise
        except Exception as e:
            logger.warning(
                f"Could not check disk space: {e}. Proceeding with download..."
            )

        # Attempt to download the model
        try:
            path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
                local_dir=self.cache_dir,
            )
            logger.info(f"Successfully downloaded model to {path}")
            return Path(path)
        except FileNotFoundError as e:
            raise ModelNotFoundError(
                f"Model file '{self.model_filename}' not found in repository "
                f"'{self.model_repo}'. Please verify the model name is correct.",
                model_name=self.model_name,
                repo_id=self.model_repo,
                filename=self.model_filename,
            ) from e
        except (ConnectionError, TimeoutError, OSError) as e:
            # Check if it's a network-related OSError
            # errno 28=ENOSPC (no space), 13=EACCES (permission denied)
            if isinstance(e, OSError) and e.errno not in (28, 13):
                # Not a disk space or permission error, treat as network error
                raise NetworkError(
                    f"Network error while downloading model: {e}. "
                    f"Please check your internet connection and try again.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            elif isinstance(e, OSError) and e.errno == 13:  # EACCES - Permission denied
                raise CachePermissionError(
                    f"Permission denied while downloading model: {e}. "
                    f"Please check write permissions for {self.cache_dir}.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            elif isinstance(e, OSError) and e.errno == 28:  # ENOSPC - No space left
                raise InsufficientSpaceError(
                    f"Insufficient disk space while downloading model: {e}. "
                    f"Please free up space in {self.cache_dir}.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            else:
                raise NetworkError(
                    f"Network error while downloading model: {e}. "
                    f"Please check your internet connection and try again.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
        except Exception as e:
            # Catch Hugging Face Hub specific exceptions
            error_msg = str(e).lower()
            if "not found" in error_msg or "repository" in error_msg:
                raise ModelNotFoundError(
                    f"Model repository or file not found: {e}. "
                    f"Repository: {self.model_repo}, File: {self.model_filename}",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            elif (
                "network" in error_msg
                or "connection" in error_msg
                or "timeout" in error_msg
            ):
                raise NetworkError(
                    f"Network error while downloading model: {e}. "
                    f"Please check your internet connection and try again.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            elif "permission" in error_msg or "access" in error_msg:
                raise CachePermissionError(
                    f"Permission error while downloading model: {e}. "
                    f"Please check write permissions for {self.cache_dir}.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            elif "space" in error_msg or "disk" in error_msg or "full" in error_msg:
                raise InsufficientSpaceError(
                    f"Insufficient disk space while downloading model: {e}. "
                    f"Please free up space in {self.cache_dir}.",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e
            else:
                raise ModelDownloadError(
                    f"Failed to download model '{self.model_name}': {e}. "
                    f"Repository: {self.model_repo}, File: {self.model_filename}",
                    model_name=self.model_name,
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                ) from e

    def _get_json_grammar(self) -> LlamaGrammar:
        """
        Returns a GBNF grammar enforcing
        {"reasoning": <string>, "value": <number>, "unit": <string>}.

        Loads grammar from resources/grammars/json.gbnf using importlib.resources
        for zip-safe compatibility.

        Returns:
            LlamaGrammar instance for JSON extraction.
        """
        from loclean.utils.resources import load_grammar

        grammar_str = load_grammar("json.gbnf")
        return LlamaGrammar.from_string(grammar_str)

    def clean_batch(
        self,
        items: List[str],
        instruction: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process a batch of strings and extract structured data using GBNF.

        Result is cached to avoid re-computation.

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the task.

        Returns:
            Dictionary mapping original_string ->
            {"reasoning": str, "value": float, "unit": str} or None.
        """
        cached_results = self.cache.get_batch(items, instruction)
        misses = [item for item in items if item not in cached_results]

        if not misses:
            return cached_results

        logger.info(f"Cache miss for {len(misses)} items. Running inference...")
        new_results: Dict[str, Optional[Dict[str, Any]]] = {}

        stop_tokens = self.adapter.get_stop_tokens()

        for item in misses:
            prompt = self.adapter.format(instruction, item)

            output: Any = None
            text: Optional[str] = None
            try:
                output = self.llm.create_completion(
                    prompt=prompt,
                    grammar=self.grammar,
                    max_tokens=256,
                    stop=stop_tokens,
                    echo=False,
                )

                # llama-cpp-python can return either dict or iterator depending on
                # version/config. Handle both cases to maintain compatibility.
                if isinstance(output, dict) and "choices" in output:
                    text = str(output["choices"][0]["text"]).strip()
                else:
                    if hasattr(output, "__iter__") and not isinstance(
                        output, (str, bytes)
                    ):
                        first_item = next(iter(output), None)
                        if isinstance(first_item, dict) and "choices" in first_item:
                            text = str(first_item["choices"][0]["text"]).strip()

                if text is None:
                    logger.warning(f"No text extracted for item '{item}'")
                    new_results[item] = None
                    continue

                data = json.loads(text)

                if "value" in data and "unit" in data and "reasoning" in data:
                    new_results[item] = data
                else:
                    logger.warning(f"Result for '{item}' missing keys. Raw: {text}")
                    new_results[item] = None

            except json.JSONDecodeError as e:
                raw_text = text if "text" in locals() else "N/A"
                logger.warning(
                    f"Failed to decode JSON for item '{item}': {e}. "
                    f"Raw text: '{raw_text}'"
                )
                new_results[item] = None
            except Exception as e:
                logger.error(f"Inference error for item '{item}': {e}")
                new_results[item] = None

        # Only cache valid results to avoid polluting cache with None values.
        # Invalid results will be recomputed on next request, allowing for retry logic.
        valid_new_results = {k: v for k, v in new_results.items() if v is not None}
        if valid_new_results:
            self.cache.set_batch(
                list(valid_new_results.keys()), instruction, valid_new_results
            )

        return {**cached_results, **new_results}
