"""Factory for creating inference engines.

This module provides a factory function to instantiate the correct InferenceEngine
based on EngineConfig, with lazy loading of heavy dependencies.
"""

import logging
from typing import TYPE_CHECKING

from loclean.inference.config import EngineConfig
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


def create_engine(config: EngineConfig) -> "InferenceEngine":
    """
    Create an inference engine instance based on configuration.

    Uses lazy loading to import heavy dependencies only when needed,
    ensuring fast startup for users who don't need specific backends.

    Args:
        config: EngineConfig instance with engine selection and parameters

    Returns:
        InferenceEngine instance (LlamaCppEngine, OpenAIEngine, etc.)

    Raises:
        ValueError: If engine type is not supported
        ImportError: If required dependencies are not installed

    Example:
        >>> from loclean.inference.config import load_config
        >>> config = load_config(engine="llama-cpp", model="phi-3-mini")
        >>> engine = create_engine(config)
        >>> isinstance(engine, LlamaCppEngine)
        True
    """
    engine_type = config.engine

    if engine_type == "llama-cpp":
        # Lazy import to avoid loading llama-cpp-python unless needed
        try:
            from loclean.inference.local.llama_cpp import LlamaCppEngine
        except ImportError as e:
            raise ImportError(
                f"Failed to import LlamaCppEngine. "
                f"Please ensure 'llama-cpp-python' is installed: {e}"
            ) from e

        logger.info(f"Creating LlamaCppEngine with model: {config.model}")
        try:
            return LlamaCppEngine(
                model_name=config.model,
                cache_dir=config.cache_dir,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
            )
        except Exception as e:
            logger.error(f"Failed to create LlamaCppEngine: {e}")
            raise

    elif engine_type == "openai":
        # Placeholder for future OpenAI implementation
        raise NotImplementedError(
            "OpenAI engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'llama-cpp' engine."
        )

    elif engine_type == "anthropic":
        # Placeholder for future Anthropic implementation
        raise NotImplementedError(
            "Anthropic engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'llama-cpp' engine."
        )

    elif engine_type == "gemini":
        # Placeholder for future Gemini implementation
        raise NotImplementedError(
            "Gemini engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'llama-cpp' engine."
        )

    else:
        raise ValueError(
            f"Unsupported engine type: {engine_type}. "
            "Supported engines: llama-cpp, openai, anthropic, gemini"
        )
