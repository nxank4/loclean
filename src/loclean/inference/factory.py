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
    """Create an inference engine instance based on configuration.

    Args:
        config: EngineConfig instance with engine selection and parameters.

    Returns:
        InferenceEngine instance (OllamaEngine, etc.)

    Raises:
        ValueError: If engine type is not supported.
        ConnectionError: If Ollama is not reachable.
    """
    engine_type = config.engine

    if engine_type == "ollama":
        from loclean.inference.ollama_engine import OllamaEngine

        logger.info(f"Creating OllamaEngine with model: {config.model}")
        return OllamaEngine(
            model=config.model,
            host=config.host,
            verbose=config.verbose,
        )

    elif engine_type == "openai":
        raise NotImplementedError(
            "OpenAI engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'ollama' engine."
        )

    elif engine_type == "anthropic":
        raise NotImplementedError(
            "Anthropic engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'ollama' engine."
        )

    elif engine_type == "gemini":
        raise NotImplementedError(
            "Gemini engine is not yet implemented. "
            "It will be available in a future release. "
            "For now, please use 'ollama' engine."
        )

    else:
        raise ValueError(
            f"Unsupported engine type: {engine_type}. "
            "Supported engines: ollama, openai, anthropic, gemini"
        )
