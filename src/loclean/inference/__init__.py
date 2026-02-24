"""Inference engine package for loclean.

Provides the InferenceEngine abstract base class and the OllamaEngine
implementation for local inference via a running Ollama instance.
"""

from loclean.inference.base import InferenceEngine
from loclean.inference.ollama_engine import OllamaEngine

__all__ = ["InferenceEngine", "OllamaEngine"]
