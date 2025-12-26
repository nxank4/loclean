"""Local inference engine implementations.

This package contains implementations of InferenceEngine for local execution,
such as LlamaCppEngine for GGUF models.
"""

from semantix.inference.local.llama_cpp import LlamaCppEngine

__all__ = ["LlamaCppEngine"]
