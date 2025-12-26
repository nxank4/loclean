"""Deprecated module for backward compatibility.

This module is kept for backward compatibility. The actual implementation
has been moved to semantix.inference.local.llama_cpp.

This module will be removed in a future version.
"""

import warnings
from pathlib import Path
from typing import Optional

# Import from new location
from semantix.inference.local.llama_cpp import LlamaCppEngine

# Re-export for backward compatibility
__all__ = ["LlamaCppEngine", "LocalInferenceEngine"]


class LocalInferenceEngine(LlamaCppEngine):
    """
    Deprecated alias for LlamaCppEngine.

    This class is kept for backward compatibility but will be removed in a future version.
    Use LlamaCppEngine instead.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize LocalInferenceEngine (deprecated).

        Args:
            cache_dir: Optional custom directory for caching models.
                       Defaults to ~/.cache/semantix.
            model_name: Optional model name. If not provided, defaults to "phi-3-mini".
            **kwargs: Additional arguments passed to LlamaCppEngine.

        Deprecated:
            This class is deprecated. Use LlamaCppEngine instead.
        """
        warnings.warn(
            "LocalInferenceEngine is deprecated and will be removed in a future version. "
            "Use LlamaCppEngine instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For backward compatibility: if model_name not provided, use default
        if model_name is None:
            model_name = "phi-3-mini"

        super().__init__(model_name=model_name, cache_dir=cache_dir, **kwargs)

        # Keep old class attributes for backward compatibility
        self.MODEL_REPO = self.model_repo
        self.MODEL_FILENAME = self.model_filename
