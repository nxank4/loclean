"""Base abstract class for inference engines.

This module defines the InferenceEngine abstract base class that all
inference backends must implement, ensuring a consistent interface across
local and cloud providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class InferenceEngine(ABC):
    """
    Abstract base class for all inference engines.

    All inference engines (local GGUF models, OpenAI, Anthropic, Gemini, etc.)
    must inherit from this class and implement the clean_batch method to ensure
    consistent behavior across backends.
    """

    @abstractmethod
    def clean_batch(
        self,
        items: List[str],
        instruction: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process a batch of strings and extract structured data.

        This method takes a list of input strings and an instruction, then returns
        a dictionary mapping each input string to its extracted structured data
        (reasoning, value, unit) or None if extraction failed.

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the extraction task.

        Returns:
            Dictionary mapping original_string -> {
                "reasoning": str,
                "value": float,
                "unit": str
            } or None if extraction failed.

        Raises:
            InferenceError: If the inference operation fails critically.
                Subclasses should raise appropriate exceptions for their backend.
        """
        pass
