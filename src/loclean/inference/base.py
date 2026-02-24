"""Base abstract class for inference engines.

This module defines the InferenceEngine abstract base class that all
inference backends must implement, ensuring a consistent interface across
local and cloud providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class InferenceEngine(ABC):
    """Abstract base class for all inference engines.

    All inference engines (Ollama, OpenAI, Anthropic, Gemini, etc.)
    must inherit from this class and implement the required methods.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt / instruction.
            schema: Optional Pydantic model for structured JSON output.

        Returns:
            Raw text response from the model.
        """
        ...

    @abstractmethod
    def clean_batch(
        self,
        items: List[str],
        instruction: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Process a batch of strings and extract structured data.

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the extraction task.

        Returns:
            Dictionary mapping original_string →
            {"reasoning": str, "value": float, "unit": str} or None.
        """
        ...
