"""Prompt adapters for different model formats.

This module implements the Strategy Pattern for prompt construction, allowing
different models to use their native prompt formats (Phi-3, Qwen, Llama, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict

from jinja2 import Template

from loclean.utils.resources import load_template


class PromptAdapter(ABC):
    """
    Abstract base class for prompt adapters.

    Each adapter implements the prompt format specific to a model family,
    ensuring correct tokenization and instruction following.
    """

    @abstractmethod
    def format(self, instruction: str, item: str) -> str:
        """
        Format a prompt for the model.

        Args:
            instruction: User-defined instruction for the extraction task.
            item: The input item string to process.

        Returns:
            Formatted prompt string ready for model inference.
        """
        pass

    @abstractmethod
    def get_stop_tokens(self) -> list[str]:
        """
        Get stop tokens specific to this model format.

        Returns:
            List of stop token strings to prevent model from generating
            beyond the response.
        """
        pass


class Phi3Adapter(PromptAdapter):
    """
    Prompt adapter for Phi-3 models.

    Uses the Phi-3 Instruct format with <|user|> and <|assistant|> tags.
    """

    def __init__(self) -> None:
        """Initialize Phi3Adapter and load templates."""
        content_template_str = load_template("cleaning_instruction.j2")
        format_template_str = load_template("phi3_format.j2")
        self.content_template = Template(content_template_str)
        self.format_template = Template(format_template_str)
        self.system_message = (
            "You are a helpful assistant that extracts structured data from text."
        )

    def format(self, instruction: str, item: str) -> str:
        """
        Format prompt in Phi-3 Instruct format.

        Args:
            instruction: User-defined instruction for the extraction task.
            item: The input item string to process.

        Returns:
            Formatted prompt string in Phi-3 format.
        """
        # Build business content (Template A)
        user_content = self.content_template.render(instruction=instruction, item=item)

        # Build messages
        messages = [{"role": "user", "content": user_content}]

        # Render with format template (Template B)
        return self.format_template.render(
            messages=messages,
            system_message=self.system_message,
            add_generation_prompt=True,
        )

    def get_stop_tokens(self) -> list[str]:
        """Get stop tokens for Phi-3 format."""
        return ["<|end|>", "<|user|>"]


class QwenAdapter(PromptAdapter):
    """
    Prompt adapter for Qwen models.

    Uses ChatML format with <|im_start|> and <|im_end|> tags.
    """

    def __init__(self) -> None:
        """Initialize QwenAdapter and load templates."""
        content_template_str = load_template("cleaning_instruction.j2")
        format_template_str = load_template("qwen_format.j2")
        self.content_template = Template(content_template_str)
        self.format_template = Template(format_template_str)
        self.system_message = (
            "You are a helpful assistant that extracts structured data from text."
        )

    def format(self, instruction: str, item: str) -> str:
        """
        Format prompt in Qwen ChatML format.

        Args:
            instruction: User-defined instruction for the extraction task.
            item: The input item string to process.

        Returns:
            Formatted prompt string in Qwen ChatML format.
        """
        # Build business content (Template A)
        user_content = self.content_template.render(instruction=instruction, item=item)

        # Build messages
        messages = [{"role": "user", "content": user_content}]

        # Render with format template (Template B)
        result = self.format_template.render(
            messages=messages,
            system_message=self.system_message,
            add_generation_prompt=True,
        )
        if not result.endswith("\n"):
            result += "\n"
        return result

    def get_stop_tokens(self) -> list[str]:
        """Get stop tokens for Qwen ChatML format."""
        return ["<|im_end|>", "<|im_start|>"]


class LlamaAdapter(PromptAdapter):
    """
    Prompt adapter for Llama models.

    Uses the Llama-3 instruction format with <|start_header_id|> tags.
    """

    def __init__(self) -> None:
        """Initialize LlamaAdapter and load templates."""
        content_template_str = load_template("cleaning_instruction.j2")
        format_template_str = load_template("llama_format.j2")
        self.content_template = Template(content_template_str)
        self.format_template = Template(format_template_str)
        self.system_message = (
            "You are a helpful assistant that extracts structured data from text."
        )

    def format(self, instruction: str, item: str) -> str:
        """
        Format prompt in Llama-3 instruction format.

        Args:
            instruction: User-defined instruction for the extraction task.
            item: The input item string to process.

        Returns:
            Formatted prompt string in Llama-3 format.
        """
        # Build business content (Template A)
        user_content = self.content_template.render(instruction=instruction, item=item)

        # Build messages
        messages = [{"role": "user", "content": user_content}]

        # Render with format template (Template B)
        result = self.format_template.render(
            messages=messages,
            system_message=self.system_message,
            add_generation_prompt=True,
        )
        # Ensure exactly 2 newlines at the end
        result = result.rstrip() + "\n\n"
        return result

    def get_stop_tokens(self) -> list[str]:
        """Get stop tokens for Llama-3 format."""
        return ["<|eot_id|>", "<|start_header_id|>"]


# Model name to adapter mapping
_MODEL_ADAPTER_MAP: Dict[str, type[PromptAdapter]] = {
    # Phi-3 models
    "phi-3-mini": Phi3Adapter,
    "phi-3-mini-4k-instruct": Phi3Adapter,
    "phi-3-medium": Phi3Adapter,
    "phi-3": Phi3Adapter,
    # Qwen models
    "qwen": QwenAdapter,
    "qwen3": QwenAdapter,
    "qwen3-4b": QwenAdapter,
    "qwen-2.5": QwenAdapter,
    "qwen-2": QwenAdapter,
    # Llama models
    "llama": LlamaAdapter,
    "llama-3": LlamaAdapter,
    "llama-2": LlamaAdapter,
    "llama3": LlamaAdapter,
    "llama2": LlamaAdapter,
    # Gemma models (use Llama format)
    "gemma": LlamaAdapter,
    "gemma-3": LlamaAdapter,
    "gemma-2": LlamaAdapter,
    # DeepSeek models (use Qwen format)
    "deepseek": QwenAdapter,
    "deepseek-r1": QwenAdapter,
}


def get_adapter(model_name: str) -> PromptAdapter:
    """
    Get the appropriate prompt adapter for a given model name.

    This function performs case-insensitive matching and partial matching
    to find the correct adapter. Falls back to Phi3Adapter if no match is found.

    Args:
        model_name: Name or identifier of the model (e.g., "phi-3-mini", "qwen3-4b").

    Returns:
        PromptAdapter instance for the model.

    Examples:
        >>> adapter = get_adapter("phi-3-mini-4k-instruct")
        >>> isinstance(adapter, Phi3Adapter)
        True

        >>> adapter = get_adapter("qwen3-4b")
        >>> isinstance(adapter, QwenAdapter)
        True
    """
    model_lower = model_name.lower()

    # Direct match
    if model_lower in _MODEL_ADAPTER_MAP:
        adapter_class = _MODEL_ADAPTER_MAP[model_lower]
        return adapter_class()

    # Partial match - check if model name contains any key
    for key, adapter_class in _MODEL_ADAPTER_MAP.items():
        if key in model_lower or model_lower in key:
            return adapter_class()

    # Fallback to Phi-3 adapter (default)
    return Phi3Adapter()
