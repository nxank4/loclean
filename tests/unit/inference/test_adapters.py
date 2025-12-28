"""Test cases for prompt adapters."""

import pytest

from loclean.inference.adapters import (
    LlamaAdapter,
    Phi3Adapter,
    PromptAdapter,
    QwenAdapter,
    get_adapter,
)


class TestPromptAdapter:
    """Test cases for PromptAdapter abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that PromptAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            PromptAdapter()  # type: ignore[abstract]

        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower() or "format" in error_msg.lower()

    def test_subclass_without_implementation_raises_error(self) -> None:
        """Test that subclass without implementation cannot be instantiated."""

        class IncompleteAdapter(PromptAdapter):
            """Subclass that doesn't implement methods."""

            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteAdapter()  # type: ignore[abstract]

        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower()


class TestPhi3Adapter:
    """Test cases for Phi3Adapter."""

    def test_format_basic(self) -> None:
        """Test Phi3Adapter format with basic input."""
        adapter = Phi3Adapter()
        instruction = "Extract value and unit"
        item = "10kg"

        result = adapter.format(instruction, item)

        # Check format structure
        assert "<|user|>" in result
        assert "<|end|>" in result
        assert "<|assistant|>" in result
        assert instruction in result
        assert item in result
        assert "Task:" in result
        assert "Input Item:" in result

    def test_format_structure(self) -> None:
        """Test Phi3Adapter format structure is correct."""
        adapter = Phi3Adapter()
        instruction = "Convert to USD"
        item = "100 EUR"

        result = adapter.format(instruction, item)

        # Check order: user tag -> content -> end tag -> assistant tag
        assert result.startswith("<|user|>")
        assert result.endswith("<|assistant|>")
        assert "<|end|>" in result
        assert result.find("<|end|>") < result.find("<|assistant|>")

    def test_format_contains_instruction(self) -> None:
        """Test that instruction is properly included in format."""
        adapter = Phi3Adapter()
        instruction = "Extract the numeric value and unit"
        item = "5.5kg"

        result = adapter.format(instruction, item)

        assert f"Task: {instruction}" in result
        assert instruction in result

    def test_format_contains_item(self) -> None:
        """Test that item is properly included in format."""
        adapter = Phi3Adapter()
        instruction = "Extract value"
        item = "20.5 USD"

        result = adapter.format(instruction, item)

        assert f'Input Item: "{item}"' in result
        assert item in result

    def test_format_contains_steps(self) -> None:
        """Test that format contains the extraction steps."""
        adapter = Phi3Adapter()
        result = adapter.format("test", "test")

        assert "Step 1:" in result
        assert "Step 2:" in result
        assert "Step 3:" in result
        assert "Step 4:" in result
        assert "Step 5:" in result
        assert "Output JSON" in result
        assert '"reasoning"' in result
        assert '"value"' in result
        assert '"unit"' in result

    def test_get_stop_tokens(self) -> None:
        """Test Phi3Adapter stop tokens."""
        adapter = Phi3Adapter()
        stop_tokens = adapter.get_stop_tokens()

        assert isinstance(stop_tokens, list)
        assert "<|end|>" in stop_tokens
        assert "<|user|>" in stop_tokens
        assert len(stop_tokens) == 2

    def test_format_with_special_characters(self) -> None:
        """Test Phi3Adapter format with special characters in input."""
        adapter = Phi3Adapter()
        instruction = "Extract $ value"
        item = 'Price: "$100.50"'

        result = adapter.format(instruction, item)

        assert "$100.50" in result
        assert "$ value" in result

    def test_format_with_empty_strings(self) -> None:
        """Test Phi3Adapter format with empty strings."""
        adapter = Phi3Adapter()
        result = adapter.format("", "")

        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert 'Input Item: ""' in result


class TestQwenAdapter:
    """Test cases for QwenAdapter."""

    def test_format_basic(self) -> None:
        """Test QwenAdapter format with basic input."""
        adapter = QwenAdapter()
        instruction = "Extract value and unit"
        item = "10kg"

        result = adapter.format(instruction, item)

        # Check ChatML format structure
        assert "<|im_start|>user" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>assistant" in result
        assert instruction in result
        assert item in result

    def test_format_structure(self) -> None:
        """Test QwenAdapter format structure is correct."""
        adapter = QwenAdapter()
        instruction = "Convert to USD"
        item = "100 EUR"

        result = adapter.format(instruction, item)

        # Check order: system -> user -> content -> im_end -> assistant
        assert result.startswith("<|im_start|>system")
        assert "<|im_start|>user" in result
        assert result.endswith("<|im_start|>assistant\n")
        assert "<|im_end|>" in result
        assert result.find("<|im_end|>") < result.find("<|im_start|>assistant")

    def test_format_contains_instruction(self) -> None:
        """Test that instruction is properly included in format."""
        adapter = QwenAdapter()
        instruction = "Extract the numeric value and unit"
        item = "5.5kg"

        result = adapter.format(instruction, item)

        assert f"Task: {instruction}" in result
        assert instruction in result

    def test_format_contains_item(self) -> None:
        """Test that item is properly included in format."""
        adapter = QwenAdapter()
        instruction = "Extract value"
        item = "20.5 USD"

        result = adapter.format(instruction, item)

        assert f'Input Item: "{item}"' in result
        assert item in result

    def test_format_contains_steps(self) -> None:
        """Test that format contains the extraction steps."""
        adapter = QwenAdapter()
        result = adapter.format("test", "test")

        assert "Step 1:" in result
        assert "Step 2:" in result
        assert "Step 3:" in result
        assert "Step 4:" in result
        assert "Step 5:" in result
        assert "Output JSON" in result

    def test_get_stop_tokens(self) -> None:
        """Test QwenAdapter stop tokens."""
        adapter = QwenAdapter()
        stop_tokens = adapter.get_stop_tokens()

        assert isinstance(stop_tokens, list)
        assert "<|im_end|>" in stop_tokens
        assert "<|im_start|>" in stop_tokens
        assert len(stop_tokens) == 2

    def test_format_with_special_characters(self) -> None:
        """Test QwenAdapter format with special characters."""
        adapter = QwenAdapter()
        instruction = "Extract $ value"
        item = 'Price: "$100.50"'

        result = adapter.format(instruction, item)

        assert "$100.50" in result
        assert "$ value" in result

    def test_format_chatml_structure(self) -> None:
        """Test that QwenAdapter uses proper ChatML structure."""
        adapter = QwenAdapter()
        result = adapter.format("test", "test")

        # Should have system, user and assistant tags
        assert result.count("<|im_start|>") == 3
        assert result.count("<|im_end|>") == 2
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result


class TestLlamaAdapter:
    """Test cases for LlamaAdapter."""

    def test_format_basic(self) -> None:
        """Test LlamaAdapter format with basic input."""
        adapter = LlamaAdapter()
        instruction = "Extract value and unit"
        item = "10kg"

        result = adapter.format(instruction, item)

        # Check Llama-3 format structure
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>" in result
        assert "<|end_header_id|>" in result
        assert "<|eot_id|>" in result
        assert instruction in result
        assert item in result

    def test_format_structure(self) -> None:
        """Test LlamaAdapter format structure is correct."""
        adapter = LlamaAdapter()
        instruction = "Convert to USD"
        item = "100 EUR"

        result = adapter.format(instruction, item)

        # Check order: begin_of_text -> system -> user -> assistant
        assert result.startswith("<|begin_of_text|>")
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        assert "<|start_header_id|>system" in result
        assert "<|start_header_id|>user" in result
        assert "<|start_header_id|>assistant" in result

    def test_format_contains_system_message(self) -> None:
        """Test that LlamaAdapter includes system message."""
        adapter = LlamaAdapter()
        result = adapter.format("test", "test")

        assert "You are a helpful assistant" in result
        assert "system" in result

    def test_format_contains_instruction(self) -> None:
        """Test that instruction is properly included in format."""
        adapter = LlamaAdapter()
        instruction = "Extract the numeric value and unit"
        item = "5.5kg"

        result = adapter.format(instruction, item)

        assert f"Task: {instruction}" in result
        assert instruction in result

    def test_format_contains_item(self) -> None:
        """Test that item is properly included in format."""
        adapter = LlamaAdapter()
        instruction = "Extract value"
        item = "20.5 USD"

        result = adapter.format(instruction, item)

        assert f'Input Item: "{item}"' in result
        assert item in result

    def test_format_contains_steps(self) -> None:
        """Test that format contains the extraction steps."""
        adapter = LlamaAdapter()
        result = adapter.format("test", "test")

        assert "Step 1:" in result
        assert "Step 2:" in result
        assert "Step 3:" in result
        assert "Step 4:" in result
        assert "Step 5:" in result
        assert "Output JSON" in result

    def test_get_stop_tokens(self) -> None:
        """Test LlamaAdapter stop tokens."""
        adapter = LlamaAdapter()
        stop_tokens = adapter.get_stop_tokens()

        assert isinstance(stop_tokens, list)
        assert "<|eot_id|>" in stop_tokens
        assert "<|start_header_id|>" in stop_tokens
        assert len(stop_tokens) == 2

    def test_format_with_special_characters(self) -> None:
        """Test LlamaAdapter format with special characters."""
        adapter = LlamaAdapter()
        instruction = "Extract $ value"
        item = 'Price: "$100.50"'

        result = adapter.format(instruction, item)

        assert "$100.50" in result
        assert "$ value" in result

    def test_format_header_structure(self) -> None:
        """Test that LlamaAdapter has correct header structure."""
        adapter = LlamaAdapter()
        result = adapter.format("test", "test")

        # Should have system, user, and assistant headers
        assert result.count("<|start_header_id|>") == 3
        assert result.count("<|end_header_id|>") == 3
        assert result.count("<|eot_id|>") == 2


class TestGetAdapter:
    """Test cases for get_adapter factory function."""

    def test_get_phi3_adapter_direct_match(self) -> None:
        """Test getting Phi3Adapter with direct model name match."""
        adapter = get_adapter("phi-3-mini")
        assert isinstance(adapter, Phi3Adapter)

    def test_get_phi3_adapter_variations(self) -> None:
        """Test getting Phi3Adapter with various Phi-3 model names."""
        for model_name in [
            "phi-3-mini",
            "phi-3-mini-4k-instruct",
            "phi-3-medium",
            "phi-3",
        ]:
            adapter = get_adapter(model_name)
            assert isinstance(adapter, Phi3Adapter), f"Failed for {model_name}"

    def test_get_qwen_adapter_direct_match(self) -> None:
        """Test getting QwenAdapter with direct model name match."""
        adapter = get_adapter("qwen3-4b")
        assert isinstance(adapter, QwenAdapter)

    def test_get_qwen_adapter_variations(self) -> None:
        """Test getting QwenAdapter with various Qwen model names."""
        for model_name in ["qwen", "qwen3", "qwen3-4b", "qwen-2.5", "qwen-2"]:
            adapter = get_adapter(model_name)
            assert isinstance(adapter, QwenAdapter), f"Failed for {model_name}"

    def test_get_llama_adapter_direct_match(self) -> None:
        """Test getting LlamaAdapter with direct model name match."""
        adapter = get_adapter("llama-3")
        assert isinstance(adapter, LlamaAdapter)

    def test_get_llama_adapter_variations(self) -> None:
        """Test getting LlamaAdapter with various Llama model names."""
        for model_name in ["llama", "llama-3", "llama-2", "llama3", "llama2"]:
            adapter = get_adapter(model_name)
            assert isinstance(adapter, LlamaAdapter), f"Failed for {model_name}"

    def test_get_gemma_adapter(self) -> None:
        """Test that Gemma models use LlamaAdapter."""
        for model_name in ["gemma", "gemma-3", "gemma-2"]:
            adapter = get_adapter(model_name)
            assert isinstance(adapter, LlamaAdapter), f"Failed for {model_name}"

    def test_get_deepseek_adapter(self) -> None:
        """Test that DeepSeek models use QwenAdapter."""
        for model_name in ["deepseek", "deepseek-r1"]:
            adapter = get_adapter(model_name)
            assert isinstance(adapter, QwenAdapter), f"Failed for {model_name}"

    def test_get_adapter_case_insensitive(self) -> None:
        """Test that get_adapter is case-insensitive."""
        adapter1 = get_adapter("PHI-3-MINI")
        adapter2 = get_adapter("phi-3-mini")
        adapter3 = get_adapter("Phi-3-Mini")

        assert isinstance(adapter1, Phi3Adapter)
        assert isinstance(adapter2, Phi3Adapter)
        assert isinstance(adapter3, Phi3Adapter)

    def test_get_adapter_partial_match(self) -> None:
        """Test that get_adapter works with partial matches."""
        adapter1 = get_adapter("microsoft/Phi-3-mini-4k-instruct-gguf")
        adapter2 = get_adapter("unsloth/Qwen3-4B-Instruct-2507-GGUF")
        adapter3 = get_adapter("meta-llama/Llama-3-8B-Instruct")

        assert isinstance(adapter1, Phi3Adapter)
        assert isinstance(adapter2, QwenAdapter)
        assert isinstance(adapter3, LlamaAdapter)

    def test_get_adapter_fallback(self) -> None:
        """Test that get_adapter falls back to Phi3Adapter for unknown models."""
        adapter = get_adapter("unknown-model-xyz")
        assert isinstance(adapter, Phi3Adapter)

    def test_get_adapter_empty_string(self) -> None:
        """Test that get_adapter handles empty string."""
        adapter = get_adapter("")
        assert isinstance(adapter, Phi3Adapter)

    def test_get_adapter_returns_new_instance(self) -> None:
        """Test that get_adapter returns a new instance each time."""
        adapter1 = get_adapter("phi-3-mini")
        adapter2 = get_adapter("phi-3-mini")

        assert adapter1 is not adapter2
        assert isinstance(adapter1, Phi3Adapter)
        assert isinstance(adapter2, Phi3Adapter)


class TestAdapterIntegration:
    """Integration tests for adapters."""

    def test_all_adapters_format_same_content(self) -> None:
        """Test that all adapters format the same content correctly."""
        instruction = "Extract value and unit"
        item = "10.5kg"

        phi3_adapter = Phi3Adapter()
        qwen_adapter = QwenAdapter()
        llama_adapter = LlamaAdapter()

        phi3_result = phi3_adapter.format(instruction, item)
        qwen_result = qwen_adapter.format(instruction, item)
        llama_result = llama_adapter.format(instruction, item)

        # All should contain the same instruction and item
        assert instruction in phi3_result
        assert instruction in qwen_result
        assert instruction in llama_result

        assert item in phi3_result
        assert item in qwen_result
        assert item in llama_result

        # But should have different format markers
        assert "<|user|>" in phi3_result
        assert "<|im_start|>" in qwen_result
        assert "<|begin_of_text|>" in llama_result

    def test_all_adapters_have_stop_tokens(self) -> None:
        """Test that all adapters return stop tokens."""
        adapters = [Phi3Adapter(), QwenAdapter(), LlamaAdapter()]

        for adapter in adapters:
            stop_tokens = adapter.get_stop_tokens()
            assert isinstance(stop_tokens, list)
            assert len(stop_tokens) > 0
            assert all(isinstance(token, str) for token in stop_tokens)
