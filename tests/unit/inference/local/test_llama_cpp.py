"""Test cases for LlamaCppEngine."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from loclean.inference.adapters import LlamaAdapter, Phi3Adapter, QwenAdapter
from loclean.inference.local.llama_cpp import LlamaCppEngine


@pytest.fixture
def temp_cache_dir(tmp_path: Any) -> Any:
    """Create a temporary cache directory for testing."""
    return tmp_path / "test_cache"


@pytest.fixture
def mock_llama() -> Any:
    """Create a mock Llama instance."""
    mock_llama_instance = Mock()
    mock_llama_instance.create_completion = Mock()
    return mock_llama_instance


@pytest.fixture
def mock_llama_class(mock_llama: Any) -> Any:
    """Patch Llama class to return mock instance."""
    with patch("loclean.inference.local.llama_cpp.Llama", return_value=mock_llama):
        yield mock_llama


@pytest.fixture
def mock_grammar() -> Any:
    """Create a mock LlamaGrammar instance."""
    mock_grammar_instance = Mock()
    return mock_grammar_instance


@pytest.fixture
def mock_grammar_class(mock_grammar: Any) -> Any:
    """Patch LlamaGrammar class to return mock instance."""
    with patch(
        "loclean.inference.local.llama_cpp.LlamaGrammar", return_value=mock_grammar
    ):
        with patch(
            "loclean.inference.local.llama_cpp.LlamaGrammar.from_string",
            return_value=mock_grammar,
        ):
            yield mock_grammar


@pytest.fixture
def mock_cache() -> Any:
    """Create a mock LocleanCache instance."""
    mock_cache_instance = Mock()
    mock_cache_instance.get_batch = Mock(return_value={})
    mock_cache_instance.set_batch = Mock()
    mock_cache_instance._hash = Mock(
        side_effect=lambda text, instruction: f"hash_{text}_{instruction}"
    )
    return mock_cache_instance


@pytest.fixture
def mock_cache_class(mock_cache: Any) -> Any:
    """Patch LocleanCache class to return mock instance."""
    with patch("loclean.cache.LocleanCache", return_value=mock_cache):
        yield mock_cache


@pytest.fixture
def mock_model_path(temp_cache_dir: Any) -> Any:
    """Create a mock model file path."""
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_cache_dir / "Phi-3-mini-4k-instruct-q4.gguf"
    model_path.touch()
    return model_path


@pytest.fixture
def mock_hf_download(mock_model_path: Any) -> Any:
    """Patch hf_hub_download to return mock model path."""
    with patch(
        "loclean.inference.local.llama_cpp.hf_hub_download",
        return_value=str(mock_model_path),
    ):
        yield


class TestLlamaCppEngine:
    """Test cases for LlamaCppEngine."""

    def test_init_with_custom_cache_dir(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with custom cache directory."""
        engine = LlamaCppEngine(cache_dir=temp_cache_dir)

        assert engine.cache_dir == temp_cache_dir
        assert engine.model_path == mock_model_path
        assert engine.llm is not None
        assert engine.grammar is not None
        assert engine.cache is not None
        assert engine.model_name == "phi-3-mini"

    def test_init_with_default_cache_dir(
        self,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with default cache directory."""
        with patch(
            "loclean.inference.local.llama_cpp.Path.home",
            return_value=Path("/home/test"),
        ):
            with patch("pathlib.Path.mkdir"):  # Patch mkdir to avoid permission errors
                with patch(
                    "loclean.inference.local.llama_cpp.hf_hub_download",
                    return_value=str(mock_model_path),
                ):
                    engine = LlamaCppEngine()

                    assert engine.cache_dir == Path("/home/test") / ".cache" / "loclean"
                    assert engine.model_path == mock_model_path

    def test_init_with_model_name(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with specific model name."""
        with patch("loclean.inference.local.llama_cpp.Path.exists", return_value=True):
            engine = LlamaCppEngine(model_name="qwen3-4b", cache_dir=temp_cache_dir)

            assert engine.model_name == "qwen3-4b"
            assert engine.model_repo == "unsloth/Qwen3-4B-Instruct-2507-GGUF"
            assert engine.model_filename == "Qwen3-4B-Instruct-2507-GGUF.q4_k_m.gguf"

    def test_init_with_unknown_model_falls_back(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine falls back to default model when
        unknown model is provided."""
        with patch("loclean.inference.local.llama_cpp.logger") as mock_logger:
            engine = LlamaCppEngine(
                model_name="unknown-model", cache_dir=temp_cache_dir
            )

            assert engine.model_name == "phi-3-mini"
            mock_logger.warning.assert_called_once()

    def test_init_with_n_ctx_and_n_gpu_layers(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with custom n_ctx and n_gpu_layers."""
        with patch("loclean.inference.local.llama_cpp.Llama") as mock_llama_constructor:
            mock_llama_instance = Mock()
            mock_llama_constructor.return_value = mock_llama_instance

            engine = LlamaCppEngine(
                cache_dir=temp_cache_dir, n_ctx=8192, n_gpu_layers=10
            )

            assert engine.llm is not None
            mock_llama_constructor.assert_called_once_with(
                model_path=str(mock_model_path),
                n_ctx=8192,
                n_gpu_layers=10,
                verbose=False,
            )

    def test_get_model_path_existing_file(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_hf_download: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
    ) -> None:
        """Test _get_model_path returns existing file."""
        with patch("loclean.inference.local.llama_cpp.Llama"):
            with patch("loclean.inference.local.llama_cpp.LlamaGrammar"):
                with patch("loclean.cache.LocleanCache"):
                    engine = LlamaCppEngine(cache_dir=temp_cache_dir)
                    path = engine._get_model_path()

                    assert path == mock_model_path

    def test_get_model_path_downloads_when_missing(self, temp_cache_dir: Any) -> None:
        """Test _get_model_path downloads model when missing."""
        with patch("loclean.inference.local.llama_cpp.Path.exists", return_value=False):
            with patch("loclean.inference.local.llama_cpp.Llama"):
                with patch("loclean.inference.local.llama_cpp.LlamaGrammar"):
                    with patch("loclean.cache.LocleanCache"):
                        with patch(
                            "loclean.inference.local.llama_cpp.hf_hub_download"
                        ) as mock_download:
                            mock_download.return_value = str(
                                temp_cache_dir / "downloaded.gguf"
                            )
                            engine = LlamaCppEngine(cache_dir=temp_cache_dir)
                            path = engine._get_model_path()

                            assert mock_download.called
                            assert path == Path(temp_cache_dir / "downloaded.gguf")

    def test_get_json_grammar(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test _get_json_grammar loads grammar from resources."""
        with patch("loclean.utils.resources.load_grammar") as mock_load_grammar:
            mock_load_grammar.return_value = "root ::= object"
            with patch(
                "loclean.inference.local.llama_cpp.LlamaGrammar"
            ) as mock_grammar_class:
                mock_grammar_instance = Mock()
                mock_grammar_class.from_string = Mock(
                    return_value=mock_grammar_instance
                )

                engine = LlamaCppEngine(cache_dir=temp_cache_dir)
                # Reset mocks after initialization (grammar is loaded in __init__)
                mock_load_grammar.reset_mock()
                mock_grammar_class.from_string.reset_mock()

                grammar = engine._get_json_grammar()

                assert grammar == mock_grammar_instance
                mock_load_grammar.assert_called_once_with("json.gbnf")
                mock_grammar_class.from_string.assert_called_once()

    def test_adapter_selection_for_different_models(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that correct adapter is selected for different models."""
        with patch("loclean.inference.local.llama_cpp.Path.exists", return_value=True):
            # Test Phi-3 model
            engine_phi = LlamaCppEngine(
                model_name="phi-3-mini", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_phi.adapter, Phi3Adapter)

            # Test Qwen model
            engine_qwen = LlamaCppEngine(
                model_name="qwen3-4b", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_qwen.adapter, QwenAdapter)

            # Test Gemma model (should use LlamaAdapter)
            engine_gemma = LlamaCppEngine(
                model_name="gemma-3-4b", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_gemma.adapter, LlamaAdapter)

    def test_clean_batch_uses_adapter(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that clean_batch uses adapter to format prompts."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [{"text": '{"reasoning": "test", "value": 1.0, "unit": "kg"}'}]
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        engine.clean_batch(["5.5kg"], "Extract value and unit")

        # Verify adapter.format was called (indirectly through prompt formatting)
        assert mock_llama.create_completion.called
        call_args = mock_llama.create_completion.call_args
        assert "prompt" in call_args.kwargs

    def test_clean_batch_all_cached(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch returns cached results when all items are cached."""
        mock_cache = mock_cache_class
        mock_cache.get_batch.return_value = {
            "5.5kg": {"reasoning": "cached", "value": 5.5, "unit": "kg"}
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg"], "Extract value and unit")

        assert result == {"5.5kg": {"reasoning": "cached", "value": 5.5, "unit": "kg"}}
        mock_llama_class.create_completion.assert_not_called()

    def test_clean_batch_partial_cache(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles partial cache hits."""
        mock_cache = mock_cache_class
        mock_cache.get_batch.return_value = {
            "5.5kg": {"reasoning": "cached", "value": 5.5, "unit": "kg"}
        }

        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [{"text": '{"reasoning": "new", "value": 10.0, "unit": "m"}'}]
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg", "10m"], "Extract value and unit")

        assert "5.5kg" in result
        assert "10m" in result
        assert result["5.5kg"] is not None
        assert result["10m"] is not None
        assert result["5.5kg"]["value"] == 5.5
        assert result["10m"]["value"] == 10.0

    def test_clean_batch_successful_inference(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch successfully processes items through inference."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [
                {"text": '{"reasoning": "Extracted", "value": 5.5, "unit": "kg"}'}
            ]
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg"], "Extract value and unit")

        assert result["5.5kg"] == {
            "reasoning": "Extracted",
            "value": 5.5,
            "unit": "kg",
        }
        mock_cache_class.set_batch.assert_called_once()

    def test_clean_batch_json_decode_error(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles JSON decode errors gracefully."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "invalid json"}]
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg"], "Extract value and unit")

        assert result["5.5kg"] is None
        mock_cache_class.set_batch.assert_not_called()

    def test_clean_batch_missing_keys(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles missing required keys in JSON."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [{"text": '{"value": 5.5, "unit": "kg"}'}]  # Missing "reasoning"
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg"], "Extract value and unit")

        assert result["5.5kg"] is None

    def test_clean_batch_inference_exception(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles inference exceptions gracefully."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.side_effect = Exception("Inference error")

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["5.5kg"], "Extract value and unit")

        assert result["5.5kg"] is None

    def test_clean_batch_multiple_items_with_mixed_results(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles multiple items with mixed success/failure."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.side_effect = [
            {"choices": [{"text": '{"reasoning": "ok", "value": 5.5, "unit": "kg"}'}]},
            {"choices": [{"text": "invalid json"}]},
            {"choices": [{"text": '{"reasoning": "ok", "value": 10.0, "unit": "m"}'}]},
        ]

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(
            ["5.5kg", "invalid", "10m"], "Extract value and unit"
        )

        assert result["5.5kg"] == {"reasoning": "ok", "value": 5.5, "unit": "kg"}
        assert result["invalid"] is None
        assert result["10m"] == {"reasoning": "ok", "value": 10.0, "unit": "m"}

    def test_clean_batch_only_caches_valid_results(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that only valid results are cached."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.side_effect = [
            {"choices": [{"text": '{"reasoning": "ok", "value": 5.5, "unit": "kg"}'}]},
            {"choices": [{"text": "invalid json"}]},
        ]

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        engine.clean_batch(["5.5kg", "invalid"], "Extract value and unit")

        # Verify set_batch was called only once with valid result
        assert mock_cache_class.set_batch.call_count == 1
        call_args = mock_cache_class.set_batch.call_args
        assert "5.5kg" in call_args[0][0]  # First arg is list of keys
        assert "invalid" not in call_args[0][0]

    def test_clean_batch_uses_stop_tokens(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that clean_batch uses stop tokens from adapter."""
        mock_llama = mock_llama_class
        mock_llama.create_completion.return_value = {
            "choices": [{"text": '{"reasoning": "test", "value": 1.0, "unit": "kg"}'}]
        }

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        engine.clean_batch(["5.5kg"], "Extract value and unit")

        call_args = mock_llama.create_completion.call_args
        assert "stop" in call_args.kwargs
        # Phi3Adapter should provide stop tokens
        assert call_args.kwargs["stop"] is not None

    def test_model_registry_contains_expected_models(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that model registry contains all expected models."""
        from loclean.inference.local.llama_cpp import _MODEL_REGISTRY

        expected_models = ["phi-3-mini", "qwen3-4b", "gemma-3-4b", "deepseek-r1"]
        for model_name in expected_models:
            assert model_name in _MODEL_REGISTRY
            assert "repo" in _MODEL_REGISTRY[model_name]
            assert "filename" in _MODEL_REGISTRY[model_name]

    def test_clean_batch_empty_items_list(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test clean_batch handles empty items list."""
        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch([], "Extract value and unit")

        assert result == {}
        mock_llama_class.create_completion.assert_not_called()
