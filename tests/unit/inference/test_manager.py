"""Test cases for LlamaCppEngine and LocalInferenceEngine."""

import warnings
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from loclean.inference.adapters import LlamaAdapter, Phi3Adapter, QwenAdapter
from loclean.inference.manager import LlamaCppEngine, LocalInferenceEngine


@pytest.fixture
def temp_cache_dir(tmp_path: Any) -> Any:
    return tmp_path / "test_cache"


@pytest.fixture
def mock_llama() -> Any:
    mock_llama_instance = Mock()
    mock_llama_instance.create_completion = Mock()
    return mock_llama_instance


@pytest.fixture
def mock_llama_class(mock_llama: Any) -> Any:
    with patch("loclean.inference.local.llama_cpp.Llama", return_value=mock_llama):
        yield mock_llama


@pytest.fixture
def mock_grammar() -> Any:
    mock_grammar_instance = Mock()
    return mock_grammar_instance


@pytest.fixture
def mock_grammar_class(mock_grammar: Any) -> Any:
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
    mock_cache_instance = Mock()
    mock_cache_instance.get_batch = Mock(return_value={})
    mock_cache_instance.set_batch = Mock()
    mock_cache_instance._hash = Mock(
        side_effect=lambda text, instruction: f"hash_{text}_{instruction}"
    )
    return mock_cache_instance


@pytest.fixture
def mock_cache_class(mock_cache: Any) -> Any:
    with patch("loclean.cache.LocleanCache", return_value=mock_cache):
        yield mock_cache


@pytest.fixture
def mock_model_path(temp_cache_dir: Any) -> Any:
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_cache_dir / "test_model.gguf"
    model_path.touch()
    return model_path


@pytest.fixture
def mock_hf_download(mock_model_path: Any) -> Any:
    with patch(
        "loclean.inference.local.llama_cpp.hf_hub_download",
        return_value=str(mock_model_path),
    ):
        yield


# ==================== LlamaCppEngine Tests ====================


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
        assert isinstance(engine.adapter, Phi3Adapter)

    def test_init_with_default_cache_dir(
        self,
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
            with patch("pathlib.Path.mkdir"):
                expected_cache_dir = Path("/home/test/.cache/loclean")
                mock_model_path = expected_cache_dir / "Phi-3-mini-4k-instruct-q4.gguf"

                def mock_exists(*args: Any) -> bool:
                    path_self = args[0] if args else None
                    return (
                        str(path_self) == str(mock_model_path) if path_self else False
                    )

                with patch("pathlib.Path.exists", side_effect=mock_exists):
                    engine = LlamaCppEngine()

                    assert engine.cache_dir == expected_cache_dir
                    assert engine.model_name == "phi-3-mini"

    def test_init_with_model_name(
        self,
        temp_cache_dir: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with specific model name."""
        with patch("pathlib.Path.exists", return_value=True):
            engine = LlamaCppEngine(model_name="qwen3-4b", cache_dir=temp_cache_dir)

            assert engine.model_name == "qwen3-4b"
            assert engine.model_repo == "unsloth/Qwen3-4B-Instruct-2507-GGUF"
            assert isinstance(engine.adapter, QwenAdapter)

    def test_init_with_unknown_model_falls_back(
        self,
        temp_cache_dir: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that unknown model name falls back to phi-3-mini."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("loclean.inference.local.llama_cpp.logger") as mock_logger:
                engine = LlamaCppEngine(
                    model_name="unknown-model", cache_dir=temp_cache_dir
                )

                assert engine.model_name == "phi-3-mini"
                mock_logger.warning.assert_called()

    def test_init_with_n_ctx_and_n_gpu_layers(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test LlamaCppEngine initialization with n_ctx and n_gpu_layers."""
        with patch("loclean.inference.local.llama_cpp.Llama") as mock_llama_class:
            mock_llama_instance = Mock()
            mock_llama_class.return_value = mock_llama_instance

            engine = LlamaCppEngine(
                cache_dir=temp_cache_dir, n_ctx=8192, n_gpu_layers=10
            )

            assert engine.llm is not None
            mock_llama_class.assert_called_once_with(
                model_path=str(mock_model_path),
                n_ctx=8192,
                n_gpu_layers=10,
                verbose=False,
            )

    def test_get_model_path_existing_file(
        self, temp_cache_dir: Any, mock_model_path: Any, mock_hf_download: Any
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
                                temp_cache_dir / "downloaded_model.gguf"
                            )

                            engine = LlamaCppEngine(cache_dir=temp_cache_dir)
                            expected_repo = engine.model_repo
                            expected_filename = engine.model_filename
                            mock_download.reset_mock()
                            engine._get_model_path()

                            mock_download.assert_called_once_with(
                                repo_id=expected_repo,
                                filename=expected_filename,
                                local_dir=temp_cache_dir,
                            )

    def test_get_json_grammar(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test _get_json_grammar returns correct grammar."""
        with patch(
            "loclean.inference.local.llama_cpp.LlamaGrammar"
        ) as mock_grammar_class:
            mock_grammar_instance = Mock()
            mock_grammar_class.from_string = Mock(return_value=mock_grammar_instance)

            engine = LlamaCppEngine(cache_dir=temp_cache_dir)
            mock_grammar_class.from_string.reset_mock()
            grammar = engine._get_json_grammar()

            assert grammar == mock_grammar_instance
            mock_grammar_class.from_string.assert_called_once()

    def test_adapter_selection_for_different_models(
        self,
        temp_cache_dir: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that correct adapter is selected for different models."""
        with patch("pathlib.Path.exists", return_value=True):
            # Phi-3 model
            engine_phi = LlamaCppEngine(
                model_name="phi-3-mini", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_phi.adapter, Phi3Adapter)

            # Qwen model
            engine_qwen = LlamaCppEngine(
                model_name="qwen3-4b", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_qwen.adapter, QwenAdapter)

            # Gemma model (uses LlamaAdapter)
            engine_gemma = LlamaCppEngine(
                model_name="gemma-3-4b", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_gemma.adapter, LlamaAdapter)

            # DeepSeek model (uses QwenAdapter)
            engine_deepseek = LlamaCppEngine(
                model_name="deepseek-r1", cache_dir=temp_cache_dir
            )
            assert isinstance(engine_deepseek.adapter, QwenAdapter)

    def test_clean_batch_uses_adapter(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test that clean_batch uses adapter for prompt formatting."""
        llm_output = {
            "choices": [{"text": '{"reasoning": "test", "value": 10.0, "unit": "kg"}'}]
        }
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg"], "Extract weight")

        # Verify adapter was used
        assert mock_llama_class.create_completion.called
        call_kwargs = mock_llama_class.create_completion.call_args[1]
        assert "prompt" in call_kwargs
        assert "<|user|>" in call_kwargs["prompt"]  # Phi-3 format
        assert call_kwargs["stop"] == ["<|end|>", "<|user|>"]  # Phi-3 stop tokens
        assert "10kg" in result
        assert result["10kg"] is not None
        assert result["10kg"]["value"] == 10.0

    def test_clean_batch_all_cached(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch returns cached results when all items are cached."""
        cached_results = {
            "10kg": {"reasoning": "test", "value": 10.0, "unit": "kg"},
            "500g": {"reasoning": "test", "value": 500.0, "unit": "g"},
        }

        mock_cache_class.get_batch = Mock(return_value=cached_results)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg", "500g"], "Extract weight")

        assert result == cached_results
        assert not engine.llm.create_completion.called  # type: ignore[attr-defined]
        assert not mock_cache_class.set_batch.called

    def test_clean_batch_partial_cache(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch handles partial cache hits."""
        cached_results = {"10kg": {"reasoning": "test", "value": 10.0, "unit": "kg"}}
        mock_cache_class.get_batch = Mock(return_value=cached_results)

        llm_output = {
            "choices": [{"text": '{"reasoning": "test", "value": 500.0, "unit": "g"}'}]
        }
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg", "500g"], "Extract weight")

        assert "10kg" in result
        assert "500g" in result
        assert result["10kg"] == cached_results["10kg"]
        assert result["500g"] is not None
        assert result["500g"]["value"] == 500.0
        assert result["500g"]["unit"] == "g"
        mock_cache_class.set_batch.assert_called_once()

    def test_clean_batch_successful_inference(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch successful inference."""
        llm_output = {
            "choices": [
                {
                    "text": (
                        '{"reasoning": "Extracted weight", "value": 10.0, "unit": "kg"}'
                    )
                }
            ]
        }
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg"], "Extract weight")

        assert "10kg" in result
        assert result["10kg"] is not None
        assert result["10kg"]["value"] == 10.0
        assert result["10kg"]["unit"] == "kg"
        assert result["10kg"]["reasoning"] == "Extracted weight"
        mock_cache_class.set_batch.assert_called_once()

    def test_clean_batch_json_decode_error(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch handles JSON decode errors."""
        llm_output = {"choices": [{"text": "invalid json {"}]}
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg"], "Extract weight")

        assert "10kg" in result
        assert result["10kg"] is None
        assert not mock_cache_class.set_batch.called

    def test_clean_batch_missing_keys(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch handles missing keys in response."""
        llm_output = {"choices": [{"text": '{"value": 10.0}'}]}
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg"], "Extract weight")

        assert "10kg" in result
        assert result["10kg"] is None
        assert not mock_cache_class.set_batch.called

    def test_clean_batch_inference_exception(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch handles inference exceptions."""
        mock_llama_class.create_completion = Mock(side_effect=Exception("LLM error"))

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["10kg"], "Extract weight")

        assert "10kg" in result
        assert result["10kg"] is None
        assert not mock_cache_class.set_batch.called

    def test_clean_batch_multiple_items_with_mixed_results(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test clean_batch handles multiple items with mixed results."""
        cached_results = {"item1": {"reasoning": "cached", "value": 1.0, "unit": "kg"}}
        mock_cache_class.get_batch = Mock(return_value=cached_results)

        def create_completion_side_effect(*args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt", "")
            if "item2" in prompt:
                return {
                    "choices": [
                        {"text": '{"reasoning": "inferred", "value": 2.0, "unit": "g"}'}
                    ]
                }
            elif "item3" in prompt:
                return {"choices": [{"text": "invalid json"}]}
            return {
                "choices": [
                    {"text": '{"reasoning": "test", "value": 0.0, "unit": "kg"}'}
                ]
            }

        mock_llama_class.create_completion = Mock(
            side_effect=create_completion_side_effect
        )

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)
        result = engine.clean_batch(["item1", "item2", "item3"], "Extract weight")

        assert result["item1"] is not None
        assert result["item2"] is not None
        assert result["item1"]["value"] == 1.0
        assert result["item2"]["value"] == 2.0
        assert result["item3"] is None
        assert mock_llama_class.create_completion.call_count == 2

    def test_clean_batch_only_caches_valid_results(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test that only valid results are cached."""

        def create_completion_side_effect(*args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt", "")
            if '"valid_item"' in prompt:
                return {
                    "choices": [
                        {"text": '{"reasoning": "ok", "value": 10.0, "unit": "kg"}'}
                    ]
                }
            return {"choices": [{"text": "invalid json {"}]}

        mock_llama_class.create_completion = Mock(
            side_effect=create_completion_side_effect
        )
        mock_cache_class.get_batch = Mock(return_value={})
        mock_cache_class.set_batch.reset_mock()

        engine = LlamaCppEngine(cache_dir=temp_cache_dir)

        assert engine.cache is mock_cache_class

        result = engine.clean_batch(["valid_item", "invalid_item"], "Extract weight")

        assert engine.cache.set_batch.called  # type: ignore[attr-defined]

        call_args = engine.cache.set_batch.call_args  # type: ignore[attr-defined]
        assert call_args is not None

        cached_items = call_args[0][0]
        cached_results = call_args[0][2]

        assert "valid_item" in cached_items
        assert "invalid_item" not in cached_items
        assert "valid_item" in cached_results
        assert "invalid_item" not in cached_results

        assert result["valid_item"] is not None
        assert result["invalid_item"] is None


# ==================== LocalInferenceEngine Tests
# (Backward Compatibility) ====================


class TestLocalInferenceEngine:
    """Test cases for LocalInferenceEngine (deprecated alias)."""

    def test_deprecation_warning(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that LocalInferenceEngine raises DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = LocalInferenceEngine(cache_dir=temp_cache_dir)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert isinstance(engine, LlamaCppEngine)

    def test_backward_compatibility_attributes(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that LocalInferenceEngine maintains backward
        compatibility attributes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = LocalInferenceEngine(cache_dir=temp_cache_dir)

            # Check that old attributes exist
            assert hasattr(engine, "MODEL_REPO")
            assert hasattr(engine, "MODEL_FILENAME")
            assert engine.MODEL_REPO == engine.model_repo
            assert engine.MODEL_FILENAME == engine.model_filename

    def test_backward_compatibility_init_signature(
        self,
        temp_cache_dir: Any,
        mock_model_path: Any,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
    ) -> None:
        """Test that LocalInferenceEngine accepts old init signature."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Old signature: only cache_dir
            engine1 = LocalInferenceEngine(cache_dir=temp_cache_dir)
            assert engine1.model_name == "phi-3-mini"

            # New signature: with model_name
            engine2 = LocalInferenceEngine(
                cache_dir=temp_cache_dir, model_name="qwen3-4b"
            )
            assert engine2.model_name == "qwen3-4b"

    def test_clean_batch_still_works(
        self,
        mock_llama_class: Any,
        mock_grammar_class: Any,
        mock_cache_class: Any,
        mock_hf_download: Any,
        temp_cache_dir: Any,
        mock_model_path: Any,
    ) -> None:
        """Test that clean_batch still works with LocalInferenceEngine."""
        llm_output = {
            "choices": [{"text": '{"reasoning": "test", "value": 10.0, "unit": "kg"}'}]
        }
        mock_llama_class.create_completion = Mock(return_value=llm_output)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
            result = engine.clean_batch(["10kg"], "Extract weight")

            assert "10kg" in result
            assert result["10kg"] is not None
            assert result["10kg"]["value"] == 10.0
            assert result["10kg"]["unit"] == "kg"
