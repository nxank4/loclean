"""Test cases for inference engine factory."""

from unittest.mock import Mock, patch

import pytest

from semantix.inference.base import InferenceEngine
from semantix.inference.config import EngineConfig
from semantix.inference.factory import create_engine
from semantix.inference.manager import LlamaCppEngine


@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "test_cache"


@pytest.fixture
def mock_llama():
    mock_llama_instance = Mock()
    mock_llama_instance.create_completion = Mock()
    return mock_llama_instance


@pytest.fixture
def mock_llama_class(mock_llama):
    with patch("semantix.inference.manager.Llama", return_value=mock_llama):
        yield mock_llama


@pytest.fixture
def mock_grammar_class():
    with patch("semantix.inference.manager.LlamaGrammar"):
        with patch("semantix.inference.manager.LlamaGrammar.from_string"):
            yield


@pytest.fixture
def mock_cache_class():
    with patch("semantix.cache.SemantixCache"):
        yield


@pytest.fixture
def mock_model_path(temp_cache_dir):
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_cache_dir / "test_model.gguf"
    model_path.touch()
    return model_path


@pytest.fixture
def mock_hf_download(mock_model_path):
    with patch("semantix.inference.manager.hf_hub_download", return_value=str(mock_model_path)):
        yield


class TestCreateEngine:
    """Test cases for create_engine factory function."""

    def test_create_llama_cpp_engine_with_defaults(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test creating LlamaCppEngine with default config."""
        config = EngineConfig(cache_dir=temp_cache_dir)
        engine = create_engine(config)

        assert isinstance(engine, LlamaCppEngine)
        assert isinstance(engine, InferenceEngine)
        # Note: config.model might fallback to "phi-3-mini" if not in registry
        assert engine.model_name in ["phi-3-mini", config.model]
        assert engine.cache_dir == temp_cache_dir

    def test_create_llama_cpp_engine_with_custom_model(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test creating LlamaCppEngine with custom model name."""
        config = EngineConfig(
            engine="llama-cpp",
            model="qwen3-4b",
            cache_dir=temp_cache_dir,
        )
        with patch("pathlib.Path.exists", return_value=True):
            engine = create_engine(config)

            assert isinstance(engine, LlamaCppEngine)
            assert engine.model_name == "qwen3-4b"

    def test_create_llama_cpp_engine_with_custom_params(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test creating LlamaCppEngine with custom n_ctx and n_gpu_layers."""
        config = EngineConfig(
            engine="llama-cpp",
            cache_dir=temp_cache_dir,
            n_ctx=8192,
            n_gpu_layers=10,
        )

        with patch("semantix.inference.manager.Llama") as mock_llama:
            mock_llama_instance = Mock()
            mock_llama.return_value = mock_llama_instance

            engine = create_engine(config)

            assert isinstance(engine, LlamaCppEngine)
            mock_llama.assert_called_once_with(
                model_path=str(mock_model_path),
                n_ctx=8192,
                n_gpu_layers=10,
                verbose=False,
            )

    def test_create_openai_engine_raises_not_implemented(self):
        """Test that creating OpenAI engine raises NotImplementedError."""
        config = EngineConfig(engine="openai", model="gpt-4o", api_key="sk-test")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "OpenAI engine is not yet implemented" in str(exc_info.value)
        assert "llama-cpp" in str(exc_info.value)

    def test_create_anthropic_engine_raises_not_implemented(self):
        """Test that creating Anthropic engine raises NotImplementedError."""
        config = EngineConfig(engine="anthropic", model="claude-3", api_key="sk-test")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "Anthropic engine is not yet implemented" in str(exc_info.value)
        assert "llama-cpp" in str(exc_info.value)

    def test_create_gemini_engine_raises_not_implemented(self):
        """Test that creating Gemini engine raises NotImplementedError."""
        config = EngineConfig(engine="gemini", model="gemini-pro", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "Gemini engine is not yet implemented" in str(exc_info.value)
        assert "llama-cpp" in str(exc_info.value)

    def test_create_engine_passes_all_config_params(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test that all config parameters are passed to engine."""
        config = EngineConfig(
            engine="llama-cpp",
            model="phi-3-mini",
            cache_dir=temp_cache_dir,
            n_ctx=2048,
            n_gpu_layers=5,
        )

        with patch("semantix.inference.manager.LlamaCppEngine") as mock_engine_class:
            mock_engine_instance = Mock()
            mock_engine_class.return_value = mock_engine_instance

            create_engine(config)

            mock_engine_class.assert_called_once_with(
                model_name="phi-3-mini",
                cache_dir=temp_cache_dir,
                n_ctx=2048,
                n_gpu_layers=5,
            )

    def test_create_engine_lazy_loading(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test that LlamaCppEngine is lazily imported."""
        config = EngineConfig(cache_dir=temp_cache_dir)

        # Verify that LlamaCppEngine is not imported at module level
        import semantix.inference.factory as factory_module

        # LlamaCppEngine should not be in factory module's namespace
        assert "LlamaCppEngine" not in dir(factory_module)

        # But should be importable when create_engine is called
        with patch("semantix.inference.manager.Llama"):
            engine = create_engine(config)
            assert isinstance(engine, LlamaCppEngine)

    def test_create_engine_with_different_models(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test creating engines with different model names."""
        models = ["phi-3-mini", "qwen3-4b", "gemma-3-4b", "deepseek-r1"]

        with patch("pathlib.Path.exists", return_value=True):
            for model_name in models:
                config = EngineConfig(
                    engine="llama-cpp",
                    model=model_name,
                    cache_dir=temp_cache_dir,
                )
                engine = create_engine(config)

                assert isinstance(engine, LlamaCppEngine)
                assert engine.model_name == model_name

    def test_create_engine_logs_info(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test that create_engine logs info message."""
        config = EngineConfig(
            engine="llama-cpp",
            model="phi-3-mini",
            cache_dir=temp_cache_dir,
        )

        with patch("semantix.inference.factory.logger") as mock_logger:
            create_engine(config)

            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "Creating LlamaCppEngine" in call_args
            assert "phi-3-mini" in call_args

    def test_create_engine_returns_inference_engine_instance(
        self,
        temp_cache_dir,
        mock_model_path,
        mock_llama_class,
        mock_grammar_class,
        mock_cache_class,
        mock_hf_download,
    ):
        """Test that create_engine returns InferenceEngine instance."""
        config = EngineConfig(cache_dir=temp_cache_dir)
        engine = create_engine(config)

        assert isinstance(engine, InferenceEngine)
        assert hasattr(engine, "clean_batch")
        assert callable(engine.clean_batch)
