"""Test cases for inference engine factory."""

from unittest.mock import Mock, patch

import pytest

from loclean.inference.base import InferenceEngine
from loclean.inference.config import EngineConfig
from loclean.inference.factory import create_engine


class TestCreateEngine:
    """Test cases for create_engine factory function."""

    @patch("loclean.inference.ollama_engine.OllamaEngine")
    def test_create_ollama_engine_with_defaults(self, mock_cls: Mock) -> None:
        """Test creating OllamaEngine with default config."""
        mock_engine = Mock(spec=InferenceEngine)
        mock_cls.return_value = mock_engine

        config = EngineConfig()
        engine = create_engine(config)

        assert engine is mock_engine
        mock_cls.assert_called_once_with(
            model="phi3",
            host="http://localhost:11434",
            verbose=False,
        )

    @patch("loclean.inference.ollama_engine.OllamaEngine")
    def test_create_ollama_engine_with_custom_model(self, mock_cls: Mock) -> None:
        """Test creating OllamaEngine with a custom model."""
        mock_engine = Mock(spec=InferenceEngine)
        mock_cls.return_value = mock_engine

        config = EngineConfig(engine="ollama", model="llama3")
        engine = create_engine(config)

        assert engine is mock_engine
        mock_cls.assert_called_once_with(
            model="llama3",
            host="http://localhost:11434",
            verbose=False,
        )

    @patch("loclean.inference.ollama_engine.OllamaEngine")
    def test_create_ollama_engine_with_custom_host(self, mock_cls: Mock) -> None:
        """Test creating OllamaEngine with custom host."""
        mock_engine = Mock(spec=InferenceEngine)
        mock_cls.return_value = mock_engine

        config = EngineConfig(host="http://remote:8080")
        engine = create_engine(config)

        mock_cls.assert_called_once_with(
            model="phi3",
            host="http://remote:8080",
            verbose=False,
        )
        assert engine is mock_engine

    def test_create_openai_engine_raises_not_implemented(self) -> None:
        """Test that creating OpenAI engine raises NotImplementedError."""
        config = EngineConfig(engine="openai", model="gpt-4o", api_key="sk-test")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "OpenAI engine is not yet implemented" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    def test_create_anthropic_engine_raises_not_implemented(self) -> None:
        """Test that creating Anthropic engine raises NotImplementedError."""
        config = EngineConfig(engine="anthropic", model="claude-3", api_key="sk-test")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "Anthropic engine is not yet implemented" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    def test_create_gemini_engine_raises_not_implemented(self) -> None:
        """Test that creating Gemini engine raises NotImplementedError."""
        config = EngineConfig(engine="gemini", model="gemini-pro", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            create_engine(config)

        assert "Gemini engine is not yet implemented" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    @patch("loclean.inference.ollama_engine.OllamaEngine")
    def test_create_engine_logs_info(self, mock_cls: Mock) -> None:
        """Test that create_engine logs info message."""
        mock_cls.return_value = Mock(spec=InferenceEngine)
        config = EngineConfig()

        with patch("loclean.inference.factory.logger") as mock_logger:
            create_engine(config)

            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "OllamaEngine" in call_args
            assert "phi3" in call_args

    @patch("loclean.inference.ollama_engine.OllamaEngine")
    def test_create_engine_returns_inference_engine(self, mock_cls: Mock) -> None:
        """Test that create_engine returns InferenceEngine instance."""
        mock_engine = Mock(spec=InferenceEngine)
        mock_engine.clean_batch = Mock()
        mock_engine.generate = Mock()
        mock_cls.return_value = mock_engine

        config = EngineConfig()
        engine = create_engine(config)

        assert hasattr(engine, "clean_batch")
        assert hasattr(engine, "generate")
        assert callable(engine.clean_batch)
        assert callable(engine.generate)
