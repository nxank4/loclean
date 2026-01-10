"""Tests for grammar utilities."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from loclean.extraction.grammar_utils import get_grammar_from_schema


class SimpleProduct(BaseModel):
    """Simple product schema for testing."""

    name: str
    price: int
    color: str


def test_get_grammar_from_schema_success() -> None:
    """Test successful grammar generation from Pydantic schema."""
    with patch("llama_cpp.LlamaGrammar") as mock_grammar_class:
        mock_grammar = Mock()
        mock_grammar_class.from_json_schema.return_value = mock_grammar

        grammar = get_grammar_from_schema(SimpleProduct)

        assert grammar == mock_grammar
        mock_grammar_class.from_json_schema.assert_called_once()
        # Verify JSON schema was generated
        call_args = mock_grammar_class.from_json_schema.call_args[0][0]
        assert "properties" in call_args
        assert "name" in call_args["properties"]
        assert "price" in call_args["properties"]


def test_get_grammar_from_schema_invalid_type() -> None:
    """Test that non-BaseModel classes raise ValueError."""
    with pytest.raises(ValueError, match="must be a Pydantic BaseModel"):
        get_grammar_from_schema(str)  # type: ignore[arg-type]


def test_get_grammar_from_schema_caching() -> None:
    """Test that grammar generation is cached."""
    with patch("llama_cpp.LlamaGrammar") as mock_grammar_class:
        mock_grammar = Mock()
        mock_grammar_class.from_json_schema.return_value = mock_grammar

        # Call twice
        grammar1 = get_grammar_from_schema(SimpleProduct)
        grammar2 = get_grammar_from_schema(SimpleProduct)

        # Should be same instance (cached)
        assert grammar1 is grammar2
        # But should only call from_json_schema once (cached)
        assert mock_grammar_class.from_json_schema.call_count == 1
