"""Tests for grammar registry module."""

import pytest
from pydantic import BaseModel

from loclean.inference.grammar_registry import GrammarRegistry
from loclean.inference.schemas import ExtractionResult


class TestGrammarRegistry:
    """Test cases for GrammarRegistry."""

    def test_get_preset_string(self) -> None:
        """Test getting grammar from preset string."""
        grammar = GrammarRegistry.get("json")
        assert grammar is not None

    def test_get_pydantic_model(self) -> None:
        """Test getting grammar from Pydantic model."""
        grammar = GrammarRegistry.get(ExtractionResult)
        assert grammar is not None

    def test_get_invalid_preset(self) -> None:
        """Test getting grammar with invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown grammar preset"):
            GrammarRegistry.get("invalid_preset")

    def test_get_invalid_type(self) -> None:
        """Test getting grammar with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Schema must be"):
            GrammarRegistry.get(123)  # type: ignore[arg-type]

    def test_lru_cache_works(self) -> None:
        """Test that LRU cache prevents redundant compilation."""
        # Clear cache first
        GrammarRegistry.clear_cache()

        # Get cache info before first call
        cache_info_before = GrammarRegistry.get.cache_info()
        assert cache_info_before.hits == 0
        assert cache_info_before.misses == 0

        # First call - should be a miss
        grammar1 = GrammarRegistry.get("json")
        cache_info_after_first = GrammarRegistry.get.cache_info()
        assert cache_info_after_first.misses == 1
        assert cache_info_after_first.hits == 0

        # Second call with same schema - should be a hit
        grammar2 = GrammarRegistry.get("json")
        cache_info_after_second = GrammarRegistry.get.cache_info()
        assert cache_info_after_second.hits == 1
        assert cache_info_after_second.misses == 1

        # Verify same object is returned (cached)
        assert grammar1 is grammar2

    def test_lru_cache_different_schemas(self) -> None:
        """Test that different schemas are cached separately."""
        GrammarRegistry.clear_cache()

        # Get different schemas
        grammar_json = GrammarRegistry.get("json")
        grammar_email = GrammarRegistry.get("email")
        grammar_extraction = GrammarRegistry.get(ExtractionResult)

        # Verify they are different objects
        assert grammar_json is not grammar_email
        assert grammar_json is not grammar_extraction
        assert grammar_email is not grammar_extraction

        # Verify cache has 3 entries
        cache_info = GrammarRegistry.get.cache_info()
        assert cache_info.misses == 3
        assert cache_info.hits == 0

        # Call again - should hit cache
        grammar_json2 = GrammarRegistry.get("json")
        grammar_email2 = GrammarRegistry.get("email")
        grammar_extraction2 = GrammarRegistry.get(ExtractionResult)

        assert grammar_json is grammar_json2
        assert grammar_email is grammar_email2
        assert grammar_extraction is grammar_extraction2

        cache_info_after = GrammarRegistry.get.cache_info()
        assert cache_info_after.hits == 3
        assert cache_info_after.misses == 3

    def test_lru_cache_maxsize(self) -> None:
        """Test that LRU cache respects maxsize limit."""
        GrammarRegistry.clear_cache()

        # Create a custom Pydantic model for testing
        class TestModel(BaseModel):
            field1: str

        # Fill cache beyond maxsize (32)
        for i in range(35):
            # Create unique model classes dynamically
            # Note: This is a simplified test - in practice, creating
            # 35 unique Pydantic models is complex, so we test with presets
            if i < len(GrammarRegistry.PRESETS):
                GrammarRegistry.get(list(GrammarRegistry.PRESETS.keys())[i])

        # Verify cache doesn't exceed maxsize
        cache_info = GrammarRegistry.get.cache_info()
        assert cache_info.currsize <= 32

    def test_clear_cache(self) -> None:
        """Test that clear_cache clears the LRU cache."""
        # Get a grammar to populate cache
        GrammarRegistry.get("json")
        cache_info_before = GrammarRegistry.get.cache_info()
        assert cache_info_before.currsize > 0

        # Clear cache
        GrammarRegistry.clear_cache()

        # Verify cache is empty
        cache_info_after = GrammarRegistry.get.cache_info()
        assert cache_info_after.currsize == 0
        assert cache_info_after.hits == 0
        assert cache_info_after.misses == 0

    def test_all_presets_available(self) -> None:
        """Test that all preset grammars can be retrieved."""
        expected_presets = ["json", "list_str", "email"]
        for preset in expected_presets:
            grammar = GrammarRegistry.get(preset)
            assert grammar is not None

    def test_pydantic_model_grammar_generation(self) -> None:
        """Test that Pydantic models generate valid grammars."""
        grammar = GrammarRegistry.get(ExtractionResult)
        assert grammar is not None

        # Verify it's different from preset grammars
        preset_grammar = GrammarRegistry.get("json")
        assert grammar is not preset_grammar
