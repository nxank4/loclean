"""Test cases for resource loading utilities."""

import pytest

from loclean.utils.resources import (
    get_grammar_preset,
    list_grammar_presets,
    list_grammars,
    list_templates,
    load_grammar,
    load_template,
)


class TestLoadGrammar:
    """Test cases for load_grammar function."""

    def test_load_json_grammar(self) -> None:
        """Test loading json.gbnf grammar file."""
        grammar = load_grammar("json.gbnf")

        assert isinstance(grammar, str)
        assert len(grammar) > 0
        assert "root" in grammar
        assert "object" in grammar
        assert "reasoning" in grammar
        assert "value" in grammar
        assert "unit" in grammar

    def test_load_nonexistent_grammar_raises_error(self) -> None:
        """Test that loading nonexistent grammar raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_grammar("nonexistent.gbnf")

        assert "nonexistent.gbnf" in str(exc_info.value)

    def test_grammar_content_structure(self) -> None:
        """Test that grammar has correct structure."""
        grammar = load_grammar("json.gbnf")

        # Should contain root rule
        assert "root" in grammar
        # Should contain object rule
        assert "object" in grammar
        # Should contain number rule
        assert "number" in grammar
        # Should contain string rule
        assert "string" in grammar
        # Should contain whitespace rule
        assert "ws" in grammar


class TestLoadTemplate:
    """Test cases for load_template function."""

    def test_load_nonexistent_template_raises_error(self) -> None:
        """Test that loading nonexistent template raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_template("nonexistent.j2")

        assert "nonexistent.j2" in str(exc_info.value)


class TestListResources:
    """Test cases for list functions."""

    def test_list_grammars(self) -> None:
        """Test listing available grammar files."""
        grammars = list_grammars()

        assert isinstance(grammars, list)
        assert "json.gbnf" in grammars
        assert all(g.endswith(".gbnf") for g in grammars)

    def test_list_templates(self) -> None:
        """Test listing available template files."""
        templates = list_templates()

        assert isinstance(templates, list)
        # Currently no templates, but function should work
        assert all(t.endswith(".j2") for t in templates) if templates else True


class TestGrammarPresets:
    """Test cases for Grammar Registry presets."""

    def test_list_grammar_presets(self) -> None:
        """Test listing available grammar presets."""
        presets = list_grammar_presets()

        assert isinstance(presets, list)
        assert len(presets) >= 3
        assert "json" in presets
        assert "list[str]" in presets
        assert "email" in presets

    def test_get_grammar_preset_json(self) -> None:
        """Test getting json preset grammar."""
        grammar = get_grammar_preset("json")

        assert isinstance(grammar, str)
        assert len(grammar) > 0
        assert "root" in grammar
        assert "value" in grammar
        assert "object" in grammar
        assert "array" in grammar

    def test_get_grammar_preset_list_str(self) -> None:
        """Test getting list[str] preset grammar."""
        grammar = get_grammar_preset("list[str]")

        assert isinstance(grammar, str)
        assert len(grammar) > 0
        assert "root" in grammar
        assert "array" in grammar
        assert "string" in grammar

    def test_get_grammar_preset_email(self) -> None:
        """Test getting email preset grammar."""
        grammar = get_grammar_preset("email")

        assert isinstance(grammar, str)
        assert len(grammar) > 0
        assert "root" in grammar
        assert "email" in grammar
        assert "local" in grammar
        assert "domain" in grammar

    def test_get_grammar_preset_nonexistent_raises_keyerror(self) -> None:
        """Test that getting nonexistent preset raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_grammar_preset("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "Grammar Registry" in str(exc_info.value)
