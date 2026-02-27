"""Tests for loclean.utils.source_sanitizer module."""

from loclean.utils.source_sanitizer import sanitize_source


class TestStripMarkdownFences:
    """Markdown code fences should be removed."""

    def test_python_fences(self) -> None:
        source = "```python\ndef f():\n    return 1\n```"
        result = sanitize_source(source)
        assert "```" not in result
        assert "def f():" in result

    def test_triple_backtick_only(self) -> None:
        source = "```\ndef f():\n    return 1\n```"
        result = sanitize_source(source)
        assert "```" not in result
        assert "def f():" in result

    def test_no_fences_passthrough(self) -> None:
        source = "def f():\n    return 1"
        assert sanitize_source(source) == source


class TestStripProse:
    """Leading/trailing prose should be removed."""

    def test_leading_explanation(self) -> None:
        source = (
            "Here is the corrected function:\n\n"
            "def generate_features(row):\n"
            "    return {'a': row['x'] * 2}\n"
        )
        result = sanitize_source(source)
        assert result.startswith("def generate_features")

    def test_trailing_explanation(self) -> None:
        source = (
            "def f(row):\n"
            "    return {'a': 1}\n\n"
            "This function computes a simple feature by multiplying the value."
        )
        result = sanitize_source(source)
        assert "This function computes" not in result
        assert "def f(row):" in result

    def test_import_preserved(self) -> None:
        source = "import math\n\ndef f(row):\n    return {'a': math.log(1)}"
        result = sanitize_source(source)
        assert result.startswith("import math")


class TestFixUnicodeOperators:
    """Non-ASCII math operators should be replaced."""

    def test_multiplication(self) -> None:
        source = "def f(row):\n    return row['a'] \u00d7 row['b']"
        result = sanitize_source(source)
        assert "\u00d7" not in result
        assert "row['a'] * row['b']" in result

    def test_division(self) -> None:
        source = "def f(row):\n    return row['a'] \u00f7 row['b']"
        result = sanitize_source(source)
        assert "/" in result

    def test_minus_sign(self) -> None:
        source = "def f(row):\n    return row['a'] \u2212 row['b']"
        result = sanitize_source(source)
        assert "\u2212" not in result
        assert "-" in result

    def test_smart_quotes(self) -> None:
        source = "def f(row):\n    return row[\u2018name\u2019]"
        result = sanitize_source(source)
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_comparison_operators(self) -> None:
        source = "def f(x):\n    return x \u2264 10"
        result = sanitize_source(source)
        assert "<=" in result


class TestFixNumericLiterals:
    """Invalid numeric literals should be fixed."""

    def test_invalid_binary_digit(self) -> None:
        source = "def f():\n    x = 0b2\n    return x"
        result = sanitize_source(source)
        assert "0b2" not in result
        assert "2" in result

    def test_trailing_currency(self) -> None:
        source = "def f():\n    return 100$"
        result = sanitize_source(source)
        assert "$" not in result
        assert "100" in result

    def test_valid_binary_untouched(self) -> None:
        source = "def f():\n    return 0b101"
        result = sanitize_source(source)
        assert "0b101" in result


class TestStripBackticks:
    """Stray inline backticks should be removed."""

    def test_wrapped_expression(self) -> None:
        source = "def f(row):\n    return `math.log(row['x'])`"
        result = sanitize_source(source)
        assert "`" not in result
        assert "math.log(row['x'])" in result

    def test_no_backticks_passthrough(self) -> None:
        source = "def f():\n    return 42"
        assert sanitize_source(source) == source


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_complete_cleanup(self) -> None:
        source = (
            "Here is your function:\n\n"
            "```python\n"
            "import math\n\n"
            "def generate_features(row):\n"
            "    result = {}\n"
            "    result[\u2018log_price\u2019] = math.log(row[\u2018price\u2019])\n"
            "    result['ratio'] = row['a'] \u00d7 row['b']\n"
            "    return result\n"
            "```\n\n"
            "This function generates two features."
        )
        result = sanitize_source(source)

        assert "```" not in result
        assert "Here is your function" not in result
        assert "This function generates" not in result
        assert "\u2018" not in result
        assert "\u00d7" not in result
        assert "import math" in result
        assert "def generate_features(row):" in result
        assert "math.log" in result
        assert "* row['b']" in result
