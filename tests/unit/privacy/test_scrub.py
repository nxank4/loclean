"""Tests for scrub functions."""

import pytest

from loclean.privacy.scrub import replace_entities, scrub_string
from loclean.privacy.schemas import PIIEntity


class TestReplaceEntities:
    """Test cases for replace_entities function."""

    def test_mask_mode(self) -> None:
        """Test masking mode."""
        text = "Contact 0909123456"
        entities = [
            PIIEntity(type="phone", value="0909123456", start=8, end=18),
        ]
        result = replace_entities(text, entities, mode="mask")

        assert "[PHONE]" in result
        assert "0909123456" not in result

    def test_fake_mode_requires_faker(self) -> None:
        """Test that fake mode raises error if faker not installed."""
        text = "Contact 0909123456"
        entities = [
            PIIEntity(type="phone", value="0909123456", start=8, end=18),
        ]

        try:
            from faker import Faker  # noqa: F401

            # Faker is installed, should work
            result = replace_entities(text, entities, mode="fake", locale="vi_VN")
            assert result != text
            assert "0909123456" not in result
        except ImportError:
            # Faker not installed, should raise error
            with pytest.raises(ImportError):
                replace_entities(text, entities, mode="fake", locale="vi_VN")

    def test_multiple_entities(self) -> None:
        """Test replacement with multiple entities."""
        text = "Contact 0909123456 or support@example.com"
        entities = [
            PIIEntity(type="phone", value="0909123456", start=8, end=18),
            PIIEntity(type="email", value="support@example.com", start=22, end=41),
        ]
        result = replace_entities(text, entities, mode="mask")

        assert "[PHONE]" in result
        assert "[EMAIL]" in result
        assert "0909123456" not in result
        assert "support@example.com" not in result

    def test_no_entities(self) -> None:
        """Test with no entities."""
        text = "Just regular text"
        result = replace_entities(text, [], mode="mask")

        assert result == text


class TestScrubString:
    """Test cases for scrub_string function."""

    def test_scrub_string_mask_mode(self) -> None:
        """Test scrubbing string in mask mode."""
        text = "Contact 0909123456"
        result = scrub_string(text, strategies=["phone"], mode="mask")

        assert "[PHONE]" in result
        assert "0909123456" not in result

    def test_scrub_string_empty(self) -> None:
        """Test with empty string."""
        result = scrub_string("", strategies=["phone"])
        assert result == ""

    def test_scrub_string_no_pii(self) -> None:
        """Test with text containing no PII."""
        text = "Just regular text"
        result = scrub_string(text, strategies=["phone", "email"])

        assert result == text

