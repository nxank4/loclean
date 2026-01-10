"""Tests for hybrid PII detector with overlap resolution."""

from loclean.privacy.detector import PIIDetector, resolve_overlaps
from loclean.privacy.schemas import PIIEntity


class TestResolveOverlaps:
    """Test cases for overlap resolution."""

    def test_no_overlaps(self) -> None:
        """Test when entities don't overlap."""
        entities = [
            PIIEntity(type="phone", value="0909123456", start=0, end=10),
            PIIEntity(type="email", value="test@example.com", start=20, end=35),
        ]
        resolved = resolve_overlaps(entities)

        assert len(resolved) == 2
        assert resolved[0].type == "phone"
        assert resolved[1].type == "email"

    def test_overlap_longer_wins(self) -> None:
        """Test that longer entity wins when overlapping."""
        # "nam" overlaps with "nam@gmail.com"
        entities = [
            PIIEntity(type="person", value="nam", start=18, end=21),
            PIIEntity(type="email", value="nam@gmail.com", start=18, end=31),
        ]
        resolved = resolve_overlaps(entities)

        # Email (longer) should win
        assert len(resolved) == 1
        assert resolved[0].type == "email"
        assert resolved[0].value == "nam@gmail.com"

    def test_overlap_same_length_first_wins(self) -> None:
        """Test that first entity wins when same length."""
        entities = [
            PIIEntity(type="phone", value="123", start=0, end=3),
            PIIEntity(type="email", value="456", start=0, end=3),
        ]
        resolved = resolve_overlaps(entities)

        # First one should win (phone)
        assert len(resolved) == 1
        assert resolved[0].type == "phone"

    def test_empty_list(self) -> None:
        """Test with empty entity list."""
        resolved = resolve_overlaps([])
        assert len(resolved) == 0


class TestPIIDetector:
    """Test cases for PIIDetector."""

    def test_detect_regex_only(self) -> None:
        """Test detection with regex strategies only."""
        detector = PIIDetector(inference_engine=None)
        text = "Contact 0909123456 or support@example.com"
        entities = detector.detect(text, strategies=["phone", "email"])

        assert len(entities) >= 2
        types = {e.type for e in entities}
        assert "phone" in types
        assert "email" in types

    def test_detect_no_llm_engine(self) -> None:
        """Test that LLM strategies are skipped when no engine provided."""
        detector = PIIDetector(inference_engine=None)
        text = "Contact anh Nam"
        entities = detector.detect(text, strategies=["person"])

        # Should return empty since no LLM engine
        assert len(entities) == 0

    def test_detect_mixed_strategies(self) -> None:
        """Test detection with both regex and LLM strategies."""
        detector = PIIDetector(inference_engine=None)
        text = "Contact 0909123456 or anh Nam"
        entities = detector.detect(text, strategies=["phone", "person"])

        # Should detect phone (regex) but not person (needs LLM)
        phone_entities = [e for e in entities if e.type == "phone"]
        assert len(phone_entities) >= 1
