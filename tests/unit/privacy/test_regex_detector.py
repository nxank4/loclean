"""Tests for regex-based PII detection."""

from loclean.privacy.regex_detector import RegexDetector


class TestRegexDetector:
    """Test cases for RegexDetector."""

    def test_detect_email(self) -> None:
        """Test email detection."""
        detector = RegexDetector()
        text = "Contact us at support@example.com or info@test.org"
        entities = detector.detect_email(text)

        assert len(entities) == 2
        assert entities[0].type == "email"
        assert entities[0].value == "support@example.com"
        assert entities[1].value == "info@test.org"

    def test_detect_phone_vietnamese(self) -> None:
        """Test Vietnamese phone number detection."""
        detector = RegexDetector()
        text = "Call 0909123456 or +84901234567 or 84912345678"
        entities = detector.detect_phone(text)

        assert len(entities) >= 2
        assert all(e.type == "phone" for e in entities)
        assert any("0909123456" in e.value for e in entities)

    def test_detect_credit_card(self) -> None:
        """Test credit card detection."""
        detector = RegexDetector()
        text = "Card: 4532-1234-5678-9010"
        entities = detector.detect_credit_card(text)

        assert len(entities) >= 1
        assert entities[0].type == "credit_card"

    def test_detect_ip_address_ipv4(self) -> None:
        """Test IPv4 address detection and validation."""
        detector = RegexDetector()
        text = "Server at 192.168.1.1 and 10.0.0.1"
        entities = detector.detect_ip_address(text)

        assert len(entities) == 2
        assert all(e.type == "ip_address" for e in entities)
        assert any("192.168.1.1" in e.value for e in entities)

    def test_detect_ip_address_invalid(self) -> None:
        """Test that invalid IP addresses are filtered out."""
        detector = RegexDetector()
        text = "Invalid: 999.999.999.999"
        entities = detector.detect_ip_address(text)

        # Invalid IP should be filtered out
        assert len(entities) == 0

    def test_detect_no_pii(self) -> None:
        """Test detection when no PII is present."""
        detector = RegexDetector()
        text = "This is just regular text with no sensitive information."
        email_entities = detector.detect_email(text)
        phone_entities = detector.detect_phone(text)

        assert len(email_entities) == 0
        assert len(phone_entities) == 0
