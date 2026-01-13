"""Fast regex-based PII detection for structured data types."""

import ipaddress
import re
from typing import List

from loclean.privacy.schemas import PIIEntity

# Email pattern (RFC 5322 compliant, simplified)
EMAIL_PATTERN = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

# International phone patterns
# Supports multiple formats:
# - International: +1-555-123-4567, +44 20 7946 0958, +33 1 23 45 67 89
# - US/Canada: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567
# - Vietnamese: 0909123456, +84901234567, 84901234567
# - UK: 020 7946 0958, 07946 095 123
# Pattern requires minimum 7 digits to avoid false positives (dates, short numbers)
PHONE_PATTERN = (
    r"(?<!\d)(?:"  # Negative lookbehind to avoid matching mid-number
    # International: +44 20 7946 0958, +33 1 23 45 67 89
    r"(?:\+\d{1,4}[\s.-]?\d{1,4}(?:[\s.-]?\d{1,4}){1,})(?!\d)"
    r"|"  # OR
    # US/Canada: (555) 123-4567, 555-123-4567
    r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    r"|"  # OR
    r"\d{3}[\s.-]?\d{4}(?!\d)"  # Short US: 555-1234 (7 digits minimum)
    r"|"  # OR
    r"(?:\+84|84|0)[3-9]\d{8,9}(?!\d)"  # Vietnamese: 0909123456, +84901234567
    r"|"  # OR
    r"0\d{2,3}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"  # UK: 020 7946 0958
    r"|"  # OR
    r"\d{10,}(?!\d)"  # Long numbers (10+ digits) - likely phone numbers
    r")"
)

# Credit card patterns (Visa, MasterCard, Amex)
# Visa: 13-16 digits starting with 4
# MasterCard: 16 digits starting with 5
# Amex: 15 digits starting with 34 or 37
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"

# IPv4 pattern
IPV4_PATTERN = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

# IPv6 pattern (RFC 4291 - simplified)
IPV6_PATTERN = r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|::1|::"


class RegexDetector:
    """Fast regex-based detector for structured PII types."""

    @staticmethod
    def detect_email(text: str) -> List[PIIEntity]:
        """
        Detect email addresses in text.

        Args:
            text: Input text to scan

        Returns:
            List of detected email entities
        """
        entities: List[PIIEntity] = []
        for match in re.finditer(EMAIL_PATTERN, text):
            entities.append(
                PIIEntity(
                    type="email",
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
        return entities

    @staticmethod
    def detect_phone(text: str) -> List[PIIEntity]:
        """
        Detect international phone numbers in text.

        Supports multiple formats:
        - International: +1-555-123-4567, +44 20 7946 0958, +33 1 23 45 67 89
        - US/Canada: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567
        - Vietnamese: 0909123456, +84901234567, 84901234567
        - UK: 020 7946 0958, 07946 095 123
        - General formats with country codes and various separators

        Args:
            text: Input text to scan

        Returns:
            List of detected phone entities
        """
        entities: List[PIIEntity] = []
        for match in re.finditer(PHONE_PATTERN, text):
            entities.append(
                PIIEntity(
                    type="phone",
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
        return entities

    @staticmethod
    def detect_credit_card(text: str) -> List[PIIEntity]:
        """
        Detect credit card numbers in text.

        Supports Visa, MasterCard, and Amex formats.

        Args:
            text: Input text to scan

        Returns:
            List of detected credit card entities
        """
        entities: List[PIIEntity] = []
        for match in re.finditer(CREDIT_CARD_PATTERN, text):
            # Remove separators for validation
            card_number = re.sub(r"[-\s]", "", match.group())
            # Basic validation: check length and starting digits
            if len(card_number) >= 13 and len(card_number) <= 19:
                entities.append(
                    PIIEntity(
                        type="credit_card",
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return entities

    @staticmethod
    def detect_ip_address(text: str) -> List[PIIEntity]:
        """
        Detect IP addresses (IPv4 and IPv6) in text.

        Uses Python's ipaddress library for validation.

        Args:
            text: Input text to scan

        Returns:
            List of detected IP address entities
        """
        entities: List[PIIEntity] = []

        # Detect and validate IPv4
        for match in re.finditer(IPV4_PATTERN, text):
            try:
                ipaddress.IPv4Address(match.group())
                entities.append(
                    PIIEntity(
                        type="ip_address",
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
            except ValueError:
                continue  # Invalid IP, skip

        # Detect and validate IPv6
        for match in re.finditer(IPV6_PATTERN, text):
            try:
                ipaddress.IPv6Address(match.group())
                entities.append(
                    PIIEntity(
                        type="ip_address",
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
            except ValueError:
                continue  # Invalid IP, skip

        return entities
