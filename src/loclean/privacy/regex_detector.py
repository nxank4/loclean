"""Fast regex-based PII detection for structured data types."""

import ipaddress
import re
from typing import List

from loclean.privacy.schemas import PIIEntity

# Email pattern (RFC 5322 compliant, simplified)
EMAIL_PATTERN = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

# Vietnamese phone patterns
# Formats: 0909123456, 0912345678, +84901234567, 84901234567
PHONE_PATTERN = r"\b(?:\+84|84|0)[3-9]\d{8,9}\b"

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
        Detect Vietnamese phone numbers in text.

        Supports formats: 0909123456, 0912345678, +84901234567, 84901234567

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
