"""Pydantic schemas for PII detection results."""

from typing import Literal

from pydantic import BaseModel

PIIType = Literal["person", "phone", "email", "credit_card", "address", "ip_address"]


class PIIEntity(BaseModel):
    """Represents a detected PII entity in text."""

    type: PIIType
    value: str  # LLM returns this, we find positions
    start: int  # Calculated by Python code
    end: int  # Calculated by Python code

    @property
    def length(self) -> int:
        """Length of entity value for overlap resolution."""
        return self.end - self.start


class PIIDetectionResult(BaseModel):
    """Result of PII detection for a single text item."""

    entities: list[PIIEntity]
    reasoning: str | None = None
