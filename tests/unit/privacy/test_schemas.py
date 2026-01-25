"""Test cases for PII detection schemas."""

import pytest
from pydantic import ValidationError

from loclean.privacy.schemas import PIIDetectionResult, PIIEntity


class TestPIIEntity:
    """Test cases for PIIEntity class."""

    def test_entity_creation_with_all_fields(self) -> None:
        """Test entity creation with all fields."""
        entity = PIIEntity(type="phone", value="0909123456", start=0, end=10)

        assert entity.type == "phone"
        assert entity.value == "0909123456"
        assert entity.start == 0
        assert entity.end == 10

    def test_entity_with_different_pii_types(self) -> None:
        """Test entity with different PII types."""
        from typing import Literal

        PIIType = Literal[
            "person", "phone", "email", "credit_card", "address", "ip_address"
        ]
        types: list[PIIType] = [
            "person",
            "phone",
            "email",
            "credit_card",
            "address",
            "ip_address",
        ]

        for pii_type in types:
            entity = PIIEntity(type=pii_type, value="test", start=0, end=4)
            assert entity.type == pii_type

    def test_length_property_calculation(self) -> None:
        """Test length property calculation."""
        entity = PIIEntity(type="phone", value="0909123456", start=0, end=10)

        assert entity.length == 10
        assert entity.length == entity.end - entity.start

    def test_length_property_with_different_positions(self) -> None:
        """Test length property with different positions."""
        entity = PIIEntity(type="email", value="test@example.com", start=5, end=20)

        assert entity.length == 15
        assert entity.length == entity.end - entity.start

    def test_length_property_zero_length(self) -> None:
        """Test length property with zero length."""
        entity = PIIEntity(type="phone", value="", start=0, end=0)

        assert entity.length == 0

    def test_validation_error_invalid_type(self) -> None:
        """Test validation error for invalid type."""
        with pytest.raises(ValidationError):
            PIIEntity(type="invalid_type", value="test", start=0, end=4)  # type: ignore[arg-type]

    def test_validation_error_missing_fields(self) -> None:
        """Test validation error for missing required fields."""
        with pytest.raises(ValidationError):
            PIIEntity(type="phone")  # type: ignore[call-arg]

    def test_entity_serialization(self) -> None:
        """Test entity serialization to dict."""
        entity = PIIEntity(type="phone", value="0909123456", start=0, end=10)

        data = entity.model_dump()

        assert data["type"] == "phone"
        assert data["value"] == "0909123456"
        assert data["start"] == 0
        assert data["end"] == 10

    def test_entity_from_dict(self) -> None:
        """Test entity creation from dict."""
        data = {
            "type": "email",
            "value": "test@example.com",
            "start": 0,
            "end": 16,
        }

        entity = PIIEntity.model_validate(data)

        assert entity.type == "email"
        assert entity.value == "test@example.com"
        assert entity.start == 0
        assert entity.end == 16


class TestPIIDetectionResult:
    """Test cases for PIIDetectionResult class."""

    def test_result_creation_with_entities(self) -> None:
        """Test result creation with entities."""
        entities = [PIIEntity(type="phone", value="0909123456", start=0, end=10)]
        result = PIIDetectionResult(entities=entities)

        assert len(result.entities) == 1
        assert result.entities[0].type == "phone"
        assert result.reasoning is None

    def test_result_creation_with_reasoning(self) -> None:
        """Test result creation with reasoning."""
        entities = [PIIEntity(type="phone", value="0909123456", start=0, end=10)]
        result = PIIDetectionResult(entities=entities, reasoning="Found phone number")

        assert len(result.entities) == 1
        assert result.reasoning == "Found phone number"

    def test_result_creation_with_empty_entities(self) -> None:
        """Test result creation with empty entities."""
        result = PIIDetectionResult(entities=[])

        assert len(result.entities) == 0
        assert result.reasoning is None

    def test_result_creation_with_multiple_entities(self) -> None:
        """Test result creation with multiple entities."""
        entities = [
            PIIEntity(type="phone", value="0909123456", start=0, end=10),
            PIIEntity(type="email", value="test@example.com", start=20, end=36),
        ]
        result = PIIDetectionResult(entities=entities)

        assert len(result.entities) == 2
        assert result.entities[0].type == "phone"
        assert result.entities[1].type == "email"

    def test_result_reasoning_none_by_default(self) -> None:
        """Test result reasoning is None by default."""
        entities = [PIIEntity(type="phone", value="0909123456", start=0, end=10)]
        result = PIIDetectionResult(entities=entities)

        assert result.reasoning is None

    def test_result_validation_error_missing_entities(self) -> None:
        """Test validation error for missing entities."""
        with pytest.raises(ValidationError):
            PIIDetectionResult()  # type: ignore[call-arg]

    def test_result_serialization(self) -> None:
        """Test result serialization to dict."""
        entities = [PIIEntity(type="phone", value="0909123456", start=0, end=10)]
        result = PIIDetectionResult(entities=entities, reasoning="Found phone")

        data = result.model_dump()

        assert "entities" in data
        assert len(data["entities"]) == 1
        assert data["reasoning"] == "Found phone"

    def test_result_from_dict(self) -> None:
        """Test result creation from dict."""
        data = {
            "entities": [
                {
                    "type": "phone",
                    "value": "0909123456",
                    "start": 0,
                    "end": 10,
                }
            ],
            "reasoning": "Found phone number",
        }

        result = PIIDetectionResult.model_validate(data)

        assert len(result.entities) == 1
        assert result.entities[0].type == "phone"
        assert result.reasoning == "Found phone number"

    def test_result_with_none_reasoning(self) -> None:
        """Test result with None reasoning explicitly."""
        entities = [PIIEntity(type="phone", value="0909123456", start=0, end=10)]
        result = PIIDetectionResult(entities=entities, reasoning=None)

        assert result.reasoning is None
