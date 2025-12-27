import pytest
from pydantic import ValidationError

from loclean.inference.schemas import ExtractionResult


class TestExtractionResult:
    """Test cases for ExtractionResult Pydantic model."""

    def test_valid_creation(self) -> None:
        """Test creating ExtractionResult with valid data."""
        result = ExtractionResult(
            reasoning="Extracted 5.5kg from input string", value=5.5, unit="kg"
        )
        assert result.reasoning == "Extracted 5.5kg from input string"
        assert result.value == 5.5
        assert result.unit == "kg"

    def test_valid_creation_with_negative_value(self) -> None:
        """Test creating ExtractionResult with negative value."""
        result = ExtractionResult(
            reasoning="Temperature is -10 degrees", value=-10.0, unit="C"
        )
        assert result.value == -10.0
        assert result.unit == "C"

    def test_valid_creation_with_zero_value(self) -> None:
        """Test creating ExtractionResult with zero value."""
        result = ExtractionResult(
            reasoning="Zero value extracted", value=0.0, unit="USD"
        )
        assert result.value == 0.0

    def test_valid_creation_with_decimal_value(self) -> None:
        """Test creating ExtractionResult with decimal value."""
        result = ExtractionResult(
            reasoning="Decimal value extracted", value=3.14159, unit="m"
        )
        assert result.value == 3.14159

    def test_missing_reasoning(self) -> None:
        """Test that missing reasoning field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(value=5.5, unit="kg")  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("reasoning",) for error in errors)

    def test_missing_value(self) -> None:
        """Test that missing value field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(reasoning="Test", unit="kg")  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("value",) for error in errors)

    def test_missing_unit(self) -> None:
        """Test that missing unit field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(reasoning="Test", value=5.5)  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("unit",) for error in errors)

    def test_invalid_value_type(self) -> None:
        """Test that non-numeric value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(reasoning="Test", value="not_a_number", unit="kg")  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("value",) for error in errors)

    def test_invalid_reasoning_type(self) -> None:
        """Test that non-string reasoning raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(reasoning=123, value=5.5, unit="kg")  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("reasoning",) for error in errors)

    def test_invalid_unit_type(self) -> None:
        """Test that non-string unit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractionResult(reasoning="Test", value=5.5, unit=123)  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("unit",) for error in errors)

    def test_empty_reasoning(self) -> None:
        """Test that empty reasoning string is allowed."""
        result = ExtractionResult(reasoning="", value=5.5, unit="kg")
        assert result.reasoning == ""

    def test_empty_unit(self) -> None:
        """Test that empty unit string is allowed."""
        result = ExtractionResult(reasoning="Test", value=5.5, unit="")
        assert result.unit == ""

    def test_json_serialization(self) -> None:
        """Test that ExtractionResult can be serialized to JSON."""
        result = ExtractionResult(reasoning="Test reasoning", value=5.5, unit="kg")
        json_dict = result.model_dump()
        assert json_dict == {"reasoning": "Test reasoning", "value": 5.5, "unit": "kg"}

    def test_json_deserialization(self) -> None:
        """Test that ExtractionResult can be created from JSON dict."""
        json_dict = {"reasoning": "Test reasoning", "value": 5.5, "unit": "kg"}
        result = ExtractionResult(**json_dict)  # type: ignore[arg-type]
        assert result.reasoning == "Test reasoning"
        assert result.value == 5.5
        assert result.unit == "kg"

    def test_model_dump_json(self) -> None:
        """Test model_dump_json method."""
        result = ExtractionResult(reasoning="Test reasoning", value=5.5, unit="kg")
        json_str = result.model_dump_json()
        assert '"reasoning":"Test reasoning"' in json_str
        assert '"value":5.5' in json_str
        assert '"unit":"kg"' in json_str

    def test_model_validate(self) -> None:
        """Test model_validate class method."""
        data = {"reasoning": "Validated from dict", "value": 10.0, "unit": "USD"}
        result = ExtractionResult.model_validate(data)
        assert isinstance(result, ExtractionResult)
        assert result.value == 10.0

    def test_model_validate_json(self) -> None:
        """Test model_validate_json class method."""
        json_str = '{"reasoning":"From JSON","value":7.5,"unit":"m"}'
        result = ExtractionResult.model_validate_json(json_str)
        assert isinstance(result, ExtractionResult)
        assert result.value == 7.5
        assert result.unit == "m"

    def test_int_value_converted_to_float(self) -> None:
        """Test that integer values are automatically converted to float."""
        result = ExtractionResult(
            reasoning="Integer value",
            value=10,  # int instead of float
            unit="kg",
        )
        assert isinstance(result.value, float)
        assert result.value == 10.0

    def test_very_large_value(self) -> None:
        """Test that very large values are handled correctly."""
        result = ExtractionResult(reasoning="Large value", value=1e10, unit="km")
        assert result.value == 1e10

    def test_very_small_value(self) -> None:
        """Test that very small values are handled correctly."""
        result = ExtractionResult(reasoning="Small value", value=1e-10, unit="mm")
        assert result.value == 1e-10

    def test_unicode_in_reasoning(self) -> None:
        """Test that unicode characters in reasoning are handled correctly."""
        result = ExtractionResult(
            reasoning="Test với unicode: 测试", value=5.5, unit="kg"
        )
        assert "测试" in result.reasoning

    def test_unicode_in_unit(self) -> None:
        """Test that unicode characters in unit are handled correctly."""
        result = ExtractionResult(reasoning="Test", value=5.5, unit="°C")
        assert result.unit == "°C"

    def test_special_characters_in_unit(self) -> None:
        """Test that special characters in unit are handled correctly."""
        result = ExtractionResult(reasoning="Test", value=5.5, unit="$/kg")
        assert result.unit == "$/kg"
