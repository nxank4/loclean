from pydantic import BaseModel, ConfigDict, Field


class ExtractionResult(BaseModel):
    """
    Structured output format for semantic data extraction.

    This model represents the result of extracting structured information
    from unstructured text, containing the reasoning, numeric value, and unit.

    Attributes:
        reasoning: Explanation of how the value and unit were extracted.
        value: The extracted numeric value.
        unit: The unit of measurement (e.g., "kg", "USD", "C").
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reasoning": "The input '5.5kg' contains value 5.5 and unit 'kg'",
                "value": 5.5,
                "unit": "kg",
            }
        }
    )

    reasoning: str = Field(
        ..., description="Explanation of how the value and unit were extracted from the input"
    )
    value: float = Field(..., description="The extracted numeric value")
    unit: str = Field(..., description="The unit of measurement (e.g., 'kg', 'USD', 'C', 'm')")
