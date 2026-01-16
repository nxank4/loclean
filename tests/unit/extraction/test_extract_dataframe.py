from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

from loclean.extraction.extract_dataframe import extract_dataframe
from loclean.extraction.extractor import Extractor


class Product(BaseModel):
    name: str
    price: int


class ComplexSchema(BaseModel):
    is_valid: bool
    score: float
    tags: list[str]


@pytest.fixture
def mock_extractor() -> Any:
    extractor = MagicMock(spec=Extractor)
    # Default behavior: return a dict mapping input string to a Product model
    extractor.extract_batch.side_effect = lambda values, schema, instruction: {
        val: Product(name=f"Clean {val}", price=100) for val in values
    }
    return extractor


def test_polars_output_dict(mock_extractor: Any) -> None:
    df = pl.DataFrame({"raw": ["item1", "item2"]})

    result = extract_dataframe(
        df,
        target_col="raw",
        schema=Product,
        extractor=mock_extractor,
        output_type="dict",
    )

    assert "raw_extracted" in result.columns
    assert result["raw_extracted"].dtype == pl.Struct

    # Check values
    first_row = result.row(0, named=True)
    assert first_row["raw_extracted"]["name"] == "Clean item1"
    assert first_row["raw_extracted"]["price"] == 100


def test_polars_output_pydantic(mock_extractor: Any) -> None:
    df = pl.DataFrame({"raw": ["item1"]})

    result = extract_dataframe(
        df,
        target_col="raw",
        schema=Product,
        extractor=mock_extractor,
        output_type="pydantic",
    )

    assert "raw_extracted" in result.columns
    # When using pydantic output type with Polars, it stores as Object because
    # values are Python objects
    assert result["raw_extracted"].dtype == pl.Object

    val = result["raw_extracted"][0]
    # Should be the actual Pydantic model instance
    assert isinstance(val, Product)
    assert val.name == "Clean item1"


def test_pandas_output_dict(mock_extractor: Any) -> None:
    df = pd.DataFrame({"raw": ["item1"]})

    result = extract_dataframe(
        df,
        target_col="raw",
        schema=Product,
        extractor=mock_extractor,
        output_type="dict",
    )

    assert "raw_extracted" in result.columns
    # Pandas stores struct/dict as object usually
    val = result["raw_extracted"].iloc[0]
    assert isinstance(val, dict)
    assert val["name"] == "Clean item1"


def test_pandas_output_pydantic(mock_extractor: Any) -> None:
    df = pd.DataFrame({"raw": ["item1"]})

    result = extract_dataframe(
        df,
        target_col="raw",
        schema=Product,
        extractor=mock_extractor,
        output_type="pydantic",
    )

    assert "raw_extracted" in result.columns
    val = result["raw_extracted"].iloc[0]
    # Should be the actual Pydantic model instance
    assert isinstance(val, Product)
    assert val.name == "Clean item1"


def test_missing_column(mock_extractor: Any) -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Column 'b' not found"):
        extract_dataframe(df, "b", Product, extractor=mock_extractor)


def test_empty_dataframe(mock_extractor: Any) -> None:
    df = pl.DataFrame({"raw": []}, schema={"raw": pl.Utf8})
    result = extract_dataframe(df, "raw", Product, extractor=mock_extractor)

    # Should return original dataframe if no values to process
    assert result.shape == (0, 1)
    assert "raw_extracted" not in result.columns


def test_no_valid_values(mock_extractor: Any) -> None:
    # Dataframe with None and empty strings
    df = pl.DataFrame({"raw": [None, "", "   "]})
    result = extract_dataframe(df, "raw", Product, extractor=mock_extractor)

    # Should return original dataframe
    assert "raw_extracted" not in result.columns


def test_extractor_auto_creation() -> None:
    mock_engine = MagicMock()
    df = pl.DataFrame({"raw": ["val"]})

    # Mock Extractor class to verify it is initialized
    with patch("loclean.extraction.extract_dataframe.Extractor") as MockExtractorCls:
        mock_instance = MockExtractorCls.return_value
        mock_instance.extract_batch.return_value = {
            "val": Product(name="Clean val", price=100)
        }

        extract_dataframe(
            df, "raw", Product, extractor=None, inference_engine=mock_engine
        )

        MockExtractorCls.assert_called_once()
        mock_instance.extract_batch.assert_called()


def test_extraction_none_value(mock_extractor: Any) -> None:
    # Verify handling of None results from extractor
    df = pl.DataFrame({"raw": ["valid", "invalid"]})

    mock_extractor.extract_batch.side_effect = lambda values, *args: {
        "valid": Product(name="Good", price=1),
        "invalid": None,
    }

    result = extract_dataframe(
        df, "raw", Product, extractor=mock_extractor, output_type="dict"
    )

    valid_row = result.filter(pl.col("raw") == "valid").row(0, named=True)
    invalid_row = result.filter(pl.col("raw") == "invalid").row(0, named=True)

    assert valid_row["raw_extracted"]["name"] == "Good"
    assert invalid_row["raw_extracted"] is None


def test_schema_types(mock_extractor: Any) -> None:
    # Verify Polars type mapping for various fields
    df = pl.DataFrame({"raw": ["test"]})

    mock_extractor.extract_batch.side_effect = lambda values, *args: {
        "test": ComplexSchema(is_valid=True, score=9.5, tags=["a", "b"])
    }

    result = extract_dataframe(
        df, "raw", ComplexSchema, extractor=mock_extractor, output_type="dict"
    )

    val = result["raw_extracted"][0]
    assert val["is_valid"] is True
    assert val["score"] == 9.5
    assert val["tags"] == ["a", "b"]
    # Polars usually handles lists differently, let's see.
    # Current implementation falls back to Utf8 for complex types (list),
    # so it might be str representation.
    # Looking at code: struct_fields[field_name] = pl.Utf8 for fallback.
    # So yes, it will likely be coerced to string or fail if polars can't auto-cast
    # list to utf8 struct field?
    # Actually, if we pass a dict value (list) to a pl.Utf8 field, Polars might
    # error or cast.
    # Let's verify what happens.
    # If it fails, we know we hit the fallback line.
