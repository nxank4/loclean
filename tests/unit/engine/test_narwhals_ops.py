from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest

from loclean.engine.narwhals_ops import NarwhalsEngine

# Optional import for pandas test
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.fixture
def mock_inference_engine() -> Any:
    """Mock LocalInferenceEngine to avoid running real LLM."""
    mock_engine = Mock()
    # Mock clean_batch to return predefined results
    mock_engine.clean_batch = Mock(
        side_effect=lambda items, instruction: {
            item: {
                "value": float(item.replace("kg", "").replace("g", "")),
                "unit": "kg" if "kg" in item else "g",
            }
            for item in items
        }
    )
    return mock_engine


@pytest.fixture
def sample_polars_df() -> pl.DataFrame:
    """Sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "weight": ["10kg", "500g", "10kg", "2kg", "500g"],
            "other_col": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def sample_pandas_df() -> Any:
    """Sample pandas DataFrame for testing."""
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")
    return pd.DataFrame(
        {
            "weight": ["10kg", "500g", "10kg", "2kg", "500g"],
            "other_col": [1, 2, 3, 4, 5],
        }
    )


def test_process_column_polars_basic(
    sample_polars_df: Any, mock_inference_engine: Any
) -> None:
    """Test process_column with Polars DataFrame - basic case."""
    result = NarwhalsEngine.process_column(
        sample_polars_df, "weight", mock_inference_engine, "Extract weight"
    )

    # Verify result is Polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Verify columns are added
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns
    assert "weight" in result.columns
    assert "other_col" in result.columns

    # Verify row count unchanged
    assert len(result) == len(sample_polars_df)

    # Verify inference engine was called
    assert mock_inference_engine.clean_batch.called


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_process_column_pandas_basic(
    sample_pandas_df: Any, mock_inference_engine: Any
) -> None:
    """Test process_column with pandas DataFrame to verify multi-backend support."""
    result = NarwhalsEngine.process_column(
        sample_pandas_df, "weight", mock_inference_engine, "Extract weight"
    )

    # Verify result is pandas DataFrame (returns native type)
    assert isinstance(result, pd.DataFrame)

    # Verify columns are added
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns
    assert "weight" in result.columns
    assert "other_col" in result.columns

    # Verify row count unchanged
    assert len(result) == len(sample_pandas_df)


def test_process_column_column_not_found(
    sample_polars_df: Any, mock_inference_engine: Any
) -> None:
    """Test validation when column does not exist."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        NarwhalsEngine.process_column(
            sample_polars_df, "nonexistent", mock_inference_engine, "Extract weight"
        )


def test_process_column_empty_unique_values(mock_inference_engine: Any) -> None:
    """Test when there are no valid unique values."""
    # DataFrame with only None and empty strings
    df = pl.DataFrame({"weight": [None, "", "   ", None]})

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Should return original DataFrame
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(df)
    # Inference engine should not be called because there are no unique values
    assert not mock_inference_engine.clean_batch.called


def test_process_column_batch_processing(mock_inference_engine: Any) -> None:
    """Test batch processing with many unique values."""
    # Create DataFrame with many unique values to test batching
    unique_values = [f"{i}kg" for i in range(60)]  # 60 unique values
    # Repeat to have many rows
    weight_col = unique_values * 2  # 120 rows total
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,  # 60 values will be split into 2 batches
    )

    # Verify inference engine was called correct number of times
    assert (
        mock_inference_engine.clean_batch.call_count == 2
    )  # 60 values / 50 batch_size = 2 batches

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns


def test_process_column_join_logic(
    sample_polars_df: Any, mock_inference_engine: Any
) -> None:
    """Test join logic - verify mapping is joined correctly."""

    # Custom mock to return specific results
    def custom_clean_batch(items: Any, instruction: Any) -> Any:
        return {
            "10kg": {"value": 10.0, "unit": "kg"},
            "500g": {"value": 500.0, "unit": "g"},
            "2kg": {"value": 2.0, "unit": "kg"},
        }

    mock_inference_engine.clean_batch = Mock(side_effect=custom_clean_batch)

    result = NarwhalsEngine.process_column(
        sample_polars_df, "weight", mock_inference_engine, "Extract weight"
    )

    # Verify join is correct - all rows with "10kg" should have clean_value = 10.0
    rows_10kg = result.filter(pl.col("weight") == "10kg")
    assert len(rows_10kg) == 2  # There are 2 rows with "10kg"
    assert all(rows_10kg["clean_value"] == 10.0)
    assert all(rows_10kg["clean_unit"] == "kg")

    # Verify rows with "500g"
    rows_500g = result.filter(pl.col("weight") == "500g")
    assert len(rows_500g) == 2
    assert all(rows_500g["clean_value"] == 500.0)
    assert all(rows_500g["clean_unit"] == "g")


def test_process_column_with_none_results(mock_inference_engine: Any) -> None:
    """Test when inference engine returns None for some items."""
    df = pl.DataFrame({"weight": ["10kg", "invalid", "500g"]})

    def mock_clean_batch_with_none(items: Any, instruction: Any) -> Any:
        return {
            "10kg": {"value": 10.0, "unit": "kg"},
            "invalid": None,  # Return None
            "500g": {"value": 500.0, "unit": "g"},
        }

    mock_inference_engine.clean_batch = Mock(side_effect=mock_clean_batch_with_none)

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Verify result still has columns
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns

    # Row with "invalid" should have clean_value and clean_unit = None
    invalid_row = result.filter(pl.col("weight") == "invalid")
    assert invalid_row["clean_value"][0] is None
    assert invalid_row["clean_unit"][0] is None


def test_process_column_no_keys_extracted(mock_inference_engine: Any) -> None:
    """Test when no keys are extracted (all are None)."""
    df = pl.DataFrame({"weight": ["item1", "item2"]})

    def mock_clean_batch_all_none(items: Any, instruction: Any) -> Any:
        return {"item1": None, "item2": None}

    mock_inference_engine.clean_batch = Mock(side_effect=mock_clean_batch_all_none)

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Should return original DataFrame when there are no valid keys
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(df)


def test_process_column_parallel_processing(mock_inference_engine: Any) -> None:
    """Test parallel processing with multiple batches."""
    # Create DataFrame with many unique values to test parallel processing
    unique_values = [f"{i}kg" for i in range(150)]  # 150 unique values
    weight_col = unique_values * 2  # 300 rows total
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,  # 150 values will be split into 3 batches
        parallel=True,
        max_workers=2,
    )

    # Verify inference engine was called correct number of times
    assert (
        mock_inference_engine.clean_batch.call_count == 3
    )  # 150 values / 50 batch_size = 3 batches

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns
    assert "clean_reasoning" in result.columns


def test_process_column_parallel_disabled(mock_inference_engine: Any) -> None:
    """Test backward compatibility - parallel=False should work as before."""
    unique_values = [f"{i}kg" for i in range(60)]
    weight_col = unique_values * 2
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,
        parallel=False,  # Explicitly disable parallel
    )

    # Should work exactly like before
    assert mock_inference_engine.clean_batch.call_count == 2
    assert isinstance(result, pl.DataFrame)
    assert "clean_value" in result.columns


def test_process_column_parallel_auto_workers(mock_inference_engine: Any) -> None:
    """Test parallel processing with auto-detected max_workers."""
    unique_values = [f"{i}kg" for i in range(100)]
    weight_col = unique_values * 2
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,  # 100 values = 2 batches
        parallel=True,
        max_workers=None,  # Auto-detect
    )

    # Should process all batches
    assert mock_inference_engine.clean_batch.call_count == 2
    assert isinstance(result, pl.DataFrame)


def test_process_column_parallel_single_batch(mock_inference_engine: Any) -> None:
    """Test that parallel processing falls back to sequential for single batch."""
    unique_values = [f"{i}kg" for i in range(30)]  # Less than batch_size
    df = pl.DataFrame({"weight": unique_values})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,
        parallel=True,  # Even with parallel=True, should use sequential for 1 batch
    )

    # Should still work correctly
    assert mock_inference_engine.clean_batch.call_count == 1
    assert isinstance(result, pl.DataFrame)


def test_process_column_parallel_max_workers_one(
    mock_inference_engine: Any,
) -> None:
    """Test that max_workers=1 falls back to sequential processing."""
    unique_values = [f"{i}kg" for i in range(100)]
    weight_col = unique_values * 2
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,
        parallel=True,
        max_workers=1,  # Should fallback to sequential
    )

    # Should process sequentially
    assert mock_inference_engine.clean_batch.call_count == 2
    assert isinstance(result, pl.DataFrame)
