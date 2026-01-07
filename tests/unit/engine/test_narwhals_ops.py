from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest

from loclean.engine.narwhals_ops import NarwhalsEngine

# Optional import cho pandas test
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.fixture
def mock_inference_engine() -> Any:
    """Mock LocalInferenceEngine để tránh phải chạy LLM thật."""
    mock_engine = Mock()
    # Mock clean_batch để trả về kết quả giả định
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
    """Sample Polars DataFrame cho testing."""
    return pl.DataFrame(
        {
            "weight": ["10kg", "500g", "10kg", "2kg", "500g"],
            "other_col": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def sample_pandas_df() -> Any:
    """Sample pandas DataFrame cho testing."""
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
    """Test process_column với Polars DataFrame - basic case."""
    result = NarwhalsEngine.process_column(
        sample_polars_df, "weight", mock_inference_engine, "Extract weight"
    )

    # Kiểm tra result là Polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Kiểm tra columns được thêm vào
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns
    assert "weight" in result.columns
    assert "other_col" in result.columns

    # Kiểm tra số rows không đổi
    assert len(result) == len(sample_polars_df)

    # Kiểm tra inference engine được gọi
    assert mock_inference_engine.clean_batch.called


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_process_column_pandas_basic(
    sample_pandas_df: Any, mock_inference_engine: Any
) -> None:
    """Test process_column với pandas DataFrame để verify multi-backend support."""
    result = NarwhalsEngine.process_column(
        sample_pandas_df, "weight", mock_inference_engine, "Extract weight"
    )

    # Kiểm tra result là pandas DataFrame (return về native type)
    assert isinstance(result, pd.DataFrame)

    # Kiểm tra columns được thêm vào
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns
    assert "weight" in result.columns
    assert "other_col" in result.columns

    # Kiểm tra số rows không đổi
    assert len(result) == len(sample_pandas_df)


def test_process_column_column_not_found(
    sample_polars_df: Any, mock_inference_engine: Any
) -> None:
    """Test validation khi column không tồn tại."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        NarwhalsEngine.process_column(
            sample_polars_df, "nonexistent", mock_inference_engine, "Extract weight"
        )


def test_process_column_empty_unique_values(mock_inference_engine: Any) -> None:
    """Test khi không có unique values hợp lệ."""
    # DataFrame với chỉ None và empty strings
    df = pl.DataFrame({"weight": [None, "", "   ", None]})

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Nên trả về DataFrame gốc
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(df)
    # Inference engine không nên được gọi vì không có unique values
    assert not mock_inference_engine.clean_batch.called


def test_process_column_batch_processing(mock_inference_engine: Any) -> None:
    """Test batch processing với nhiều unique values."""
    # Tạo DataFrame với nhiều unique values để test batching
    unique_values = [f"{i}kg" for i in range(60)]  # 60 unique values
    # Repeat để có nhiều rows
    weight_col = unique_values * 2  # 120 rows total
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,  # 60 values sẽ chia thành 2 batches
    )

    # Kiểm tra inference engine được gọi đúng số lần batch
    assert (
        mock_inference_engine.clean_batch.call_count == 2
    )  # 60 values / 50 batch_size = 2 batches

    # Kiểm tra result
    assert isinstance(result, pl.DataFrame)
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns


def test_process_column_join_logic(
    sample_polars_df: Any, mock_inference_engine: Any
) -> None:
    """Test join logic - verify mapping được join đúng."""

    # Custom mock để trả về kết quả cụ thể
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

    # Kiểm tra join đúng - tất cả rows với "10kg" nên có clean_value = 10.0
    rows_10kg = result.filter(pl.col("weight") == "10kg")
    assert len(rows_10kg) == 2  # Có 2 rows với "10kg"
    assert all(rows_10kg["clean_value"] == 10.0)
    assert all(rows_10kg["clean_unit"] == "kg")

    # Kiểm tra rows với "500g"
    rows_500g = result.filter(pl.col("weight") == "500g")
    assert len(rows_500g) == 2
    assert all(rows_500g["clean_value"] == 500.0)
    assert all(rows_500g["clean_unit"] == "g")


def test_process_column_with_none_results(mock_inference_engine: Any) -> None:
    """Test khi inference engine trả về None cho một số items."""
    df = pl.DataFrame({"weight": ["10kg", "invalid", "500g"]})

    def mock_clean_batch_with_none(items: Any, instruction: Any) -> Any:
        return {
            "10kg": {"value": 10.0, "unit": "kg"},
            "invalid": None,  # Trả về None
            "500g": {"value": 500.0, "unit": "g"},
        }

    mock_inference_engine.clean_batch = Mock(side_effect=mock_clean_batch_with_none)

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Kiểm tra result vẫn có columns
    assert "clean_value" in result.columns
    assert "clean_unit" in result.columns

    # Row với "invalid" nên có clean_value và clean_unit = None
    invalid_row = result.filter(pl.col("weight") == "invalid")
    assert invalid_row["clean_value"][0] is None
    assert invalid_row["clean_unit"][0] is None


def test_process_column_no_keys_extracted(mock_inference_engine: Any) -> None:
    """Test khi không có keys nào được extract (tất cả đều None)."""
    df = pl.DataFrame({"weight": ["item1", "item2"]})

    def mock_clean_batch_all_none(items: Any, instruction: Any) -> Any:
        return {"item1": None, "item2": None}

    mock_inference_engine.clean_batch = Mock(side_effect=mock_clean_batch_all_none)

    result = NarwhalsEngine.process_column(
        df, "weight", mock_inference_engine, "Extract weight"
    )

    # Nên trả về DataFrame gốc khi không có keys hợp lệ
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(df)


def test_process_column_parallel_processing(mock_inference_engine: Any) -> None:
    """Test parallel processing với nhiều batches."""
    # Tạo DataFrame với nhiều unique values để test parallel processing
    unique_values = [f"{i}kg" for i in range(150)]  # 150 unique values
    weight_col = unique_values * 2  # 300 rows total
    df = pl.DataFrame({"weight": weight_col})

    result = NarwhalsEngine.process_column(
        df,
        "weight",
        mock_inference_engine,
        "Extract weight",
        batch_size=50,  # 150 values sẽ chia thành 3 batches
        parallel=True,
        max_workers=2,
    )

    # Kiểm tra inference engine được gọi đúng số lần batch
    assert (
        mock_inference_engine.clean_batch.call_count == 3
    )  # 150 values / 50 batch_size = 3 batches

    # Kiểm tra result
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
