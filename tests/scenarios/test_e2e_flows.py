from typing import Any, Generator
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

import loclean


@pytest.fixture
def mock_inference() -> Generator[Any, None, None]:
    """Mock the LlamaCppEngine.clean_batch to return deterministic results."""
    with patch("loclean.inference.local.llama_cpp.LlamaCppEngine") as mock_class:
        engine = mock_class.return_value

        def mock_clean_batch(items: list[str], instruction: str) -> dict[str, Any]:
            results = {}
            for item in items:
                # Logic for cleaning tests
                if "10kg" in item or "10.0kg" in item:
                    results[item] = {
                        "reasoning": "Detected 10kg",
                        "value": 10.0,
                        "unit": "kg",
                    }
                elif "500g" in item:
                    results[item] = {
                        "reasoning": "Detected 500g",
                        "value": 500.0,
                        "unit": "g",
                    }
                elif "20 EUR" in item:
                    results[item] = {
                        "reasoning": "Detected 20 EUR",
                        "value": 20.0,
                        "unit": "EUR",
                    }
                # Logic for scrubbing tests (PII spans)
                elif "John" in item:
                    results[item] = {
                        "reasoning": "Found John",
                        "pii": [
                            {"type": "person", "value": "John", "start": 8, "end": 12}
                        ],
                    }
                else:
                    results[item] = {"reasoning": "Unknown", "value": 0.0, "unit": ""}
            return results

        engine.clean_batch.side_effect = mock_clean_batch
        # Mock extractor for extract() calls
        engine.verbose = False
        yield engine


def test_pipeline_scrub_then_clean(mock_inference: Any) -> None:
    """Scenario: Scrub PII from a column, then perform semantic cleaning.

    This test verifies that the library can handle a pipeline where PII is
    first scrubbed from a column, and then semantic cleaning is performed
    on the scrubbed data.
    """
    data = {"info": ["Contact John at 10kg", "Meeting with Alice: 500g"]}
    df = pl.DataFrame(data)

    # 1. Scrub PII (Person names)
    # We mock LLMDetector.detect_batch which is called by PIIDetector
    from loclean.privacy.schemas import PIIDetectionResult, PIIEntity

    with patch("loclean.privacy.llm_detector.LLMDetector.detect_batch") as mock_detect:
        # LLMDetector.detect_batch returns a list of PIIDetectionResult
        # Each PIIDetectionResult has a list of PIIEntity
        mock_detect.return_value = [
            PIIDetectionResult(
                entities=[PIIEntity(type="person", value="John", start=0, end=0)],
                reasoning="test",
            )
        ]

        # We need to handle the second item too
        def side_effect(
            items: list[str], strategies: list[str]
        ) -> list[PIIDetectionResult]:
            results = []
            for item in items:
                if "John" in item:
                    results.append(
                        PIIDetectionResult(
                            entities=[
                                PIIEntity(type="person", value="John", start=0, end=0)
                            ],
                            reasoning="test",
                        )
                    )
                elif "Alice" in item:
                    results.append(
                        PIIDetectionResult(
                            entities=[
                                PIIEntity(type="person", value="Alice", start=0, end=0)
                            ],
                            reasoning="test",
                        )
                    )
                else:
                    results.append(PIIDetectionResult(entities=[], reasoning="none"))
            return results

        mock_detect.side_effect = side_effect

        # Scrub will use get_engine() internally if no engine provided,
        # but here we want to make sure it doesn't try to load a real model.
        with patch("loclean.get_engine", return_value=mock_inference):
            scrubbed_df: Any = loclean.scrub(
                df, target_col="info", strategies=["person"]
            )

    assert "[PERSON]" in scrubbed_df["info"][0]
    assert "John" not in scrubbed_df["info"][0]
    assert "[PERSON]" in scrubbed_df["info"][1]
    assert "Alice" not in scrubbed_df["info"][1]

    # 2. Clean the scrubbed column
    with patch("loclean.get_engine", return_value=mock_inference):
        cleaned_df: Any = loclean.clean(
            scrubbed_df, target_col="info", instruction="Extract weight"
        )

    assert "clean_value" in cleaned_df.columns
    assert cleaned_df["clean_value"][0] == 10.0
    assert cleaned_df["clean_value"][1] == 500.0
    assert "[PERSON]" in cleaned_df["info"][0]


def test_multi_column_processing(mock_inference: Any) -> None:
    """Scenario: Clean multiple columns in the same DataFrame."""
    df = pl.DataFrame(
        {
            "weight_raw": ["10kg", "500g"],
            "price_raw": [
                "10 USD",
                "20 EUR",
            ],  # 10 USD will hit 'Unknown' in our mock
        }
    )

    # Clean first column
    df_any: Any = loclean.clean(
        df, target_col="weight_raw", instruction="Extract weight"
    )
    # Clean second column
    df_any = loclean.clean(df_any, target_col="price_raw", instruction="Extract price")

    assert "clean_value" in df_any.columns
    assert "clean_unit" in df_any.columns


def test_backend_consistency_pandas_polars(mock_inference: Any) -> None:
    """Scenario: Ensure Pandas and Polars produce identical logical results."""
    data = {"raw": ["10kg", "500g"]}
    df_pl = pl.DataFrame(data)
    df_pd = pd.DataFrame(data)

    res_pl: Any = loclean.clean(df_pl, "raw")
    res_pd: Any = loclean.clean(df_pd, "raw")

    # Assert values match
    assert list(res_pl["clean_value"]) == [10.0, 500.0]
    assert list(res_pd["clean_value"]) == [10.0, 500.0]

    # Assert types match input
    assert isinstance(res_pl, pl.DataFrame)
    assert isinstance(res_pd, pd.DataFrame)

    # Assert column names match
    expected_cols = ["raw", "clean_value", "clean_unit", "clean_reasoning"]
    assert all(c in res_pl.columns for c in expected_cols)
    assert all(c in res_pd.columns for c in expected_cols)
