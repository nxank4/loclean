"""Test cases for FeatureDiscovery."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from loclean.extraction.feature_discovery import FeatureDiscovery

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

VALID_FEATURE_SOURCE = """
def generate_features(row: dict) -> dict:
    result = {}
    try:
        result["ratio_a_b"] = row.get("a", 0) / max(row.get("b", 1), 1)
    except Exception:
        result["ratio_a_b"] = None
    try:
        result["sum_a_b"] = row.get("a", 0) + row.get("b", 0)
    except Exception:
        result["sum_a_b"] = None
    try:
        val = row.get("a", 0)
        result["log_a"] = math.log(val) if val and val > 0 else 0.0
    except Exception:
        result["log_a"] = None
    try:
        result["product_a_b"] = row.get("a", 0) * row.get("b", 0)
    except Exception:
        result["product_a_b"] = None
    try:
        result["diff_a_b"] = row.get("a", 0) - row.get("b", 0)
    except Exception:
        result["diff_a_b"] = None
    return result
"""

SAMPLE_DF = pl.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "target": [0, 1, 0, 1],
    }
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def discoverer(mock_engine: MagicMock) -> FeatureDiscovery:
    return FeatureDiscovery(inference_engine=mock_engine, n_features=5, max_retries=2)


# ------------------------------------------------------------------
# __init__ validation
# ------------------------------------------------------------------


class TestInit:
    def test_rejects_zero_n_features(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="n_features"):
            FeatureDiscovery(inference_engine=mock_engine, n_features=0)

    def test_rejects_zero_max_retries(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            FeatureDiscovery(inference_engine=mock_engine, max_retries=0)

    def test_accepts_valid_params(self, mock_engine: MagicMock) -> None:
        fd = FeatureDiscovery(inference_engine=mock_engine, n_features=3, max_retries=2)
        assert fd.n_features == 3
        assert fd.max_retries == 2


# ------------------------------------------------------------------
# _extract_state
# ------------------------------------------------------------------


class TestExtractState:
    def test_extracts_columns_and_dtypes(self) -> None:
        import narwhals as nw

        df = nw.from_native(SAMPLE_DF)
        state = FeatureDiscovery._extract_state(df, "target")
        assert "a" in state["columns"]
        assert "b" in state["columns"]
        assert "target" in state["columns"]
        assert state["target_col"] == "target"
        assert len(state["dtypes"]) == 3

    def test_samples_rows(self) -> None:
        import narwhals as nw

        df = nw.from_native(SAMPLE_DF)
        state = FeatureDiscovery._extract_state(df, "target", sample_n=2)
        assert len(state["sample_rows"]) == 2

    def test_returns_all_rows_if_small(self) -> None:
        import narwhals as nw

        df = nw.from_native(SAMPLE_DF)
        state = FeatureDiscovery._extract_state(df, "target", sample_n=100)
        assert len(state["sample_rows"]) == 4


# ------------------------------------------------------------------
# _compile_function
# ------------------------------------------------------------------


class TestCompileFunction:
    def test_valid_source(self) -> None:
        fn = FeatureDiscovery._compile_function(VALID_FEATURE_SOURCE)
        result = fn({"a": 2.0, "b": 10.0})
        assert "ratio_a_b" in result
        assert "sum_a_b" in result

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Compilation failed"):
            FeatureDiscovery._compile_function("def broken(")

    def test_missing_function_raises(self) -> None:
        with pytest.raises(ValueError, match="does not define"):
            FeatureDiscovery._compile_function("x = 1")


# ------------------------------------------------------------------
# _verify_function
# ------------------------------------------------------------------


class TestVerifyFunction:
    def test_passes_valid_function(self) -> None:
        fn = FeatureDiscovery._compile_function(VALID_FEATURE_SOURCE)
        sample_rows = [
            {"a": 1.0, "b": 10.0, "target": 0},
            {"a": 2.0, "b": 20.0, "target": 1},
        ]
        ok, error = FeatureDiscovery._verify_function(fn, sample_rows)
        assert ok is True
        assert error == ""

    def test_fails_on_exception(self) -> None:
        def bad_fn(row: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        ok, error = FeatureDiscovery._verify_function(bad_fn, [{"a": 1}])
        assert ok is False
        assert "boom" in error

    def test_fails_on_non_dict_return(self) -> None:
        def bad_fn(row: dict[str, Any]) -> Any:
            return [1, 2, 3]

        ok, error = FeatureDiscovery._verify_function(bad_fn, [{"a": 1}])
        assert ok is False
        assert "dict" in error

    def test_fails_on_empty_return(self) -> None:
        def empty_fn(row: dict[str, Any]) -> dict[str, Any]:
            return {}

        ok, error = FeatureDiscovery._verify_function(empty_fn, [{"a": 1}])
        assert ok is False
        assert "empty" in error


# ------------------------------------------------------------------
# discover (end-to-end with mocks)
# ------------------------------------------------------------------


class TestDiscover:
    def test_happy_path(
        self, discoverer: FeatureDiscovery, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = VALID_FEATURE_SOURCE

        result = discoverer.discover(SAMPLE_DF, "target")
        assert len(result) == 4
        assert "ratio_a_b" in result.columns
        assert "sum_a_b" in result.columns
        assert "a" in result.columns

    def test_missing_column_raises(self, discoverer: FeatureDiscovery) -> None:
        with pytest.raises(ValueError, match="not found"):
            discoverer.discover(SAMPLE_DF, "missing")

    def test_repair_cycle(
        self, discoverer: FeatureDiscovery, mock_engine: MagicMock
    ) -> None:
        bad_source = "def generate_features(row): raise Exception('bad')"
        mock_engine.generate.side_effect = [
            bad_source,
            VALID_FEATURE_SOURCE,
        ]

        result = discoverer.discover(SAMPLE_DF, "target")
        assert "ratio_a_b" in result.columns
        assert mock_engine.generate.call_count == 2

    def test_exhausted_retries_returns_original(
        self, discoverer: FeatureDiscovery, mock_engine: MagicMock
    ) -> None:
        bad_source = "def generate_features(row): raise Exception('bad')"
        mock_engine.generate.side_effect = [
            bad_source,
            bad_source,
            bad_source,
        ]

        result = discoverer.discover(SAMPLE_DF, "target")
        assert list(result.columns) == list(SAMPLE_DF.columns)
        assert len(result) == len(SAMPLE_DF)
