"""Unit tests for the MissingnessRecognizer module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import narwhals as nw
import polars as pl

from loclean.extraction.missingness_recognizer import MissingnessRecognizer

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_engine(response: str) -> MagicMock:
    engine = MagicMock()
    engine.generate.return_value = response
    return engine


_ENCODER_SRC = (
    "def encode_missingness(row: dict) -> bool:\n"
    "    try:\n"
    "        return row.get('category') == 'electronics'\n"
    "    except Exception:\n"
    "        return False\n"
)


def _df_with_nulls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [100.0, None, 300.0, None, 500.0, None],
            "category": [
                "clothing",
                "electronics",
                "clothing",
                "electronics",
                "clothing",
                "electronics",
            ],
            "quantity": [10, 5, 20, 3, 15, 1],
        }
    )


# ------------------------------------------------------------------
# _find_null_columns
# ------------------------------------------------------------------


class TestFindNullColumns:
    def test_detects_columns_with_nulls(self) -> None:
        df = _df_with_nulls()
        df_nw = nw.from_native(df)
        result = MissingnessRecognizer._find_null_columns(df_nw)
        assert result == ["price"]

    def test_no_nulls(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_nw = nw.from_native(df)
        result = MissingnessRecognizer._find_null_columns(df_nw)
        assert result == []


# ------------------------------------------------------------------
# _sample_null_context
# ------------------------------------------------------------------


class TestSampleNullContext:
    def test_samples_rows_where_target_is_null(self) -> None:
        df = _df_with_nulls()
        df_nw = nw.from_native(df)
        sample = MissingnessRecognizer._sample_null_context(
            df_nw, "price", ["category", "quantity"]
        )
        assert len(sample) == 3
        for row in sample:
            assert "category" in row
            assert "quantity" in row
            assert "price" not in row

    def test_respects_max_rows(self) -> None:
        df = _df_with_nulls()
        df_nw = nw.from_native(df)
        sample = MissingnessRecognizer._sample_null_context(
            df_nw, "price", ["category"], max_rows=2
        )
        assert len(sample) == 2

    def test_empty_when_no_nulls(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_nw = nw.from_native(df)
        sample = MissingnessRecognizer._sample_null_context(df_nw, "a", ["b"])
        assert sample == []


# ------------------------------------------------------------------
# _build_prompt
# ------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_column_name_and_sample(self) -> None:
        prompt = MissingnessRecognizer._build_prompt(
            "price",
            ["category", "quantity"],
            [{"category": "electronics", "quantity": 5}],
        )
        assert "price" in prompt
        assert "encode_missingness" in prompt
        assert "electronics" in prompt

    def test_includes_rules(self) -> None:
        prompt = MissingnessRecognizer._build_prompt("x", ["y"], [{"y": 1}])
        assert "try/except" in prompt
        assert "boolean" in prompt.lower() or "bool" in prompt.lower()


# ------------------------------------------------------------------
# _verify_encoder
# ------------------------------------------------------------------


class TestVerifyEncoder:
    def test_valid_encoder_passes(self) -> None:
        def good_fn(row: dict[str, Any]) -> bool:
            return True

        ok, err = MissingnessRecognizer._verify_encoder(good_fn, [{"a": 1}, {"a": 2}])
        assert ok is True
        assert err == ""

    def test_non_bool_return_fails(self) -> None:
        def bad_fn(row: dict[str, Any]) -> Any:
            return "not a bool"

        ok, err = MissingnessRecognizer._verify_encoder(bad_fn, [{"a": 1}])
        assert ok is False
        assert "bool" in err.lower()


# ------------------------------------------------------------------
# recognize (integration with mock LLM)
# ------------------------------------------------------------------


class TestRecognize:
    def test_adds_mnar_column(self) -> None:
        engine = _make_engine(_ENCODER_SRC)
        recognizer = MissingnessRecognizer(inference_engine=engine, max_retries=1)
        df = _df_with_nulls()
        result, summary = recognizer.recognize(df)

        assert "price_mnar" in result.columns
        assert "price" in summary["patterns"]
        assert summary["patterns"]["price"]["encoded_as"] == "price_mnar"

    def test_no_nulls_skips(self) -> None:
        engine = _make_engine("")
        recognizer = MissingnessRecognizer(inference_engine=engine, max_retries=1)
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result, summary = recognizer.recognize(df)

        assert set(result.columns) == {"a", "b"}
        assert summary["patterns"] == {}
        engine.generate.assert_not_called()

    def test_target_cols_filter(self) -> None:
        df = pl.DataFrame(
            {
                "a": [1, None, 3],
                "b": [None, 2, None],
                "c": [10, 20, 30],
            }
        )
        engine = _make_engine(_ENCODER_SRC)
        recognizer = MissingnessRecognizer(inference_engine=engine, max_retries=1)
        _, summary = recognizer.recognize(df, target_cols=["a"])

        assert "a" in summary["patterns"]
        assert "b" not in summary["patterns"]

    def test_compile_failure_returns_none_pattern(self) -> None:
        engine = _make_engine("this is not valid python at all!!!")
        recognizer = MissingnessRecognizer(inference_engine=engine, max_retries=1)
        df = _df_with_nulls()
        result, summary = recognizer.recognize(df)

        assert "price_mnar" not in result.columns
        assert summary["patterns"]["price"] is None
