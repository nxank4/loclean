"""Test cases for SemanticOversampler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import BaseModel

from loclean.extraction.oversampler import SemanticOversampler, _row_key

# ------------------------------------------------------------------
# Test schema
# ------------------------------------------------------------------


class SampleRecord(BaseModel):
    label: str
    value: float
    category: str


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def sampler(mock_engine: MagicMock) -> SemanticOversampler:
    return SemanticOversampler(
        inference_engine=mock_engine, batch_size=5, max_retries=3
    )


# ------------------------------------------------------------------
# __init__ validation
# ------------------------------------------------------------------


class TestInit:
    def test_rejects_zero_batch_size(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            SemanticOversampler(inference_engine=mock_engine, batch_size=0)

    def test_rejects_zero_max_retries(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            SemanticOversampler(inference_engine=mock_engine, max_retries=0)

    def test_accepts_valid_params(self, mock_engine: MagicMock) -> None:
        s = SemanticOversampler(
            inference_engine=mock_engine, batch_size=3, max_retries=2
        )
        assert s.batch_size == 3
        assert s.max_retries == 2


# ------------------------------------------------------------------
# _sample_rows
# ------------------------------------------------------------------


class TestSampleRows:
    def test_filters_minority_class(self, sampler: SemanticOversampler) -> None:
        import narwhals as nw

        df = nw.from_native(
            pl.DataFrame(
                {
                    "label": ["A", "B", "A", "B", "A"],
                    "value": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        )
        rows = sampler._sample_rows(df, "label", "B")
        assert len(rows) == 2
        assert all(r["label"] == "B" for r in rows)

    def test_returns_empty_for_missing_class(
        self, sampler: SemanticOversampler
    ) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"label": ["A", "A"]}))
        rows = sampler._sample_rows(df, "label", "Z")
        assert rows == []

    def test_samples_when_exceeds_n(self, sampler: SemanticOversampler) -> None:
        import narwhals as nw

        df = nw.from_native(
            pl.DataFrame({"label": ["X"] * 100, "val": list(range(100))})
        )
        rows = sampler._sample_rows(df, "label", "X", n=5)
        assert len(rows) == 5


# ------------------------------------------------------------------
# _parse_batch_response
# ------------------------------------------------------------------


class TestParseBatchResponse:
    def test_parses_dict_with_records(self) -> None:
        raw = {"records": [{"label": "A", "value": 1.0, "category": "x"}]}
        result = SemanticOversampler._parse_batch_response(raw)
        assert len(result) == 1

    def test_parses_json_string(self) -> None:
        raw = json.dumps({"records": [{"a": 1}, {"a": 2}]})
        result = SemanticOversampler._parse_batch_response(raw)
        assert len(result) == 2

    def test_parses_raw_list(self) -> None:
        raw = json.dumps([{"a": 1}])
        result = SemanticOversampler._parse_batch_response(raw)
        assert len(result) == 1

    def test_repairs_malformed(self) -> None:
        raw = '{"records": [{"a": 1},]}'
        result = SemanticOversampler._parse_batch_response(raw)
        assert len(result) >= 1

    def test_returns_empty_on_failure(self) -> None:
        result = SemanticOversampler._parse_batch_response("completely broken")
        assert result == []


# ------------------------------------------------------------------
# _row_key / deduplication
# ------------------------------------------------------------------


class TestDeduplicate:
    def test_row_key_deterministic(self) -> None:
        r1 = {"a": 1, "b": "x"}
        r2 = {"b": "x", "a": 1}
        assert _row_key(r1) == _row_key(r2)

    def test_validate_and_filter_removes_dupes(
        self, sampler: SemanticOversampler
    ) -> None:
        existing = {_row_key({"label": "A", "value": 1.0, "category": "x"})}
        candidates = [
            {"label": "A", "value": 1.0, "category": "x"},
            {"label": "A", "value": 2.0, "category": "y"},
        ]
        result = sampler._validate_and_filter(candidates, SampleRecord, existing, [])
        assert len(result) == 1
        assert result[0]["value"] == 2.0

    def test_validate_and_filter_rejects_invalid_schema(
        self, sampler: SemanticOversampler
    ) -> None:
        candidates: list[dict[str, Any]] = [
            {"label": "A"},
        ]
        result = sampler._validate_and_filter(candidates, SampleRecord, set(), [])
        assert result == []


# ------------------------------------------------------------------
# _generate_batch
# ------------------------------------------------------------------


class TestGenerateBatch:
    def test_prompt_includes_schema_and_samples(
        self, sampler: SemanticOversampler, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"records": [{"label": "B", "value": 9.0, "category": "new"}]}
        )
        sample = [{"label": "B", "value": 1.0, "category": "old"}]
        sampler._generate_batch(sample, SampleRecord, 1)

        prompt = mock_engine.generate.call_args[0][0]
        assert "label" in prompt
        assert "value" in prompt
        assert "old" in prompt


# ------------------------------------------------------------------
# oversample (end-to-end with mocks)
# ------------------------------------------------------------------


class TestOversample:
    def test_happy_path(
        self, sampler: SemanticOversampler, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame(
            {
                "label": ["A", "A", "A", "B"],
                "value": [1.0, 2.0, 3.0, 10.0],
                "category": ["x", "y", "z", "w"],
            }
        )

        mock_engine.generate.return_value = json.dumps(
            {
                "records": [
                    {"label": "B", "value": 20.0, "category": "v"},
                    {"label": "B", "value": 30.0, "category": "u"},
                ]
            }
        )

        result = sampler.oversample(df, "label", "B", n=2, schema=SampleRecord)
        assert len(result) == 6
        assert result["label"].to_list().count("B") == 3

    def test_missing_column_raises(self, sampler: SemanticOversampler) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            sampler.oversample(df, "missing", "x", n=1, schema=SampleRecord)

    def test_empty_minority_raises(self, sampler: SemanticOversampler) -> None:
        df = pl.DataFrame({"label": ["A", "A"]})
        with pytest.raises(ValueError, match="No rows found"):
            sampler.oversample(df, "label", "Z", n=1, schema=SampleRecord)

    def test_dedup_triggers_retry(
        self, sampler: SemanticOversampler, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame(
            {
                "label": ["B"],
                "value": [1.0],
                "category": ["x"],
            }
        )

        mock_engine.generate.side_effect = [
            json.dumps({"records": [{"label": "B", "value": 1.0, "category": "x"}]}),
            json.dumps({"records": [{"label": "B", "value": 99.0, "category": "new"}]}),
        ]

        result = sampler.oversample(df, "label", "B", n=1, schema=SampleRecord)
        assert len(result) == 2
        assert mock_engine.generate.call_count == 2
