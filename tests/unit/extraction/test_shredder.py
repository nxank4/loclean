"""Test cases for RelationalShredder."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from loclean.extraction.shredder import RelationalShredder, _RelationalSchema, _TableDef

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SAMPLE_LOGS = [
    "2024-01-01 10:00:00 INFO server=web01 user=alice action=login ip=1.2.3.4",
    "2024-01-01 10:01:00 ERROR server=web01 user=bob action=timeout code=504",
    "2024-01-01 10:02:00 INFO server=db01 user=alice action=query rows=42",
]

SAMPLE_SCHEMA = _RelationalSchema(
    tables=[
        _TableDef(
            name="events",
            columns=["timestamp", "level", "server", "action"],
            primary_key="timestamp",
            foreign_key=None,
        ),
        _TableDef(
            name="metadata",
            columns=["timestamp", "user", "ip", "code", "rows"],
            primary_key="timestamp",
            foreign_key="timestamp",
        ),
    ]
)

VALID_EXTRACT_SOURCE = """
def extract_relations(log: str) -> dict[str, dict]:
    parts = log.split()
    result = {
        "events": {
            "timestamp": parts[0] + " " + parts[1] if len(parts) > 1 else "",
            "level": parts[2] if len(parts) > 2 else "",
            "server": "",
            "action": "",
        },
        "metadata": {
            "timestamp": parts[0] + " " + parts[1] if len(parts) > 1 else "",
            "user": "",
            "ip": "",
            "code": "",
            "rows": "",
        },
    }
    try:
        for part in parts[3:]:
            if "=" in part:
                k, v = part.split("=", 1)
                if k == "server":
                    result["events"]["server"] = v
                elif k == "action":
                    result["events"]["action"] = v
                elif k in ("user", "ip", "code", "rows"):
                    result["metadata"][k] = v
    except Exception:
        pass
    return result
"""


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def shredder(mock_engine: MagicMock) -> RelationalShredder:
    return RelationalShredder(
        inference_engine=mock_engine, sample_size=10, max_retries=2
    )


# ------------------------------------------------------------------
# __init__ validation
# ------------------------------------------------------------------


class TestInit:
    def test_rejects_zero_sample_size(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="sample_size"):
            RelationalShredder(inference_engine=mock_engine, sample_size=0)

    def test_rejects_zero_max_retries(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            RelationalShredder(inference_engine=mock_engine, max_retries=0)

    def test_accepts_valid_params(self, mock_engine: MagicMock) -> None:
        s = RelationalShredder(
            inference_engine=mock_engine,
            sample_size=5,
            max_retries=2,
        )
        assert s.sample_size == 5
        assert s.max_retries == 2


# ------------------------------------------------------------------
# _sample_entries
# ------------------------------------------------------------------


class TestSampleEntries:
    def test_extracts_non_empty(self, shredder: RelationalShredder) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"log": SAMPLE_LOGS + ["", None]}))
        entries = shredder._sample_entries(df, "log")
        assert len(entries) == 3
        assert all(e.strip() for e in entries)

    def test_samples_when_exceeds_limit(self, mock_engine: MagicMock) -> None:
        import narwhals as nw

        s = RelationalShredder(inference_engine=mock_engine, sample_size=2)
        df = nw.from_native(pl.DataFrame({"log": [f"entry_{i}" for i in range(100)]}))
        entries = s._sample_entries(df, "log")
        assert len(entries) == 2

    def test_empty_column(self, shredder: RelationalShredder) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"log": ["", None]}))
        entries = shredder._sample_entries(df, "log")
        assert entries == []


# ------------------------------------------------------------------
# _parse_schema_response
# ------------------------------------------------------------------


class TestInferSchema:
    def test_parses_dict(self) -> None:
        raw = SAMPLE_SCHEMA.model_dump()
        result = RelationalShredder._parse_schema_response(raw)
        assert len(result.tables) == 2

    def test_parses_json_string(self) -> None:
        raw = SAMPLE_SCHEMA.model_dump_json()
        result = RelationalShredder._parse_schema_response(raw)
        assert result.tables[0].name == "events"

    def test_raises_on_failure(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            RelationalShredder._parse_schema_response("broken")


# ------------------------------------------------------------------
# _compile_function
# ------------------------------------------------------------------


class TestCompileFunction:
    def test_valid_source(self) -> None:
        fn = RelationalShredder._compile_function(VALID_EXTRACT_SOURCE)
        result = fn(SAMPLE_LOGS[0])
        assert "events" in result
        assert "metadata" in result

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Compilation failed"):
            RelationalShredder._compile_function("def broken(")

    def test_missing_function_raises(self) -> None:
        with pytest.raises(ValueError, match="does not define"):
            RelationalShredder._compile_function("x = 1")


# ------------------------------------------------------------------
# _verify_function
# ------------------------------------------------------------------


class TestVerifyFunction:
    def test_passes_valid_function(self, shredder: RelationalShredder) -> None:
        fn = RelationalShredder._compile_function(VALID_EXTRACT_SOURCE)
        ok, error = shredder._verify_function(fn, SAMPLE_LOGS, SAMPLE_SCHEMA)
        assert ok is True
        assert error == ""

    def test_fails_on_exception(self, shredder: RelationalShredder) -> None:
        def bad_fn(log: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        ok, error = shredder._verify_function(bad_fn, SAMPLE_LOGS, SAMPLE_SCHEMA)
        assert ok is False
        assert "boom" in error

    def test_fails_on_missing_table(self, shredder: RelationalShredder) -> None:
        def partial_fn(log: str) -> dict[str, dict[str, str]]:
            return {"events": {"timestamp": "x"}}

        ok, error = shredder._verify_function(partial_fn, SAMPLE_LOGS, SAMPLE_SCHEMA)
        assert ok is False
        assert "Missing tables" in error


# ------------------------------------------------------------------
# _separate_tables
# ------------------------------------------------------------------


class TestSeparateTables:
    def test_builds_multi_table(self) -> None:
        fn = RelationalShredder._compile_function(VALID_EXTRACT_SOURCE)
        results = [fn(log) for log in SAMPLE_LOGS]

        tables = RelationalShredder._separate_tables(results, SAMPLE_SCHEMA, pl)
        assert "events" in tables
        assert "metadata" in tables
        assert len(tables["events"]) == 3
        assert len(tables["metadata"]) == 3


# ------------------------------------------------------------------
# shred (end-to-end with mocks)
# ------------------------------------------------------------------


class TestShred:
    def test_happy_path(
        self, shredder: RelationalShredder, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"log": SAMPLE_LOGS})

        mock_engine.generate.side_effect = [
            SAMPLE_SCHEMA.model_dump(),
            VALID_EXTRACT_SOURCE,
        ]

        result = shredder.shred(df, "log")
        assert isinstance(result, dict)
        assert len(result) >= 2
        assert "events" in result
        assert "metadata" in result

    def test_missing_column_raises(self, shredder: RelationalShredder) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            shredder.shred(df, "missing")

    def test_empty_column_raises(self, shredder: RelationalShredder) -> None:
        df = pl.DataFrame({"log": ["", None]})
        with pytest.raises(ValueError, match="No valid entries"):
            shredder.shred(df, "log")

    def test_repair_cycle(
        self, shredder: RelationalShredder, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"log": SAMPLE_LOGS})

        bad_source = "def extract_relations(log): raise Exception('bad')"

        mock_engine.generate.side_effect = [
            SAMPLE_SCHEMA.model_dump(),
            bad_source,
            VALID_EXTRACT_SOURCE,
        ]

        result = shredder.shred(df, "log")
        assert "events" in result
        assert mock_engine.generate.call_count == 3

    def test_exhausted_retries_returns_empty(
        self, shredder: RelationalShredder, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"log": SAMPLE_LOGS})

        bad_source = "def extract_relations(log): raise Exception('bad')"

        mock_engine.generate.side_effect = [
            SAMPLE_SCHEMA.model_dump(),
            bad_source,
            bad_source,
            bad_source,
        ]

        result = shredder.shred(df, "log")
        assert result == {}
