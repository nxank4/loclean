"""Test cases for EntityResolver."""

import json
from unittest.mock import MagicMock

import polars as pl
import pytest

from loclean.extraction.resolver import EntityResolver

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def resolver(mock_engine: MagicMock) -> EntityResolver:
    return EntityResolver(inference_engine=mock_engine, threshold=0.8)


# ------------------------------------------------------------------
# __init__ validation
# ------------------------------------------------------------------


class TestInit:
    def test_rejects_zero_threshold(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            EntityResolver(inference_engine=mock_engine, threshold=0)

    def test_rejects_negative_threshold(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            EntityResolver(inference_engine=mock_engine, threshold=-0.5)

    def test_rejects_threshold_above_one(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            EntityResolver(inference_engine=mock_engine, threshold=1.5)

    def test_accepts_valid_threshold(self, mock_engine: MagicMock) -> None:
        r = EntityResolver(inference_engine=mock_engine, threshold=0.5)
        assert r.threshold == 0.5

    def test_accepts_threshold_one(self, mock_engine: MagicMock) -> None:
        r = EntityResolver(inference_engine=mock_engine, threshold=1.0)
        assert r.threshold == 1.0


# ------------------------------------------------------------------
# _extract_unique_values
# ------------------------------------------------------------------


class TestExtractUniqueValues:
    def test_returns_unique_non_empty(self, resolver: EntityResolver) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"col": ["a", "b", "a", "c", "b"]}))
        result = resolver._extract_unique_values(df, "col")
        assert sorted(result) == ["a", "b", "c"]

    def test_filters_none_and_whitespace(self, resolver: EntityResolver) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"col": ["a", None, "", "  ", "b"]}))
        result = resolver._extract_unique_values(df, "col")
        assert sorted(result) == ["a", "b"]

    def test_empty_column(self, resolver: EntityResolver) -> None:
        import narwhals as nw

        df = nw.from_native(pl.DataFrame({"col": [None, "", "  "]}))
        result = resolver._extract_unique_values(df, "col")
        assert result == []


# ------------------------------------------------------------------
# _parse_mapping_response
# ------------------------------------------------------------------


class TestParseMappingResponse:
    def test_parses_dict(self) -> None:
        raw = {"mapping": {"NYC": "New York City", "NY": "New York City"}}
        result = EntityResolver._parse_mapping_response(raw)
        assert result["mapping"]["NYC"] == "New York City"

    def test_parses_json_string(self) -> None:
        raw = json.dumps({"mapping": {"a": "A", "b": "B"}})
        result = EntityResolver._parse_mapping_response(raw)
        assert result["mapping"] == {"a": "A", "b": "B"}

    def test_repairs_malformed_json(self) -> None:
        raw = '{"mapping": {"a": "A", "b": "B",}}'
        result = EntityResolver._parse_mapping_response(raw)
        assert "mapping" in result

    def test_returns_empty_on_total_failure(self) -> None:
        result = EntityResolver._parse_mapping_response("completely broken")
        assert result == {}


# ------------------------------------------------------------------
# _build_canonical_mapping
# ------------------------------------------------------------------


class TestBuildCanonicalMapping:
    def test_happy_path(self, resolver: EntityResolver, mock_engine: MagicMock) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"mapping": {"NYC": "New York City", "New York": "New York City"}}
        )
        result = resolver._build_canonical_mapping(["NYC", "New York"])
        assert result["NYC"] == "New York City"
        assert result["New York"] == "New York City"

    def test_unmapped_values_keep_original(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"mapping": {"NYC": "New York City"}}
        )
        result = resolver._build_canonical_mapping(["NYC", "Tokyo"])
        assert result["NYC"] == "New York City"
        assert result["Tokyo"] == "Tokyo"

    def test_fallback_on_unparseable_response(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = "not json at all"
        result = resolver._build_canonical_mapping(["a", "b"])
        assert result == {"a": "a", "b": "b"}

    def test_filters_empty_canonical_values(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"mapping": {"a": "", "b": "   ", "c": "Canon"}}
        )
        result = resolver._build_canonical_mapping(["a", "b", "c"])
        assert result["a"] == "a"
        assert result["b"] == "b"
        assert result["c"] == "Canon"

    def test_prompt_includes_threshold(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps({"mapping": {"x": "x"}})
        resolver._build_canonical_mapping(["x"])
        prompt = mock_engine.generate.call_args[0][0]
        assert "0.8" in prompt


# ------------------------------------------------------------------
# resolve (end-to-end with mocks)
# ------------------------------------------------------------------


class TestResolve:
    def test_adds_canonical_column(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"city": ["NYC", "New York", "NYC", "LA"]})

        mock_engine.generate.return_value = json.dumps(
            {
                "mapping": {
                    "LA": "Los Angeles",
                    "NYC": "New York City",
                    "New York": "New York City",
                }
            }
        )

        result = resolver.resolve(df, "city")
        assert "city_canonical" in result.columns
        canonical = result["city_canonical"].to_list()
        assert canonical == [
            "New York City",
            "New York City",
            "New York City",
            "Los Angeles",
        ]

    def test_raises_on_missing_column(self, resolver: EntityResolver) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column 'b' not found"):
            resolver.resolve(df, "b")

    def test_all_null_returns_identity(self, resolver: EntityResolver) -> None:
        df = pl.DataFrame({"col": [None, None]})
        result = resolver.resolve(df, "col")
        assert "col_canonical" in result.columns

    def test_preserves_original_column(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"name": ["Alice", "Bob"]})
        mock_engine.generate.return_value = json.dumps(
            {"mapping": {"Alice": "Alice", "Bob": "Bob"}}
        )
        result = resolver.resolve(df, "name")
        assert result["name"].to_list() == ["Alice", "Bob"]
        assert "name_canonical" in result.columns

    def test_identity_mapping_when_no_groups(
        self, resolver: EntityResolver, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({"item": ["apple", "banana", "cherry"]})
        mock_engine.generate.return_value = json.dumps(
            {
                "mapping": {
                    "apple": "apple",
                    "banana": "banana",
                    "cherry": "cherry",
                }
            }
        )
        result = resolver.resolve(df, "item")
        assert result["item_canonical"].to_list() == [
            "apple",
            "banana",
            "cherry",
        ]
