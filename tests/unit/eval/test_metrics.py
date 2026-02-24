"""Unit tests for loclean.eval.metrics."""

from __future__ import annotations

import json

import pytest

from loclean.eval.metrics import (
    ExactMatch,
    PartialJSONMatch,
    PIIMaskingRecall,
    get_metric,
)


class TestExactMatch:
    """Tests for the ExactMatch metric."""

    def test_exact(self) -> None:
        score, _ = ExactMatch().score("hello", "hello")
        assert score == 1.0

    def test_mismatch(self) -> None:
        score, detail = ExactMatch().score("hello", "world")
        assert score == 0.0
        assert "expected" in detail

    def test_whitespace_normalisation(self) -> None:
        score, _ = ExactMatch().score("  hello  ", "hello")
        assert score == 1.0

    def test_case_sensitive(self) -> None:
        score, _ = ExactMatch().score("Hello", "hello")
        assert score == 0.0


class TestPartialJSONMatch:
    """Tests for the PartialJSONMatch metric."""

    def test_full_match(self) -> None:
        obj = json.dumps({"a": 1, "b": 2})
        score, detail = PartialJSONMatch().score(obj, obj)
        assert score == 1.0
        assert "2/2" in detail

    def test_partial_match(self) -> None:
        actual = json.dumps({"a": 1, "b": 99})
        expected = json.dumps({"a": 1, "b": 2})
        score, detail = PartialJSONMatch().score(actual, expected)
        assert score == 0.5
        assert "1/2" in detail

    def test_no_match(self) -> None:
        actual = json.dumps({"x": 1})
        expected = json.dumps({"a": 1})
        score, _ = PartialJSONMatch().score(actual, expected)
        assert score == 0.0

    def test_invalid_json(self) -> None:
        score, detail = PartialJSONMatch().score("notjson", '{"a": 1}')
        assert score == 0.0
        assert "JSON parse error" in detail

    def test_non_dict_json(self) -> None:
        score, detail = PartialJSONMatch().score("[1,2]", "[1,2]")
        assert score == 0.0
        assert "must be JSON objects" in detail

    def test_empty_expected(self) -> None:
        score, _ = PartialJSONMatch().score("{}", "{}")
        assert score == 1.0


class TestPIIMaskingRecall:
    """Tests for the PIIMaskingRecall metric."""

    def test_all_masked(self) -> None:
        meta = {"pii_tokens": ["John", "555-1234"]}
        score, detail = PIIMaskingRecall().score(
            "Contact [PERSON] at [PHONE]", "", meta
        )
        assert score == 1.0
        assert "2/2" in detail

    def test_none_masked(self) -> None:
        meta = {"pii_tokens": ["John", "555-1234"]}
        score, detail = PIIMaskingRecall().score("Contact John at 555-1234", "", meta)
        assert score == 0.0
        assert "0/2" in detail

    def test_partial(self) -> None:
        meta = {"pii_tokens": ["John", "555-1234"]}
        score, _ = PIIMaskingRecall().score("Contact [PERSON] at 555-1234", "", meta)
        assert score == 0.5

    def test_no_tokens(self) -> None:
        score, detail = PIIMaskingRecall().score("text", "", {})
        assert score == 1.0
        assert "no PII tokens" in detail

    def test_no_metadata(self) -> None:
        score, _ = PIIMaskingRecall().score("text", "")
        assert score == 1.0


class TestGetMetric:
    """Tests for the metric registry."""

    def test_known(self) -> None:
        m = get_metric("exact_match")
        assert m.name == "exact_match"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent")
