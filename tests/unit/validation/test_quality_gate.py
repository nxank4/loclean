"""Test cases for the QualityGate module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import polars as pl
import pytest

from loclean.validation.quality_gate import QualityGate, QualityReport, _RowCompliance

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def gate(mock_engine: MagicMock) -> QualityGate:
    return QualityGate(inference_engine=mock_engine, batch_size=5, sample_size=50)


# ------------------------------------------------------------------
# _RowCompliance schema
# ------------------------------------------------------------------


class TestRowCompliance:
    def test_valid(self) -> None:
        rc = _RowCompliance(compliant=True, reasoning="All good")
        assert rc.compliant is True
        assert rc.reasoning == "All good"

    def test_non_compliant(self) -> None:
        rc = _RowCompliance(compliant=False, reasoning="Missing value")
        assert rc.compliant is False


# ------------------------------------------------------------------
# QualityReport
# ------------------------------------------------------------------


class TestQualityReport:
    def test_full_compliance(self) -> None:
        r = QualityReport(total_rows=10, passed_rows=10, compliance_rate=1.0)
        assert r.compliance_rate == 1.0
        assert r.failures == []

    def test_partial_compliance(self) -> None:
        r = QualityReport(
            total_rows=10,
            passed_rows=7,
            compliance_rate=0.7,
            failures=[{"row": {"a": 1}, "reasoning": "bad"}],
        )
        assert r.compliance_rate == 0.7
        assert len(r.failures) == 1

    def test_model_dump(self) -> None:
        r = QualityReport(total_rows=2, passed_rows=1, compliance_rate=0.5)
        d = r.model_dump()
        assert d["total_rows"] == 2
        assert d["compliance_rate"] == 0.5


# ------------------------------------------------------------------
# _parse_compliance
# ------------------------------------------------------------------


class TestParseCompliance:
    def test_parses_dict(self) -> None:
        raw = {"compliant": True, "reasoning": "ok"}
        result = QualityGate._parse_compliance(raw)
        assert result.compliant is True

    def test_parses_json_string(self) -> None:
        raw = json.dumps({"compliant": False, "reasoning": "fail"})
        result = QualityGate._parse_compliance(raw)
        assert result.compliant is False
        assert result.reasoning == "fail"

    def test_repairs_malformed_json(self) -> None:
        raw = '{"compliant": true, "reasoning": "ok",}'
        result = QualityGate._parse_compliance(raw)
        assert result.compliant is True

    def test_defaults_to_non_compliant_on_failure(self) -> None:
        result = QualityGate._parse_compliance("totally broken")
        assert result.compliant is False
        assert "Failed to parse" in result.reasoning


# ------------------------------------------------------------------
# _check_row
# ------------------------------------------------------------------


class TestCheckRow:
    def test_compliant_row(self, gate: QualityGate, mock_engine: MagicMock) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"compliant": True, "reasoning": "All rules satisfied"}
        )
        result = gate._check_row(
            {"price": 10, "currency": "USD"},
            ["Price must be positive", "Currency must be a valid ISO code"],
        )
        assert result.compliant is True

    def test_non_compliant_row(self, gate: QualityGate, mock_engine: MagicMock) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"compliant": False, "reasoning": "Price is negative"}
        )
        result = gate._check_row({"price": -5}, ["Price must be positive"])
        assert result.compliant is False
        assert "negative" in result.reasoning

    def test_prompt_contains_rules(
        self, gate: QualityGate, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"compliant": True, "reasoning": "ok"}
        )
        gate._check_row({"a": 1}, ["Rule A", "Rule B"])
        prompt = mock_engine.generate.call_args[0][0]
        assert "Rule A" in prompt
        assert "Rule B" in prompt


# ------------------------------------------------------------------
# evaluate (end-to-end with mocks)
# ------------------------------------------------------------------


class TestEvaluate:
    def test_all_pass(self, gate: QualityGate, mock_engine: MagicMock) -> None:
        df = pl.DataFrame({"val": [1, 2, 3]})
        mock_engine.generate.return_value = json.dumps(
            {"compliant": True, "reasoning": "ok"}
        )
        report = gate.evaluate(df, ["val must be positive"])
        assert report["total_rows"] == 3
        assert report["passed_rows"] == 3
        assert report["compliance_rate"] == 1.0
        assert report["failures"] == []

    def test_partial_failure(self, gate: QualityGate, mock_engine: MagicMock) -> None:
        df = pl.DataFrame({"val": [1, -1, 2]})

        responses = [
            json.dumps({"compliant": True, "reasoning": "ok"}),
            json.dumps({"compliant": False, "reasoning": "Negative value"}),
            json.dumps({"compliant": True, "reasoning": "ok"}),
        ]
        mock_engine.generate.side_effect = responses

        report = gate.evaluate(df, ["val must be positive"])
        assert report["total_rows"] == 3
        assert report["passed_rows"] == 2
        assert len(report["failures"]) == 1
        assert "Negative" in report["failures"][0]["reasoning"]

    def test_empty_dataframe(self, gate: QualityGate, mock_engine: MagicMock) -> None:
        df = pl.DataFrame({"val": []})
        report = gate.evaluate(df, ["some rule"])
        assert report["total_rows"] == 0
        assert report["compliance_rate"] == 1.0

    def test_empty_rules_raises(self, gate: QualityGate) -> None:
        df = pl.DataFrame({"val": [1]})
        with pytest.raises(ValueError, match="At least one rule"):
            gate.evaluate(df, [])

    def test_init_rejects_invalid_batch_size(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            QualityGate(inference_engine=mock_engine, batch_size=0)

    def test_init_rejects_invalid_sample_size(self, mock_engine: MagicMock) -> None:
        with pytest.raises(ValueError, match="sample_size"):
            QualityGate(inference_engine=mock_engine, sample_size=0)

    def test_sampling_limits_rows(self, mock_engine: MagicMock) -> None:
        gate = QualityGate(
            inference_engine=mock_engine,
            batch_size=5,
            sample_size=3,
        )
        df = pl.DataFrame({"val": list(range(100))})
        mock_engine.generate.return_value = json.dumps(
            {"compliant": True, "reasoning": "ok"}
        )
        report = gate.evaluate(df, ["val must exist"])
        assert report["total_rows"] == 3
