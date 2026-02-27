"""Unit tests for the TargetLeakageAuditor module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import narwhals as nw
import polars as pl
import pytest

from loclean.extraction.leakage_auditor import TargetLeakageAuditor

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_engine(response: str) -> MagicMock:
    engine = MagicMock()
    engine.generate.return_value = response
    return engine


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [25, 30, 45, 50, 35],
            "income": [50000, 60000, 80000, 90000, 55000],
            "approved_date": [
                "2024-01-15",
                "2024-01-20",
                "2024-02-01",
                "2024-02-10",
                "2024-01-25",
            ],
            "feedback_score": [4, 5, 3, 5, 4],
            "approved": [True, True, False, True, True],
        }
    )


# ------------------------------------------------------------------
# _extract_state
# ------------------------------------------------------------------


class TestExtractState:
    def test_extracts_features_and_samples(self) -> None:
        df = _sample_df()
        df_nw = nw.from_native(df)
        features = ["age", "income", "approved_date", "feedback_score"]
        state = TargetLeakageAuditor._extract_state(df_nw, "approved", features)

        assert state["target_col"] == "approved"
        assert state["features"] == features
        assert len(state["sample_rows"]) <= 10
        assert "age" in state["dtypes"]

    def test_respects_sample_n(self) -> None:
        df = _sample_df()
        df_nw = nw.from_native(df)
        state = TargetLeakageAuditor._extract_state(
            df_nw, "approved", ["age"], sample_n=2
        )
        assert len(state["sample_rows"]) == 2


# ------------------------------------------------------------------
# _build_prompt
# ------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_domain_and_target(self) -> None:
        state = {
            "target_col": "approved",
            "features": ["age", "income"],
            "dtypes": {"age": "Int64", "income": "Int64"},
            "sample_rows": [{"age": 25, "income": 50000, "approved": True}],
        }
        prompt = TargetLeakageAuditor._build_prompt(state, "loan approval prediction")
        assert "loan approval prediction" in prompt
        assert "approved" in prompt
        assert "age" in prompt
        assert "is_leakage" in prompt

    def test_no_domain(self) -> None:
        state = {
            "target_col": "y",
            "features": ["x"],
            "dtypes": {"x": "Float64"},
            "sample_rows": [{"x": 1.0, "y": 0}],
        }
        prompt = TargetLeakageAuditor._build_prompt(state, "")
        assert "Dataset domain:" not in prompt


# ------------------------------------------------------------------
# _parse_verdict
# ------------------------------------------------------------------


class TestParseVerdict:
    def test_parses_valid_json(self) -> None:
        response = json.dumps(
            [
                {"column": "approved_date", "is_leakage": True, "reason": "Post-event"},
                {"column": "age", "is_leakage": False, "reason": "Pre-event"},
            ]
        )
        verdicts = TargetLeakageAuditor._parse_verdict(response)
        assert len(verdicts) == 2
        assert verdicts[0]["column"] == "approved_date"
        assert verdicts[0]["is_leakage"] is True
        assert verdicts[1]["is_leakage"] is False

    def test_handles_extra_text(self) -> None:
        response = (
            'Analysis:\n[{"column": "x", "is_leakage": false, "reason": "ok"}]\nEnd.'
        )
        verdicts = TargetLeakageAuditor._parse_verdict(response)
        assert len(verdicts) == 1

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(ValueError, match="No JSON array"):
            TargetLeakageAuditor._parse_verdict("no json here")


# ------------------------------------------------------------------
# audit (integration with mock LLM)
# ------------------------------------------------------------------


class TestAudit:
    def test_drops_leaked_columns(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "age", "is_leakage": False, "reason": "ok"},
                {"column": "income", "is_leakage": False, "reason": "ok"},
                {"column": "approved_date", "is_leakage": True, "reason": "Post-event"},
                {
                    "column": "feedback_score",
                    "is_leakage": True,
                    "reason": "Post-event",
                },
            ]
        )
        engine = _make_engine(response)
        auditor = TargetLeakageAuditor(inference_engine=engine)

        pruned, summary = auditor.audit(df, "approved", "loan approval")

        assert "approved_date" not in pruned.columns
        assert "feedback_score" not in pruned.columns
        assert "age" in pruned.columns
        assert "income" in pruned.columns
        assert "approved" in pruned.columns
        assert "approved_date" in summary["dropped_columns"]
        assert "feedback_score" in summary["dropped_columns"]

    def test_keeps_all_if_no_leakage(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "age", "is_leakage": False, "reason": "ok"},
                {"column": "income", "is_leakage": False, "reason": "ok"},
                {"column": "approved_date", "is_leakage": False, "reason": "ok"},
                {"column": "feedback_score", "is_leakage": False, "reason": "ok"},
            ]
        )
        engine = _make_engine(response)
        auditor = TargetLeakageAuditor(inference_engine=engine)

        pruned, summary = auditor.audit(df, "approved")

        assert set(pruned.columns) == set(df.columns)
        assert summary["dropped_columns"] == []

    def test_missing_target_raises(self) -> None:
        df = _sample_df()
        engine = _make_engine("[]")
        auditor = TargetLeakageAuditor(inference_engine=engine)

        with pytest.raises(ValueError, match="not found"):
            auditor.audit(df, "nonexistent")

    def test_no_feature_columns(self) -> None:
        df = pl.DataFrame({"target": [1, 2, 3]})
        engine = _make_engine("[]")
        auditor = TargetLeakageAuditor(inference_engine=engine)

        pruned, summary = auditor.audit(df, "target")

        assert pruned.columns == ["target"]
        assert summary["dropped_columns"] == []
        engine.generate.assert_not_called()

    def test_summary_contains_verdicts(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "age", "is_leakage": False, "reason": "ok"},
            ]
        )
        engine = _make_engine(response)
        auditor = TargetLeakageAuditor(inference_engine=engine)

        _, summary = auditor.audit(df, "approved")

        assert "verdicts" in summary
        assert isinstance(summary["verdicts"], list)

    def test_domain_passed_to_prompt(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "age", "is_leakage": False, "reason": "ok"},
            ]
        )
        engine = _make_engine(response)
        auditor = TargetLeakageAuditor(inference_engine=engine)

        auditor.audit(df, "approved", domain="healthcare readmission")

        call_args = engine.generate.call_args[0][0]
        assert "healthcare readmission" in call_args
