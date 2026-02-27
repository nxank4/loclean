"""Unit tests for the TrapPruner module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import narwhals as nw
import polars as pl
import pytest

from loclean.extraction.trap_pruner import TrapPruner, _ColumnProfile

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_engine(response: str) -> MagicMock:
    engine = MagicMock()
    engine.generate.return_value = response
    return engine


def _sample_df() -> pl.DataFrame:
    """DataFrame with one real feature and one Gaussian noise column."""
    import random

    random.seed(42)
    n = 100
    prices = [200_000 + i * 5000 for i in range(n)]
    sqft = [150 + i * 10 + random.randint(-5, 5) for i in range(n)]
    noise = [random.gauss(0, 1) for _ in range(n)]

    return pl.DataFrame(
        {
            "sqft": sqft,
            "noise_feat": noise,
            "price": prices,
        }
    )


# ------------------------------------------------------------------
# _profile_columns
# ------------------------------------------------------------------


class TestProfileColumns:
    def test_basic_stats(self) -> None:
        df = _sample_df()
        df_nw = nw.from_native(df)
        profiles = TrapPruner._profile_columns(df_nw, "price", ["sqft", "noise_feat"])
        assert len(profiles) == 2

        sqft_p = next(p for p in profiles if p.name == "sqft")
        noise_p = next(p for p in profiles if p.name == "noise_feat")

        assert abs(sqft_p.corr_with_target) > 0.5
        assert abs(noise_p.corr_with_target) < 0.2

    def test_zero_variance_column(self) -> None:
        df = pl.DataFrame(
            {
                "constant": [5] * 10,
                "target": list(range(10)),
            }
        )
        df_nw = nw.from_native(df)
        profiles = TrapPruner._profile_columns(df_nw, "target", ["constant"])
        assert len(profiles) == 1
        assert profiles[0].variance == 0.0
        assert profiles[0].corr_with_target == 0.0

    def test_single_row_returns_empty(self) -> None:
        df = pl.DataFrame({"a": [1], "target": [2]})
        df_nw = nw.from_native(df)
        profiles = TrapPruner._profile_columns(df_nw, "target", ["a"])
        assert profiles == []


# ------------------------------------------------------------------
# _build_prompt
# ------------------------------------------------------------------


class TestBuildPrompt:
    def test_anonymises_column_names(self) -> None:
        profiles = [
            _ColumnProfile(
                name="secret_column",
                mean=0.0,
                std=1.0,
                variance=1.0,
                skewness=0.0,
                kurtosis=0.0,
                min_val=-3.0,
                max_val=3.0,
                corr_with_target=0.01,
            ),
        ]
        col_map, prompt = TrapPruner._build_prompt(profiles)

        assert "secret_column" not in prompt
        assert "col_0" in prompt
        assert col_map["col_0"] == "secret_column"

    def test_multiple_columns_indexed(self) -> None:
        profiles = [
            _ColumnProfile(
                name=f"feat_{i}",
                mean=float(i),
                std=1.0,
                variance=1.0,
                skewness=0.0,
                kurtosis=0.0,
                min_val=0.0,
                max_val=10.0,
                corr_with_target=0.5,
            )
            for i in range(3)
        ]
        col_map, prompt = TrapPruner._build_prompt(profiles)
        assert len(col_map) == 3
        assert "col_0" in prompt
        assert "col_1" in prompt
        assert "col_2" in prompt


# ------------------------------------------------------------------
# _parse_verdict
# ------------------------------------------------------------------


class TestParseVerdict:
    def test_maps_anonymous_to_real(self) -> None:
        col_map = {"col_0": "noise_feat", "col_1": "real_feat"}
        response = json.dumps(
            [
                {"column": "col_0", "is_trap": True, "reason": "Gaussian noise"},
                {"column": "col_1", "is_trap": False, "reason": "Correlated"},
            ]
        )

        verdicts = TrapPruner._parse_verdict(response, col_map)
        assert len(verdicts) == 2
        assert verdicts[0]["column"] == "noise_feat"
        assert verdicts[0]["is_trap"] is True
        assert verdicts[1]["column"] == "real_feat"
        assert verdicts[1]["is_trap"] is False

    def test_handles_extra_text_around_json(self) -> None:
        col_map = {"col_0": "feat_a"}
        response = (
            "Here is the analysis:\n"
            '[{"column": "col_0", "is_trap": false, "reason": "ok"}]\nDone.'
        )

        verdicts = TrapPruner._parse_verdict(response, col_map)
        assert len(verdicts) == 1
        assert verdicts[0]["column"] == "feat_a"

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(ValueError, match="No JSON array"):
            TrapPruner._parse_verdict("no json here", {})


# ------------------------------------------------------------------
# prune (integration with mock LLM)
# ------------------------------------------------------------------


class TestPrune:
    def test_removes_trap_columns(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "col_0", "is_trap": False, "reason": "Correlated"},
                {"column": "col_1", "is_trap": True, "reason": "Gaussian noise"},
            ]
        )
        engine = _make_engine(response)
        pruner = TrapPruner(inference_engine=engine)

        pruned, summary = pruner.prune(df, "price")

        assert "noise_feat" not in pruned.columns
        assert "sqft" in pruned.columns
        assert "price" in pruned.columns
        assert "noise_feat" in summary["dropped_columns"]

    def test_keeps_all_if_no_traps(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "col_0", "is_trap": False, "reason": "ok"},
                {"column": "col_1", "is_trap": False, "reason": "ok"},
            ]
        )
        engine = _make_engine(response)
        pruner = TrapPruner(inference_engine=engine)

        pruned, summary = pruner.prune(df, "price")

        assert set(pruned.columns) == set(df.columns)
        assert summary["dropped_columns"] == []

    def test_returns_summary_with_verdicts(self) -> None:
        df = _sample_df()
        response = json.dumps(
            [
                {"column": "col_0", "is_trap": False, "reason": "real"},
                {"column": "col_1", "is_trap": True, "reason": "noise"},
            ]
        )
        engine = _make_engine(response)
        pruner = TrapPruner(inference_engine=engine)

        _, summary = pruner.prune(df, "price")

        assert "dropped_columns" in summary
        assert "verdicts" in summary
        assert len(summary["verdicts"]) == 2

    def test_missing_target_raises(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        engine = _make_engine("[]")
        pruner = TrapPruner(inference_engine=engine)

        with pytest.raises(ValueError, match="not found"):
            pruner.prune(df, "nonexistent")

    def test_no_numeric_columns(self) -> None:
        df = pl.DataFrame(
            {
                "name": ["alice", "bob"],
                "target": [1, 2],
            }
        )
        engine = _make_engine("[]")
        pruner = TrapPruner(inference_engine=engine)

        pruned, summary = pruner.prune(df, "target")

        assert set(pruned.columns) == set(df.columns)
        assert summary["dropped_columns"] == []
        engine.generate.assert_not_called()
