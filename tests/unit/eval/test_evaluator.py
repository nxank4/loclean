"""Unit tests for loclean.eval.evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from loclean.eval.evaluator import Evaluator
from loclean.eval.schemas import TestCase


class TestEvaluatorRouting:
    """Verify that the evaluator dispatches to correct loclean functions."""

    @patch("loclean.eval.evaluator.loclean")
    def test_clean_dispatch(self, mock_lc: MagicMock) -> None:
        result_df = pl.DataFrame(
            {
                "input": ["5kg"],
                "clean_value": ["5"],
                "clean_unit": ["kg"],
                "clean_reasoning": ["parsed"],
            }
        )
        mock_lc.clean.return_value = result_df

        case = TestCase(
            input="5kg",
            expected_output="5",
            task="clean",
            metric="exact_match",
        )
        evaluator = Evaluator()
        summary = evaluator.run([case])

        mock_lc.clean.assert_called_once()
        assert len(summary.results) == 1
        assert summary.results[0].score == 1.0

    @patch("loclean.eval.evaluator.loclean")
    def test_scrub_dispatch(self, mock_lc: MagicMock) -> None:
        mock_lc.scrub.return_value = "Contact [PERSON] at [EMAIL]"

        case = TestCase(
            input="Contact John at john@x.com",
            expected_output="Contact [PERSON] at [EMAIL]",
            task="scrub",
            metric="exact_match",
        )
        summary = Evaluator().run([case])

        mock_lc.scrub.assert_called_once()
        assert summary.results[0].score == 1.0

    @patch("loclean.eval.evaluator.loclean")
    def test_extract_dispatch(self, mock_lc: MagicMock) -> None:
        mock_lc.extract.return_value = {"name": "shirt", "price": 50}

        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: int

        case = TestCase(
            input="red shirt $50",
            expected_output='{"name": "shirt", "price": 50}',
            task="extract",
            metric="partial_json_match",
            metadata={"schema": Item},
        )
        summary = Evaluator().run([case])

        mock_lc.extract.assert_called_once()
        assert summary.results[0].score == 1.0

    @patch("loclean.eval.evaluator.loclean")
    def test_failed_case_records_empty_output(self, mock_lc: MagicMock) -> None:
        mock_lc.scrub.side_effect = RuntimeError("boom")

        case = TestCase(
            input="text",
            expected_output="masked",
            task="scrub",
            metric="exact_match",
        )
        summary = Evaluator().run([case])

        assert summary.results[0].actual_output == ""
        assert summary.results[0].score == 0.0


class TestEvaluatorSummary:
    """Verify aggregation in EvalSummary."""

    @patch("loclean.eval.evaluator.loclean")
    def test_mean_and_pass_rate(self, mock_lc: MagicMock) -> None:
        mock_lc.scrub.side_effect = ["exact", "wrong"]

        cases = [
            TestCase(input="a", expected_output="exact", task="scrub"),
            TestCase(input="b", expected_output="other", task="scrub"),
        ]
        summary = Evaluator().run(cases)

        assert summary.mean_score == 0.5
        assert summary.pass_rate == 0.5


class TestEvaluatorWithTracker:
    """Verify tracker integration hooks are called."""

    @patch("loclean.eval.evaluator.loclean")
    def test_tracker_lifecycle(self, mock_lc: MagicMock) -> None:
        mock_lc.scrub.return_value = "ok"
        tracker = MagicMock()
        tracker.start_run.return_value = "run-1"

        case = TestCase(input="hi", expected_output="ok", task="scrub")
        Evaluator(tracker=tracker).run([case])

        tracker.start_run.assert_called_once()
        tracker.log_step.assert_called_once()
        assert tracker.log_score.call_count == 2  # per-case + mean
        tracker.end_run.assert_called_once_with("run-1")
