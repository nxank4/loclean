"""Batch evaluator that runs loclean functions over test cases."""

from __future__ import annotations

import json
import logging
from typing import Any

import narwhals as nw

import loclean
from loclean.eval.metrics import get_metric
from loclean.eval.schemas import EvalResult, EvalSummary, TestCase

logger = logging.getLogger(__name__)

# Avoid circular import at runtime; tracker is optional.
try:
    from loclean.tracking.base import BaseTracker
except ImportError:  # pragma: no cover
    BaseTracker = None  # type: ignore[assignment,misc]


class Evaluator:
    """Run a batch of ``TestCase`` instances and collect scored results.

    Args:
        tracker: Optional tracker for logging each step and score.
        **engine_kwargs: Forwarded to the underlying loclean functions
            (e.g. ``model``, ``host``, ``verbose``).
    """

    def __init__(
        self,
        tracker: Any | None = None,
        **engine_kwargs: Any,
    ) -> None:
        self._tracker = tracker
        self._engine_kwargs = engine_kwargs

    def _run_clean(self, text: str) -> str:
        """Wrap a single string in a DataFrame, clean, and extract result."""
        df: Any
        try:
            import polars as pl

            df = pl.DataFrame({"input": [text]})
        except ImportError:
            import pandas as pd

            df = pd.DataFrame({"input": [text]})

        result = loclean.clean(df, "input", **self._engine_kwargs)
        result_nw = nw.from_native(result)
        values = result_nw.get_column("clean_value").to_list()
        return str(values[0]) if values else ""

    def _run_scrub(self, text: str) -> str:
        return str(loclean.scrub(text, **self._engine_kwargs))

    def _run_extract(self, text: str, metadata: dict[str, Any]) -> str:
        schema_cls = metadata.get("schema")
        if schema_cls is None:
            raise ValueError("extract task requires metadata['schema']")
        result = loclean.extract(text, schema_cls, **self._engine_kwargs)
        if isinstance(result, dict):
            return json.dumps(result, default=str, sort_keys=True)
        return str(result)

    def _dispatch(self, case: TestCase) -> str:
        if case.task == "clean":
            return self._run_clean(case.input)
        if case.task == "scrub":
            return self._run_scrub(case.input)
        if case.task == "extract":
            return self._run_extract(case.input, case.metadata)
        raise ValueError(f"Unknown task {case.task!r}")

    def run(self, cases: list[TestCase]) -> EvalSummary:
        """Evaluate all test cases and return an aggregated summary.

        Args:
            cases: List of test case definitions.

        Returns:
            ``EvalSummary`` containing individual results and aggregate
            scores.
        """
        run_id: str | None = None
        if self._tracker is not None:
            run_id = self._tracker.start_run(
                name="loclean-eval",
                metadata={"num_cases": len(cases), **self._engine_kwargs},
            )

        results: list[EvalResult] = []
        for idx, case in enumerate(cases):
            try:
                actual = self._dispatch(case)
            except Exception as exc:
                logger.warning("Case %d failed: %s", idx, exc)
                actual = ""

            metric = get_metric(case.metric)
            score_val, detail = metric.score(
                actual, case.expected_output, case.metadata
            )

            result = EvalResult(
                test_case=case,
                actual_output=actual,
                score=score_val,
                passed=score_val >= 0.5,
                details=detail,
            )
            results.append(result)

            if self._tracker is not None and run_id is not None:
                self._tracker.log_step(
                    run_id=run_id,
                    step_name=f"case_{idx}_{case.task}",
                    input_text=case.input,
                    output_text=actual,
                    metadata={
                        "expected": case.expected_output,
                        "metric": case.metric,
                    },
                )
                self._tracker.log_score(
                    run_id=run_id,
                    name=case.metric,
                    value=score_val,
                    comment=detail,
                )

        summary = EvalSummary(results=results)

        if self._tracker is not None and run_id is not None:
            self._tracker.log_score(
                run_id=run_id,
                name="mean_score",
                value=summary.mean_score,
                comment=f"pass_rate={summary.pass_rate:.2%}",
            )
            self._tracker.end_run(run_id)

        return summary
