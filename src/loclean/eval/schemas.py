"""Evaluation schemas for Loclean test cases and results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """Single evaluation case definition.

    Args:
        input: Raw text to feed into the loclean function.
        expected_output: Ground-truth string to compare against.
        task: Which loclean function to invoke.
        metric: Name of the metric to score with.
        metadata: Extra context forwarded to the metric (e.g. pii_tokens).
    """

    input: str
    expected_output: str
    task: Literal["clean", "scrub", "extract"] = "clean"
    metric: str = "exact_match"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Outcome of evaluating a single test case."""

    test_case: TestCase
    actual_output: str
    score: float
    passed: bool
    details: str = ""


class EvalSummary(BaseModel):
    """Aggregated metrics across an evaluation run."""

    results: list[EvalResult]

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)
