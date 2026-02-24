"""Loclean evaluation framework.

Provides test-case schemas, composable metrics, and a batch evaluator
for measuring the quality of ``clean()``, ``scrub()``, and ``extract()``
outputs against ground-truth expectations.
"""

from loclean.eval.evaluator import Evaluator
from loclean.eval.metrics import (BaseMetric, ExactMatch, PartialJSONMatch,
                                  PIIMaskingRecall, get_metric)
from loclean.eval.schemas import EvalResult, EvalSummary, TestCase

__all__ = [
    "BaseMetric",
    "EvalResult",
    "EvalSummary",
    "Evaluator",
    "ExactMatch",
    "PartialJSONMatch",
    "PIIMaskingRecall",
    "TestCase",
    "get_metric",
]
