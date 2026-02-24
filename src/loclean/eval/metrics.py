"""Evaluation metrics for Loclean outputs.

Each metric implements a ``score`` method that compares an actual output
against an expected output and returns a ``(score, detail)`` tuple.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics."""

    name: str = "base"

    @abstractmethod
    def score(
        self,
        actual: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """Return ``(score_0_to_1, detail_string)``."""
        ...


class ExactMatch(BaseMetric):
    """Binary exact-match after whitespace normalisation."""

    name: str = "exact_match"

    def score(
        self,
        actual: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        if actual.strip() == expected.strip():
            return 1.0, "exact match"
        return 0.0, f"expected={expected!r}, got={actual!r}"


class PartialJSONMatch(BaseMetric):
    """Fraction of matching top-level keys between two JSON objects."""

    name: str = "partial_json_match"

    def score(
        self,
        actual: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        try:
            actual_obj = json.loads(actual)
            expected_obj = json.loads(expected)
        except json.JSONDecodeError as exc:
            return 0.0, f"JSON parse error: {exc}"

        if not isinstance(expected_obj, dict) or not isinstance(actual_obj, dict):
            return 0.0, "both sides must be JSON objects"

        if not expected_obj:
            return 1.0, "empty expected object"

        matched = sum(
            1 for k, v in expected_obj.items() if k in actual_obj and actual_obj[k] == v
        )
        total = len(expected_obj)
        ratio = matched / total
        return ratio, f"{matched}/{total} keys matched"


class PIIMaskingRecall(BaseMetric):
    """Fraction of known PII tokens successfully masked (absent from output).

    Expects ``metadata["pii_tokens"]`` to contain the list of raw PII
    strings that should have been redacted.
    """

    name: str = "pii_masking_recall"

    def score(
        self,
        actual: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        tokens: list[str] = (metadata or {}).get("pii_tokens", [])
        if not tokens:
            return 1.0, "no PII tokens to check"

        masked = sum(1 for t in tokens if t not in actual)
        total = len(tokens)
        ratio = masked / total
        return ratio, f"{masked}/{total} tokens masked"


METRICS: dict[str, BaseMetric] = {
    "exact_match": ExactMatch(),
    "partial_json_match": PartialJSONMatch(),
    "pii_masking_recall": PIIMaskingRecall(),
}


def get_metric(name: str) -> BaseMetric:
    """Retrieve a metric by name.

    Raises:
        ValueError: If the metric name is unknown.
    """
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}. Available: {list(METRICS.keys())}")
    return METRICS[name]
