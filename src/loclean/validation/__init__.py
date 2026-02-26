"""Explainable data quality gates.

Evaluate structured data against natural-language constraints and produce
a compliance report with per-row reasoning.
"""

from loclean.validation.quality_gate import QualityGate, QualityReport

__all__ = ["QualityGate", "QualityReport"]
