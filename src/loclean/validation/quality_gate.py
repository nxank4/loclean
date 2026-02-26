"""Quality gate evaluation via LLM-driven rule compliance checking.

Checks each row of a DataFrame against a set of natural-language rules
using the local Ollama engine. Produces a :class:`QualityReport` with
compliance rate, per-failure reasoning, and programmatic dict output.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import narwhals as nw
from pydantic import BaseModel, Field

from loclean.extraction.json_repair import repair_json
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class _RowCompliance(BaseModel):
    """LLM-constrained output for a single row evaluation."""

    compliant: bool = Field(
        ...,
        description="Whether the row satisfies all specified rules",
    )
    reasoning: str = Field(
        ...,
        description="Logical explanation for the compliance decision",
    )


class QualityReport(BaseModel):
    """Aggregated compliance report across evaluated rows.

    Attributes:
        total_rows: Number of rows evaluated.
        passed_rows: Number of rows that passed all rules.
        compliance_rate: Fraction of rows that passed (0.0–1.0).
        failures: List of dicts, each containing ``row`` data and
            ``reasoning`` for non-compliant rows.
    """

    total_rows: int
    passed_rows: int
    compliance_rate: float
    failures: list[dict[str, Any]] = Field(default_factory=list)


class QualityGate:
    """Evaluate data quality against natural-language rules.

    Prompts the Ollama engine to check each sampled row against the
    provided rules, collecting boolean compliance flags and reasoning.

    Args:
        inference_engine: Engine used for rule evaluation.
        batch_size: Rows processed per LLM call batch.
        sample_size: Maximum rows to evaluate (sampled if exceeded).
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        batch_size: int = 20,
        sample_size: int = 100,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")
        if sample_size < 1:
            raise ValueError("sample_size must be ≥ 1")
        self.inference_engine = inference_engine
        self.batch_size = batch_size
        self.sample_size = sample_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: IntoFrameT,
        rules: list[str],
    ) -> dict[str, Any]:
        """Evaluate a DataFrame against natural-language rules.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            rules: List of natural-language constraint strings.

        Returns:
            Dictionary representation of :class:`QualityReport`.

        Raises:
            ValueError: If *rules* is empty.
        """
        if not rules:
            raise ValueError("At least one rule must be provided")

        df_nw = nw.from_native(df)  # type: ignore[type-var]
        rows = self._sample_rows(df_nw)

        if not rows:
            report = QualityReport(
                total_rows=0,
                passed_rows=0,
                compliance_rate=1.0,
            )
            return report.model_dump()

        passed = 0
        failures: list[dict[str, Any]] = []

        for i in range(0, len(rows), self.batch_size):
            batch = rows[i : i + self.batch_size]
            for row in batch:
                compliance = self._check_row(row, rules)
                if compliance.compliant:
                    passed += 1
                else:
                    failures.append(
                        {
                            "row": row,
                            "reasoning": compliance.reasoning,
                        }
                    )

        total = len(rows)
        rate = passed / total if total > 0 else 1.0

        report = QualityReport(
            total_rows=total,
            passed_rows=passed,
            compliance_rate=round(rate, 4),
            failures=failures,
        )

        logger.info(
            f"[green]✓[/green] Quality gate: "
            f"[bold]{passed}/{total}[/bold] rows compliant "
            f"({report.compliance_rate:.1%})"
        )

        return report.model_dump()

    # ------------------------------------------------------------------
    # Row sampling
    # ------------------------------------------------------------------

    def _sample_rows(self, df_nw: nw.DataFrame[Any]) -> list[dict[str, Any]]:
        """Sample up to *sample_size* rows from the DataFrame.

        Args:
            df_nw: Narwhals DataFrame.

        Returns:
            List of row dicts.
        """
        all_rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]
        if len(all_rows) <= self.sample_size:
            return all_rows

        step = len(all_rows) / self.sample_size
        return [all_rows[int(i * step)] for i in range(self.sample_size)]

    # ------------------------------------------------------------------
    # Per-row compliance check
    # ------------------------------------------------------------------

    def _check_row(
        self,
        row: dict[str, Any],
        rules: list[str],
    ) -> _RowCompliance:
        """Prompt the engine to evaluate a single row against rules.

        Args:
            row: Row data as a dictionary.
            rules: Natural-language constraint strings.

        Returns:
            Parsed compliance result.
        """
        rules_text = "\n".join(f"  {i + 1}. {rule}" for i, rule in enumerate(rules))
        prompt = (
            "You are a data quality auditor.\n\n"
            "Given this data row:\n"
            f"{json.dumps(row, ensure_ascii=False, default=str)}\n\n"
            "Evaluate whether it satisfies ALL of the following rules:\n"
            f"{rules_text}\n\n"
            "Return a JSON object with:\n"
            '- "compliant": true/false\n'
            '- "reasoning": a brief explanation of your decision'
        )

        raw = self.inference_engine.generate(prompt, schema=_RowCompliance)
        return self._parse_compliance(raw)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_compliance(raw: Any) -> _RowCompliance:
        """Best-effort parse of the LLM response into a compliance result.

        Args:
            raw: Raw output from the inference engine.

        Returns:
            Parsed ``_RowCompliance`` (defaults to non-compliant on
            total parse failure).
        """
        if isinstance(raw, dict):
            try:
                return _RowCompliance(**raw)
            except Exception:
                pass

        text = str(raw) if not isinstance(raw, str) else raw

        try:
            parsed = json.loads(text)
            return _RowCompliance(**parsed)
        except (json.JSONDecodeError, TypeError, Exception):
            pass

        try:
            repaired = repair_json(text)
            if isinstance(repaired, dict):
                return _RowCompliance(**repaired)
            parsed_r = json.loads(repaired)  # type: ignore[arg-type]
            return _RowCompliance(**parsed_r)
        except Exception:
            pass

        logger.warning(
            "[yellow]⚠[/yellow] Could not parse compliance response. "
            "Marking row as non-compliant."
        )
        return _RowCompliance(
            compliant=False,
            reasoning="Failed to parse LLM response",
        )
