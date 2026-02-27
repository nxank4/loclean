"""Semantic target leakage detection via LLM-driven timeline evaluation.

Identifies features that mathematically or logically imply the target
variable — i.e. columns containing information generated *after* the
target event occurs.  In a deterministic leakage scenario:

    P(Y | X_i) ≈ 1

The generative engine acts as a semantic auditor, catching logical
leakage that basic statistical tests miss by evaluating the causal
timeline of each feature relative to the target outcome.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import narwhals as nw

from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class TargetLeakageAuditor:
    """Detect and remove features that leak the target variable.

    For each feature column the auditor prompts the LLM with the
    dataset domain description and a representative sample, asking
    it to evaluate whether the feature could only be known *after*
    the target outcome is determined.

    Args:
        inference_engine: Ollama (or compatible) engine.
        max_retries: LLM generation retry budget.
        sample_n: Number of sample rows to include in the prompt.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        *,
        max_retries: int = 2,
        sample_n: int = 10,
    ) -> None:
        self.inference_engine = inference_engine
        self.max_retries = max_retries
        self.sample_n = sample_n

    def audit(
        self,
        df: IntoFrameT,
        target_col: str,
        domain: str = "",
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Audit features for target leakage and drop offenders.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column name of the prediction target.
            domain: Brief text description of the dataset domain
                (e.g. ``"hospital readmission prediction"``).

        Returns:
            Tuple of ``(pruned_df, summary)`` where *summary*
            contains ``dropped_columns`` and per-column ``verdicts``.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]

        if target_col not in df_nw.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        feature_cols = [c for c in df_nw.columns if c != target_col]
        if not feature_cols:
            logger.info("No feature columns to audit.")
            return df, {"dropped_columns": [], "verdicts": []}

        state = self._extract_state(df_nw, target_col, feature_cols)
        prompt = self._build_prompt(state, domain)
        verdicts = self._evaluate_with_llm(prompt)

        leaked = [v["column"] for v in verdicts if v.get("is_leakage")]
        valid_leaked = [c for c in leaked if c in feature_cols]

        summary: dict[str, Any] = {
            "dropped_columns": valid_leaked,
            "verdicts": verdicts,
        }

        if valid_leaked:
            logger.info(
                "Dropping %d leaked feature(s): %s",
                len(valid_leaked),
                valid_leaked,
            )
            try:
                pruned_nw = df_nw.drop(valid_leaked)
                return nw.to_native(pruned_nw), summary  # type: ignore[type-var]
            except Exception as exc:
                logger.warning("Failed to drop columns: %s", exc)
                return df, summary

        logger.info("No target leakage detected.")
        return df, summary

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_state(
        df_nw: nw.DataFrame[Any],
        target_col: str,
        feature_cols: list[str],
        sample_n: int = 10,
    ) -> dict[str, Any]:
        """Build structural metadata for the LLM prompt.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Target variable column.
            feature_cols: Feature column names.
            sample_n: Number of sample rows.

        Returns:
            Dict with ``target_col``, ``features``, ``dtypes``,
            and ``sample_rows``.
        """
        n = min(df_nw.shape[0], sample_n)
        sampled = df_nw.head(n)
        sample_rows = sampled.rows(named=True)

        dtypes = {col: str(df_nw[col].dtype) for col in feature_cols}

        return {
            "target_col": target_col,
            "features": feature_cols,
            "dtypes": dtypes,
            "sample_rows": sample_rows,  # type: ignore[dict-item]
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        state: dict[str, Any],
        domain: str,
    ) -> str:
        """Build the LLM prompt for timeline evaluation."""
        target = state["target_col"]
        features = state["features"]
        dtypes = state["dtypes"]
        sample_str = json.dumps(state["sample_rows"][:10], indent=2, default=str)

        domain_line = f"Dataset domain: {domain}\n" if domain else ""

        feature_info = "\n".join(
            f"  - {f} (dtype: {dtypes.get(f, 'unknown')})" for f in features
        )

        return (
            "You are a machine learning auditor specialising in data "
            "leakage detection.\n\n"
            f"{domain_line}"
            f"Target variable: '{target}'\n\n"
            f"Feature columns:\n{feature_info}\n\n"
            f"Sample rows:\n{sample_str}\n\n"
            "Task: For each feature column, evaluate whether it could "
            "constitute **target leakage** — meaning the feature contains "
            "information that would only be available AFTER the target "
            "outcome is determined.\n\n"
            "Consider:\n"
            "- Temporal ordering: was this feature generated before or "
            "after the target event?\n"
            "- Semantic meaning: does the feature directly encode or "
            "trivially derive from the target?\n"
            "- Statistical signal: extremely high correlation may "
            "indicate leakage, not just a good predictor.\n\n"
            "Output ONLY a JSON array. For each feature, output an "
            "object with exactly three keys:\n"
            '- "column": the feature name\n'
            '- "is_leakage": boolean\n'
            '- "reason": brief explanation\n\n'
            "Output ONLY the JSON array, no other text."
        )

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_with_llm(
        self,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Send the prompt and parse the leakage verdicts."""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = str(self.inference_engine.generate(prompt)).strip()
                return self._parse_verdict(raw)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning(
                    "LLM verdict parsing failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
        logger.warning("Could not parse LLM verdicts — keeping all columns.")
        return []

    @staticmethod
    def _parse_verdict(response: str) -> list[dict[str, Any]]:
        """Parse the JSON verdict from the LLM response."""
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in LLM response")

        items: list[dict[str, Any]] = json.loads(text[start : end + 1])
        verdicts: list[dict[str, Any]] = []

        for item in items:
            verdicts.append(
                {
                    "column": str(item["column"]),
                    "is_leakage": bool(item.get("is_leakage", False)),
                    "reason": str(item.get("reason", "")),
                }
            )

        return verdicts
