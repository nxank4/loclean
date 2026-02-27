"""Automated trap feature pruning via statistical profiling and LLM verification.

Identifies columns that look like valid signals but are actually
uncorrelated Gaussian noise (trap features).  Uses Narwhals for
backend-agnostic statistical profiling and an ``InferenceEngine``
for generative verification.
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


class _ColumnProfile:
    """Statistical profile for a single numeric column."""

    __slots__ = (
        "name",
        "mean",
        "std",
        "variance",
        "skewness",
        "kurtosis",
        "min_val",
        "max_val",
        "corr_with_target",
    )

    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        variance: float,
        skewness: float,
        kurtosis: float,
        min_val: float,
        max_val: float,
        corr_with_target: float,
    ) -> None:
        self.name = name
        self.mean = mean
        self.std = std
        self.variance = variance
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.min_val = min_val
        self.max_val = max_val
        self.corr_with_target = corr_with_target

    def to_dict(self) -> dict[str, Any]:
        """Serialise profile to a plain dictionary."""
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "min": self.min_val,
            "max": self.max_val,
            "corr_with_target": self.corr_with_target,
        }


class TrapPruner:
    """Identify and remove trap features from a DataFrame.

    Trap features are columns of uncorrelated Gaussian noise that
    masquerade as valid signals.  Detection relies entirely on
    statistical distributions and target correlations — column names
    are deliberately ignored.

    Args:
        inference_engine: Ollama (or compatible) engine for verification.
        correlation_threshold: Absolute correlation below which a
            column is considered uncorrelated.  Default ``0.05``.
        max_retries: LLM generation retry budget.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        *,
        correlation_threshold: float = 0.05,
        max_retries: int = 2,
    ) -> None:
        self.inference_engine = inference_engine
        self.correlation_threshold = correlation_threshold
        self.max_retries = max_retries

    def prune(
        self,
        df: IntoFrameT,
        target_col: str,
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Profile, verify, and drop trap features.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column name of the prediction target.

        Returns:
            Tuple of ``(pruned_df, summary)`` where *summary* contains
            ``dropped_columns`` (list of removed names) and
            ``verdicts`` (per-column LLM reasoning).
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]

        if target_col not in df_nw.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric_cols = [
            c for c in df_nw.columns if c != target_col and df_nw[c].dtype.is_numeric()
        ]

        if not numeric_cols:
            logger.info("No numeric feature columns to evaluate.")
            return df, {"dropped_columns": [], "verdicts": []}

        profiles = self._profile_columns(df_nw, target_col, numeric_cols)

        col_map, prompt = self._build_prompt(profiles)

        verdicts = self._verify_with_llm(prompt, col_map)

        trap_cols = [v["column"] for v in verdicts if v.get("is_trap")]

        summary: dict[str, Any] = {
            "dropped_columns": trap_cols,
            "verdicts": verdicts,
        }

        if trap_cols:
            logger.info(
                "Dropping %d trap feature(s): %s",
                len(trap_cols),
                trap_cols,
            )
            pruned_nw = df_nw.drop(trap_cols)
            return nw.to_native(pruned_nw), summary  # type: ignore[type-var]

        logger.info("No trap features detected.")
        return df, summary

    # ------------------------------------------------------------------
    # Statistical profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_columns(
        df_nw: nw.DataFrame[Any],
        target_col: str,
        numeric_cols: list[str],
    ) -> list[_ColumnProfile]:
        """Compute distribution statistics for each numeric column.

        All operations use the Narwhals interface.  Division-by-zero
        and other math errors are caught per-column.
        """
        n = df_nw.shape[0]
        if n < 2:
            return []

        target_series = df_nw[target_col].cast(nw.Float64)
        target_mean = target_series.mean()
        target_std = target_series.std()

        profiles: list[_ColumnProfile] = []

        for col in numeric_cols:
            try:
                series = df_nw[col].cast(nw.Float64)
                col_mean = series.mean()
                col_std = series.std()

                diffs = series - col_mean
                variance = (diffs * diffs).mean()

                if col_std and col_std > 0 and target_std and target_std > 0:
                    corr = float(
                        ((series - col_mean) * (target_series - target_mean)).mean()
                        / (col_std * target_std)
                    )
                else:
                    corr = 0.0

                if col_std and col_std > 0:
                    skewness = float((diffs**3).mean() / (col_std**3))
                    kurtosis = float((diffs**4).mean() / (col_std**4)) - 3.0
                else:
                    skewness = 0.0
                    kurtosis = 0.0

                profiles.append(
                    _ColumnProfile(
                        name=col,
                        mean=float(col_mean) if col_mean is not None else 0.0,
                        std=float(col_std) if col_std is not None else 0.0,
                        variance=float(variance) if variance is not None else 0.0,
                        skewness=skewness,
                        kurtosis=kurtosis,
                        min_val=float(series.min()),
                        max_val=float(series.max()),
                        corr_with_target=corr,
                    )
                )
            except (ZeroDivisionError, ValueError, OverflowError):
                profiles.append(
                    _ColumnProfile(
                        name=col,
                        mean=0.0,
                        std=0.0,
                        variance=0.0,
                        skewness=0.0,
                        kurtosis=0.0,
                        min_val=0.0,
                        max_val=0.0,
                        corr_with_target=0.0,
                    )
                )

        return profiles

    # ------------------------------------------------------------------
    # Prompt construction (anonymised)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        profiles: list[_ColumnProfile],
    ) -> tuple[dict[str, str], str]:
        """Build the LLM verification prompt with anonymised column IDs.

        Returns:
            Tuple of ``(col_map, prompt_text)`` where *col_map* maps
            ``"col_0"`` → real column name.
        """
        col_map: dict[str, str] = {}
        lines: list[str] = []

        for i, p in enumerate(profiles):
            anon = f"col_{i}"
            col_map[anon] = p.name

            lines.append(
                f"Column {anon}: "
                f"mean={p.mean:.4f}, std={p.std:.4f}, "
                f"variance={p.variance:.4f}, "
                f"skewness={p.skewness:.4f}, kurtosis={p.kurtosis:.4f}, "
                f"min={p.min_val:.4f}, max={p.max_val:.4f}, "
                f"corr_with_target={p.corr_with_target:.4f}"
            )

        profile_block = "\n".join(lines)

        prompt = (
            "You are a statistical analyst. Below are the statistical profiles "
            "of several numeric columns from a dataset. Each column is "
            "identified only by an anonymous ID (column names are hidden).\n\n"
            f"{profile_block}\n\n"
            "A **trap feature** is a column that:\n"
            "1. Exhibits a distribution close to standard Gaussian "
            "(skewness ≈ 0, kurtosis ≈ 0, i.e. excess kurtosis near zero).\n"
            "2. Has a correlation with the target variable very close to "
            "zero (|corr| < 0.05).\n\n"
            "Analyse each column and output ONLY a JSON array. "
            "For each column output an object with exactly three keys:\n"
            '- "column": the anonymous ID (e.g. "col_0")\n'
            '- "is_trap": boolean\n'
            '- "reason": brief explanation\n\n'
            "Output ONLY the JSON array, no other text."
        )

        return col_map, prompt

    # ------------------------------------------------------------------
    # LLM verification
    # ------------------------------------------------------------------

    def _verify_with_llm(
        self,
        prompt: str,
        col_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Send the prompt to the LLM and parse the verdict."""
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.inference_engine.generate(prompt)
                return self._parse_verdict(str(raw).strip(), col_map)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning(
                    "LLM verdict parsing failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
        logger.warning("Could not parse LLM verdicts — keeping all columns.")
        return [
            {"column": real, "is_trap": False, "reason": "LLM parse failure"}
            for real in col_map.values()
        ]

    @staticmethod
    def _parse_verdict(
        response: str,
        col_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Parse the JSON verdict and map anonymous IDs back to real names."""
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in LLM response")

        items: list[dict[str, Any]] = json.loads(text[start : end + 1])
        verdicts: list[dict[str, Any]] = []

        for item in items:
            anon_id = item["column"]
            real_name = col_map.get(anon_id, anon_id)
            verdicts.append(
                {
                    "column": real_name,
                    "is_trap": bool(item.get("is_trap", False)),
                    "reason": str(item.get("reason", "")),
                }
            )

        return verdicts
