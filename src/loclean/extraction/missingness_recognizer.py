"""Missingness pattern recognition via LLM-driven MNAR detection.

Identifies Missing Not At Random (MNAR) patterns where the probability
of a value being missing in feature X depends on the value of feature Y:

    P(X_missing | Y) ≠ P(X_missing)

Uses Narwhals for backend-agnostic null analysis and an InferenceEngine
to infer structural correlations from data samples.  Detected patterns
are encoded as new boolean feature columns.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw

from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)

_SUFFIX = "_mnar"


class MissingnessRecognizer:
    """Detect MNAR patterns and encode them as boolean feature flags.

    For each column containing nulls the recognizer:

    1. Samples rows where the column is null alongside other features.
    2. Prompts the LLM to identify structural correlations.
    3. Compiles the LLM-generated ``encode_missingness`` function in a
       sandbox.
    4. Applies the function across the DataFrame to create a boolean
       ``{col}_mnar`` column.

    Args:
        inference_engine: Ollama (or compatible) engine.
        sample_size: Maximum null rows to sample per column.
        max_retries: LLM code-generation retry budget.
        timeout_s: Per-row execution timeout in seconds.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        *,
        sample_size: int = 50,
        max_retries: int = 3,
        timeout_s: float = 2.0,
    ) -> None:
        self.inference_engine = inference_engine
        self.sample_size = sample_size
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def recognize(
        self,
        df: IntoFrameT,
        target_cols: list[str] | None = None,
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Detect MNAR patterns and add boolean feature columns.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_cols: Columns to analyse for missingness.  If
                ``None``, all columns containing nulls are evaluated.

        Returns:
            Tuple of ``(augmented_df, summary)`` where *summary*
            maps each analysed column to its pattern description
            or ``None`` if no pattern was found.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]

        null_cols = self._find_null_columns(df_nw)

        if target_cols is not None:
            null_cols = [c for c in target_cols if c in null_cols]

        if not null_cols:
            logger.info("No columns with null values to analyse.")
            return df, {"patterns": {}}

        all_cols = df_nw.columns
        patterns: dict[str, Any] = {}
        new_columns: dict[str, list[bool]] = {}

        for col in null_cols:
            context_cols = [c for c in all_cols if c != col]
            sample = self._sample_null_context(df_nw, col, context_cols)

            if not sample:
                patterns[col] = None
                continue

            prompt = self._build_prompt(col, context_cols, sample)
            fn = self._generate_and_compile(prompt)

            if fn is None:
                patterns[col] = None
                continue

            ok, error = self._verify_encoder(fn, sample)
            if not ok:
                logger.warning("Encoder for '%s' failed verification: %s", col, error)
                patterns[col] = None
                continue

            flags = self._apply_encoder(df_nw, fn)
            col_name = f"{col}{_SUFFIX}"
            new_columns[col_name] = flags
            patterns[col] = {
                "encoded_as": col_name,
                "null_count": sum(1 for v in df_nw[col].to_list() if v is None),
                "pattern_flags_true": sum(flags),
            }
            logger.info(
                "Encoded MNAR pattern for '%s' → '%s' (%d flagged)",
                col,
                col_name,
                sum(flags),
            )

        if new_columns:
            native_ns = nw.get_native_namespace(df_nw)
            rows_data: dict[str, list[Any]] = {
                c: df_nw[c].to_list() for c in df_nw.columns
            }
            rows_data.update(new_columns)
            result_nw = nw.from_dict(rows_data, backend=native_ns)
            return nw.to_native(result_nw), {"patterns": patterns}

        return df, {"patterns": patterns}

    # ------------------------------------------------------------------
    # Null detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_null_columns(df_nw: nw.DataFrame[Any]) -> list[str]:
        """Return column names that contain at least one null."""
        return [col for col in df_nw.columns if df_nw[col].null_count() > 0]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_null_context(
        df_nw: nw.DataFrame[Any],
        null_col: str,
        context_cols: list[str],
        max_rows: int = 50,
    ) -> list[dict[str, Any]]:
        """Extract rows where *null_col* is null with context values.

        Returns a list of dicts, each containing the context column
        values for a row where *null_col* is missing.
        """
        null_mask = df_nw[null_col].is_null()
        null_rows = df_nw.filter(null_mask)

        if null_rows.shape[0] == 0:
            return []

        n = min(null_rows.shape[0], max_rows)
        sampled = null_rows.head(n)

        select_cols = [c for c in context_cols if c in sampled.columns]
        if not select_cols:
            return []

        return sampled.select(select_cols).rows(named=True)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        null_col: str,
        context_cols: list[str],
        sample_rows: list[dict[str, Any]],
    ) -> str:
        """Build the LLM prompt for pattern inference."""
        sample_str = json.dumps(sample_rows[:20], indent=2, default=str)

        return (
            "You are a data scientist analysing missing data patterns.\n\n"
            f"Column '{null_col}' has missing values. Below are sample rows "
            "where this column IS NULL, showing the values of the other "
            "columns:\n\n"
            f"Context columns: {context_cols}\n\n"
            f"Sample rows (where '{null_col}' is null):\n{sample_str}\n\n"
            "Task: Identify if there is a structural pattern that predicts "
            f"when '{null_col}' is missing based on other column values.\n\n"
            "Write a pure Python function with this exact signature:\n\n"
            "def encode_missingness(row: dict) -> bool:\n"
            "    ...\n\n"
            "The function receives a dict of ALL column values for a row "
            "(including the target column) and returns True if the "
            "missingness pattern is detected.\n\n"
            "Rules:\n"
            "- Use ONLY standard library modules (math, statistics, operator)\n"
            "- Wrap logic in try/except returning False on failure\n"
            "- Return a single boolean value\n"
            "- Do NOT use markdown fences, comments, or prose\n"
            "- Output ONLY the function code, nothing else\n\n"
            "Example:\n"
            "def encode_missingness(row: dict) -> bool:\n"
            "    try:\n"
            "        return row.get('category') == 'electronics' "
            "and row.get('price', 0) > 500\n"
            "    except Exception:\n"
            "        return False\n"
        )

    # ------------------------------------------------------------------
    # Code generation + compilation
    # ------------------------------------------------------------------

    def _generate_and_compile(
        self,
        prompt: str,
    ) -> Callable[[dict[str, Any]], bool] | None:
        """Generate, sanitize, and compile the encoder function."""
        import re

        from loclean.utils.sandbox import compile_sandboxed

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = str(self.inference_engine.generate(prompt)).strip()
                source = re.sub(r"```(?:python)?\s*\n?", "", raw).strip()
                fn = compile_sandboxed(
                    source,
                    "encode_missingness",
                    ["math", "statistics", "operator"],
                )
                return fn  # type: ignore[return-value]
            except (ValueError, SyntaxError) as exc:
                logger.warning(
                    "⚠ Code generation failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        logger.warning(
            "Could not compile encoder after %d retries.",
            self.max_retries,
        )
        return None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_encoder(
        fn: Callable[[dict[str, Any]], bool],
        sample_rows: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Test the encoder on sample rows."""
        from loclean.utils.sandbox import run_with_timeout

        for row in sample_rows[:5]:
            result, error = run_with_timeout(fn, (row,), 2.0)
            if error:
                return False, f"Execution error: {error}"
            if not isinstance(result, bool):
                return False, f"Expected bool, got {type(result).__name__}"

        return True, ""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def _apply_encoder(
        self,
        df_nw: nw.DataFrame[Any],
        fn: Callable[[dict[str, Any]], bool],
    ) -> list[bool]:
        """Apply the encoder across all rows."""
        from loclean.utils.sandbox import run_with_timeout

        rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]
        flags: list[bool] = []

        for row in rows:
            result, error = run_with_timeout(fn, (row,), self.timeout_s)
            if error or not isinstance(result, bool):
                flags.append(False)
            else:
                flags.append(result)

        return flags
