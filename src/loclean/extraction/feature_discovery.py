"""Generative feature cross discovery via LLM-driven transformation proposals.

Automates feature engineering by prompting the Ollama engine to propose
mathematical transformations between existing columns that maximise
mutual information :math:`I(X_{new}; Y)` with the target variable.
The proposed function is compiled via ``exec`` and applied natively
across the Narwhals DataFrame.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw

from loclean.utils.cache_keys import compute_code_key
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class FeatureDiscovery:
    """Propose and compile feature crosses using an LLM.

    Extracts structural metadata (column names, dtypes, sample rows)
    and prompts the engine to write a ``generate_features`` function
    that produces *n_features* new columns from existing ones.

    Args:
        inference_engine: Engine for generative requests.
        n_features: Number of new features to propose.
        max_retries: Repair budget for the compilation loop.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        n_features: int = 5,
        max_retries: int = 3,
        timeout_s: float = 2.0,
        cache: LocleanCache | None = None,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be ≥ 1")
        if max_retries < 1:
            raise ValueError("max_retries must be ≥ 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.inference_engine = inference_engine
        self.n_features = n_features
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        df: IntoFrameT,
        target_col: str,
    ) -> IntoFrameT:
        """Discover and apply feature crosses to the DataFrame.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column name of the target variable.

        Returns:
            DataFrame augmented with new feature columns, or the
            original DataFrame unchanged if generation fails.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]
        if target_col not in df_nw.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        state = self._extract_state(df_nw, target_col)

        cache_key = compute_code_key(
            columns=state["columns"],
            dtypes=list(state["dtypes"].values()),
            target_col=target_col,
            module_prefix="feature_discovery",
        )

        cached_source = self.cache.get_code(cache_key) if self.cache else None
        if cached_source is not None:
            fn = self._compile_function(cached_source)
            sample_rows = state["sample_rows"]
            ok, _ = self._verify_function(fn, sample_rows, self.timeout_s)
            if ok:
                logger.info("[green]✓[/green] Cache hit — reusing compiled features")
                result = self._apply_to_dataframe(df_nw, fn, self.timeout_s)
                return result.to_native()  # type: ignore[no-any-return,return-value]

        source = self._propose_features(state)
        sample_rows = state["sample_rows"]

        try:
            fn = self._compile_function(source)
            ok, error = self._verify_function(fn, sample_rows, self.timeout_s)
        except ValueError as exc:
            ok, error = False, str(exc)

        retries = 0
        while not ok and retries < self.max_retries:
            retries += 1
            logger.warning(
                f"[yellow]⚠[/yellow] Retrying code generation "
                f"({retries}/{self.max_retries}): {error}"
            )
            source = self._repair_function(source, error, state)
            try:
                fn = self._compile_function(source)
                ok, error = self._verify_function(fn, sample_rows, self.timeout_s)
            except ValueError as exc:
                ok, error = False, str(exc)

        if not ok:
            logger.warning(
                f"[yellow]⚠[/yellow] The model could not generate valid Python "
                f"code after {self.max_retries} retries. This is not a library "
                f"bug — smaller models (e.g. phi3) sometimes produce syntax "
                f"errors or invalid logic. Returning the original DataFrame.\n"
                f"  [dim]Last error: {error}[/dim]\n"
                f"  [dim]Tip: try a larger model "
                f"(model='qwen2.5-coder:7b') or increase max_retries.[/dim]"
            )
            return df

        if self.cache:
            self.cache.set_code(cache_key, source)

        result = self._apply_to_dataframe(df_nw, fn, self.timeout_s)

        logger.info(
            "[green]✓[/green] Discovered and applied "
            f"[bold]{self.n_features}[/bold] new feature columns"
        )

        return result.to_native()  # type: ignore[no-any-return,return-value]

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_state(
        df_nw: nw.DataFrame[Any],
        target_col: str,
        sample_n: int = 10,
    ) -> dict[str, Any]:
        """Build structural metadata for prompting.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Target variable column.
            sample_n: Number of sample rows to include.

        Returns:
            Dict with ``columns``, ``dtypes``, ``target_col``,
            and ``sample_rows``.
        """
        columns = df_nw.columns
        dtypes = {col: str(df_nw[col].dtype) for col in columns}
        all_rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]

        if len(all_rows) <= sample_n:
            sample_rows = all_rows
        else:
            step = len(all_rows) / sample_n
            sample_rows = [all_rows[int(i * step)] for i in range(sample_n)]

        return {
            "columns": columns,
            "dtypes": dtypes,
            "target_col": target_col,
            "sample_rows": sample_rows,
        }

    # ------------------------------------------------------------------
    # Generative proposal
    # ------------------------------------------------------------------

    def _propose_features(self, state: dict[str, Any]) -> str:
        """Prompt the engine to write a generate_features function.

        Args:
            state: Structural metadata from _extract_state.

        Returns:
            Python source code string.
        """
        col_info = json.dumps(state["dtypes"], indent=2)
        samples = json.dumps(state["sample_rows"][:5], ensure_ascii=False, default=str)

        prompt = (
            "You are an expert feature engineer.\n\n"
            "Given a dataset with these columns and types:\n"
            f"{col_info}\n\n"
            f"Target variable: {state['target_col']}\n\n"
            "Sample rows:\n"
            f"{samples}\n\n"
            f"Propose exactly {self.n_features} mathematical "
            "transformations between existing columns that would "
            "maximise mutual information I(X_new; Y) with the target.\n\n"
            "Write a pure Python function with this exact signature:\n\n"
            "def generate_features(row: dict) -> dict:\n\n"
            "EXAMPLE (for a different dataset with columns "
            "'age', 'income', 'debt'):\n\n"
            "import math\n\n"
            "def generate_features(row: dict) -> dict:\n"
            "    result = {}\n"
            "    try:\n"
            "        result['debt_to_income'] = "
            "row['debt'] / row['income'] if row['income'] else None\n"
            "    except Exception:\n"
            "        result['debt_to_income'] = None\n"
            "    try:\n"
            "        result['log_income'] = "
            "math.log(row['income']) if row['income'] and "
            "row['income'] > 0 else None\n"
            "    except Exception:\n"
            "        result['log_income'] = None\n"
            "    return result\n\n"
            "Now write yours for the dataset above. The function must:\n"
            "- Accept a dict of column_name: value pairs\n"
            f"- Return a dict with exactly {self.n_features} new "
            "key-value pairs (the new feature names and values)\n"
            "- Use ONLY standard library modules (math, statistics, operator)\n"
            "- Wrap each calculation in try/except, defaulting to "
            "None on failure\n"
            "- Use descriptive feature names like 'ratio_a_b' or "
            "'log_amount'\n\n"
            "Return ONLY the Python function code. "
            "No markdown fences, no explanations."
        )

        raw = self.inference_engine.generate(prompt)
        return str(raw).strip()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_function(
        source: str,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Compile source code in a restricted sandbox.

        Applies deterministic sanitization before compilation to fix
        common LLM output artifacts (markdown fences, non-ASCII
        operators, invalid literals, etc.).

        Args:
            source: Python source containing ``generate_features``.

        Returns:
            The compiled function.

        Raises:
            ValueError: If compilation fails or function not found.
        """
        from loclean.utils.sandbox import compile_sandboxed
        from loclean.utils.source_sanitizer import sanitize_source

        return compile_sandboxed(
            sanitize_source(source),
            "generate_features",
            ["math", "statistics", "operator"],
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_function(
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        sample_rows: list[dict[str, Any]],
        timeout_s: float = 2.0,
    ) -> tuple[bool, str]:
        """Test the compiled function against sample rows with timeout.

        Args:
            fn: Compiled feature generation function.
            sample_rows: Rows to test.
            timeout_s: Maximum seconds per row execution.

        Returns:
            Tuple of (success, error_message).
        """
        from loclean.utils.sandbox import run_with_timeout

        test_rows = sample_rows[:5]

        for row in test_rows:
            result, error = run_with_timeout(fn, (row,), timeout_s)

            if error:
                return False, (f"Function failed for row {str(row)[:100]}: {error}")

            if not isinstance(result, dict):
                return False, (f"Expected dict return, got {type(result).__name__}")

            if not result:
                return False, "Function returned empty dict"

        return True, ""

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def _repair_function(
        self,
        source: str,
        error: str,
        state: dict[str, Any],
    ) -> str:
        """Ask the engine to fix a broken feature function.

        Args:
            source: Current source code.
            error: Error message from verification.
            state: Structural metadata for context.

        Returns:
            Repaired Python source code string.
        """
        samples = json.dumps(state["sample_rows"][:3], ensure_ascii=False, default=str)
        prompt = (
            "The following Python function has a bug.\n\n"
            f"Source:\n{source}\n\n"
            f"Error:\n{error}\n\n"
            f"Sample input rows:\n{samples}\n\n"
            "Fix the function. Return ONLY the corrected Python code, "
            "no markdown fences."
        )

        raw = self.inference_engine.generate(prompt)
        return str(raw).strip()

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_to_dataframe(
        df_nw: nw.DataFrame[Any],
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        timeout_s: float = 2.0,
    ) -> nw.DataFrame[Any]:
        """Map the feature function across all rows with timeout.

        Args:
            df_nw: Narwhals DataFrame.
            fn: Compiled feature generation function.
            timeout_s: Maximum seconds per row execution.

        Returns:
            Augmented Narwhals DataFrame with new feature columns.
        """
        from loclean.utils.sandbox import run_with_timeout

        rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]

        new_col_data: dict[str, list[Any]] = {}

        for row in rows:
            result, error = run_with_timeout(fn, (row,), timeout_s)
            features: dict[str, Any] = result if isinstance(result, dict) else {}

            if error:
                logger.debug(f"Row execution failed: {error}")

            if not new_col_data:
                for key in features:
                    new_col_data[key] = []

            for key in new_col_data:
                new_col_data[key].append(features.get(key))

        if not new_col_data:
            return df_nw

        native_ns = nw.get_native_namespace(df_nw)
        new_df = nw.from_dict(new_col_data, backend=native_ns)

        original_rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]
        combined_data: dict[str, list[Any]] = {
            col: [r[col] for r in original_rows] for col in df_nw.columns
        }
        for col in new_col_data:
            combined_data[col] = new_df[col].to_list()

        return nw.from_dict(combined_data, backend=native_ns)
