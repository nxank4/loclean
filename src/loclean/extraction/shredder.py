"""Automated relational shredding for unstructured log columns.

Parses a single column of deeply nested unstructured text into multiple
relational DataFrames by:

1. Sampling representative log entries.
2. Prompting the Ollama engine to infer a relational schema
   (functional dependencies :math:`X \\rightarrow Y`).
3. Generating and compiling a pure-Python ``extract_relations`` function.
4. Applying the function across the column and separating results into
   per-table Narwhals DataFrames.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw
from pydantic import BaseModel, Field

from loclean.extraction.json_repair import repair_json
from loclean.utils.cache_keys import compute_code_key
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT

    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


# ------------------------------------------------------------------
# Pydantic schemas for LLM-inferred relational structure
# ------------------------------------------------------------------


class _TableDef(BaseModel):
    """Definition of a single relational table."""

    name: str = Field(..., description="Table name")
    columns: list[str] = Field(..., description="Column names for this table")
    primary_key: str = Field(..., description="Primary key column name")
    foreign_key: str | None = Field(
        default=None,
        description="Foreign key column referencing another table",
    )


class _RelationalSchema(BaseModel):
    """Multi-table relational schema inferred from log data."""

    tables: list[_TableDef] = Field(
        ...,
        min_length=2,
        description="At least two related tables",
    )


class RelationalShredder:
    """Shred unstructured log columns into relational DataFrames.

    Uses a two-phase LLM approach:

    1. **Schema inference** — propose tables with PKs/FKs.
    2. **Code generation** — compile a pure-Python extraction function,
       verify against samples, and repair on failure.

    Args:
        inference_engine: Engine for generative requests.
        sample_size: Number of log entries to sample.
        max_retries: Repair budget for the compilation loop.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        sample_size: int = 30,
        max_retries: int = 3,
        timeout_s: float = 2.0,
        cache: LocleanCache | None = None,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be ≥ 1")
        if max_retries < 1:
            raise ValueError("max_retries must be ≥ 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.inference_engine = inference_engine
        self.sample_size = sample_size
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shred(
        self,
        df: IntoFrameT,
        target_col: str,
    ) -> dict[str, Any]:
        """Parse a log column into multiple relational DataFrames.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column containing unstructured log text.

        Returns:
            Dictionary mapping table names to native DataFrames,
            or an empty dict if generation fails.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]
        if target_col not in df_nw.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        samples = self._sample_entries(df_nw, target_col)
        if not samples:
            raise ValueError(f"No valid entries in column '{target_col}'")

        schema = self._infer_schema(samples)

        all_columns = sorted(col for tbl in schema.tables for col in tbl.columns)
        table_names = sorted(tbl.name for tbl in schema.tables)
        cache_key = compute_code_key(
            columns=all_columns,
            dtypes=table_names,
            target_col=target_col,
            module_prefix="shredder",
        )

        cached_source = self.cache.get_code(cache_key) if self.cache else None
        if cached_source is not None:
            extract_fn = self._compile_function(cached_source)
            ok, _ = self._verify_function(extract_fn, samples, schema, self.timeout_s)
            if ok:
                logger.info("[green]✓[/green] Cache hit — reusing compiled extractor")
                results = self._apply_function(
                    df_nw, target_col, extract_fn, self.timeout_s
                )
                native_ns = nw.get_native_namespace(df_nw)
                return self._separate_tables(results, schema, native_ns)

        source = self._generate_extractor(schema, samples)

        try:
            extract_fn = self._compile_function(source)
            ok, error = self._verify_function(
                extract_fn, samples, schema, self.timeout_s
            )
        except ValueError as exc:
            ok, error = False, str(exc)

        retries = 0
        while not ok and retries < self.max_retries:
            retries += 1
            logger.warning(
                f"[yellow]⚠[/yellow] Retrying code generation "
                f"({retries}/{self.max_retries}): {error}"
            )
            source = self._repair_function(source, error, samples)
            try:
                extract_fn = self._compile_function(source)
                ok, error = self._verify_function(
                    extract_fn, samples, schema, self.timeout_s
                )
            except ValueError as exc:
                ok, error = False, str(exc)

        if not ok:
            logger.warning(
                f"[yellow]⚠[/yellow] The model could not generate valid Python "
                f"code after {self.max_retries} retries. This is not a library "
                f"bug — smaller models (e.g. phi3) sometimes produce syntax "
                f"errors or invalid logic. Returning empty result.\n"
                f"  [dim]Last error: {error}[/dim]\n"
                f"  [dim]Tip: try a larger model "
                f"(model='qwen2.5-coder:7b') or increase max_retries.[/dim]"
            )
            return {}

        if self.cache:
            self.cache.set_code(cache_key, source)

        results = self._apply_function(df_nw, target_col, extract_fn, self.timeout_s)
        native_ns = nw.get_native_namespace(df_nw)
        tables = self._separate_tables(results, schema, native_ns)

        table_summary = ", ".join(f"{name}({len(tbl)})" for name, tbl in tables.items())
        logger.info(
            f"[green]✓[/green] Shredded into "
            f"[bold]{len(tables)}[/bold] tables: {table_summary}"
        )

        return tables

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_entries(
        self,
        df_nw: nw.DataFrame[Any],
        target_col: str,
    ) -> list[str]:
        """Length-stratified sampling of non-empty log strings.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Column to sample from.

        Returns:
            List of up to *sample_size* unique strings.
        """
        raw = df_nw.unique(subset=[target_col])[target_col].to_list()
        valid: list[str] = [str(v) for v in raw if v is not None and str(v).strip()]
        if len(valid) <= self.sample_size:
            valid.sort(key=len)
            return valid

        valid.sort(key=len)
        step = len(valid) / self.sample_size
        return [valid[int(i * step)] for i in range(self.sample_size)]

    # ------------------------------------------------------------------
    # Phase 1: Schema inference
    # ------------------------------------------------------------------

    def _infer_schema(self, samples: list[str]) -> _RelationalSchema:
        """Prompt the engine to propose a relational schema.

        Args:
            samples: Representative log entries.

        Returns:
            Parsed relational schema with at least two tables.
        """
        prompt = (
            "You are a database architect analyzing log data.\n\n"
            "Here are sample log entries:\n"
            f"{json.dumps(samples[:10], ensure_ascii=False)}\n\n"
            "Analyze the structure and propose a relational schema "
            "in Third Normal Form (3NF). Identify functional "
            "dependencies (X → Y) to separate concerns.\n\n"
            "Return a JSON object with key 'tables' containing a list "
            "of at least 2 table definitions. Each table must have:\n"
            '- "name": table name\n'
            '- "columns": list of column names\n'
            '- "primary_key": primary key column\n'
            '- "foreign_key": foreign key column (null for root table)'
        )

        raw = self.inference_engine.generate(prompt, schema=_RelationalSchema)
        return self._parse_schema_response(raw)

    @staticmethod
    def _parse_schema_response(raw: Any) -> _RelationalSchema:
        """Best-effort parse of the schema inference response.

        Args:
            raw: Raw LLM output.

        Returns:
            Validated relational schema.

        Raises:
            ValueError: If parsing fails completely.
        """
        if isinstance(raw, _RelationalSchema):
            return raw

        data: dict[str, Any] | None = None

        if isinstance(raw, dict):
            data = raw
        else:
            text = str(raw) if not isinstance(raw, str) else raw
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                try:
                    repaired = repair_json(text)
                    if isinstance(repaired, dict):
                        data = repaired
                    else:
                        data = json.loads(repaired)  # type: ignore[arg-type]
                except Exception:
                    pass

        if data is None:
            raise ValueError("Failed to parse relational schema")

        return _RelationalSchema(**data)

    # ------------------------------------------------------------------
    # Phase 2: Code generation + compilation
    # ------------------------------------------------------------------

    def _generate_extractor(
        self,
        schema: _RelationalSchema,
        samples: list[str],
    ) -> str:
        """Prompt the engine to write an extract_relations function.

        Args:
            schema: Inferred relational schema.
            samples: Representative log entries.

        Returns:
            Python source code string.
        """
        table_specs = json.dumps(
            [t.model_dump() for t in schema.tables],
            indent=2,
        )
        prompt = (
            "You are an expert Python programmer.\n\n"
            "Write a pure Python function with this exact signature:\n\n"
            "def extract_relations(log: str) -> dict[str, dict]:\n\n"
            "The function must parse a single log string and return a "
            "dictionary where each key is a table name and each value "
            "is a dictionary of column_name: extracted_value pairs.\n\n"
            f"Target tables:\n{table_specs}\n\n"
            "Sample log entries:\n"
            f"{json.dumps(samples[:5], ensure_ascii=False)}\n\n"
            "EXAMPLE (for a different log format):\n\n"
            "import re\n\n"
            "def extract_relations(log: str) -> dict[str, dict]:\n"
            "    result = {}\n"
            "    try:\n"
            "        m = re.match("
            "r'(\\S+) (\\S+) \\[(.*?)\\] \"(\\S+)\"', log)\n"
            "        if m:\n"
            "            result['requests'] = {\n"
            "                'ip': m.group(1),\n"
            "                'method': m.group(4),\n"
            "            }\n"
            "    except Exception:\n"
            "        result['requests'] = "
            "{'ip': '', 'method': ''}\n"
            "    return result\n\n"
            "Now write yours for the log format above.\n\n"
            "Rules:\n"
            "- Use ONLY standard library modules (re, json, "
            "datetime, collections)\n"
            "- Wrap parsing logic in try/except blocks\n"
            "- Return empty strings for fields that cannot be parsed\n"
            "- Do NOT import any third-party libraries\n\n"
            "Return ONLY the Python function code, no markdown fences."
        )

        raw = self.inference_engine.generate(prompt)
        return str(raw).strip()

    @staticmethod
    def _compile_function(
        source: str,
    ) -> Callable[[str], dict[str, dict[str, Any]]]:
        """Compile source code in a restricted sandbox.

        Applies deterministic sanitization before compilation to fix
        common LLM output artifacts (markdown fences, non-ASCII
        operators, invalid literals, etc.).

        Args:
            source: Python source containing ``extract_relations``.

        Returns:
            The compiled function.

        Raises:
            ValueError: If compilation fails or function not found.
        """
        from loclean.utils.sandbox import compile_sandboxed
        from loclean.utils.source_sanitizer import sanitize_source

        return compile_sandboxed(
            sanitize_source(source),
            "extract_relations",
            ["re", "json", "datetime", "collections"],
        )

    def _verify_function(
        self,
        fn: Callable[[str], dict[str, dict[str, Any]]],
        samples: list[str],
        schema: _RelationalSchema,
        timeout_s: float = 2.0,
    ) -> tuple[bool, str]:
        """Test the compiled function against sample entries with timeout.

        Args:
            fn: Compiled extraction function.
            samples: Log entries to test.
            schema: Expected table structure.
            timeout_s: Maximum seconds per sample execution.

        Returns:
            Tuple of (success, error_message).
        """
        from loclean.utils.sandbox import run_with_timeout

        table_names = {t.name for t in schema.tables}
        test_samples = samples[:5]

        for sample in test_samples:
            result, error = run_with_timeout(fn, (sample,), timeout_s)

            if error:
                return False, (f"Function failed for input {sample[:100]!r}: {error}")

            if not isinstance(result, dict):
                return False, (f"Expected dict return, got {type(result).__name__}")

            missing = table_names - set(result.keys())
            if missing:
                return False, f"Missing tables in output: {missing}"

        return True, ""

    def _repair_function(
        self,
        source: str,
        error: str,
        samples: list[str],
    ) -> str:
        """Ask the engine to fix a broken extraction function.

        Args:
            source: Current source code.
            error: Error message from verification.
            samples: Log entries for context.

        Returns:
            Repaired Python source code string.
        """
        prompt = (
            "The following Python function has a bug.\n\n"
            f"Source:\n{source}\n\n"
            f"Error:\n{error}\n\n"
            "Sample inputs:\n"
            f"{json.dumps(samples[:3], ensure_ascii=False)}\n\n"
            "Fix the function. Return ONLY the corrected Python code, "
            "no markdown fences."
        )

        raw = self.inference_engine.generate(prompt)
        return str(raw).strip()

    # ------------------------------------------------------------------
    # Phase 3: Full execution + separation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_function(
        df_nw: nw.DataFrame[Any],
        target_col: str,
        fn: Callable[[str], dict[str, dict[str, Any]]],
        timeout_s: float = 2.0,
    ) -> list[dict[str, dict[str, Any]]]:
        """Apply the extraction function to every row with timeout.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Column with log text.
            fn: Compiled extraction function.
            timeout_s: Maximum seconds per row execution.

        Returns:
            List of extraction results (one per row).
        """
        from loclean.utils.sandbox import run_with_timeout

        values = df_nw[target_col].to_list()
        results: list[dict[str, dict[str, Any]]] = []

        for val in values:
            text = str(val) if val is not None else ""
            result, error = run_with_timeout(fn, (text,), timeout_s)
            if error:
                logger.debug(f"Row execution failed: {error}")
                results.append({})
            else:
                results.append(result if isinstance(result, dict) else {})

        return results

    @staticmethod
    def _separate_tables(
        results: list[dict[str, dict[str, Any]]],
        schema: _RelationalSchema,
        native_ns: Any,
    ) -> dict[str, Any]:
        """Split extraction results into per-table DataFrames.

        Args:
            results: List of extraction dicts from _apply_function.
            schema: Relational schema defining table structure.
            native_ns: Native namespace (e.g. polars module).

        Returns:
            Dict mapping table names to native DataFrames.
        """
        tables: dict[str, Any] = {}

        for table_def in schema.tables:
            col_data: dict[str, list[Any]] = {col: [] for col in table_def.columns}

            for row_result in results:
                table_row = row_result.get(table_def.name, {})
                for col in table_def.columns:
                    col_data[col].append(table_row.get(col))

            table_nw = nw.from_dict(col_data, backend=native_ns)
            tables[table_def.name] = table_nw.to_native()

        return tables
