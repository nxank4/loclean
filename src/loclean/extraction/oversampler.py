"""Semantic synthetic oversampling via LLM-driven record generation.

Replaces geometric interpolation (SMOTE) with generative modelling.
The :class:`SemanticOversampler` produces structurally valid minority-class
records that satisfy a user-provided Pydantic schema and maintain the
logical correlations present in the original data sample.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Type

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import BaseModel, Field, ValidationError

from loclean.extraction.json_repair import repair_json
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class _GeneratedBatch(BaseModel):
    """Wrapper schema for batch-generated records."""

    records: list[dict[str, Any]] = Field(
        ...,
        description="List of generated records matching the target schema",
    )


class SemanticOversampler:
    """Generate synthetic minority-class records using an LLM.

    Instead of computing $x_{new} = x_i + \\lambda(x_{neighbor} - x_i)$,
    uses semantic generation to produce records that satisfy structural
    dependencies found in a valid data sample.

    Args:
        inference_engine: Engine used for generative requests.
        batch_size: Maximum records to request per LLM call.
        max_retries: Maximum generation rounds before giving up.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        batch_size: int = 10,
        max_retries: int = 5,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")
        if max_retries < 1:
            raise ValueError("max_retries must be ≥ 1")
        self.inference_engine = inference_engine
        self.batch_size = batch_size
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def oversample(
        self,
        df: IntoFrameT,
        target_col: str,
        target_value: Any,
        n: int,
        schema: Type[BaseModel],
    ) -> IntoFrameT:
        """Generate *n* synthetic minority-class records and append them.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column identifying the class label.
            target_value: Value of the minority class to oversample.
            n: Number of synthetic records to generate.
            schema: Pydantic model defining the record structure.

        Returns:
            DataFrame of the same native type with synthetic rows appended.

        Raises:
            ValueError: If *target_col* is missing or no minority rows exist.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]
        if target_col not in df_nw.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        sample_rows = self._sample_rows(df_nw, target_col, target_value)
        if not sample_rows:
            raise ValueError(f"No rows found where '{target_col}' == {target_value!r}")

        existing_keys = self._build_key_set(df_nw)
        generated: list[dict[str, Any]] = []
        retries = 0

        while len(generated) < n and retries < self.max_retries:
            remaining = n - len(generated)
            request_count = min(remaining, self.batch_size)

            batch = self._generate_batch(sample_rows, schema, request_count)

            validated = self._validate_and_filter(
                batch, schema, existing_keys, generated
            )
            generated.extend(validated)
            retries += 1

        if len(generated) < n:
            logger.warning(
                f"[yellow]⚠[/yellow] Generated {len(generated)}/{n} "
                f"records after {self.max_retries} retries"
            )

        if not generated:
            return df_nw.to_native()  # type: ignore[return-value]

        native_ns = nw.get_native_namespace(df_nw)
        col_data: dict[str, list[Any]] = {col: [] for col in df_nw.columns}
        for rec in generated:
            for col in df_nw.columns:
                col_data[col].append(rec.get(col))

        synthetic_nw = nw.from_dict(col_data, backend=native_ns)
        result = nw.concat([df_nw, synthetic_nw])

        logger.info(
            f"[green]✓[/green] Oversampled "
            f"[bold]{len(generated)}[/bold] synthetic records "
            f"for '{target_col}' == {target_value!r}"
        )

        return result.to_native()  # type: ignore[no-any-return,return-value]

    # ------------------------------------------------------------------
    # Minority sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_rows(
        df_nw: nw.DataFrame[Any],
        target_col: str,
        target_value: Any,
        n: int = 20,
    ) -> list[dict[str, Any]]:
        """Extract a representative sample of minority-class rows.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Class-label column.
            target_value: Minority class value.
            n: Maximum sample size.

        Returns:
            List of row dicts from the minority class.
        """
        minority = df_nw.filter(nw.col(target_col) == target_value)
        all_rows: list[dict[str, Any]] = minority.rows(named=True)  # type: ignore[assignment]

        if len(all_rows) <= n:
            return all_rows

        step = len(all_rows) / n
        return [all_rows[int(i * step)] for i in range(n)]

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def _generate_batch(
        self,
        sample_rows: list[dict[str, Any]],
        schema: Type[BaseModel],
        count: int,
    ) -> list[dict[str, Any]]:
        """Prompt the engine to generate a batch of synthetic records.

        Args:
            sample_rows: Representative minority-class rows.
            schema: Pydantic model defining record structure.
            count: Number of records to request.

        Returns:
            List of raw record dicts (unvalidated).
        """
        schema_fields = {
            name: {
                "type": str(info.annotation),
                "description": (info.description if info.description else ""),
            }
            for name, info in schema.model_fields.items()
        }

        prompt = (
            "You are a synthetic data generator.\n\n"
            "Generate exactly {count} NEW records that match this schema:\n"
            "{schema}\n\n"
            "Here are real example rows for reference:\n"
            "{samples}\n\n"
            "Rules:\n"
            "- Each record MUST conform to the schema structure\n"
            "- Maintain logical correlations seen in the examples\n"
            "- Do NOT copy the example rows exactly\n"
            "- Generated values must be physically plausible\n\n"
            'Return a JSON object with key "records" containing '
            "a list of {count} record dictionaries."
        ).format(
            count=count,
            schema=json.dumps(schema_fields, indent=2),
            samples=json.dumps(sample_rows[:5], ensure_ascii=False, default=str),
        )

        raw = self.inference_engine.generate(prompt, schema=_GeneratedBatch)
        return self._parse_batch_response(raw)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch_response(raw: Any) -> list[dict[str, Any]]:
        """Best-effort parse of the LLM response into a list of records.

        Args:
            raw: Raw output from the inference engine.

        Returns:
            List of record dicts (may be empty on total failure).
        """
        if isinstance(raw, dict):
            records = raw.get("records", [])
            if isinstance(records, list):
                return records
            return []

        text = str(raw) if not isinstance(raw, str) else raw

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                records = parsed.get("records", [])
                return records if isinstance(records, list) else []
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            repaired = repair_json(text)
            if isinstance(repaired, dict):
                records = repaired.get("records", [])
                return records if isinstance(records, list) else []
            parsed_r = json.loads(repaired)  # type: ignore[arg-type]
            if isinstance(parsed_r, dict):
                records = parsed_r.get("records", [])
                return records if isinstance(records, list) else []
            if isinstance(parsed_r, list):
                return parsed_r
        except Exception:
            pass

        logger.warning(
            "[yellow]⚠[/yellow] Could not parse batch response. Returning empty batch."
        )
        return []

    # ------------------------------------------------------------------
    # Validation and deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_filter(
        candidates: list[dict[str, Any]],
        schema: Type[BaseModel],
        existing_keys: set[frozenset[tuple[str, Any]]],
        already_generated: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate candidates against the schema and deduplicate.

        Args:
            candidates: Raw record dicts from the LLM.
            schema: Pydantic model for validation.
            existing_keys: Fingerprints of original DataFrame rows.
            already_generated: Records already accepted in prior rounds.

        Returns:
            List of validated, deduplicated record dicts.
        """
        gen_keys = {_row_key(r) for r in already_generated}
        combined_keys = existing_keys | gen_keys

        valid: list[dict[str, Any]] = []
        for rec in candidates:
            try:
                instance = schema(**rec)
                clean_rec = instance.model_dump()
            except (ValidationError, TypeError, Exception):
                continue

            key = _row_key(clean_rec)
            if key in combined_keys:
                continue

            combined_keys.add(key)
            valid.append(clean_rec)

        return valid

    @staticmethod
    def _build_key_set(
        df_nw: nw.DataFrame[Any],
    ) -> set[frozenset[tuple[str, Any]]]:
        """Build a set of row fingerprints for deduplication.

        Args:
            df_nw: Narwhals DataFrame.

        Returns:
            Set of frozensets, each representing one row.
        """
        rows: list[dict[str, Any]] = df_nw.rows(named=True)  # type: ignore[assignment]
        return {_row_key(r) for r in rows}


def _row_key(row: dict[str, Any]) -> frozenset[tuple[str, Any]]:
    """Create a hashable fingerprint for a row dict."""
    return frozenset((k, str(v)) for k, v in sorted(row.items()))
