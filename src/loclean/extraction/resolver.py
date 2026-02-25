"""Semantic entity resolution via LLM-driven canonicalization.

This module provides :class:`EntityResolver`, which canonicalizes messy string
columns by:

1. Extracting unique values from a Narwhals column.
2. Prompting the local Ollama engine to group semantically similar strings
   under a single canonical label (respecting a distance threshold ε).
3. Mapping the canonical dictionary back across the original DataFrame column.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import BaseModel, Field

from loclean.extraction.json_repair import repair_json
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class _CanonicalMapping(BaseModel):
    """Internal schema used to constrain the LLM's structured output."""

    mapping: dict[str, str] = Field(
        ...,
        description=(
            "Dictionary mapping each input string variation "
            "to its canonical (authoritative) form"
        ),
    )


class EntityResolver:
    """Canonicalize messy string values via semantic entity resolution.

    Groups similar string variations into a single authoritative label
    using a generative model.  The *threshold* parameter (ε) controls
    how aggressively strings are merged: a higher value allows more
    distant strings to be grouped.
    """

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        threshold: float = 0.8,
        max_retries: int = 3,
    ) -> None:
        """Initialize the resolver.

        Args:
            inference_engine: Engine used for semantic evaluation.
            threshold: Semantic-distance threshold ε in ``(0, 1]``.
                Pairs with ``d(x, y) < ε`` are merged.
            max_retries: Currently reserved for future retry logic.
        """
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.inference_engine = inference_engine
        self.threshold = threshold
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        df: IntoFrameT,
        target_col: str,
    ) -> IntoFrameT:
        """Canonicalize a string column and return the augmented DataFrame.

        A new column ``{target_col}_canonical`` is appended containing the
        resolved canonical labels.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column with messy string values.

        Returns:
            DataFrame of the same native type with an added canonical column.

        Raises:
            ValueError: If *target_col* is not found in the DataFrame.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]
        if target_col not in df_nw.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        unique_values = self._extract_unique_values(df_nw, target_col)
        if not unique_values:
            logger.warning(
                "No valid values found in column. Returning original DataFrame."
            )
            return df_nw.with_columns(
                nw.col(target_col).alias(f"{target_col}_canonical"),
            ).to_native()  # type: ignore[return-value]

        canonical_map = self._build_canonical_mapping(unique_values)

        logger.info(
            f"[green]✓[/green] Resolved [bold]{len(unique_values)}[/bold] "
            f"unique values into [bold]{len(set(canonical_map.values()))}[/bold] "
            "canonical entities"
        )

        canonical_col = f"{target_col}_canonical"
        result = df_nw.with_columns(
            nw.col(target_col)
            .cast(nw.String)
            .replace_strict(
                old=list(canonical_map.keys()),
                new=list(canonical_map.values()),
                default=nw.col(target_col).cast(nw.String),
            )
            .alias(canonical_col),
        )

        return result.to_native()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Unique-value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_unique_values(
        df_nw: nw.DataFrame[Any],
        target_col: str,
    ) -> list[str]:
        """Return sorted unique non-empty string values from *target_col*.

        Args:
            df_nw: Narwhals DataFrame.
            target_col: Column to extract from.

        Returns:
            Deduplicated list of non-empty strings.
        """
        raw = df_nw.unique(subset=[target_col])[target_col].to_list()
        valid: list[str] = [str(v) for v in raw if v is not None and str(v).strip()]
        valid.sort()
        return valid

    # ------------------------------------------------------------------
    # Canonical mapping via LLM
    # ------------------------------------------------------------------

    def _build_canonical_mapping(
        self,
        values: list[str],
    ) -> dict[str, str]:
        """Prompt the engine to produce a variation → canonical mapping.

        Args:
            values: List of unique string values to canonicalize.

        Returns:
            Dictionary mapping every input value to its canonical form.
        """
        prompt = (
            "You are a data-quality expert performing entity resolution.\n\n"
            "Given this list of string values extracted from a dataset column:\n"
            f"{json.dumps(values, ensure_ascii=False)}\n\n"
            "Group strings that refer to the same real-world entity. "
            "Two strings x and y should be grouped only when their semantic "
            f"distance d(x, y) falls below the threshold ε = {self.threshold}.\n\n"
            "For each group, choose the cleanest, most complete string as "
            "the canonical label. Strings that do not closely match any "
            "other string should map to themselves.\n\n"
            "Return a JSON object with key 'mapping' whose value is a "
            "dictionary mapping EVERY input string to its canonical form."
        )

        raw = self.inference_engine.generate(prompt, schema=_CanonicalMapping)
        data = self._parse_mapping_response(raw)
        mapping: dict[str, str] = data.get("mapping", {})

        validated: dict[str, str] = {}
        for val in values:
            canonical = mapping.get(val)
            if isinstance(canonical, str) and canonical.strip():
                validated[val] = canonical.strip()
            else:
                validated[val] = val

        return validated

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mapping_response(raw: Any) -> dict[str, Any]:
        """Best-effort parse of the LLM response into a dict.

        Handles raw dicts, valid JSON strings, and malformed JSON (via
        ``repair_json``).

        Args:
            raw: Raw output from the inference engine.

        Returns:
            Parsed dict (may be empty on total failure).
        """
        if isinstance(raw, dict):
            return raw

        try:
            parsed: dict[str, Any] = json.loads(raw)  # type: ignore[arg-type]
            return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            repaired = repair_json(raw)
            if isinstance(repaired, dict):
                return repaired
            parsed_repaired: dict[str, Any] = json.loads(repaired)  # type: ignore[arg-type]
            return parsed_repaired
        except (json.JSONDecodeError, TypeError, Exception):
            logger.warning(
                "[yellow]⚠[/yellow] Could not parse canonical mapping "
                "response. Falling back to identity mapping."
            )
            return {}
