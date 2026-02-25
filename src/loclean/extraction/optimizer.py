"""Automated instruction optimization via a generate-evaluate-select loop.

This module provides :class:`InstructionOptimizer`, which refines data-extraction
instructions by:

1. Generating structural prompt variations through the local Ollama engine.
2. Evaluating each variation against a validation sample drawn from a
   Narwhals column using the core :class:`Extractor`.
3. Scoring results with the harmonic mean of field-level precision and recall.
4. Returning the highest-reward instruction.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import BaseModel, Field

from loclean.extraction.extract_dataframe import _sample_diverse_rows
from loclean.extraction.extractor import Extractor
from loclean.extraction.json_repair import repair_json
from loclean.utils.logging import configure_module_logger

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = configure_module_logger(__name__, level=logging.INFO)


class _InstructionVariations(BaseModel):
    """Internal schema used to constrain the LLM's structured output."""

    variations: list[str] = Field(
        ..., description="List of distinct instruction string variations"
    )


class InstructionOptimizer:
    """Optimise extraction instructions through a reward-driven feedback loop.

    The optimizer treats each candidate instruction as an *action*, runs it
    against a validation sample (*environment*), computes a field-level F1
    *reward*, and selects the best-performing variant (*policy update*).
    """

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        max_retries: int = 3,
    ) -> None:
        """Initialize the optimizer.

        Args:
            inference_engine: Engine used for both variation generation and
                extraction evaluation.
            max_retries: Retry budget forwarded to the internal
                :class:`Extractor` instances.
        """
        self.inference_engine = inference_engine
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        df: IntoFrameT,
        target_col: str,
        schema: type[BaseModel],
        baseline_instruction: str | None = None,
        sample_size: int = 20,
    ) -> str:
        """Run one optimization cycle and return the best instruction.

        Args:
            df: Input DataFrame (pandas, Polars, etc.).
            target_col: Column containing the text to extract from.
            schema: Pydantic BaseModel class defining the target structure.
            baseline_instruction: Starting instruction. When ``None`` a
                default is built from *schema*.
            sample_size: Number of validation rows to sample.

        Returns:
            The instruction string that achieved the highest F1 reward.

        Raises:
            ValueError: If *target_col* is not found in the DataFrame.
        """
        df_nw = nw.from_native(df)  # type: ignore[type-var]
        if target_col not in df_nw.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        if baseline_instruction is None:
            baseline_instruction = self._build_default_instruction(schema)

        sample = _sample_diverse_rows(df_nw, target_col, n=sample_size)
        if not sample:
            logger.warning(
                "No valid sample rows found. Returning baseline instruction."
            )
            return baseline_instruction

        variations = self._generate_variations(baseline_instruction, schema)

        candidates = [baseline_instruction, *variations]

        best_instruction = baseline_instruction
        best_score = -1.0

        for candidate in candidates:
            score = self._evaluate_variation(candidate, sample, schema)
            logger.info(
                f"[cyan]Instruction:[/cyan] "
                f"[dim]'{candidate[:60]}…'[/dim]  "
                f"[bold]F1={score:.4f}[/bold]"
            )
            if score > best_score:
                best_score = score
                best_instruction = candidate

        logger.info(
            f"[green]✓[/green] Best instruction "
            f"(F1=[bold yellow]{best_score:.4f}[/bold yellow]): "
            f"[dim]'{best_instruction[:80]}…'[/dim]"
        )

        return best_instruction

    # ------------------------------------------------------------------
    # Action Generation
    # ------------------------------------------------------------------

    def _generate_variations(
        self,
        baseline: str,
        schema: type[BaseModel],
        n: int = 3,
    ) -> list[str]:
        """Ask the inference engine for *n* instruction variations.

        Args:
            baseline: The current instruction to rephrase.
            schema: Target Pydantic schema (included in the prompt for
                context).
            n: Number of variations to request.

        Returns:
            A list of exactly *n* instruction strings.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        prompt = (
            "You are an expert prompt engineer. Given this baseline "
            "instruction for structured data extraction:\n\n"
            f'"{baseline}"\n\n'
            f"And the target output schema:\n{schema_json}\n\n"
            f"Generate exactly {n} distinct structural variations of this "
            "instruction. Each variation should use a different strategy "
            "(e.g. field-specific emphasis, step-by-step decomposition, "
            "negative constraints, or output format reminders).\n\n"
            f"Return a JSON object with key 'variations' containing "
            f"an array of exactly {n} instruction strings."
        )

        raw = self.inference_engine.generate(prompt, schema=_InstructionVariations)

        data = self._parse_variations_response(raw)
        variations = data.get("variations", [])

        valid: list[str] = [v for v in variations if isinstance(v, str) and v.strip()]
        while len(valid) < n:
            valid.append(baseline)
        return valid[:n]

    # ------------------------------------------------------------------
    # Environment Execution
    # ------------------------------------------------------------------

    def _evaluate_variation(
        self,
        instruction: str,
        sample: list[str],
        schema: type[BaseModel],
    ) -> float:
        """Extract *sample* rows using *instruction* and return the reward.

        Args:
            instruction: Candidate instruction to evaluate.
            sample: Validation text rows.
            schema: Target Pydantic schema.

        Returns:
            Harmonic-mean (F1) reward in ``[0, 1]``.
        """
        extractor = Extractor(
            inference_engine=self.inference_engine,
            max_retries=self.max_retries,
        )
        try:
            results = extractor.extract_batch(sample, schema, instruction)
        except Exception as exc:
            logger.warning(f"[yellow]⚠[/yellow] Evaluation failed: [red]{exc}[/red]")
            return 0.0

        return self._score_extraction(results, schema)

    # ------------------------------------------------------------------
    # Reward Calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _score_extraction(
        results: dict[str, BaseModel | None],
        schema: type[BaseModel],
    ) -> float:
        """Compute the harmonic mean of field-level precision and recall.

        * **Precision** = correctly populated fields / total extracted fields
        * **Recall** = correctly populated fields / expected total fields

        A field is *correctly populated* when it holds a substantive
        (non-None, non-empty) value.

        Args:
            results: Mapping of input text → validated model or ``None``.
            schema: Pydantic schema (used to enumerate expected fields).

        Returns:
            F1 score in ``[0, 1]``.
        """
        total_samples = len(results)
        if total_samples == 0:
            return 0.0

        field_names = list(schema.model_fields.keys())
        num_fields = len(field_names)
        if num_fields == 0:
            return 0.0

        expected_total = total_samples * num_fields
        correctly_populated = 0
        total_extracted = 0

        for model_instance in results.values():
            if model_instance is None:
                continue
            data = model_instance.model_dump()
            for field in field_names:
                total_extracted += 1
                if InstructionOptimizer._is_field_populated(data.get(field)):
                    correctly_populated += 1

        if total_extracted == 0 or expected_total == 0:
            return 0.0

        precision = correctly_populated / total_extracted
        recall = correctly_populated / expected_total

        denominator = precision + recall
        if denominator == 0:
            return 0.0

        return 2 * precision * recall / denominator

    @staticmethod
    def _is_field_populated(value: Any) -> bool:
        """Return ``True`` when *value* carries meaningful content.

        Args:
            value: A single field value from a model dump.

        Returns:
            ``False`` for ``None``, empty strings, and empty collections.
        """
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, (list, dict, set)) and len(value) == 0:
            return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_instruction(schema: type[BaseModel]) -> str:
        """Build a baseline instruction from the schema definition.

        Args:
            schema: Pydantic BaseModel class.

        Returns:
            Instruction string.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        return (
            "Extract structured information from the text and return it as "
            f"JSON matching this schema: {schema_json}. "
            "All required fields must be present and correctly typed."
        )

    @staticmethod
    def _parse_variations_response(raw: Any) -> dict[str, Any]:
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
                "[yellow]⚠[/yellow] Could not parse instruction variations "
                "response. Falling back to baseline."
            )
            return {}
