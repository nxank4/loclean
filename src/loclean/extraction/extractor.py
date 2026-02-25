"""Core extraction logic for structured data extraction using Pydantic schemas.

This module provides the Extractor class that orchestrates LLM inference,
JSON repair, Pydantic validation, and retry logic to ensure 100% schema compliance.

It also exposes a generative compilation path: the ``compile`` method
synthesises a pure-Python ``extract_data`` function via the inference engine,
verifies it against sample rows, and returns a callable that can be mapped
natively over Narwhals columns without further LLM round-trips.
"""

import json
import logging
import re
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ValidationError

from loclean.extraction.json_repair import repair_json

if TYPE_CHECKING:
    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


class Extractor:
    """Extractor for structured data extraction using Pydantic schemas.

    Ensures LLM outputs strictly conform to user-defined Pydantic models
    through Ollama's structured output support, JSON repair, retry logic,
    and Pydantic validation.
    """

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        cache: "LocleanCache | None" = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Extractor.

        Args:
            inference_engine: Inference engine instance (e.g. OllamaEngine).
            cache: Optional cache instance for storing extraction results.
            max_retries: Maximum retry attempts on validation failure.
        """
        self.inference_engine = inference_engine
        self.cache = cache
        self.max_retries = max_retries

        if hasattr(self.inference_engine, "verbose") and self.inference_engine.verbose:
            logger.setLevel(logging.DEBUG)

    def extract(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str | None = None,
    ) -> BaseModel:
        """Extract structured data from text using a Pydantic schema.

        Args:
            text: Input text to extract from.
            schema: Pydantic BaseModel class defining the output structure.
            instruction: Optional custom instruction.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValueError: If extraction fails after max_retries.
        """
        if not issubclass(schema, BaseModel):
            raise ValueError(
                f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
            )

        final_instruction = self._build_instruction(schema, instruction)

        if self.cache:
            cache_key = self._get_cache_key(text, schema, final_instruction)
            cached = self.cache.get_batch([text], cache_key)
            if text in cached:
                try:
                    return schema.model_validate(cached[text])
                except ValidationError:
                    logger.warning(
                        f"[yellow]⚠[/yellow] Cache entry for "
                        f"[dim]'{text[:50]}...'[/dim] failed validation, "
                        f"[cyan]recomputing[/cyan]"
                    )

        result = self._extract_with_retry(
            text, schema, final_instruction, retry_count=0
        )

        if result is None:
            msg = (
                f"Failed to extract valid {schema.__name__} from text "
                f"after {self.max_retries} retries"
            )
            raise ValueError(msg) from None

        if self.cache:
            cache_key = self._get_cache_key(text, schema, final_instruction)
            self.cache.set_batch([text], cache_key, {text: result.model_dump()})

        return result

    def extract_batch(
        self,
        items: list[str],
        schema: type[BaseModel],
        instruction: str | None = None,
    ) -> dict[str, BaseModel | None]:
        """Extract structured data from a batch of texts.

        Args:
            items: List of input texts to extract from.
            schema: Pydantic BaseModel class defining the output structure.
            instruction: Optional custom instruction.

        Returns:
            Dictionary mapping input_text → BaseModel instance or None.
        """
        if not items:
            return {}

        start_time = time.time()
        final_instruction = self._build_instruction(schema, instruction)

        results: dict[str, BaseModel | None] = {}
        misses: list[str] = []

        if self.cache:
            cache_key = self._get_cache_key("", schema, final_instruction)
            cached = self.cache.get_batch(items, cache_key)
            for item in items:
                if item in cached:
                    try:
                        results[item] = schema.model_validate(cached[item])
                    except ValidationError:
                        logger.warning(
                            f"Cache entry for '{item}' failed validation, recomputing"
                        )
                        misses.append(item)
                else:
                    misses.append(item)

            from loclean.utils.rich_output import log_cache_stats

            log_cache_stats(
                total_items=len(items),
                cache_hits=len(items) - len(misses),
                cache_misses=len(misses),
                context="Extraction",
            )
        else:
            misses = items

        errors: list[dict[str, Any]] = []
        for item in misses:
            try:
                result = self._extract_with_retry(
                    item, schema, final_instruction, retry_count=0
                )
                results[item] = result
            except ValidationError as e:
                errors.append({"type": "ValidationError", "item": item})
                if len(errors) <= 3:
                    logger.warning(
                        f"[yellow]⚠[/yellow] Failed to extract from "
                        f"[dim]'{item[:50]}...'[/dim]: "
                        f"[red]{str(e)[:60]}[/red]"
                    )
                results[item] = None
            except Exception as e:
                errors.append({"type": type(e).__name__, "item": item})
                if len(errors) <= 3:
                    logger.warning(
                        f"[yellow]⚠[/yellow] Failed to extract from "
                        f"[dim]'{item[:50]}...'[/dim]: "
                        f"[red]{str(e)[:60]}[/red]"
                    )
                results[item] = None

        if self.cache and misses:
            cache_key = self._get_cache_key("", schema, final_instruction)
            valid_results = {
                item: result.model_dump()
                for item, result in results.items()
                if result is not None and item in misses
            }
            if valid_results:
                self.cache.set_batch(
                    list(valid_results.keys()), cache_key, valid_results
                )

        elapsed_time = time.time() - start_time
        successful = sum(1 for r in results.values() if r is not None)
        failed = len(results) - successful

        from loclean.utils.rich_output import log_error_summary, log_processing_summary

        log_processing_summary(
            total_processed=len(items),
            successful=successful,
            failed=failed,
            time_taken=elapsed_time,
            context="Extraction",
        )

        if len(errors) > 3:
            from collections import Counter

            error_counts = Counter(e["type"] for e in errors)
            error_summary = []
            for error_type, count in error_counts.items():
                sample_items = [
                    e["item"][:50] for e in errors if e["type"] == error_type
                ][:3]
                error_summary.append(
                    {
                        "type": error_type,
                        "count": count,
                        "sample_items": sample_items,
                    }
                )
            log_error_summary(error_summary, max_display=5, context="Extraction")

        return results

    def _extract_with_retry(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str,
        retry_count: int,
    ) -> BaseModel | None:
        """Extract with retry logic on validation failures.

        Args:
            text: Input text to extract from.
            schema: Pydantic BaseModel class.
            instruction: Extraction instruction.
            retry_count: Current retry attempt number.

        Returns:
            Validated Pydantic model instance or None.
        """
        if retry_count >= self.max_retries:
            logger.error(
                f"[red]❌[/red] Max retries "
                f"([yellow]{self.max_retries}[/yellow]) exceeded "
                f"for text: [dim]'{text[:50]}...'[/dim]"
            )
            return None

        try:
            prompt = f"{instruction}\n\nInput: {text}"
            raw_output = self.inference_engine.generate(prompt, schema=schema)

            verbose = getattr(self.inference_engine, "verbose", False)
            if verbose:
                logger.debug(
                    f"[bold green]EXTRACTION RAW OUTPUT:[/bold green]\n{raw_output}"
                )

            if not raw_output:
                logger.warning(
                    f"[yellow]⚠[/yellow] Empty response for [dim]'{text[:50]}...'[/dim]"
                )
                return self._retry_extraction(text, schema, instruction, retry_count)

            return self._parse_and_validate(
                raw_output, schema, text, instruction, retry_count
            )

        except Exception as e:
            logger.warning(
                f"[yellow]⚠[/yellow] Extraction attempt "
                f"[cyan]{retry_count + 1}[/cyan] failed: "
                f"[red]{str(e)[:60]}[/red]"
            )
            return self._retry_extraction(text, schema, instruction, retry_count)

    def _parse_and_validate(
        self,
        text_output: str,
        schema: type[BaseModel],
        original_text: str,
        instruction: str,
        retry_count: int,
    ) -> BaseModel:
        """Parse JSON and validate against Pydantic schema.

        Args:
            text_output: Raw JSON text from Ollama.
            schema: Pydantic BaseModel class.
            original_text: Original input text (for retry context).
            instruction: Extraction instruction (for retry context).
            retry_count: Current retry count (for retry context).

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If parsing or validation fails.
        """
        try:
            if isinstance(text_output, dict):
                data = text_output
            else:
                data = json.loads(text_output)
        except (json.JSONDecodeError, TypeError) as e:
            if isinstance(text_output, dict):
                data = text_output
            else:
                repaired = repair_json(text_output)
                if isinstance(repaired, dict):
                    data = repaired
                else:
                    try:
                        data = json.loads(repaired)
                    except (json.JSONDecodeError, TypeError) as parse_error:
                        logger.warning(
                            f"[yellow]⚠[/yellow] JSON decode failed even "
                            f"after repair: [red]{str(parse_error)[:60]}[/red]. "
                            f"Original: [dim]{str(e)[:40]}[/dim]"
                        )
                        raise ValueError(
                            f"Invalid JSON: {str(text_output)[:100]}"
                        ) from parse_error

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            logger.warning(
                f"[yellow]⚠[/yellow] Pydantic validation failed: "
                f"[red]{str(e)[:80]}[/red]"
            )
            raise

    def _retry_extraction(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str,
        retry_count: int,
    ) -> BaseModel | None:
        """Retry extraction with adjusted prompt.

        Args:
            text: Input text.
            schema: Pydantic BaseModel class.
            instruction: Current instruction.
            retry_count: Current retry count.

        Returns:
            Validated Pydantic model instance or None.
        """
        adjusted_instruction = (
            f"{instruction}\n\n"
            f"IMPORTANT: The output MUST strictly match the JSON Schema "
            f"for {schema.__name__}. "
            f"All required fields must be present and correctly typed."
        )

        return self._extract_with_retry(
            text, schema, adjusted_instruction, retry_count + 1
        )

    def _build_instruction(
        self, schema: type[BaseModel], custom_instruction: str | None
    ) -> str:
        """Build extraction instruction from schema and optional custom instruction.

        Args:
            schema: Pydantic BaseModel class.
            custom_instruction: Optional custom instruction.

        Returns:
            Final instruction string.
        """
        if custom_instruction:
            return custom_instruction

        schema_json = schema.model_json_schema()
        return (
            f"Extract structured information from the text and return it as JSON "
            f"matching this schema: {json.dumps(schema_json, indent=2)}. "
            f"All required fields must be present and correctly typed."
        )

    def _get_cache_key(
        self, text: str, schema: type[BaseModel], instruction: str
    ) -> str:
        """Generate cache key for extraction.

        Args:
            text: Input text (empty string for batch operations).
            schema: Pydantic BaseModel class.
            instruction: Extraction instruction.

        Returns:
            Cache key string.
        """
        schema_name = schema.__name__
        return f"extract_v1::{schema_name}::{instruction}"

    # ------------------------------------------------------------------
    # Generative Compilation
    # ------------------------------------------------------------------

    def compile(
        self,
        schema: type[BaseModel],
        sample_rows: list[str],
        instruction: str | None = None,
        max_repair_attempts: int = 3,
    ) -> Callable[[str], dict[str, Any]]:
        """Compile a pure-Python extraction function via code generation.

        Synthesises a ``extract_data(text: str) -> dict`` function using
        the inference engine, then verifies it against *sample_rows*.
        On failure the traceback is fed back for heuristic repair, up to
        *max_repair_attempts* times.

        Args:
            schema: Pydantic BaseModel class defining the target structure.
            sample_rows: Representative text rows used for verification.
            instruction: Optional domain-specific hint prepended to the prompt.
            max_repair_attempts: Maximum code-repair iterations.

        Returns:
            A callable ``(str) -> dict`` that can be mapped over a column.

        Raises:
            ValueError: If a valid function cannot be produced within the
                retry budget.
        """
        if not issubclass(schema, BaseModel):
            raise ValueError(
                f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
            )

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        sample_block = "\n".join(f"  - {repr(row)}" for row in sample_rows[:20])

        prompt = (
            "Write a pure Python function with this exact signature:\n\n"
            "def extract_data(text: str) -> dict:\n\n"
            "The function must parse the input text string and return a "
            "dictionary matching this JSON schema:\n"
            f"{schema_json}\n\n"
            f"Here are example input strings:\n{sample_block}\n\n"
            "Requirements:\n"
            "- Use only the Python standard library (re, json, etc.)\n"
            "- Return a dict with ALL required keys from the schema\n"
            "- Handle edge cases gracefully (missing data, unexpected formats)\n"
            "- Do NOT import any external packages\n\n"
            "Output ONLY the Python function definition, no explanation or "
            "markdown fences."
        )

        if instruction:
            prompt = f"{instruction}\n\n{prompt}"

        generated_code = self.inference_engine.generate(prompt)

        for attempt in range(max_repair_attempts):
            fn, error = self._try_load_and_verify(generated_code, sample_rows)
            if fn is not None:
                logger.info(
                    f"[green]✓[/green] Compiled extraction function "
                    f"verified on [cyan]{len(sample_rows)}[/cyan] samples "
                    f"(attempt [yellow]{attempt + 1}[/yellow])"
                )
                return fn

            logger.warning(
                f"[yellow]⚠[/yellow] Compilation attempt "
                f"[cyan]{attempt + 1}[/cyan] failed: "
                f"[red]{str(error)[:120]}[/red]"
            )

            repair_prompt = (
                "The following Python function has an error:\n\n"
                f"```python\n{generated_code}\n```\n\n"
                f"Error traceback:\n{error}\n\n"
                "Fix the function. Return ONLY the corrected Python code, "
                "no explanation or markdown fences."
            )
            generated_code = self.inference_engine.generate(repair_prompt)

        raise ValueError(
            f"Failed to compile a valid extraction function after "
            f"{max_repair_attempts} repair attempts"
        )

    def _try_load_and_verify(
        self,
        code: str,
        sample_rows: list[str],
    ) -> tuple[Callable[[str], dict[str, Any]] | None, str | None]:
        """Load generated code via ``exec`` and verify against sample rows.

        Args:
            code: Python source containing an ``extract_data`` function.
            sample_rows: Text rows to test the function against.

        Returns:
            A tuple of ``(function, None)`` on success or
            ``(None, error_description)`` on failure.
        """
        code = self._strip_code_fences(code)

        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)  # noqa: S102
        except Exception:
            return None, traceback.format_exc()

        fn = namespace.get("extract_data")
        if fn is None or not callable(fn):
            return None, "No callable 'extract_data' found in generated code"

        for row in sample_rows:
            try:
                result = fn(row)
                if not isinstance(result, dict):
                    return None, (
                        f"extract_data({repr(row[:60])}) returned "
                        f"{type(result).__name__}, expected dict"
                    )
            except Exception:
                return None, traceback.format_exc()

        return fn, None

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove markdown code fences from LLM-generated code."""
        stripped = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()
