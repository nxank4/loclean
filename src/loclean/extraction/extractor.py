"""Core extraction logic for structured data extraction using Pydantic schemas.

This module provides the Extractor class that orchestrates LLM inference,
JSON repair, Pydantic validation, and retry logic to ensure 100% schema compliance.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any

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
                    f"[yellow]⚠[/yellow] Empty response for "
                    f"[dim]'{text[:50]}...'[/dim]"
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
