"""Core extraction logic for structured data extraction using Pydantic schemas.

This module provides the Extractor class that orchestrates LLM inference,
JSON repair, Pydantic validation, and retry logic to ensure 100% schema compliance.
"""

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from loclean.extraction.grammar_utils import get_grammar_from_schema
from loclean.extraction.json_repair import repair_json

if TYPE_CHECKING:
    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class Extractor:
    """
    Extractor for structured data extraction using Pydantic schemas.

    Ensures LLM outputs strictly conform to user-defined Pydantic models through:
    - Dynamic GBNF grammar generation from Pydantic schemas
    - JSON repair for malformed outputs
    - Retry logic with prompt adjustment on validation failures
    - Caching for performance optimization
    """

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        cache: "LocleanCache | None" = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Extractor.

        Args:
            inference_engine: Inference engine instance (e.g., LlamaCppEngine).
            cache: Optional cache instance for storing extraction results.
            max_retries: Maximum number of retry attempts on validation failure.
                        Defaults to 3.
        """
        self.inference_engine = inference_engine
        self.cache = cache
        self.max_retries = max_retries

    def extract(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str | None = None,
    ) -> BaseModel:
        """
        Extract structured data from text using a Pydantic schema.

        Args:
            text: Input text to extract from.
            schema: Pydantic BaseModel class defining the output structure.
            instruction: Optional custom instruction.
                        If None, auto-generated from schema.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If extraction fails after max_retries attempts.
            ValueError: If schema is not a Pydantic BaseModel subclass.
        """
        if not issubclass(schema, BaseModel):
            raise ValueError(
                f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
            )

        # Build instruction
        final_instruction = self._build_instruction(schema, instruction)

        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(text, schema, final_instruction)
            cached = self.cache.get_batch([text], cache_key)
            if text in cached:
                try:
                    return schema.model_validate(cached[text])
                except ValidationError:
                    logger.warning(
                        f"Cache entry for '{text}' failed validation, recomputing"
                    )

        # Extract with retry
        result = self._extract_with_retry(
            text, schema, final_instruction, retry_count=0
        )

        if result is None:
            raise ValidationError(
                f"Failed to extract valid {schema.__name__} from text "
                f"after {self.max_retries} retries"
            )

        # Cache result
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
        """
        Extract structured data from a batch of texts.

        Args:
            items: List of input texts to extract from.
            schema: Pydantic BaseModel class defining the output structure.
            instruction: Optional custom instruction. If None, auto-generated from schema.

        Returns:
            Dictionary mapping input_text -> BaseModel instance or None if extraction failed.
        """
        if not items:
            return {}

        final_instruction = self._build_instruction(schema, instruction)

        # Check cache
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
        else:
            misses = items

        # Process misses
        for item in misses:
            try:
                result = self._extract_with_retry(
                    item, schema, final_instruction, retry_count=0
                )
                results[item] = result
            except ValidationError as e:
                logger.warning(f"Failed to extract from '{item}': {e}")
                results[item] = None

        # Cache successful results
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

        return results

    def _extract_with_retry(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str,
        retry_count: int,
    ) -> BaseModel | None:
        """
        Extract with retry logic on validation failures.

        Args:
            text: Input text to extract from.
            schema: Pydantic BaseModel class.
            instruction: Extraction instruction.
            retry_count: Current retry attempt number.

        Returns:
            Validated Pydantic model instance or None if max retries exceeded.
        """
        if retry_count >= self.max_retries:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded for text: '{text[:50]}...'"
            )
            return None

        try:
            # Get grammar from schema
            grammar = get_grammar_from_schema(schema)

            # Check if inference engine has direct LLM access (LlamaCppEngine)
            if hasattr(self.inference_engine, "llm") and hasattr(
                self.inference_engine, "adapter"
            ):
                # Direct LLM access for custom grammar
                prompt = self.inference_engine.adapter.format(instruction, text)
                stop_tokens = self.inference_engine.adapter.get_stop_tokens()

                output = self.inference_engine.llm.create_completion(
                    prompt=prompt,
                    grammar=grammar,
                    max_tokens=512,
                    stop=stop_tokens,
                    echo=False,
                )

                # Extract text from output
                text_output: str | None = None
                if isinstance(output, dict) and "choices" in output:
                    text_output = str(output["choices"][0]["text"]).strip()
                else:
                    if hasattr(output, "__iter__") and not isinstance(
                        output, (str, bytes)
                    ):
                        first_item = next(iter(output), None)
                        if isinstance(first_item, dict) and "choices" in first_item:
                            text_output = str(first_item["choices"][0]["text"]).strip()

                if text_output is None:
                    logger.warning(f"No text extracted for '{text[:50]}...'")
                    return self._retry_extraction(
                        text, schema, instruction, retry_count
                    )

                # Parse and validate
                return self._parse_and_validate(
                    text_output, schema, text, instruction, retry_count
                )

            else:
                # Fallback: use inference engine's clean_batch method
                # This won't use custom grammar, but provides compatibility
                logger.warning(
                    "Inference engine does not support custom grammar. "
                    "Using default extraction method."
                )
                batch_results = self.inference_engine.clean_batch([text], instruction)
                if text in batch_results and batch_results[text] is not None:
                    # Try to validate against schema
                    try:
                        return schema.model_validate(batch_results[text])
                    except ValidationError:
                        return self._retry_extraction(
                            text, schema, instruction, retry_count
                        )
                return self._retry_extraction(text, schema, instruction, retry_count)

        except Exception as e:
            logger.warning(f"Extraction attempt {retry_count + 1} failed: {e}")
            return self._retry_extraction(text, schema, instruction, retry_count)

    def _parse_and_validate(
        self,
        text_output: str,
        schema: type[BaseModel],
        original_text: str,
        instruction: str,
        retry_count: int,
    ) -> BaseModel:
        """
        Parse JSON and validate against Pydantic schema.

        Args:
            text_output: Raw JSON text from LLM.
            schema: Pydantic BaseModel class.
            original_text: Original input text (for retry context).
            instruction: Extraction instruction (for retry context).
            retry_count: Current retry count (for retry context).

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If parsing or validation fails.
        """
        # Try to parse JSON
        try:
            data = json.loads(text_output)
        except json.JSONDecodeError:
            # Try to repair JSON
            repaired = repair_json(text_output)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed even after repair: {e}")
                raise ValidationError(f"Invalid JSON: {text_output[:100]}") from e

        # Validate against schema
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed: {e}")
            raise

    def _retry_extraction(
        self,
        text: str,
        schema: type[BaseModel],
        instruction: str,
        retry_count: int,
    ) -> BaseModel | None:
        """
        Retry extraction with adjusted prompt.

        Args:
            text: Input text.
            schema: Pydantic BaseModel class.
            instruction: Current instruction.
            retry_count: Current retry count.

        Returns:
            Validated Pydantic model instance or None.
        """
        # Adjust instruction to emphasize schema requirements
        adjusted_instruction = (
            f"{instruction}\n\n"
            f"IMPORTANT: The output MUST strictly match the JSON Schema for {schema.__name__}. "
            f"All required fields must be present and correctly typed."
        )

        return self._extract_with_retry(
            text, schema, adjusted_instruction, retry_count + 1
        )

    def _build_instruction(
        self, schema: type[BaseModel], custom_instruction: str | None
    ) -> str:
        """
        Build extraction instruction from schema and optional custom instruction.

        Args:
            schema: Pydantic BaseModel class.
            custom_instruction: Optional custom instruction.

        Returns:
            Final instruction string.
        """
        if custom_instruction:
            return custom_instruction

        # Auto-generate instruction from schema
        schema_json = schema.model_json_schema()
        return (
            f"Extract structured information from the text and return it as JSON "
            f"matching this schema: {json.dumps(schema_json, indent=2)}. "
            f"All required fields must be present and correctly typed."
        )

    def _get_cache_key(
        self, text: str, schema: type[BaseModel], instruction: str
    ) -> str:
        """
        Generate cache key for extraction.

        Args:
            text: Input text (empty string for batch operations).
            schema: Pydantic BaseModel class.
            instruction: Extraction instruction.

        Returns:
            Cache key string.
        """
        # Include schema name in cache key to invalidate on schema changes
        schema_name = schema.__name__
        return f"extract_v1::{schema_name}::{instruction}"
