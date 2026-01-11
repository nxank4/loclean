"""LLM-based semantic PII detection for unstructured data types."""

import json
import logging
from typing import TYPE_CHECKING, List

from jinja2 import Template

from loclean.cache import LocleanCache
from loclean.privacy.schemas import PIIDetectionResult
from loclean.utils.resources import get_grammar_preset, load_template

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class LLMDetector:
    """LLM-based detector for semantic PII types (person names, addresses)."""

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        cache: LocleanCache | None = None,
    ) -> None:
        """
        Initialize LLM detector.

        Args:
            inference_engine: Inference engine instance for LLM calls
            cache: Optional cache instance for caching results
        """
        self.inference_engine = inference_engine
        self.cache = cache or LocleanCache()

        # Load grammar and template
        from loclean.utils.resources import load_grammar

        self.grammar_str = load_grammar("pii_detection.gbnf")
        self.template_str = load_template("pii_detection.j2")
        self.template = Template(self.template_str)

    def detect_batch(
        self, items: List[str], strategies: List[str]
    ) -> List[PIIDetectionResult]:
        """
        Detect PII entities in a batch of text items using LLM.

        Args:
            items: List of text items to process
            strategies: List of PII types to detect (e.g., ["person", "address"])

        Returns:
            List of detection results, one per input item
        """
        # Filter to only LLM-based strategies
        llm_strategies = [s for s in strategies if s in ["person", "address"]]
        if not llm_strategies:
            return [PIIDetectionResult(entities=[], reasoning=None) for _ in items]

        # Check cache - use a consistent instruction key
        # We'll use a simple instruction string for caching
        cache_instruction = f"Detect {', '.join(llm_strategies)}"
        cached_results = self.cache.get_batch(items, cache_instruction)
        misses = [item for item in items if item not in cached_results]

        results: List[PIIDetectionResult] = []

        # Process cached items
        for item in items:
            if item in cached_results:
                cached_data = cached_results[item]
                if cached_data:
                    try:
                        result = PIIDetectionResult(**cached_data)
                        results.append(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse cached result for '{item}': {e}"
                        )
                        results.append(PIIDetectionResult(entities=[], reasoning=None))
                else:
                    results.append(PIIDetectionResult(entities=[], reasoning=None))
            else:
                # Placeholder for misses - will be replaced below
                results.append(PIIDetectionResult(entities=[], reasoning=None))

        # Process misses using inference engine
        if misses:
            logger.info(f"Cache miss for {len(misses)} items. Running LLM inference...")

            # Use the inference engine's clean_batch method
            # We'll adapt it for PII detection
            batch_results = self._detect_with_llm(misses, llm_strategies)

            # Cache valid results
            valid_results = {
                item: result.model_dump()
                for item, result in zip(misses, batch_results, strict=False)
                if result.entities
            }
            if valid_results:
                self.cache.set_batch(
                    list(valid_results.keys()), cache_instruction, valid_results
                )

            # Replace placeholder results with actual results
            miss_index = 0
            for i, item in enumerate(items):
                if item in misses:
                    results[i] = batch_results[miss_index]
                    miss_index += 1

        return results

    def _detect_with_llm(
        self, items: List[str], strategies: List[str]
    ) -> List[PIIDetectionResult]:
        """
        Detect PII using LLM inference.

        Args:
            items: List of text items
            strategies: List of PII types to detect

        Returns:
            List of detection results
        """
        results: List[PIIDetectionResult] = []

        # We need to use the inference engine directly
        # This requires accessing the underlying llama-cpp-python instance
        if hasattr(self.inference_engine, "llm") and hasattr(
            self.inference_engine, "adapter"
        ):
            # Load PII detection grammar using JSON schema (more reliable)
            from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]
            from loclean.privacy.schemas import PIIDetectionResult
            import json

            # Use JSON schema approach like extraction
            json_schema = PIIDetectionResult.model_json_schema()
            json_schema_str = json.dumps(json_schema)
            pii_grammar = LlamaGrammar.from_json_schema(json_schema_str)

            # Direct LLM access (for LlamaCppEngine)
            for item in items:
                try:
                    # Build instruction from template for this item
                    instruction = self.template.render(strategies=strategies, item=item)
                    # Format prompt using adapter
                    # The adapter expects instruction and item separately
                    prompt = self.inference_engine.adapter.format(instruction, item)
                    stop_tokens = self.inference_engine.adapter.get_stop_tokens()

                    output = self.inference_engine.llm.create_completion(
                        prompt=prompt,
                        grammar=pii_grammar,
                        max_tokens=256,
                        stop=stop_tokens,
                        echo=False,
                    )

                    # Parse output
                    text: str | None = None
                    if isinstance(output, dict) and "choices" in output:
                        text = str(output["choices"][0]["text"]).strip()
                    else:
                        if hasattr(output, "__iter__") and not isinstance(
                            output, (str, bytes)
                        ):
                            first_item = next(iter(output), None)
                            if isinstance(first_item, dict) and "choices" in first_item:
                                text = str(first_item["choices"][0]["text"]).strip()

                    if text:
                        data = json.loads(text)
                        result = PIIDetectionResult(**data)
                        results.append(result)
                    else:
                        results.append(PIIDetectionResult(entities=[], reasoning=None))

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON for item '{item}': {e}")
                    results.append(PIIDetectionResult(entities=[], reasoning=None))
                except Exception as e:
                    logger.error(f"Inference error for item '{item}': {e}")
                    results.append(PIIDetectionResult(entities=[], reasoning=None))
        else:
            # Fallback: return empty results
            logger.warning("Inference engine does not support direct LLM access")
            results = [PIIDetectionResult(entities=[], reasoning=None) for _ in items]

        return results
