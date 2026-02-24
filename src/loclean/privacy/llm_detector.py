"""LLM-based semantic PII detection for unstructured data types."""

import json
import logging
from typing import TYPE_CHECKING, List

from loclean.cache import LocleanCache
from loclean.privacy.schemas import PIIDetectionResult

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


class LLMDetector:
    """LLM-based detector for semantic PII types (person names, addresses)."""

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        cache: LocleanCache | None = None,
    ) -> None:
        """Initialize LLM detector.

        Args:
            inference_engine: Inference engine instance for LLM calls.
            cache: Optional cache instance for caching results.
        """
        self.inference_engine = inference_engine
        self.cache = cache or LocleanCache()

    def detect_batch(
        self, items: List[str], strategies: List[str]
    ) -> List[PIIDetectionResult]:
        """Detect PII entities in a batch of text items using LLM.

        Args:
            items: List of text items to process.
            strategies: PII types to detect (e.g. ["person", "address"]).

        Returns:
            List of detection results, one per input item.
        """
        llm_strategies = [s for s in strategies if s in ["person", "address"]]
        if not llm_strategies:
            return [PIIDetectionResult(entities=[], reasoning=None) for _ in items]

        cache_instruction = f"Detect {', '.join(llm_strategies)}"
        cached_results = self.cache.get_batch(items, cache_instruction)
        misses = [item for item in items if item not in cached_results]

        results: List[PIIDetectionResult] = []

        for item in items:
            if item in cached_results:
                cached_data = cached_results[item]
                if cached_data:
                    try:
                        results.append(PIIDetectionResult(**cached_data))
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse cached result for '{item}': {e}"
                        )
                        results.append(PIIDetectionResult(entities=[], reasoning=None))
                else:
                    results.append(PIIDetectionResult(entities=[], reasoning=None))
            else:
                results.append(PIIDetectionResult(entities=[], reasoning=None))

        if misses:
            from loclean.utils.rich_output import log_cache_stats

            log_cache_stats(
                total_items=len(items),
                cache_hits=len(items) - len(misses),
                cache_misses=len(misses),
                context="PII Detection",
            )

            batch_results = self._detect_with_llm(misses, llm_strategies)

            valid_results = {
                item: result.model_dump()
                for item, result in zip(misses, batch_results, strict=False)
                if result.entities
            }
            if valid_results:
                self.cache.set_batch(
                    list(valid_results.keys()), cache_instruction, valid_results
                )

            miss_index = 0
            for i, item in enumerate(items):
                if item in misses:
                    results[i] = batch_results[miss_index]
                    miss_index += 1

        return results

    def _detect_with_llm(
        self, items: List[str], strategies: List[str]
    ) -> List[PIIDetectionResult]:
        """Detect PII using LLM inference via the engine's generate method.

        Args:
            items: List of text items.
            strategies: List of PII types to detect.

        Returns:
            List of detection results.
        """
        results: List[PIIDetectionResult] = []
        strategy_names = ", ".join(strategies)

        for item in items:
            try:
                prompt = (
                    f"Detect the following PII types in the text: {strategy_names}.\n"
                    f"Return a JSON object with 'entities' (list of "
                    f"{{'type': str, 'value': str, 'start': int, 'end': int}}) "
                    f"and 'reasoning' (str or null).\n\n"
                    f"Text: {item}"
                )

                raw = self.inference_engine.generate(
                    prompt, schema=PIIDetectionResult
                )
                data = json.loads(raw)
                results.append(PIIDetectionResult(**data))

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode JSON for item '{item}': {e}")
                results.append(PIIDetectionResult(entities=[], reasoning=None))
            except Exception as e:
                logger.error(f"Inference error for item '{item}': {e}")
                results.append(PIIDetectionResult(entities=[], reasoning=None))

        return results
