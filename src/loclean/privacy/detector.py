"""Hybrid PII detector combining regex and LLM-based detection."""

import logging
from typing import TYPE_CHECKING, List

from loclean.privacy.llm_detector import LLMDetector
from loclean.privacy.regex_detector import RegexDetector
from loclean.privacy.schemas import PIIEntity

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = logging.getLogger(__name__)

# Strategies handled by regex (fast)
REGEX_STRATEGIES = {"email", "phone", "credit_card", "ip_address"}

# Strategies handled by LLM (slow but accurate)
LLM_STRATEGIES = {"person", "address"}


def find_all_positions(text: str, value: str) -> List[tuple[int, int]]:
    """
    Find all occurrences of value in text.

    Privacy-first: mask all occurrences.

    Args:
        text: Text to search in
        value: Value to find

    Returns:
        List of (start, end) position tuples
    """
    positions: List[tuple[int, int]] = []
    start = 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(value)))
        start = pos + 1
    return positions


def resolve_overlaps(entities: List[PIIEntity]) -> List[PIIEntity]:
    """
    Resolve overlapping entities using "Longer Match Wins" strategy.

    If entities overlap, keep the longer one. If same length, keep the first.

    Args:
        entities: List of PII entities (may have overlaps)

    Returns:
        List of non-overlapping entities, sorted by start position
    """
    if not entities:
        return []

    # Sort by length (descending), then by start position
    sorted_entities = sorted(entities, key=lambda e: (-e.length, e.start))

    resolved: List[PIIEntity] = []
    for entity in sorted_entities:
        # Check if this entity overlaps with any already resolved entity
        overlaps = False
        for resolved_entity in resolved:
            if (
                entity.start < resolved_entity.end
                and entity.end > resolved_entity.start
            ):
                overlaps = True
                break

        if not overlaps:
            resolved.append(entity)

    # Sort by start position for final output
    return sorted(resolved, key=lambda e: e.start)


class PIIDetector:
    """Hybrid detector combining regex (fast) and LLM (accurate) detection."""

    def __init__(
        self,
        inference_engine: "InferenceEngine | None" = None,
        cache=None,
    ) -> None:
        """
        Initialize PII detector.

        Args:
            inference_engine: Optional inference engine for LLM detection.
                             If None, LLM strategies will be skipped.
            cache: Optional cache instance for LLM results
        """
        self.regex_detector = RegexDetector()
        self.llm_detector: LLMDetector | None = None

        if inference_engine is not None:
            self.llm_detector = LLMDetector(inference_engine, cache)

    def detect(self, text: str, strategies: List[str]) -> List[PIIEntity]:
        """
        Detect PII entities in text using hybrid approach.

        Args:
            text: Input text to scan
            strategies: List of PII types to detect
                       (e.g., ["person", "phone", "email"])

        Returns:
            List of detected PII entities (non-overlapping, sorted by position)
        """
        all_entities: List[PIIEntity] = []

        # Filter strategies into regex vs LLM
        regex_strategies = [s for s in strategies if s in REGEX_STRATEGIES]
        llm_strategies = [s for s in strategies if s in LLM_STRATEGIES]

        # Run regex detection (fast)
        if regex_strategies:
            if "email" in regex_strategies:
                all_entities.extend(self.regex_detector.detect_email(text))
            if "phone" in regex_strategies:
                all_entities.extend(self.regex_detector.detect_phone(text))
            if "credit_card" in regex_strategies:
                all_entities.extend(self.regex_detector.detect_credit_card(text))
            if "ip_address" in regex_strategies:
                all_entities.extend(self.regex_detector.detect_ip_address(text))

        # Run LLM detection (slow but accurate)
        if llm_strategies and self.llm_detector is not None:
            llm_results = self.llm_detector.detect_batch([text], llm_strategies)
            if llm_results:
                result = llm_results[0]
                # LLM returns entities with type and value only
                # We need to find positions in the original text
                for entity_data in result.entities:
                    # Find all occurrences of the value
                    positions = find_all_positions(text, entity_data.value)
                    for start, end in positions:
                        all_entities.append(
                            PIIEntity(
                                type=entity_data.type,
                                value=entity_data.value,
                                start=start,
                                end=end,
                            )
                        )
        elif llm_strategies:
            logger.warning(
                "LLM strategies requested but no inference engine provided. "
                "Skipping LLM detection."
            )

        # Resolve overlaps using "Longer Match Wins"
        resolved_entities = resolve_overlaps(all_entities)

        return resolved_entities

