"""Main scrub functions for PII detection and masking/fake data generation."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from loclean.privacy.detector import PIIDetector
from loclean.privacy.generator import FakeDataGenerator
from loclean.privacy.schemas import PIIEntity

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


def replace_entities(
    text: str,
    entities: List[PIIEntity],
    mode: str,
    locale: str = "vi_VN",
) -> str:
    """
    Replace entities in text with masks or fake data.

    Args:
        text: Original text
        entities: List of PII entities (already resolved for overlaps)
        mode: "mask" (replace with [TYPE]) or "fake" (replace with fake data)
        locale: Faker locale for fake data generation

    Returns:
        Scrubbed text with entities replaced
    """
    if not entities:
        return text

    # Sort by start position (reverse for safe replacement)
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

    if mode == "mask":
        generator = None
    else:  # mode == "fake"
        generator = FakeDataGenerator(locale=locale)

    result = text
    for entity in sorted_entities:
        if mode == "mask":
            replacement = f"[{entity.type.upper()}]"
        else:  # mode == "fake"
            replacement = generator.generate_fake(entity)

        result = result[: entity.start] + replacement + result[entity.end :]

    return result


def scrub_string(
    text: str,
    strategies: List[str],
    mode: str = "mask",
    locale: str = "vi_VN",
    inference_engine: Optional["InferenceEngine"] = None,
    **kwargs: Any,
) -> str:
    """
    Scrub PII from a string.

    Args:
        text: Input text to scrub
        strategies: List of PII types to detect (e.g., ["person", "phone", "email"])
        mode: "mask" (replace with [TYPE]) or "fake" (replace with fake data)
        locale: Faker locale for fake data generation (default: "vi_VN")
        inference_engine: Optional inference engine for LLM detection.
                         If None, LLM strategies will be skipped.
        **kwargs: Additional arguments (unused, for API compatibility)

    Returns:
        Scrubbed text with PII replaced
    """
    if not text or not text.strip():
        return text

    # Initialize detector
    detector = PIIDetector(inference_engine=inference_engine)

    # Detect entities
    entities = detector.detect(text, strategies)

    # Replace entities
    scrubbed = replace_entities(text, entities, mode, locale)

    return scrubbed


def scrub_dataframe(
    df: IntoFrameT,
    target_col: str,
    strategies: List[str],
    mode: str = "mask",
    locale: str = "vi_VN",
    inference_engine: Optional["InferenceEngine"] = None,
    **kwargs: Any,
) -> IntoFrameT:
    """
    Scrub PII from a DataFrame column.

    Args:
        df: Input DataFrame
        target_col: Name of the column to scrub
        strategies: List of PII types to detect
        mode: "mask" (replace with [TYPE]) or "fake" (replace with fake data)
        locale: Faker locale for fake data generation (default: "vi_VN")
        inference_engine: Optional inference engine for LLM detection.
                         If None, LLM strategies will be skipped.
        **kwargs: Additional arguments (unused, for API compatibility)

    Returns:
        DataFrame with scrubbed column (same type as input)
    """
    df_nw = nw.from_native(df)  # type: ignore[type-var]
    if target_col not in df_nw.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    # Get unique values for batch processing
    unique_df = df_nw.unique(subset=[target_col])
    unique_values = unique_df[target_col].to_list()

    # Filter out None and empty strings
    unique_values = [
        str(x) for x in unique_values if x is not None and str(x).strip() != ""
    ]

    if not unique_values:
        logger.warning("No valid values found. Returning original DataFrame.")
        return df

    # Initialize detector
    detector = PIIDetector(inference_engine=inference_engine)

    # Process each unique value
    scrubbed_map: dict[str, str] = {}
    for value in unique_values:
        entities = detector.detect(value, strategies)
        scrubbed_map[value] = replace_entities(value, entities, mode, locale)

    # Apply mapping to DataFrame
    df_nw = df_nw.with_columns(
        nw.col(target_col).map_elements(
            lambda x: scrubbed_map.get(str(x), str(x)) if x is not None else x
        )
    )

    return df_nw.to_native()

