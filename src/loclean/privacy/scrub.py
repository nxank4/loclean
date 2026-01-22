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

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


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

    generator: FakeDataGenerator | None = None
    if mode == "fake":
        generator = FakeDataGenerator(locale=locale)

    result = text
    for entity in sorted_entities:
        if mode == "mask":
            replacement = f"[{entity.type.upper()}]"
        else:
            assert generator is not None  # Should not happen if mode is validated
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
    unique_values_raw = unique_df[target_col].to_list()

    # Filter out None and empty strings, convert to string
    unique_values: list[str] = []
    unique_map: dict[str, Any] = {}  # Map string representation to original value
    for x in unique_values_raw:
        if x is not None and str(x).strip() != "":
            str_val = str(x)
            unique_values.append(str_val)
            unique_map[str_val] = x

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

    # Create mapping DataFrame and join (similar to narwhals_ops.py approach)
    mapping_keys = list(scrubbed_map.keys())
    mapping_values = [scrubbed_map[k] for k in mapping_keys]

    # Detect backend and create mapping DataFrame
    native_df_cls = type(df_nw.to_native())
    module_name = native_df_cls.__module__

    map_df_native: Any
    if "polars" in module_name:
        import polars as pl

        map_df_native = pl.DataFrame(
            {
                target_col: mapping_keys,
                f"{target_col}_scrubbed": mapping_values,
            },
            schema={
                target_col: pl.String,
                f"{target_col}_scrubbed": pl.String,
            },
        )
    elif "pandas" in module_name:
        import pandas as pd

        map_df_native = pd.DataFrame(
            {
                target_col: mapping_keys,
                f"{target_col}_scrubbed": mapping_values,
            }
        )
    else:
        raise ValueError(
            f"Unsupported dataframe type: {module_name}. "
            "Loclean currently explicitly supports 'pandas' and 'polars' "
            "for this operation."
        )

    map_df = nw.from_native(map_df_native)

    # Join and replace column
    result_df = (
        df_nw.with_columns(
            nw.col(target_col).cast(nw.String).alias(f"{target_col}_join_key")
        )
        .join(
            map_df,  # type: ignore[arg-type]
            left_on=f"{target_col}_join_key",
            right_on=target_col,
            how="left",
        )
        .with_columns(
            nw.coalesce([nw.col(f"{target_col}_scrubbed"), nw.col(target_col)]).alias(
                target_col
            )
        )
        .drop([f"{target_col}_join_key", f"{target_col}_scrubbed"])
        .to_native()
    )

    return result_df
