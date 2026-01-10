"""DataFrame integration for structured extraction using Pydantic schemas.

This module provides functions to extract structured data from DataFrame columns,
with support for both structured output (dict/Struct) for optimal performance
and Pydantic model instances for advanced use cases.
"""

import logging
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import BaseModel

from loclean.extraction.extractor import Extractor

if TYPE_CHECKING:
    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


def extract_dataframe(
    df: IntoFrameT,
    target_col: str,
    schema: type[BaseModel],
    instruction: str | None = None,
    output_type: Literal["dict", "pydantic"] = "dict",
    extractor: Extractor | None = None,
    inference_engine: "InferenceEngine | None" = None,
    cache: "LocleanCache | None" = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> IntoFrameT:
    """
    Extract structured data from a DataFrame column using a Pydantic schema.

    Args:
        df: Input DataFrame (pandas, Polars, etc.)
        target_col: Name of the column to extract from
        schema: Pydantic BaseModel class defining the output structure
        instruction: Optional custom instruction. If None, auto-generated from schema
        output_type: Output format ("dict" or "pydantic")
                   - "dict" (default): Structured data (Polars Struct / Pandas dict)
                                     for optimal performance and vectorized operations
                   - "pydantic": Pydantic model instances (slower, breaks vectorization)
        extractor: Optional Extractor instance. If None, creates a new one.
        inference_engine: Optional inference engine. Required if extractor is None.
        cache: Optional cache instance. Used if extractor is None.
        max_retries: Maximum retry attempts on validation failure (default: 3)
        **kwargs: Additional arguments (unused, for API compatibility)

    Returns:
        DataFrame with added column `{target_col}_extracted` containing extracted data.
        Return type matches input type (pandas -> pandas, Polars -> Polars, etc.)

    Raises:
        ValueError: If target_col is not found in DataFrame or schema is invalid.

    Example:
        >>> from pydantic import BaseModel
        >>> import polars as pl
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: int
        >>> df = pl.DataFrame({"description": ["Bán áo thun đỏ giá 50k"]})
        >>> result = extract_dataframe(df, "description", Product)
        >>> # Query with Polars Struct
        >>> result.filter(pl.col("description_extracted").struct.field("price") > 50000)
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

    # Initialize extractor if not provided
    if extractor is None:
        if inference_engine is None:
            raise ValueError("Either extractor or inference_engine must be provided")
        extractor = Extractor(
            inference_engine=inference_engine, cache=cache, max_retries=max_retries
        )

    # Extract from unique values
    extracted_map: dict[str, BaseModel | None] = extractor.extract_batch(
        unique_values, schema, instruction
    )

    # Convert to output format
    if output_type == "dict":
        output_map: dict[str, dict[str, Any] | None] = _convert_to_dict_format(
            extracted_map, schema
        )
    else:  # output_type == "pydantic"
        # For pydantic output, convert BaseModel to dict for DataFrame storage
        output_map = {
            k: (v.model_dump() if v is not None else None)
            for k, v in extracted_map.items()
        }

    # Create mapping DataFrame and join
    mapping_keys = list(output_map.keys())
    mapping_values = [output_map[k] for k in mapping_keys]

    # Detect backend and create mapping DataFrame
    native_df_cls = type(df_nw.to_native())
    module_name = native_df_cls.__module__

    map_df_native: Any
    if "polars" in module_name:
        map_df_native = _create_polars_mapping_df(
            target_col, mapping_keys, mapping_values, schema, output_type
        )
    elif "pandas" in module_name:
        map_df_native = _create_pandas_mapping_df(
            target_col, mapping_keys, mapping_values, output_type
        )
    else:
        # Fallback: use pandas

        map_df_native = _create_pandas_mapping_df(
            target_col, mapping_keys, mapping_values, output_type
        )

    map_df = nw.from_native(map_df_native)

    # Join and add extracted column
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
            nw.coalesce([nw.col(f"{target_col}_extracted"), nw.lit(None)]).alias(
                f"{target_col}_extracted"
            )
        )
        .drop([f"{target_col}_join_key"])
        .to_native()
    )

    return result_df


def _convert_to_dict_format(
    extracted_map: dict[str, BaseModel | None], schema: type[BaseModel]
) -> dict[str, dict[str, Any] | None]:
    """
    Convert Pydantic models to dict format for structured output.

    Args:
        extracted_map: Dictionary mapping text -> BaseModel or None
        schema: Pydantic BaseModel class (for type hints)

    Returns:
        Dictionary mapping text -> dict or None
    """
    result: dict[str, dict[str, Any] | None] = {}
    for key, value in extracted_map.items():
        if value is None:
            result[key] = None
        else:
            result[key] = value.model_dump()
    return result


def _create_polars_mapping_df(
    target_col: str,
    mapping_keys: list[str],
    mapping_values: list[Any],
    schema: type[BaseModel],
    output_type: Literal["dict", "pydantic"],
) -> Any:
    """
    Create Polars DataFrame with structured output.

    For output_type="dict", creates a Struct column with typed fields for
    optimal performance and vectorized operations.

    Args:
        target_col: Target column name
        mapping_keys: List of keys (original text values)
        mapping_values: List of extracted values (dict or BaseModel)
        schema: Pydantic BaseModel class
        output_type: Output format ("dict" or "pydantic")

    Returns:
        Polars DataFrame
    """
    import polars as pl

    if output_type == "dict":
        # Create Struct schema from Pydantic model
        struct_fields: dict[str, Any] = {}
        for field_name, field_info in schema.model_fields.items():
            field_type = field_info.annotation
            # Map Python types to Polars types
            if field_type == str or (
                hasattr(field_type, "__origin__")
                and str in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Utf8
            elif field_type == int or (
                hasattr(field_type, "__origin__")
                and int in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Int64
            elif field_type == float or (
                hasattr(field_type, "__origin__")
                and float in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Float64
            elif field_type == bool or (
                hasattr(field_type, "__origin__")
                and bool in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Boolean
            else:
                # Fallback to Utf8 for complex types
                struct_fields[field_name] = pl.Utf8

        # Create Struct column
        struct_values = [
            pl.Struct(mapping_values[i]) if mapping_values[i] is not None else None
            for i in range(len(mapping_values))
        ]

        return pl.DataFrame(
            {
                target_col: mapping_keys,
                f"{target_col}_extracted": pl.Series(
                    struct_values, dtype=pl.Struct(struct_fields)
                ),
            }
        )
    else:
        # output_type == "pydantic": Store as object (slower)
        return pl.DataFrame(
            {
                target_col: mapping_keys,
                f"{target_col}_extracted": mapping_values,
            }
        )


def _create_pandas_mapping_df(
    target_col: str,
    mapping_keys: list[str],
    mapping_values: list[Any],
    output_type: Literal["dict", "pydantic"],
) -> Any:
    """
    Create Pandas DataFrame with structured output.

    Args:
        target_col: Target column name
        mapping_keys: List of keys (original text values)
        mapping_values: List of extracted values (dict or BaseModel)
        output_type: Output format ("dict" or "pydantic")

    Returns:
        Pandas DataFrame
    """
    import pandas as pd

    return pd.DataFrame(
        {
            target_col: mapping_keys,
            f"{target_col}_extracted": mapping_values,
        }
    )
