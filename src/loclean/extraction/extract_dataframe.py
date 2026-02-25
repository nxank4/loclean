"""DataFrame integration for structured extraction using Pydantic schemas.

This module provides functions to extract structured data from DataFrame columns,
with support for both structured output (dict/Struct) for optimal performance
and Pydantic model instances for advanced use cases.

The ``extract_dataframe_compiled`` function offers a generative-compilation
path: it samples diverse rows, synthesises a pure-Python extractor via the
LLM, verifies it, then maps it natively across the column without further
inference round-trips.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import BaseModel

from loclean.extraction.extractor import Extractor

if TYPE_CHECKING:
    from loclean.cache import LocleanCache
    from loclean.inference.base import InferenceEngine

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


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
        >>> df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
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
    output_map: dict[str, Any]
    if output_type == "dict":
        output_map = _convert_to_dict_format(extracted_map, schema)
    else:
        # For pydantic output, keep as Pydantic model for DataFrame storage
        output_map = {k: v for k, v in extracted_map.items()}

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
        raise ValueError(
            f"Unsupported dataframe backend: {module_name}. "
            "Loclean currently explicitly supports 'pandas' and 'polars' "
            "for this operation."
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
            # Use get_origin for generic types, direct comparison for simple types
            from typing import get_origin

            origin = get_origin(field_type)
            # Check for List types first to avoid 'list[str]' matching the
            # generic 'str' check below
            if (
                origin is list
                or field_type is list
                or (origin is not None and list in getattr(field_type, "__args__", []))
                or "List" in str(field_type)
                or "list" in str(field_type)
            ):
                from typing import get_args

                args = get_args(field_type)
                inner_type = args[0] if args else str
                inner_pl: Any
                if inner_type is int:
                    inner_pl = pl.Int64
                elif inner_type is float:
                    inner_pl = pl.Float64
                elif inner_type is bool:
                    inner_pl = pl.Boolean
                else:
                    inner_pl = pl.Utf8
                struct_fields[field_name] = pl.List(inner_pl)
            elif field_type is str or (
                origin is not None and str in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Utf8
            elif field_type is int or (
                origin is not None and int in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Int64
            elif field_type is float or (
                origin is not None and float in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Float64
            elif field_type is bool or (
                origin is not None and bool in getattr(field_type, "__args__", [])
            ):
                struct_fields[field_name] = pl.Boolean
            else:
                # Fallback to Object for complex types (nested models, etc.)
                struct_fields[field_name] = pl.Object

        # Create Struct column
        # Pass dict values directly - Polars will convert to Struct based on dtype
        struct_values = mapping_values

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
            },
            schema_overrides={f"{target_col}_extracted": pl.Object},
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


# ------------------------------------------------------------------
# Generative Compilation Path
# ------------------------------------------------------------------


def _sample_diverse_rows(
    df_nw: nw.DataFrame[Any],
    target_col: str,
    n: int = 50,
) -> list[str]:
    """Select a diverse sample of approximately *n* unique rows.

    Uses length-stratified sampling: unique non-empty values are sorted
    by character length and then *n* evenly-spaced entries are picked.
    This ensures coverage across short, medium, and long strings.

    Args:
        df_nw: Narwhals DataFrame.
        target_col: Column to sample from.
        n: Desired sample size (default 50).

    Returns:
        List of up to *n* unique string values.
    """
    raw_values = df_nw.unique(subset=[target_col])[target_col].to_list()
    valid: list[str] = [str(x) for x in raw_values if x is not None and str(x).strip()]

    if len(valid) <= n:
        return valid

    valid.sort(key=len)
    step = len(valid) / n
    return [valid[int(i * step)] for i in range(n)]


def extract_dataframe_compiled(
    df: IntoFrameT,
    target_col: str,
    schema: type[BaseModel],
    instruction: str | None = None,
    extractor: Extractor | None = None,
    inference_engine: "InferenceEngine | None" = None,
    max_retries: int = 3,
    sample_size: int = 50,
    **kwargs: Any,
) -> IntoFrameT:
    """Extract structured data using a compiled Python function.

    Instead of calling the LLM for every row, this function:

    1. Samples ~*sample_size* diverse rows from *target_col*.
    2. Asks the inference engine to synthesise a pure-Python
       ``extract_data`` function matching *schema*.
    3. Verifies the function against the sample (with up to
       *max_retries* repair iterations).
    4. Maps the compiled function over every unique value and joins
       the results back to the original DataFrame.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column to extract from.
        schema: Pydantic BaseModel class defining the output structure.
        instruction: Optional domain hint forwarded to code generation.
        extractor: Optional pre-built Extractor instance.
        inference_engine: Required if *extractor* is ``None``.
        max_retries: Repair budget for the verification loop.
        sample_size: Number of diverse rows to sample (default 50).
        **kwargs: Reserved for API compatibility.

    Returns:
        DataFrame with an added ``{target_col}_extracted`` column
        containing dicts. Return type matches the input backend.

    Raises:
        ValueError: If *target_col* is missing, or compilation fails.
    """
    df_nw = nw.from_native(df)  # type: ignore[type-var]

    if target_col not in df_nw.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    sample_rows = _sample_diverse_rows(df_nw, target_col, n=sample_size)
    if not sample_rows:
        logger.warning("No valid values found. Returning original DataFrame.")
        return df

    if extractor is None:
        if inference_engine is None:
            raise ValueError("Either extractor or inference_engine must be provided")
        extractor = Extractor(
            inference_engine=inference_engine, max_retries=max_retries
        )

    extract_fn = extractor.compile(
        schema, sample_rows, instruction, max_repair_attempts=max_retries
    )

    # --- Native execution across all unique values -----------------------
    unique_raw = df_nw.unique(subset=[target_col])[target_col].to_list()
    unique_values: list[str] = [
        str(x) for x in unique_raw if x is not None and str(x).strip()
    ]

    start_time = time.time()

    output_map: dict[str, dict[str, Any] | None] = {}
    successful = 0
    failed = 0
    for val in unique_values:
        try:
            output_map[val] = extract_fn(val)
            successful += 1
        except Exception as exc:
            logger.warning(
                f"[yellow]⚠[/yellow] Compiled function failed for "
                f"[dim]'{val[:50]}'[/dim]: [red]{exc}[/red]"
            )
            output_map[val] = None
            failed += 1

    elapsed_time = time.time() - start_time

    from loclean.utils.rich_output import log_processing_summary

    log_processing_summary(
        total_processed=len(unique_values),
        successful=successful,
        failed=failed,
        time_taken=elapsed_time,
        context="Compiled Extraction",
    )

    if not output_map:
        logger.warning("No values extracted. Returning original DataFrame.")
        return df

    # --- Build mapping DataFrame using the same backend as input ---------
    mapping_keys = list(output_map.keys())
    mapping_values: list[Any] = [output_map[k] for k in mapping_keys]

    native_df_cls = type(df_nw.to_native())
    module_name = native_df_cls.__module__

    map_df_native: Any
    if "polars" in module_name:
        map_df_native = _create_polars_mapping_df(
            target_col, mapping_keys, mapping_values, schema, "dict"
        )
    elif "pandas" in module_name:
        map_df_native = _create_pandas_mapping_df(
            target_col, mapping_keys, mapping_values, "dict"
        )
    else:
        raise ValueError(
            f"Unsupported dataframe backend: {module_name}. "
            "Loclean currently supports 'pandas' and 'polars'."
        )

    map_df = nw.from_native(map_df_native)

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
        .drop([f"{target_col}_join_key"])
        .to_native()
    )

    return result_df
