import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT
from tqdm import tqdm

if TYPE_CHECKING:
    from loclean.inference.manager import LocalInferenceEngine

logger = logging.getLogger(__name__)


class NarwhalsEngine:
    """
    Narwhals-based engine for efficient semantic data cleaning.
    """

    @staticmethod
    def process_column(
        df_native: IntoFrameT,
        col_name: str,
        inference_engine: "LocalInferenceEngine",
        instruction: str,
        batch_size: int = 50,
    ) -> IntoFrameT:
        """
        Clean a specific column using the inference engine with batching
        and progress tracking.

        Args:
            df_native: Input DataFrame (pandas, Polars, PyArrow, etc.).
            col_name: Name of the column to clean.
            inference_engine: Inference engine instance for semantic extraction.
            instruction: Instruction to guide the LLM extraction.
            batch_size: Number of unique values to process per batch. Defaults to 50.

        Returns:
            DataFrame with added 'clean_value' and 'clean_unit' columns.
            Return type matches input type (pandas -> pandas, Polars -> Polars, etc.)

        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        df = nw.from_native(df_native)  # type: ignore[type-var]

        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")

        # Cast to String before unique() to ensure consistent types for join later.
        # This prevents type mismatches (e.g., Int64 vs String) when joining results.
        unique_df_native = (
            df.select(nw.col(col_name).cast(nw.String)).unique().to_native()
        )

        if hasattr(unique_df_native, "to_pandas"):
            unique_df_native = unique_df_native.to_pandas()
            col_values = unique_df_native[col_name].tolist()
        elif hasattr(unique_df_native, "to_series"):
            col_values = unique_df_native[col_name].to_series().to_list()  # type: ignore[index]
        else:
            col_values = unique_df_native[col_name].to_list()  # type: ignore[index]

        uniques: List[str] = [
            str(x) for x in col_values if x is not None and str(x).strip() != ""
        ]

        logger.info(f"Found {len(uniques)} unique patterns to clean in '{col_name}'.")

        if not uniques:
            logger.warning(
                "No valid unique values found. Returning original DataFrame."
            )
            return df_native

        mapping_results: Dict[str, Optional[Dict[str, Any]]] = {}

        chunks: List[List[str]] = [
            uniques[i : i + batch_size] for i in range(0, len(uniques), batch_size)
        ]

        print(f"🧠 Semantic Cleaning: Processing {len(uniques)} unique patterns...")

        for chunk in tqdm(chunks, desc="Inference Batches", unit="batch"):
            batch_result: Dict[str, Optional[Dict[str, Any]]] = (
                inference_engine.clean_batch(chunk, instruction=instruction)
            )
            mapping_results.update(batch_result)

        keys: List[str] = []
        clean_values: List[Optional[float]] = []
        clean_units: List[Optional[str]] = []

        for original_val, clean_data in mapping_results.items():
            keys.append(original_val)
            if clean_data:
                clean_values.append(clean_data.get("value"))
                clean_units.append(clean_data.get("unit"))
            else:
                clean_values.append(None)
                clean_units.append(None)

        if not keys:
            logger.warning(
                "No concepts were successfully extracted. Returning original DataFrame."
            )
            return df_native

        # Detect backend type to create mapping DataFrame with matching native type.
        # This ensures type consistency and avoids unnecessary conversions.
        native_df_cls = type(df_native)
        module_name = native_df_cls.__module__

        if "polars" in module_name:
            import polars as pl

            map_df_native = pl.DataFrame(
                {
                    col_name: keys,
                    "clean_value": clean_values,
                    "clean_unit": clean_units,
                },
                schema={
                    col_name: pl.String,
                    "clean_value": pl.Float64,
                    "clean_unit": pl.String,
                },
            )
        elif "pandas" in module_name:
            import pandas as pd

            map_df_native = pd.DataFrame(  # type: ignore[assignment]
                {
                    col_name: keys,
                    "clean_value": clean_values,
                    "clean_unit": clean_units,
                }
            )
        else:
            # Fallback: try native constructor first
            try:
                map_df_native = native_df_cls(  # type: ignore[call-arg,assignment]
                    {
                        col_name: keys,
                        "clean_value": clean_values,
                        "clean_unit": clean_units,
                    }
                )
            except (TypeError, ValueError):
                import pandas as pd

                logger.warning(
                    f"Could not create {native_df_cls.__name__} DataFrame, "
                    "falling back to pandas. Narwhals will handle conversion."
                )
                map_df_native = pd.DataFrame(  # type: ignore[assignment]
                    {
                        col_name: keys,
                        "clean_value": clean_values,
                        "clean_unit": clean_units,
                    }
                )

        map_df = nw.from_native(map_df_native)

        try:
            # Create temporary join key by casting to String to handle type mismatches.
            # Original column might be Int/Float while mapping keys are String.
            result_df = (
                df.with_columns(
                    nw.col(col_name).cast(nw.String).alias(f"{col_name}_join_key")
                )
                .join(
                    map_df,  # type: ignore[arg-type]
                    left_on=f"{col_name}_join_key",
                    right_on=col_name,
                    how="left",
                )
                .drop(f"{col_name}_join_key")
                .to_native()
            )
            return result_df
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise
