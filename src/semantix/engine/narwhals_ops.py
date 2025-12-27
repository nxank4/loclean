import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT
from tqdm import tqdm

if TYPE_CHECKING:
    from semantix.inference.manager import LocalInferenceEngine

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
        """
        # Wrap native DataFrame into Narwhals DataFrame
        df = nw.from_native(df_native)  # type: ignore[type-var]

        # Input validation for col_name
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")

        # 1. Extract unique values using fluent API
        # We explicitly cast to String to ensure consistency with the AI's input
        unique_df_native = (
            df.select(nw.col(col_name).cast(nw.String)).unique().to_native()
        )

        # Extract column values - handle different backends
        # Polars: unique_df_native[col_name].to_list()
        # pandas: unique_df_native[col_name].tolist()
        # PyArrow: convert to pandas first
        if hasattr(unique_df_native, "to_pandas"):
            # PyArrow Table
            unique_df_native = unique_df_native.to_pandas()
            col_values = unique_df_native[col_name].tolist()
        elif hasattr(unique_df_native, "to_series"):
            # Some backends have to_series()
            col_values = unique_df_native[col_name].to_series().to_list()  # type: ignore[index]
        else:
            # Polars or other backends
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

        # 2. Batch Processing with Progress Bar
        # Type the results dictionary: Key is original string,
        # Value is the JSON dict from AI
        mapping_results: Dict[str, Optional[Dict[str, Any]]] = {}

        # Type the chunks explicitly
        chunks: List[List[str]] = [
            uniques[i : i + batch_size] for i in range(0, len(uniques), batch_size)
        ]

        # Use instruction in log message if helpful, or just generic
        print(f"🧠 Semantic Cleaning: Processing {len(uniques)} unique patterns...")

        for chunk in tqdm(chunks, desc="Inference Batches", unit="batch"):
            # clean_batch returns Dict[str, Dict[str, Any]]
            batch_result: Dict[str, Optional[Dict[str, Any]]] = (
                inference_engine.clean_batch(chunk, instruction=instruction)
            )
            mapping_results.update(batch_result)

        # 3. Construct Result Columns
        keys: List[str] = []
        clean_values: List[Optional[float]] = []
        clean_units: List[Optional[str]] = []

        for original_val, clean_data in mapping_results.items():
            keys.append(original_val)
            if clean_data:
                # Safely get values using .get() which returns Optional types
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

        # 4. Create Mapping DataFrame using the same native backend as the input
        native_df_cls = type(df_native)
        try:
            map_df_native = native_df_cls(  # type: ignore[call-arg]
                {
                    col_name: keys,
                    "clean_value": clean_values,
                    "clean_unit": clean_units,
                }
            )
        except TypeError:
            # Fallback: if constructor signature is different,
            # try pandas-style DataFrame
            import pandas as pd

            logger.warning("Fallback to pandas-style DataFrame")
            map_df_native = pd.DataFrame(
                {col_name: keys, "clean_value": clean_values, "clean_unit": clean_units}
            )

        map_df = nw.from_native(map_df_native)

        # 5. Perform Left Join using fluent API
        try:
            # Create a temporary join key to handle type mismatches (e.g. Int vs String)
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
