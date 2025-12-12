import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import polars as pl
from tqdm import tqdm

if TYPE_CHECKING:
    from semantix.inference.manager import LocalInferenceEngine

logger = logging.getLogger(__name__)

class PolarsEngine:
    """
    Polars-based engine for efficient semantic data cleaning.
    """

    @staticmethod
    def process_column(
        df: pl.DataFrame, 
        col_name: str, 
        inference_engine: "LocalInferenceEngine",
        instruction: str,
        batch_size: int = 50
    ) -> pl.DataFrame:
        """
        Clean a specific column using the inference engine with batching and progress tracking.
        """
        # Input validation for col_name
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        
        # 1. Extract unique values
        # We explicitly cast to String to ensure consistency with the AI's input
        unique_series = df.select(pl.col(col_name).cast(pl.String)).unique().to_series()
        
        # Explicitly type the list of uniques
        uniques: List[str] = [
            str(x) for x in unique_series.to_list() 
            if x is not None and str(x).strip() != ""
        ]

        logger.info(f"Found {len(uniques)} unique patterns to clean in '{col_name}'.")

        if not uniques:
            logger.warning("No valid unique values found. Returning original DataFrame.")
            return df

        # 2. Batch Processing with Progress Bar
        # Type the results dictionary: Key is original string, Value is the JSON dict from AI
        mapping_results: Dict[str, Optional[Dict[str, Any]]] = {}
        
        # Type the chunks explicitly
        chunks: List[List[str]] = [
            uniques[i:i + batch_size] 
            for i in range(0, len(uniques), batch_size)
        ]

        # Use instruction in log message if helpful, or just generic
        print(f"🧠 Semantic Cleaning: Processing {len(uniques)} unique patterns...")
        
        for chunk in tqdm(chunks, desc="Inference Batches", unit="batch"):
            # clean_batch returns Dict[str, Dict[str, Any]]
            batch_result: Dict[str, Optional[Dict[str, Any]]] = inference_engine.clean_batch(chunk, instruction=instruction)
            mapping_results.update(batch_result)

        # 3. Construct Result Columns
        keys: List[str] = []
        clean_values: List[Optional[float]] = []
        clean_units: List[Optional[str]] = []

        for original_val, clean_data in mapping_results.items():
            keys.append(original_val)
            if clean_data:
                # Safely get values using .get() which returns Optional types
                clean_values.append(clean_data.get("value"))  # type: ignore
                clean_units.append(clean_data.get("unit"))    # type: ignore
            else:
                clean_values.append(None)
                clean_units.append(None)

        if not keys:
            logger.warning("No concepts were successfully extracted. Returning original DataFrame.")
            return df

        # 4. Create Mapping DataFrame
        map_df = pl.DataFrame({
            col_name: keys,
            "clean_value": clean_values,
            "clean_unit": clean_units
        }, schema={
            col_name: pl.String, 
            "clean_value": pl.Float64,
            "clean_unit": pl.String
        })

        # 5. Perform Left Join
        try:
            # Create a temporary join key to handle type mismatches (e.g. Int vs String)
            result_df = df.with_columns(
                pl.col(col_name).cast(pl.String).alias(f"{col_name}_join_key")
            ).join(
                map_df, 
                left_on=f"{col_name}_join_key", 
                right_on=col_name, 
                how="left"
            ).drop(f"{col_name}_join_key")
            
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise

        return result_df