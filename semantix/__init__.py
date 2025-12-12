from typing import Optional
import polars as pl
from semantix.engine.polars_ops import PolarsEngine
from semantix.inference.manager import LocalInferenceEngine

# Global singleton instance
# Note: This singleton pattern is not thread-safe. Do not call get_engine() 
# from multiple threads simultaneously during initialization.
_ENGINE_INSTANCE: Optional[LocalInferenceEngine] = None

def get_engine() -> LocalInferenceEngine:
    """
    Get or create the global LocalInferenceEngine instance.
    
    Note: This function is not thread-safe during first initialization.
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = LocalInferenceEngine()
    return _ENGINE_INSTANCE

def clean(
    df: pl.DataFrame, 
    target_col: str, 
    instruction: str = "Extract the numeric value and unit as-is."
) -> pl.DataFrame:
    """
    Clean a column in a Polars DataFrame using semantic extraction.

    Args:
        df: Input Polars DataFrame.
        target_col: Name of the column to clean.
        instruction: Instruction to guide the LLM (e.g. 'Extract the numeric value and unit as-is.').

    Returns:
        DataFrame with added 'clean_value' and 'clean_unit' columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    engine = get_engine()
    return PolarsEngine.process_column(df, target_col, engine, instruction)
