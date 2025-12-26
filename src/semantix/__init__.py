from typing import Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from semantix._version import __version__
from semantix.engine.narwhals_ops import NarwhalsEngine
from semantix.inference.manager import LocalInferenceEngine

__all__ = ["__version__", "clean", "get_engine"]

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
    df: IntoFrameT, target_col: str, instruction: str = "Extract the numeric value and unit as-is."
) -> IntoFrameT:
    """
    Clean a column in a DataFrame using semantic extraction.
    Supports pandas, Polars, Modin, cuDF, PyArrow, and other backends via Narwhals.

    Args:
        df: Input DataFrame (pandas, Polars, Modin, cuDF, PyArrow, etc.).
        target_col: Name of the column to clean.
        instruction: Instruction to guide the LLM (e.g. 'Extract the numeric value and unit as-is.').

    Returns:
        DataFrame with added 'clean_value' and 'clean_unit' columns.
        Return type matches input type (pandas -> pandas, Polars -> Polars, etc.)
    """
    df_nw = nw.from_native(df)
    if target_col not in df_nw.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    engine = get_engine()
    return NarwhalsEngine.process_column(df, target_col, engine, instruction)
