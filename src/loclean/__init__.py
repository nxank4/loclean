import warnings
from pathlib import Path
from typing import Any, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from loclean._version import __version__
from loclean.engine.narwhals_ops import NarwhalsEngine
from loclean.inference.manager import LocalInferenceEngine

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _ENGINE_INSTANCE = LocalInferenceEngine()
    return _ENGINE_INSTANCE


def clean(
    df: IntoFrameT,
    target_col: str,
    instruction: str = "Extract the numeric value and unit as-is.",
    *,
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    batch_size: int = 50,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """
    Clean a column in a DataFrame using semantic extraction.
    Supports pandas, Polars, Modin, cuDF, PyArrow, and other backends via Narwhals.

    This function provides a high-level API while still allowing advanced users
    to customize the underlying inference engine when needed.

    Args:
        df: Input DataFrame (pandas, Polars, Modin, cuDF, PyArrow, etc.).
        target_col: Name of the column to clean.
        instruction: Instruction to guide the LLM
            (e.g. 'Extract the numeric value and unit as-is.').
        model_name: Optional model identifier to override the default model
            (e.g. 'phi-3-mini', 'qwen2-1.5b'). If not provided, the default
            model configured in the engine will be used.
        cache_dir: Optional custom directory for caching model weights.
            Defaults to ~/.cache/loclean when not provided.
        n_ctx: Optional context window size override for the underlying model.
        n_gpu_layers: Optional number of GPU layers to use (0 = CPU only).
        batch_size: Number of unique values to process per batch. Defaults to 50.
        parallel: Enable parallel processing using ThreadPoolExecutor.
                 Defaults to False for backward compatibility.
        max_workers: Maximum number of worker threads for parallel processing.
                    If None, auto-detected as min(cpu_count, num_batches).
                    If 1 or parallel=False, uses sequential processing.
                    Defaults to None.
        **engine_kwargs: Additional keyword arguments forwarded to the
            underlying LocalInferenceEngine for advanced configuration.

    Returns:
        DataFrame with added 'clean_value', 'clean_unit', and 'clean_reasoning'
        columns. Return type matches input type
        (pandas -> pandas, Polars -> Polars, etc.)
    """
    df_nw = nw.from_native(df)  # type: ignore[type-var]
    if target_col not in df_nw.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    # If no engine configuration overrides are provided, reuse the global engine
    if (
        model_name is None
        and cache_dir is None
        and n_ctx is None
        and n_gpu_layers is None
        and not engine_kwargs
    ):
        engine = get_engine()
    else:
        # When users provide configuration, create a dedicated engine instance
        # so that global singleton behavior remains unchanged.
        engine = LocalInferenceEngine(
            cache_dir=cache_dir,
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            **engine_kwargs,
        )

    return NarwhalsEngine.process_column(
        df,
        target_col,
        engine,
        instruction,
        batch_size=batch_size,
        parallel=parallel,
        max_workers=max_workers,
    )
