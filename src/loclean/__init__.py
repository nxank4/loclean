from pathlib import Path
from typing import Any, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from loclean._version import __version__
from loclean.engine.narwhals_ops import NarwhalsEngine
from loclean.inference.local.llama_cpp import LlamaCppEngine

__all__ = ["__version__", "clean", "get_engine", "scrub", "extract"]

# Global singleton instance
# Note: This singleton pattern is not thread-safe. Do not call get_engine()
# from multiple threads simultaneously during initialization.
_ENGINE_INSTANCE: Optional[LlamaCppEngine] = None


def get_engine() -> LlamaCppEngine:
    """
    Get or create the global LlamaCppEngine instance.

    Note: This function is not thread-safe during first initialization.
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = LlamaCppEngine()
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
    verbose: Optional[bool] = None,
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
        verbose: Enable detailed logging of prompts and outputs.
        batch_size: Number of unique values to process per batch. Defaults to 50.
        parallel: Enable parallel processing using ThreadPoolExecutor.
                 Defaults to False for backward compatibility.
        max_workers: Maximum number of worker threads for parallel processing.
                    If None, auto-detected as min(cpu_count, num_batches).
                    If 1 or parallel=False, uses sequential processing.
                    Defaults to None.
        **engine_kwargs: Additional keyword arguments forwarded to the
            underlying LlamaCppEngine for advanced configuration.

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
        and verbose is None
        and not engine_kwargs
    ):
        engine = get_engine()
    else:
        # When users provide configuration, create a dedicated engine instance
        # so that global singleton behavior remains unchanged.
        engine_kwargs_filtered: dict[str, Any] = {}
        if model_name is not None:
            engine_kwargs_filtered["model_name"] = model_name
        if cache_dir is not None:
            engine_kwargs_filtered["cache_dir"] = cache_dir
        if n_ctx is not None:
            engine_kwargs_filtered["n_ctx"] = n_ctx
        if n_gpu_layers is not None:
            engine_kwargs_filtered["n_gpu_layers"] = n_gpu_layers
        if verbose is not None:
            engine_kwargs_filtered["verbose"] = verbose
        engine_kwargs_filtered.update(engine_kwargs)
        engine = LlamaCppEngine(**engine_kwargs_filtered)

    return NarwhalsEngine.process_column(
        df,
        target_col,
        engine,
        instruction,
        batch_size=batch_size,
        parallel=parallel,
        max_workers=max_workers,
    )


def scrub(
    input_data: str | IntoFrameT,
    strategies: list[str] | None = None,
    mode: str = "mask",
    locale: str = "vi_VN",
    *,
    target_col: str | None = None,
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> str | IntoFrameT:
    """
    Scrub PII from text or DataFrame column.

    This function detects and masks or replaces Personally Identifiable Information
    (PII) such as names, phone numbers, emails, credit cards, and addresses.

    Uses a hybrid approach:
    - Fast regex detection for structured PII (email, phone, credit_card, ip_address)
    - LLM-based detection for semantic PII (person names, addresses)

    Args:
        input_data: String or DataFrame to scrub
        strategies: List of PII types to detect.
                   Default: ["person", "phone", "email"]
                   Options: "person", "phone", "email", "credit_card",
                            "address", "ip_address"
        mode: "mask" (replace with [TYPE]) or "fake" (replace with fake data).
              Default: "mask"
        locale: Faker locale for fake data generation (default: "vi_VN").
                Only used when mode="fake"
        target_col: Column name for DataFrame input (required for DataFrame)
        model_name: Optional model identifier for LLM detection.
                   If None, uses default model from get_engine()
        cache_dir: Optional custom directory for caching models and results
        n_ctx: Optional context window size override
        n_gpu_layers: Optional number of GPU layers to use (0 = CPU only)
        verbose: Enable detailed logging of prompts and outputs.
        **engine_kwargs: Additional arguments forwarded to inference engine

    Returns:
        Scrubbed string or DataFrame (same type as input)

    Examples:
        >>> import loclean
        >>> text = "Contact John at 555-1234"
        >>> loclean.scrub(text, strategies=["person", "phone"])
        'Contact [PERSON] at [PHONE]'

        >>> loclean.scrub(
        ...     text, strategies=["person", "phone"], mode="fake", locale="en_US"
        ... )
        'Contact Jane Smith at 555-5678'
    """
    from loclean.privacy.scrub import scrub_dataframe, scrub_string

    # Get inference engine if needed (for LLM strategies)
    strategies_list = strategies or ["person", "phone", "email"]
    needs_llm = any(s in ["person", "address"] for s in strategies_list)

    inference_engine = None
    if needs_llm:
        if (
            model_name is None
            and cache_dir is None
            and n_ctx is None
            and n_gpu_layers is None
            and verbose is None
            and not engine_kwargs
        ):
            inference_engine = get_engine()
        else:
            engine_kwargs_filtered: dict[str, Any] = {}
            if model_name is not None:
                engine_kwargs_filtered["model_name"] = model_name
            if cache_dir is not None:
                engine_kwargs_filtered["cache_dir"] = cache_dir
            if n_ctx is not None:
                engine_kwargs_filtered["n_ctx"] = n_ctx
            if n_gpu_layers is not None:
                engine_kwargs_filtered["n_gpu_layers"] = n_gpu_layers
            if verbose is not None:
                engine_kwargs_filtered["verbose"] = verbose
            engine_kwargs_filtered.update(engine_kwargs)
            inference_engine = LlamaCppEngine(**engine_kwargs_filtered)

    if isinstance(input_data, str):
        return scrub_string(
            input_data,
            strategies_list,
            mode,
            locale,
            inference_engine=inference_engine,
        )
    else:
        if target_col is None:
            raise ValueError("target_col required for DataFrame input")
        return scrub_dataframe(
            input_data,
            target_col,
            strategies_list,
            mode,
            locale,
            inference_engine=inference_engine,
        )


def extract(
    input_data: str | IntoFrameT,
    schema: type[Any],
    instruction: str | None = None,
    *,
    target_col: str | None = None,
    output_type: str = "dict",
    max_retries: int = 3,
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> Any:
    """
    Extract structured data from text or DataFrame column using Pydantic schema.

    Ensures 100% compliance with the provided Pydantic model through:
    - Dynamic GBNF grammar generation from Pydantic schemas
    - JSON repair for malformed outputs
    - Retry logic with prompt adjustment on validation failures

    Args:
        input_data: String or DataFrame to extract from
        schema: Pydantic BaseModel class defining the output structure
        instruction: Optional custom instruction. If None, auto-generated from schema
        target_col: Column name for DataFrame input (required for DataFrame)
        output_type: Output format for DataFrame ("dict" or "pydantic"). Default: "dict"
                   - "dict": Structured data (Polars Struct / Pandas dict) for optimal
                             performance and vectorized operations
                   - "pydantic": Pydantic model instances (slower, breaks vectorization)
        max_retries: Maximum retry attempts on validation failure (default: 3)
        model_name: Optional model identifier override
        cache_dir: Optional custom cache directory
        n_ctx: Optional context window size override
        n_gpu_layers: Optional number of GPU layers to use (0 = CPU only)
        verbose: Enable detailed logging of prompts and outputs.
        **engine_kwargs: Additional arguments forwarded to inference engine

    Returns:
        For string input: Validated Pydantic model instance
        For DataFrame input: DataFrame with added column `{target_col}_extracted`

    Raises:
        ValueError: If target_col is not provided for DataFrame input or
            schema is invalid
        ValidationError: If extraction fails after max_retries attempts

    Examples:
        >>> from pydantic import BaseModel
        >>> import loclean
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: int
        ...     color: str
        >>> # Extract from text
        >>> item = loclean.extract("Selling red t-shirt for 50k", schema=Product)
        >>> print(item.name, item.price, item.color)
        't-shirt' 50000 'red'
        >>> # Extract from DataFrame (default: structured dict for performance)
        >>> import polars as pl
        >>> df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
        >>> result = loclean.extract(df, schema=Product, target_col="description")
        >>> # Query with Polars Struct
        >>> result.filter(pl.col("description_extracted").struct.field("price") > 50000)
    """
    from pydantic import BaseModel

    if not issubclass(schema, BaseModel):
        raise ValueError(
            f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
        )

    from loclean.cache import LocleanCache
    from loclean.extraction.extract_dataframe import extract_dataframe
    from loclean.extraction.extractor import Extractor

    # Get or create inference engine
    if (
        model_name is None
        and cache_dir is None
        and n_ctx is None
        and n_gpu_layers is None
        and verbose is None
        and not engine_kwargs
    ):
        inference_engine = get_engine()
        cache = inference_engine.cache if hasattr(inference_engine, "cache") else None
    else:
        engine_kwargs_filtered: dict[str, Any] = {}
        if model_name is not None:
            engine_kwargs_filtered["model_name"] = model_name
        if cache_dir is not None:
            engine_kwargs_filtered["cache_dir"] = cache_dir
        if n_ctx is not None:
            engine_kwargs_filtered["n_ctx"] = n_ctx
        if n_gpu_layers is not None:
            engine_kwargs_filtered["n_gpu_layers"] = n_gpu_layers
        if verbose is not None:
            engine_kwargs_filtered["verbose"] = verbose
        engine_kwargs_filtered.update(engine_kwargs)
        inference_engine = LlamaCppEngine(**engine_kwargs_filtered)
        cache = inference_engine.cache if hasattr(inference_engine, "cache") else None
        if cache is None and cache_dir:
            cache = LocleanCache(cache_dir=cache_dir)

    if isinstance(input_data, str):
        # String extraction
        extractor = Extractor(
            inference_engine=inference_engine, cache=cache, max_retries=max_retries
        )
        return extractor.extract(input_data, schema, instruction)
    else:
        # DataFrame extraction
        if target_col is None:
            raise ValueError("target_col required for DataFrame input")
        return extract_dataframe(
            input_data,
            target_col,
            schema,
            instruction,
            output_type=output_type,  # type: ignore[arg-type]
            inference_engine=inference_engine,
            cache=cache,
            max_retries=max_retries,
        )
