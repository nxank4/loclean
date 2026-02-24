from typing import Any, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from loclean._version import __version__
from loclean.engine.narwhals_ops import NarwhalsEngine
from loclean.inference.ollama_engine import OllamaEngine

__all__ = ["__version__", "Loclean", "clean", "get_engine", "scrub", "extract"]

_ENGINE_INSTANCE: Optional[OllamaEngine] = None


def get_engine(
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> OllamaEngine:
    """Get or create the global OllamaEngine instance.

    When called without arguments, returns a singleton instance.
    When called with overrides, creates a new dedicated instance.

    Args:
        model: Ollama model tag (e.g. "phi3", "llama3").
        host: Ollama server URL.
        verbose: Enable detailed logging.

    Returns:
        OllamaEngine instance.
    """
    global _ENGINE_INSTANCE

    if model is None and host is None and verbose is None:
        if _ENGINE_INSTANCE is None:
            _ENGINE_INSTANCE = OllamaEngine()
        return _ENGINE_INSTANCE

    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if host is not None:
        kwargs["host"] = host
    if verbose is not None:
        kwargs["verbose"] = verbose
    return OllamaEngine(**kwargs)


class Loclean:
    """Primary user-facing API for structured data extraction via Ollama.

    Connects to a running Ollama instance and uses Pydantic schemas to
    enforce structured JSON output from LLMs.

    Example::

        from loclean import Loclean
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int

        cleaner = Loclean(model="phi3")
        result = cleaner.extract(
            "My name is John and I am 30 years old.",
            UserInfo,
        )
        print(result)  # UserInfo(name='John', age=30)
    """

    def __init__(
        self,
        model: str = "phi3",
        host: str = "http://localhost:11434",
        verbose: bool = False,
    ) -> None:
        """Initialize a Loclean instance.

        Args:
            model: Ollama model tag (e.g. "phi3", "llama3", "gemma2").
            host: Ollama server URL.
            verbose: Enable detailed logging.

        Raises:
            ConnectionError: If Ollama is not running at *host*.
        """
        self.engine = OllamaEngine(model=model, host=host, verbose=verbose)

    def extract(
        self,
        text: str,
        schema: type[Any],
        instruction: str | None = None,
        max_retries: int = 3,
    ) -> Any:
        """Extract structured data from text using a Pydantic schema.

        Args:
            text: Input text to extract from.
            schema: Pydantic BaseModel class defining the output structure.
            instruction: Optional custom instruction.
            max_retries: Maximum retry attempts on validation failure.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValueError: If extraction fails after max_retries.
        """
        from pydantic import BaseModel

        if not issubclass(schema, BaseModel):
            raise ValueError(
                f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
            )

        from loclean.extraction.extractor import Extractor

        extractor = Extractor(inference_engine=self.engine, max_retries=max_retries)
        return extractor.extract(text, schema, instruction)


def clean(
    df: IntoFrameT,
    target_col: str,
    instruction: str = "Extract the numeric value and unit as-is.",
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    batch_size: int = 50,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """Clean a column in a DataFrame using semantic extraction.

    Supports pandas, Polars, Modin, cuDF, PyArrow, and other backends
    via Narwhals.

    Args:
        df: Input DataFrame.
        target_col: Name of the column to clean.
        instruction: Instruction to guide the LLM.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        batch_size: Number of unique values per batch. Defaults to 50.
        parallel: Enable parallel processing.
        max_workers: Maximum worker threads for parallel processing.
        **engine_kwargs: Additional keyword arguments forwarded to
            OllamaEngine.

    Returns:
        DataFrame with added 'clean_value', 'clean_unit', and
        'clean_reasoning' columns.
    """
    df_nw = nw.from_native(df)  # type: ignore[type-var]
    if target_col not in df_nw.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    if model is None and host is None and verbose is None and not engine_kwargs:
        engine = get_engine()
    else:
        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if host is not None:
            kwargs["host"] = host
        if verbose is not None:
            kwargs["verbose"] = verbose
        kwargs.update(engine_kwargs)
        engine = OllamaEngine(**kwargs)

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
    locale: str = "en_US",
    *,
    target_col: str | None = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> str | IntoFrameT:
    """Scrub PII from text or DataFrame column.

    Detects and masks or replaces Personally Identifiable Information
    (PII) such as names, phone numbers, emails, credit cards, and
    addresses.

    Args:
        input_data: String or DataFrame to scrub.
        strategies: List of PII types to detect.
            Default: ["person", "phone", "email"].
        mode: "mask" or "fake". Default: "mask".
        locale: Faker locale for fake data generation.
        target_col: Column name for DataFrame input.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to the engine.

    Returns:
        Scrubbed string or DataFrame (same type as input).
    """
    from loclean.privacy.scrub import scrub_dataframe, scrub_string

    strategies_list = strategies or ["person", "phone", "email"]
    needs_llm = any(s in ["person", "address"] for s in strategies_list)

    inference_engine = None
    if needs_llm:
        if model is None and host is None and verbose is None and not engine_kwargs:
            inference_engine = get_engine()
        else:
            kwargs_filtered: dict[str, Any] = {}
            if model is not None:
                kwargs_filtered["model"] = model
            if host is not None:
                kwargs_filtered["host"] = host
            if verbose is not None:
                kwargs_filtered["verbose"] = verbose
            kwargs_filtered.update(engine_kwargs)
            inference_engine = OllamaEngine(**kwargs_filtered)

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
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> Any:
    """Extract structured data from text or DataFrame column.

    Uses a Pydantic schema to enforce strict JSON output through
    Ollama's structured output support.

    Args:
        input_data: String or DataFrame to extract from.
        schema: Pydantic BaseModel class defining the output structure.
        instruction: Optional custom instruction.
        target_col: Column name for DataFrame input.
        output_type: Output format for DataFrame ("dict" or "pydantic").
        max_retries: Maximum retry attempts on validation failure.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to the engine.

    Returns:
        For string input: Validated Pydantic model instance.
        For DataFrame input: DataFrame with added extraction column.
    """
    from pydantic import BaseModel

    if not issubclass(schema, BaseModel):
        raise ValueError(
            f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
        )

    from loclean.cache import LocleanCache
    from loclean.extraction.extract_dataframe import extract_dataframe
    from loclean.extraction.extractor import Extractor

    if model is None and host is None and verbose is None and not engine_kwargs:
        inference_engine = get_engine()
    else:
        kwargs_filtered: dict[str, Any] = {}
        if model is not None:
            kwargs_filtered["model"] = model
        if host is not None:
            kwargs_filtered["host"] = host
        if verbose is not None:
            kwargs_filtered["verbose"] = verbose
        kwargs_filtered.update(engine_kwargs)
        inference_engine = OllamaEngine(**kwargs_filtered)

    cache = LocleanCache()

    if isinstance(input_data, str):
        extractor = Extractor(
            inference_engine=inference_engine,
            cache=cache,
            max_retries=max_retries,
        )
        return extractor.extract(input_data, schema, instruction)
    else:
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
