from typing import Any, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

from loclean._version import __version__
from loclean.engine.narwhals_ops import NarwhalsEngine
from loclean.inference.ollama_engine import OllamaEngine

__all__ = [
    "__version__",
    "Loclean",
    "audit_leakage",
    "clean",
    "discover_features",
    "extract",
    "extract_compiled",
    "get_engine",
    "optimize_instruction",
    "oversample",
    "prune_traps",
    "recognize_missingness",
    "resolve_entities",
    "scrub",
    "shred_to_relations",
    "validate_quality",
]

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


def _resolve_engine(
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> OllamaEngine:
    """Return a shared or custom OllamaEngine instance.

    When all arguments are ``None`` and there are no extra kwargs,
    returns the process-wide singleton.  Otherwise builds a fresh
    client with the supplied overrides.

    Args:
        model: Ollama model tag.
        host: Ollama server URL.
        verbose: Enable detailed logging.
        **engine_kwargs: Forwarded to ``OllamaEngine``.

    Returns:
        OllamaEngine instance.
    """
    if model is None and host is None and verbose is None and not engine_kwargs:
        return get_engine()

    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if host is not None:
        kwargs["host"] = host
    if verbose is not None:
        kwargs["verbose"] = verbose
    kwargs.update(engine_kwargs)
    return OllamaEngine(**kwargs)


class Loclean:
    """Primary user-facing API for structured data extraction via Ollama.

    Connects to a running Ollama instance and uses Pydantic schemas to
    enforce structured JSON output from LLMs.  A single ``OllamaEngine``
    instance is shared across every wrapper method, preventing redundant
    network sockets and reducing memory overhead.

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

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Advanced capabilities
    # ------------------------------------------------------------------

    def clean(
        self,
        df: IntoFrameT,
        target_col: str,
        instruction: str = "Extract the numeric value and unit as-is.",
        *,
        batch_size: int = 50,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> IntoFrameT:
        """Clean a column in a DataFrame using semantic extraction.

        Args:
            df: Input DataFrame.
            target_col: Column to clean.
            instruction: Instruction to guide the LLM.
            batch_size: Unique values per batch.
            parallel: Enable parallel processing.
            max_workers: Worker threads for parallel processing.

        Returns:
            DataFrame with added cleaning columns.
        """
        return NarwhalsEngine.process_column(
            df,
            target_col,
            self.engine,
            instruction,
            batch_size=batch_size,
            parallel=parallel,
            max_workers=max_workers,
        )

    def resolve_entities(
        self,
        df: IntoFrameT,
        target_col: str,
        *,
        threshold: float = 0.8,
    ) -> IntoFrameT:
        """Canonicalize a messy string column via entity resolution.

        Args:
            df: Input DataFrame.
            target_col: Column with messy string values.
            threshold: Semantic-distance threshold ε in ``(0, 1]``.

        Returns:
            DataFrame with an added ``{target_col}_canonical`` column.
        """
        from loclean.extraction.resolver import EntityResolver

        resolver = EntityResolver(inference_engine=self.engine, threshold=threshold)
        return resolver.resolve(df, target_col)

    def oversample(
        self,
        df: IntoFrameT,
        target_col: str,
        target_value: Any,
        n: int,
        schema: type,
        *,
        batch_size: int = 10,
    ) -> IntoFrameT:
        """Generate synthetic minority-class records.

        Args:
            df: Input DataFrame.
            target_col: Column identifying the class label.
            target_value: Minority class value.
            n: Number of synthetic records.
            schema: Pydantic model defining record structure.
            batch_size: Records per LLM batch.

        Returns:
            DataFrame with synthetic records appended.
        """
        from loclean.extraction.oversampler import SemanticOversampler

        sampler = SemanticOversampler(
            inference_engine=self.engine, batch_size=batch_size
        )
        return sampler.oversample(df, target_col, target_value, n, schema)

    def shred_to_relations(
        self,
        df: IntoFrameT,
        target_col: str,
        *,
        sample_size: int = 30,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Shred a log column into relational DataFrames.

        Args:
            df: Input DataFrame.
            target_col: Column with unstructured log text.
            sample_size: Entries to sample for inference.
            max_retries: Repair budget for code generation.

        Returns:
            Dict mapping table names to native DataFrames.
        """
        from loclean.extraction.shredder import RelationalShredder

        shredder = RelationalShredder(
            inference_engine=self.engine,
            sample_size=sample_size,
            max_retries=max_retries,
        )
        return shredder.shred(df, target_col)

    def discover_features(
        self,
        df: IntoFrameT,
        target_col: str,
        *,
        n_features: int = 5,
        max_retries: int = 3,
    ) -> IntoFrameT:
        """Discover and apply feature crosses.

        Args:
            df: Input DataFrame.
            target_col: Target variable column.
            n_features: Number of features to propose.
            max_retries: Repair budget for code generation.

        Returns:
            DataFrame augmented with new feature columns.
        """
        from loclean.extraction.feature_discovery import FeatureDiscovery

        discoverer = FeatureDiscovery(
            inference_engine=self.engine,
            n_features=n_features,
            max_retries=max_retries,
        )
        return discoverer.discover(df, target_col)

    def prune_traps(
        self,
        df: IntoFrameT,
        target_col: str,
        *,
        correlation_threshold: float = 0.05,
        max_retries: int = 2,
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Identify and remove trap features.

        Trap features are columns of uncorrelated Gaussian noise
        that masquerade as valid signals.

        Args:
            df: Input DataFrame.
            target_col: Target variable column.
            correlation_threshold: Absolute correlation below which
                a column is considered uncorrelated.
            max_retries: LLM retry budget.

        Returns:
            Tuple of (pruned DataFrame, summary dict).
        """
        from loclean.extraction.trap_pruner import TrapPruner

        pruner = TrapPruner(
            inference_engine=self.engine,
            correlation_threshold=correlation_threshold,
            max_retries=max_retries,
        )
        return pruner.prune(df, target_col)

    def recognize_missingness(
        self,
        df: IntoFrameT,
        target_cols: list[str] | None = None,
        *,
        sample_size: int = 50,
        max_retries: int = 3,
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Detect MNAR patterns and encode as boolean features.

        Args:
            df: Input DataFrame.
            target_cols: Columns to analyse (default: all with nulls).
            sample_size: Max null rows to sample per column.
            max_retries: LLM retry budget.

        Returns:
            Tuple of (augmented DataFrame, summary dict).
        """
        from loclean.extraction.missingness_recognizer import MissingnessRecognizer

        recognizer = MissingnessRecognizer(
            inference_engine=self.engine,
            sample_size=sample_size,
            max_retries=max_retries,
        )
        return recognizer.recognize(df, target_cols)

    def audit_leakage(
        self,
        df: IntoFrameT,
        target_col: str,
        domain: str = "",
        *,
        max_retries: int = 2,
        sample_n: int = 10,
    ) -> tuple[IntoFrameT, dict[str, Any]]:
        """Detect and remove target-leaking features.

        Args:
            df: Input DataFrame.
            target_col: Target variable column.
            domain: Dataset domain description.
            max_retries: LLM retry budget.
            sample_n: Sample rows for the prompt.

        Returns:
            Tuple of (pruned DataFrame, summary dict).
        """
        from loclean.extraction.leakage_auditor import TargetLeakageAuditor

        auditor = TargetLeakageAuditor(
            inference_engine=self.engine,
            max_retries=max_retries,
            sample_n=sample_n,
        )
        return auditor.audit(df, target_col, domain)

    def validate_quality(
        self,
        df: IntoFrameT,
        rules: list[str],
        *,
        batch_size: int = 20,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Evaluate data quality against natural-language rules.

        Args:
            df: Input DataFrame.
            rules: Natural-language constraint strings.
            batch_size: Rows per processing batch.
            sample_size: Maximum rows to evaluate.

        Returns:
            Dict with compliance report.
        """
        from loclean.validation.quality_gate import QualityGate

        gate = QualityGate(
            inference_engine=self.engine,
            batch_size=batch_size,
            sample_size=sample_size,
        )
        return gate.evaluate(df, rules)


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

    engine = _resolve_engine(model, host, verbose, **engine_kwargs)

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
        inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

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

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

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


def extract_compiled(
    df: IntoFrameT,
    target_col: str,
    schema: type[Any],
    instruction: str | None = None,
    *,
    max_retries: int = 3,
    sample_size: int = 50,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """Extract structured data from a DataFrame column using generative compilation.

    Synthesises a pure-Python extraction function via the local Ollama engine,
    verifies it against a diverse sample, then maps it natively across the
    column — no per-row LLM calls required.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column to extract from.
        schema: Pydantic BaseModel class defining the output structure.
        instruction: Optional domain hint forwarded to code generation.
        max_retries: Repair budget for the verification loop.
        sample_size: Number of diverse rows to sample (default 50).
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        DataFrame with an added ``{target_col}_extracted`` column.
    """
    from pydantic import BaseModel as _BM

    if not issubclass(schema, _BM):
        raise ValueError(
            f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
        )

    from loclean.extraction.extract_dataframe import extract_dataframe_compiled

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    return extract_dataframe_compiled(
        df,
        target_col,
        schema,
        instruction,
        inference_engine=inference_engine,
        max_retries=max_retries,
        sample_size=sample_size,
    )


def optimize_instruction(
    df: IntoFrameT,
    target_col: str,
    schema: type[Any],
    baseline_instruction: str | None = None,
    *,
    sample_size: int = 20,
    max_retries: int = 3,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> str:
    """Optimize an extraction instruction using a reward-driven feedback loop.

    Generates structural prompt variations via the local Ollama engine,
    evaluates each against a validation sample from *target_col*, and
    returns the instruction that achieves the highest field-level F1 score.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column containing the text to extract from.
        schema: Pydantic BaseModel class defining the target structure.
        baseline_instruction: Starting instruction. When ``None`` a
            default is built from *schema*.
        sample_size: Number of validation rows to sample.
        max_retries: Retry budget for extraction.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        The instruction string with the highest reward.
    """
    from pydantic import BaseModel as _BM

    if not issubclass(schema, _BM):
        raise ValueError(
            f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
        )

    from loclean.extraction.optimizer import InstructionOptimizer

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    optimizer = InstructionOptimizer(
        inference_engine=inference_engine,
        max_retries=max_retries,
    )
    return optimizer.optimize(
        df,
        target_col,
        schema,
        baseline_instruction=baseline_instruction,
        sample_size=sample_size,
    )


def resolve_entities(
    df: IntoFrameT,
    target_col: str,
    *,
    threshold: float = 0.8,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """Canonicalize a messy string column via semantic entity resolution.

    Groups similar string variations under a single authoritative label
    using the local Ollama engine.  A new ``{target_col}_canonical``
    column is appended to the returned DataFrame.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column containing messy string values.
        threshold: Semantic-distance threshold ε in ``(0, 1]``.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        DataFrame with an added ``{target_col}_canonical`` column.
    """
    from loclean.extraction.resolver import EntityResolver

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    resolver = EntityResolver(
        inference_engine=inference_engine,
        threshold=threshold,
    )
    return resolver.resolve(df, target_col)


def validate_quality(
    df: IntoFrameT,
    rules: list[str],
    *,
    batch_size: int = 20,
    sample_size: int = 100,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate data quality against natural-language rules.

    Checks sampled rows for compliance and returns a structured
    report with compliance rate and per-failure reasoning.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        rules: Natural-language constraint strings.
        batch_size: Rows per processing batch.
        sample_size: Maximum rows to evaluate.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        Dictionary with ``total_rows``, ``passed_rows``,
        ``compliance_rate``, and ``failures``.
    """
    from loclean.validation.quality_gate import QualityGate

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    gate = QualityGate(
        inference_engine=inference_engine,
        batch_size=batch_size,
        sample_size=sample_size,
    )
    return gate.evaluate(df, rules)


def oversample(
    df: IntoFrameT,
    target_col: str,
    target_value: Any,
    n: int,
    schema: type,
    *,
    batch_size: int = 10,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """Generate synthetic minority-class records and append them.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column identifying the class label.
        target_value: Value of the minority class to oversample.
        n: Number of synthetic records to generate.
        schema: Pydantic model defining the record structure.
        batch_size: Records per LLM generation batch.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        DataFrame with synthetic records appended.
    """
    from loclean.extraction.oversampler import SemanticOversampler

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    sampler = SemanticOversampler(
        inference_engine=inference_engine,
        batch_size=batch_size,
    )
    return sampler.oversample(df, target_col, target_value, n, schema)


def shred_to_relations(
    df: IntoFrameT,
    target_col: str,
    *,
    sample_size: int = 30,
    max_retries: int = 3,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Shred an unstructured log column into relational DataFrames.

    Uses the Ollama engine to infer a relational schema, generate
    a parsing function, and separate the column into multiple tables.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column containing unstructured log text.
        sample_size: Number of entries to sample for inference.
        max_retries: Repair budget for code generation.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        Dictionary mapping table names to native DataFrames.
    """
    from loclean.extraction.shredder import RelationalShredder

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    shredder = RelationalShredder(
        inference_engine=inference_engine,
        sample_size=sample_size,
        max_retries=max_retries,
    )
    return shredder.shred(df, target_col)


def discover_features(
    df: IntoFrameT,
    target_col: str,
    *,
    n_features: int = 5,
    max_retries: int = 3,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> IntoFrameT:
    """Discover and apply feature crosses to a DataFrame.

    Uses the Ollama engine to propose mathematical transformations
    that maximise mutual information with the target variable.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column name of the target variable.
        n_features: Number of new features to propose.
        max_retries: Repair budget for code generation.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        DataFrame augmented with new feature columns.
    """
    from loclean.extraction.feature_discovery import FeatureDiscovery

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    discoverer = FeatureDiscovery(
        inference_engine=inference_engine,
        n_features=n_features,
        max_retries=max_retries,
    )
    return discoverer.discover(df, target_col)


def prune_traps(
    df: IntoFrameT,
    target_col: str,
    *,
    correlation_threshold: float = 0.05,
    max_retries: int = 2,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> tuple[IntoFrameT, dict[str, Any]]:
    """Identify and remove trap features from a DataFrame.

    Trap features are columns of uncorrelated Gaussian noise that
    masquerade as valid signals.  Detection relies on statistical
    distributions and target correlations — column names are ignored.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column name of the prediction target.
        correlation_threshold: Absolute correlation threshold.
        max_retries: LLM retry budget.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        Tuple of ``(pruned_df, summary)`` where *summary* contains
        ``dropped_columns`` and ``verdicts``.
    """
    from loclean.extraction.trap_pruner import TrapPruner

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    pruner = TrapPruner(
        inference_engine=inference_engine,
        correlation_threshold=correlation_threshold,
        max_retries=max_retries,
    )
    return pruner.prune(df, target_col)


def recognize_missingness(
    df: IntoFrameT,
    target_cols: list[str] | None = None,
    *,
    sample_size: int = 50,
    max_retries: int = 3,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> tuple[IntoFrameT, dict[str, Any]]:
    """Detect MNAR patterns and encode as boolean feature flags.

    Identifies Missing Not At Random patterns where the probability
    of a value being missing depends on other feature values.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_cols: Columns to analyse (default: all with nulls).
        sample_size: Max null rows to sample per column.
        max_retries: LLM retry budget.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        Tuple of ``(augmented_df, summary)`` where *summary* maps
        each analysed column to its pattern description.
    """
    from loclean.extraction.missingness_recognizer import MissingnessRecognizer

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    recognizer = MissingnessRecognizer(
        inference_engine=inference_engine,
        sample_size=sample_size,
        max_retries=max_retries,
    )
    return recognizer.recognize(df, target_cols)


def audit_leakage(
    df: IntoFrameT,
    target_col: str,
    domain: str = "",
    *,
    max_retries: int = 2,
    sample_n: int = 10,
    model: Optional[str] = None,
    host: Optional[str] = None,
    verbose: Optional[bool] = None,
    **engine_kwargs: Any,
) -> tuple[IntoFrameT, dict[str, Any]]:
    """Detect and remove target-leaking features.

    Identifies features that contain information generated after the
    target event, where P(Y | X_i) ≈ 1.  Uses semantic timeline
    evaluation via the LLM.

    Args:
        df: Input DataFrame (pandas, Polars, etc.).
        target_col: Column name of the prediction target.
        domain: Brief dataset domain description.
        max_retries: LLM retry budget.
        sample_n: Sample rows for the prompt.
        model: Optional Ollama model tag override.
        host: Optional Ollama server URL override.
        verbose: Enable detailed logging.
        **engine_kwargs: Additional arguments forwarded to OllamaEngine.

    Returns:
        Tuple of ``(pruned_df, summary)`` with ``dropped_columns``
        and ``verdicts``.
    """
    from loclean.extraction.leakage_auditor import TargetLeakageAuditor

    inference_engine = _resolve_engine(model, host, verbose, **engine_kwargs)

    auditor = TargetLeakageAuditor(
        inference_engine=inference_engine,
        max_retries=max_retries,
        sample_n=sample_n,
    )
    return auditor.audit(df, target_col, domain)
