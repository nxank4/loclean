---
title: API Reference
description: Complete API documentation for Loclean functions.
---

## loclean.clean()

Clean a column in a DataFrame using semantic extraction.

```python
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
) -> IntoFrameT
```

### Parameters

- **df** (`IntoFrameT`): Input DataFrame (pandas, Polars, Modin, cuDF, PyArrow, etc.)
- **target_col** (`str`): Name of the column to clean
- **instruction** (`str`): Instruction to guide the LLM extraction. Default: `"Extract the numeric value and unit as-is."`
- **model_name** (`Optional[str]`): Model identifier (e.g., `'phi-3-mini'`, `'tinyllama'`). Default: uses configured default
- **cache_dir** (`Optional[Path]`): Custom directory for caching models. Default: `~/.cache/loclean`
- **n_ctx** (`Optional[int]`): Context window size override
- **n_gpu_layers** (`Optional[int]`): Number of GPU layers to use (0 = CPU only)
- **batch_size** (`int`): Number of unique values to process per batch. Default: `50`
- **parallel** (`bool`): Enable parallel processing. Default: `False`
- **max_workers** (`Optional[int]`): Maximum worker threads. Default: `None` (auto-detect)
- **engine_kwargs** (`Any`): Additional arguments forwarded to `LlamaCppEngine`

### Returns

DataFrame with added columns:
- `{target_col}_clean_value`: Extracted numeric value (float)
- `{target_col}_clean_unit`: Extracted unit (string)
- `{target_col}_clean_reasoning`: LLM reasoning (string)

Return type matches input type (pandas → pandas, Polars → Polars, etc.)

### Example

```python
import loclean
import polars as pl

df = pl.DataFrame({"weight": ["5kg", "3.5 kg", "5000g"]})
result = loclean.clean(df, target_col="weight")
print(result)
```

**Output:**
```
shape: (3, 4)
┌────────┬───────────────────┬──────────────────┬──────────────────────┐
│ weight ┆ weight_clean_value ┆ weight_clean_unit ┆ weight_clean_reasoning │
│ ---    ┆ ---                ┆ ---                ┆ ---                   │
│ str    ┆ f64                ┆ str                ┆ str                   │
╞════════╪════════════════════╪════════════════════╪══════════════════════╡
│ 5kg    ┆ 5.0                ┆ kg                 ┆ Extracted numeric... │
│ 3.5 kg ┆ 3.5                ┆ kg                 ┆ Extracted numeric... │
│ 5000g  ┆ 5.0                ┆ kg                 ┆ Converted 5000g...  │
└────────┴───────────────────┴──────────────────┴──────────────────────┘
```

---

## loclean.extract()

Extract structured data from unstructured text using Pydantic schemas.

```python
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
    **engine_kwargs: Any,
) -> Any | IntoFrameT
```

### Parameters

- **input_data** (`str | IntoFrameT`): Input text or DataFrame
- **schema** (`type[Any]`): Pydantic model class defining the extraction schema
- **instruction** (`Optional[str]`): Custom instruction. Default: auto-generated from schema
- **target_col** (`Optional[str]`): DataFrame column name (required if `input_data` is DataFrame)
- **output_type** (`str`): Output format - `"dict"` or `"pydantic"`. Default: `"dict"`
- **max_retries** (`int`): Maximum retry attempts on validation failure. Default: `3`
- **model_name** (`Optional[str]`): Model identifier
- **cache_dir** (`Optional[Path]`): Custom cache directory
- **n_ctx** (`Optional[int]`): Context window size
- **n_gpu_layers** (`Optional[int]`): GPU layers
- **engine_kwargs** (`Any`): Additional engine arguments

### Returns

- If `input_data` is `str`: Returns extracted data (dict or Pydantic model)
- If `input_data` is DataFrame: Returns DataFrame with `{target_col}_extracted` column

### Example

```python
from pydantic import BaseModel
import loclean

class Product(BaseModel):
    name: str
    price: int
    color: str

# From text
item = loclean.extract("Selling red t-shirt for 50k", schema=Product)
print(item.name, item.price, item.color)

# From DataFrame
df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
result = loclean.extract(df, schema=Product, target_col="description")
print(result)
```

**Output:**
```
t-shirt 50000 red
shape: (1, 2)
┌─────────────────────────────┬──────────────────────────────────────┐
│ description                 ┆ description_extracted                 │
│ ---                         ┆ struct[3]                            │
│ str                         ┆ {name: str, price: i64, color: str}   │
╞═════════════════════════════╪══════════════════════════════════════╡
│ Selling red t-shirt for 50k ┆ {t-shirt, 50000, red}                 │
└─────────────────────────────┴──────────────────────────────────────┘
```

---

## loclean.scrub()

Scrub sensitive PII data from text or DataFrames.

```python
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
    **engine_kwargs: Any,
) -> str | IntoFrameT
```

### Parameters

- **input_data** (`str | IntoFrameT`): Input text or DataFrame
- **strategies** (`Optional[list[str]]`): List of PII types to scrub. Default: `None` (all types)
- **mode** (`str`): Scrubbing mode - `"mask"` or `"replace"`. Default: `"mask"`
- **locale** (`str`): Locale for faker generation. Default: `"vi_VN"`
- **target_col** (`Optional[str]`): DataFrame column name (required if `input_data` is DataFrame)
- **model_name** (`Optional[str]`): Model identifier
- **cache_dir** (`Optional[Path]`): Custom cache directory
- **n_ctx** (`Optional[int]`): Context window size
- **n_gpu_layers** (`Optional[int]`): GPU layers
- **engine_kwargs** (`Any`): Additional engine arguments

### Returns

- If `input_data` is `str`: Returns scrubbed string
- If `input_data` is DataFrame: Returns DataFrame with scrubbed column

### Example

```python
import loclean
import polars as pl

# Scrub text
cleaned = loclean.scrub("Contact John Doe at john@example.com")
print(cleaned)

# Scrub DataFrame
df = pl.DataFrame({"text": ["Contact John Doe at john@example.com"]})
result = loclean.scrub(df, target_col="text")
print(result)
```

**Output:**
```
Contact [REDACTED] at [REDACTED]
shape: (1, 1)
┌──────────────────────────────┐
│ text                          │
│ ---                           │
│ str                           │
╞═══════════════════════════════╡
│ Contact [REDACTED] at [REDACTED] │
└──────────────────────────────┘
```
df = pl.DataFrame({"text": ["Contact John Doe at john@example.com"]})
result = loclean.scrub(df, target_col="text")
```

---

## loclean.get_engine()

Get or create the global inference engine instance.

```python
def get_engine() -> LlamaCppEngine
```

### Returns

Singleton `LlamaCppEngine` instance with default configuration.

### Example

```python
import loclean

engine = loclean.get_engine()
# Use engine for advanced operations
```

### Notes

- First call creates and caches the engine
- Subsequent calls return the same instance
- Engine uses default model (`phi-3-mini`) and configuration

---

## Type Hints

### IntoFrameT

Type alias for supported DataFrame types:

```python
IntoFrameT = pd.DataFrame | pl.DataFrame | pl.LazyFrame | Any
```

Supports:
- `pandas.DataFrame`
- `polars.DataFrame`
- `polars.LazyFrame`
- Other Narwhals-compatible backends

## Related Topics

- [Quick Start Guide](/loclean/getting-started/quick-start/) - Basic usage examples
- [Data Cleaning](/loclean/getting-started/data-cleaning/) - Using `clean()` function
- [Privacy Scrubbing](/loclean/guides/privacy/) - Using `scrub()` function
- [Structured Extraction](/loclean/guides/extraction/) - Using `extract()` function
- [Configuration](/loclean/reference/configuration/) - Engine configuration
- [Model Management](/loclean/guides/models/) - Model selection
- [Performance Optimization](/loclean/guides/performance/) - Performance tips
