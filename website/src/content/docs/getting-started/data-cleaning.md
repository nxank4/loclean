---
title: Data Cleaning
description: Clean and normalize data using semantic extraction with clean() function.
sidebar:
  order: 3
---

## Overview

The `clean()` function is Loclean's core data cleaning API. It uses semantic extraction to clean and normalize messy data in DataFrame columns, making it perfect for handling inconsistent formats, units, and representations.

> **See also:** [Quick Start](/loclean/getting-started/quick-start/) | [Structured Extraction](/loclean/guides/extraction/) | [API Reference](/loclean/reference/api/)

## Basic Usage

```python
import loclean
import polars as pl

# Create a DataFrame with messy data
df = pl.DataFrame({
    "weight": ["5kg", "3.5 kg", "5000g", "2.2kg"]
})

# Clean the weight column
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract the numeric value and unit as-is."
)

# Result includes new columns:
# - clean_value: The extracted numeric value
# - clean_unit: The extracted unit
# - clean_reasoning: LLM's reasoning for the extraction
print(result)
```

**Output:**
```
shape: (4, 4)
┌────────┬────────────────────┬───────────────────┬────────────────────────┐
│ weight ┆ weight_clean_value ┆ weight_clean_unit ┆ weight_clean_reasoning │
│ ---    ┆ ---                ┆ ---               ┆ ---                    │
│ str    ┆ f64                ┆ str               ┆ str                    │
╞════════╪════════════════════╪═══════════════════╪════════════════════════╡
│ 5kg    ┆ 5.0                ┆ kg                ┆ Extracted numeric...   │
│ 3.5 kg ┆ 3.5                ┆ kg                ┆ Extracted numeric...   │
│ 5000g  ┆ 5.0                ┆ kg                ┆ Converted 5000g...     │
│ 2.2kg  ┆ 2.2                ┆ kg                ┆ Extracted numeric...   │
└────────┴────────────────────┴───────────────────┴────────────────────────┘
```

## Understanding the Output

The `clean()` function adds three new columns to your DataFrame:

- **`clean_value`**: The extracted numeric value (as float)
- **`clean_unit`**: The extracted unit (as string)
- **`clean_reasoning`**: The LLM's reasoning process (as string)

## Custom Instructions

You can provide custom instructions to guide the extraction:

```python
# Extract price with currency
df = pl.DataFrame({
    "price": ["$50", "50 USD", "€45", "100 dollars"]
})

result = loclean.clean(
    df,
    target_col="price",
    instruction="Extract the numeric value and currency code (USD, EUR, etc.)"
)
print(result.select(["price", "price_clean_value", "price_clean_unit"]))
```

**Output:**
```
shape: (4, 3)
┌─────────────┬───────────────────┬──────────────────┐
│ price       ┆ price_clean_value ┆ price_clean_unit │
│ ---         ┆ ---               ┆ ---              │
│ str         ┆ f64               ┆ str              │
╞═════════════╪═══════════════════╪══════════════════╡
│ $50         ┆ 50.0              ┆ USD              │
│ 50 USD      ┆ 50.0              ┆ USD              │
│ €45         ┆ 45.0              ┆ EUR              │
│ 100 dollars ┆ 100.0             ┆ USD              │
└─────────────┴───────────────────┴──────────────────┘
```

## Working with Different Backends

### Pandas

```python
import pandas as pd
import loclean

df = pd.DataFrame({
    "temperature": ["25°C", "77F", "298K"]
})

result = loclean.clean(
    df,
    target_col="temperature",
    instruction="Extract temperature value and unit"
)

# Result is a pandas DataFrame
print(type(result))  # <class 'pandas.core.frame.DataFrame'>
```

### Polars

```python
import polars as pl
import loclean

df = pl.DataFrame({
    "distance": ["5km", "3 miles", "1000m"]
})

result = loclean.clean(
    df,
    target_col="distance",
    instruction="Extract distance value and unit"
)

# Result is a Polars DataFrame (lazy evaluation preserved)
print(type(result))  # <class 'polars.dataframe.frame.DataFrame'>
```

## Batch Processing

For large datasets, `clean()` automatically processes data in batches:

```python
# Process 100 items per batch (default: 50)
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight value and unit",
    batch_size=100
)
```

## Parallel Processing

Enable parallel processing for faster execution on large datasets:

```python
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight value and unit",
    parallel=True,  # Enable parallel processing
    max_workers=4   # Use 4 worker threads
)
```

## Model Selection

You can specify a different model for cleaning:

```python
# Use a faster model for simple extractions
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight value and unit",
    model_name="tinyllama"  # Faster, smaller model
)
```

## Advanced Configuration

For advanced use cases, you can configure the underlying engine:

```python
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight value and unit",
    n_ctx=8192,        # Larger context window
    n_gpu_layers=10,   # Use GPU acceleration
    cache_dir="/custom/cache/path"
)
```

## Best Practices

1. **Start with simple instructions**: Begin with basic instructions and refine as needed
2. **Use appropriate batch sizes**: Larger batches = faster but more memory usage
3. **Enable parallel for large datasets**: Use `parallel=True` for datasets with >1000 rows
4. **Choose the right model**: Use `tinyllama` for speed, `phi-3-mini` for balance, larger models for complex extractions
5. **Cache results**: Results are automatically cached to avoid redundant processing

## Common Patterns

### Extracting Multiple Fields

```python
# For complex extractions, use extract() instead
from pydantic import BaseModel
import loclean

class Measurement(BaseModel):
    value: float
    unit: str
    type: str  # "weight", "length", "temperature", etc.

result = loclean.extract(
    df,
    schema=Measurement,
    target_col="measurement"
)
```

### Handling Missing Values

```python
# clean() handles missing values gracefully
df = pl.DataFrame({
    "weight": ["5kg", None, "3kg", ""]
})

result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight value and unit"
)
```

**Note:** Missing values result in `None` for `clean_value`, `clean_unit`, and `clean_reasoning`.

## Related Topics

- [Quick Start Guide](/loclean/getting-started/quick-start/) - Basic usage examples
- [Structured Extraction](/loclean/guides/extraction/) - Extract complex data with Pydantic schemas
- [Performance Optimization](/loclean/guides/performance/) - Tips for faster processing
- [API Reference](/loclean/reference/api/) - Complete `clean()` function documentation
- [Use Cases](/loclean/guides/use-cases/) - Real-world examples
