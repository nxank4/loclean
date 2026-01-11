---
title: Quick Start
description: Get started with Loclean in minutes.
sidebar:
  order: 2
---

## Installation

```bash
pip install loclean
```

Or with uv (recommended):

```bash
uv pip install loclean
```

> **Install from PyPI:** [pypi.org/project/loclean](https://pypi.org/project/loclean) | **Full Installation Guide:** [Installation](/loclean/getting-started/installation/)

## Structured Extraction with Pydantic

Extract structured data from unstructured text with guaranteed schema compliance:

```python
from pydantic import BaseModel
import loclean

class Product(BaseModel):
    name: str
    price: int
    color: str

# Extract from text
item = loclean.extract("Selling red t-shirt for 50k", schema=Product)
print(item.name)   # "t-shirt"
print(item.price)  # 50000
print(item.color)  # "red"
```

**Output:**
```
t-shirt
50000
red
```

## Working with Tabular Data

Process entire DataFrames with automatic batch processing:

```python
import polars as pl
import loclean

# Create DataFrame with messy data
df = pl.DataFrame({
    "weight": ["5kg", "3.5 kg", "5000g", "2.2kg"]
})

# Clean the entire column
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Convert all weights to kg"
)

# View results
print(result.select(["weight", "weight_clean_value", "weight_clean_unit"]))
```

**Output:**
```
shape: (4, 3)
┌────────┬────────────────────┬───────────────────┐
│ weight ┆ weight_clean_value ┆ weight_clean_unit │
│ ---    ┆ ---                ┆ ---               │
│ str    ┆ f64                ┆ str               │
╞════════╪════════════════════╪═══════════════════╡
│ 5kg    ┆ 5.0                ┆ kg                │
│ 3.5 kg ┆ 3.5                ┆ kg                │
│ 5000g  ┆ 5.0                ┆ kg                │
│ 2.2kg  ┆ 2.2                ┆ kg                │
└────────┴────────────────────┴───────────────────┘
```

## Data Cleaning

Clean messy data in DataFrame columns:

```python
import polars as pl
import loclean

# Messy weight data
df = pl.DataFrame({
    "weight": ["5kg", "3.5 kg", "5000g", "2.2kg"]
})

# Clean the column
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract the numeric value and unit as-is."
)

# Result includes: clean_value, clean_unit, clean_reasoning
print(result.select(["weight", "weight_clean_value", "weight_clean_unit"]))
```

**Output:**
```
shape: (4, 3)
┌────────┬────────────────────┬───────────────────┐
│ weight ┆ weight_clean_value ┆ weight_clean_unit │
│ ---    ┆ ---                ┆ ---               │
│ str    ┆ f64                ┆ str               │
╞════════╪════════════════════╪═══════════════════╡
│ 5kg    ┆ 5.0                ┆ kg                │
│ 3.5 kg ┆ 3.5                ┆ kg                │
│ 5000g  ┆ 5.0                ┆ kg                │
│ 2.2kg  ┆ 2.2                ┆ kg                │
└────────┴────────────────────┴───────────────────┘
```

## Privacy Scrubbing

Scrub sensitive PII data:

```python
import loclean

# Text with PII
text = "Contact John Doe at john@example.com or 555-1234"

# Scrub PII
cleaned = loclean.scrub(text, mode="mask")
print(cleaned)
```

**Output:**
```
Contact [REDACTED] at [REDACTED] or [REDACTED]
```

## Working with DataFrames

### Pandas

```python
import pandas as pd
import loclean

df = pd.DataFrame({"description": ["Selling red t-shirt for 50k"]})
result = loclean.extract(df, schema=Product, target_col="description")
# Returns pandas DataFrame
```

### Polars

```python
import polars as pl
import loclean

df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
result = loclean.extract(df, schema=Product, target_col="description")

# Query with Polars Struct (vectorized operations)
result.filter(
    pl.col("description_extracted").struct.field("price") > 50000
)
```

## How It Works

The `extract()` function ensures 100% compliance with your Pydantic schema through:

- **Dynamic GBNF Grammar Generation**: Automatically converts Pydantic schemas to GBNF grammars
- **JSON Repair**: Automatically fixes malformed JSON output from LLMs
- **Retry Logic**: Retries with adjusted prompts when validation fails

## Backend Agnostic (Zero-Copy)

Built on **Narwhals**, Loclean supports **Pandas**, **Polars**, and **PyArrow** natively.

- Running Polars? We keep it lazy.
- Running Pandas? We handle it seamlessly.
- **No heavy dependency lock-in.**

## Interactive Demo

📓 **Try the Interactive Demo:** See [examples/demo.ipynb](https://github.com/nxank4/loclean/blob/main/examples/demo.ipynb) for a Jupyter notebook with runnable examples, including:
- Weight normalization (g → kg)
- Currency conversion (EUR → USD)
- Temperature conversion (Fahrenheit → Celsius)
- Caching demonstrations

## Next Steps

- Learn about [Data Cleaning](/loclean/getting-started/data-cleaning/) in detail
- Explore [Privacy Scrubbing](/loclean/guides/privacy/) for PII removal
- See [Structured Extraction](/loclean/guides/extraction/) for complex schemas
- Check [Use Cases](/loclean/guides/use-cases/) for real-world examples
- Review the [API Reference](/loclean/reference/api/) for complete documentation
- Understand [How It Works](/loclean/concepts/how-it-works/) architecture
