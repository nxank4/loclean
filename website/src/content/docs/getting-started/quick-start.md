---
title: Quick Start
description: Get started with Loclean in minutes.
sidebar:
  order: 2
---

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
print(item.name)  # "t-shirt"
print(item.price)  # 50000

# Extract from DataFrame (default: structured dict for performance)
import polars as pl
df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
result = loclean.extract(df, schema=Product, target_col="description")

# Query with Polars Struct (vectorized operations)
result.filter(pl.col("description_extracted").struct.field("price") > 50000)
```

The `extract()` function ensures 100% compliance with your Pydantic schema through:

- **Dynamic GBNF Grammar Generation**: Automatically converts Pydantic schemas to GBNF grammars
- **JSON Repair**: Automatically fixes malformed JSON output from LLMs
- **Retry Logic**: Retries with adjusted prompts when validation fails

## Backend Agnostic (Zero-Copy)

Built on **Narwhals**, Loclean supports **Pandas**, **Polars**, and **PyArrow** natively.

* Running Polars? We keep it lazy.
* Running Pandas? We handle it seamlessly.
* **No heavy dependency lock-in.**
