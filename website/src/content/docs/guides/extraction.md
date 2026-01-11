---
title: Structured Extraction
description: Extract structured data from unstructured text with Pydantic schemas.
---

## Overview

Loclean's structured extraction feature allows you to extract complex, structured data from unstructured text with 100% schema compliance using Pydantic models and GBNF grammars.

## How It Works

1. **Define Your Schema**: Create a Pydantic model that represents the data structure you want to extract
2. **Provide Input**: Pass unstructured text or a DataFrame column
3. **Automatic Extraction**: Loclean uses LLM with GBNF grammar to extract data matching your schema
4. **Guaranteed Compliance**: If the output doesn't match your schema, it retries automatically

## Example

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
print(item.color)  # "red"
```

## Working with DataFrames

```python
import polars as pl

df = pl.DataFrame({
    "description": [
        "Selling red t-shirt for 50k",
        "Blue jeans available for 30k"
    ]
})

result = loclean.extract(df, schema=Product, target_col="description")

# Query extracted data
result.filter(
    pl.col("description_extracted").struct.field("price") > 40000
)
```

## Benefits

- **Type Safety**: Pydantic ensures type validation
- **Schema Compliance**: GBNF grammars force valid JSON output
- **Automatic Retry**: Failed extractions are retried with adjusted prompts
- **Backend Agnostic**: Works with Pandas, Polars, and PyArrow
