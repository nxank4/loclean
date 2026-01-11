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

## Complex Schema Examples

### Nested Schemas

```python
from pydantic import BaseModel
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    address: Address
    phone_numbers: List[str]
    notes: Optional[str] = None

text = """
John Doe, age 35, email: john@example.com
Lives at 123 Main St, New York, NY 10001
Phones: 555-1234, 555-5678
Notes: Preferred contact method is email
"""

person = loclean.extract(text, schema=Person)
print(person.name)              # "John Doe"
print(person.address.city)      # "New York"
print(person.phone_numbers)     # ["555-1234", "555-5678"]
```

### Union Types

```python
from pydantic import BaseModel
from typing import Union

class EmailContact(BaseModel):
    type: str = "email"
    address: str

class PhoneContact(BaseModel):
    type: str = "phone"
    number: str

Contact = Union[EmailContact, PhoneContact]

class Customer(BaseModel):
    name: str
    contact: Contact

# LLM will extract the appropriate type
customer = loclean.extract(
    "Customer: John Doe, email: john@example.com",
    schema=Customer
)
```

## Error Handling

Loclean automatically retries failed extractions:

```python
# Automatic retry on validation failure
result = loclean.extract(
    text,
    schema=ComplexSchema,
    max_retries=3  # Default: 3 retries
)
```

If all retries fail, a `ValidationError` is raised with details about what went wrong.

## Performance Tips

### Batch Processing

For large datasets, extraction is automatically batched:

```python
# Process in batches of 50 (default)
result = loclean.extract(
    df,
    schema=Product,
    target_col="description"
)
```

### Parallel Processing

Enable parallel processing for faster execution:

```python
# Note: extract() doesn't support parallel directly
# Use clean() for parallel processing, or process chunks manually
```

### Caching

Results are automatically cached:

```python
# First call - processes and caches
result1 = loclean.extract(text, schema=Product)

# Second call - uses cache (instant)
result2 = loclean.extract(text, schema=Product)
```

## Benefits

- **Type Safety**: Pydantic ensures type validation
- **Schema Compliance**: GBNF grammars force valid JSON output
- **Automatic Retry**: Failed extractions are retried with adjusted prompts
- **Backend Agnostic**: Works with Pandas, Polars, and PyArrow
- **Cached**: Results are cached for performance
- **Deterministic**: Same input = same output (when cached)