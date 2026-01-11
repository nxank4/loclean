---
title: Use Cases
description: Real-world scenarios and examples using Loclean.
sidebar:
  order: 1
---

## E-commerce Product Data Cleaning

Clean inconsistent product data from multiple sources:

```python
import loclean
import polars as pl

# Product data from different vendors
df = pl.DataFrame({
    "product": [
        "Red T-Shirt - Size M - $25.99",
        "Blue Jeans, Size 32, Price: 49.99 USD",
        "Sneakers - $80 - Size 10"
    ]
})

# Extract structured product information
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    size: str
    price: float
    currency: str

result = loclean.extract(
    df,
    schema=Product,
    target_col="product"
)

# Now you can query and analyze clean data
expensive_products = result.filter(
    pl.col("product_extracted").struct.field("price") > 50
)
```

## Medical Records Anonymization

Scrub sensitive PII from medical records before analysis:

```python
import loclean
import polars as pl

# Medical records with PII
df = pl.DataFrame({
    "record": [
        "Patient John Doe, DOB: 01/15/1980, SSN: 123-45-6789, diagnosed with condition X",
        "Mary Smith, born 1985-03-20, phone 555-1234, condition Y",
        "Dr. Johnson treated patient at 123 Main St, NYC on 2024-01-10"
    ]
})

# Scrub all PII
anonymized = loclean.scrub(
    df,
    target_col="record",
    mode="mask"  # Replace with [REDACTED] or similar
)

# Safe for analysis without privacy concerns
print(anonymized)
```

## Financial Data Extraction

Extract structured financial data from unstructured text:

```python
import loclean
from pydantic import BaseModel

class Transaction(BaseModel):
    amount: float
    currency: str
    date: str
    description: str
    category: str

# Bank statement text
transactions = [
    "Paid $50.00 to Amazon on 2024-01-15 for Electronics",
    "Received 100 EUR from Client ABC on Jan 20, 2024 - Invoice payment",
    "Withdrawal: $200 cash on 2024-01-18"
]

df = pl.DataFrame({"text": transactions})

result = loclean.extract(
    df,
    schema=Transaction,
    target_col="text"
)

# Analyze spending patterns
total_spending = result.filter(
    pl.col("text_extracted").struct.field("category") == "Expense"
).select(
    pl.col("text_extracted").struct.field("amount").sum()
)
```

## Customer Support Ticket Classification

Extract and structure information from support tickets:

```python
import loclean
from pydantic import BaseModel

class SupportTicket(BaseModel):
    issue_type: str
    priority: str
    customer_id: str
    description: str
    requested_action: str

tickets = [
    "URGENT: Customer #12345 needs refund for order #67890 - payment issue",
    "Low priority: User #98765 asking about shipping status for order #11111",
    "HIGH: Customer #55555 reports login problem - cannot access account"
]

df = pl.DataFrame({"ticket": tickets})

result = loclean.extract(
    df,
    schema=SupportTicket,
    target_col="ticket"
)

# Route tickets by priority
urgent_tickets = result.filter(
    pl.col("ticket_extracted").struct.field("priority").str.to_uppercase() == "URGENT"
)
```

## Scientific Data Normalization

Normalize scientific measurements from various formats:

```python
import loclean

# Measurements in different formats
df = pl.DataFrame({
    "measurement": [
        "25.5°C",
        "298.15 K",
        "77.9°F",
        "Temperature: 20 degrees Celsius"
    ]
})

# Normalize to standard format
result = loclean.clean(
    df,
    target_col="measurement",
    instruction="Extract temperature value in Celsius and convert if needed"
)

# All temperatures now in consistent format
print(result.select(["measurement", "clean_value", "clean_unit"]))
```

## Address Standardization

Standardize addresses from various formats:

```python
import loclean
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

addresses = [
    "123 Main St, New York, NY 10001, USA",
    "456 Oak Avenue, Los Angeles, California 90001",
    "789 Pine Road, Chicago IL 60601 United States"
]

df = pl.DataFrame({"address": addresses})

result = loclean.extract(
    df,
    schema=Address,
    target_col="address"
)

# Standardized addresses ready for geocoding or analysis
print(result)
```

## Recipe Ingredient Extraction

Extract structured ingredient data from recipe text:

```python
import loclean
from pydantic import BaseModel
from typing import List

class Ingredient(BaseModel):
    name: str
    amount: float
    unit: str

class Recipe(BaseModel):
    name: str
    ingredients: List[Ingredient]
    servings: int

recipe_text = """
Chocolate Cake Recipe
Serves 8
Ingredients:
- 2 cups flour
- 1.5 cups sugar
- 3 eggs
- 1 cup milk
- 0.5 cup butter
"""

result = loclean.extract(
    recipe_text,
    schema=Recipe
)

print(result.name)
print(result.ingredients)
```

## Related Topics

- [Quick Start Guide](/loclean/getting-started/quick-start/) - Basic usage examples
- [Data Cleaning](/loclean/getting-started/data-cleaning/) - Clean and normalize data
- [Privacy Scrubbing](/loclean/guides/privacy/) - Remove PII data
- [Structured Extraction](/loclean/guides/extraction/) - Extract complex data
- [Performance Optimization](/loclean/guides/performance/) - Tips for faster processing
- [API Reference](/loclean/reference/api/) - Complete function documentation
