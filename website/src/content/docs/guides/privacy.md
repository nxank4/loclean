---
title: Privacy Scrubbing
description: Scrub sensitive PII data locally using Regex & LLMs.
---

## Overview

Loclean provides privacy-first data cleaning by scrubbing sensitive PII (Personally Identifiable Information) data locally. No data leaves your machine.

## Features

- **Local Processing**: All processing happens on your machine using local LLMs
- **Hybrid Approach**: Combines Regex patterns and LLM-based detection for accuracy
- **Zero Cost**: No cloud API calls, completely free to use
- **Privacy Guaranteed**: Your sensitive data never leaves your infrastructure

## Usage

```python
import loclean

# Scrub PII from text
cleaned = loclean.scrub("Contact John Doe at john@example.com or 555-1234")
# Returns scrubbed text with PII replaced

# Scrub from DataFrame
import polars as pl
df = pl.DataFrame({"text": ["Contact John Doe at john@example.com"]})
result = loclean.scrub(df, target_col="text")
```

## Supported PII Types

- Email addresses
- Phone numbers
- Names
- Addresses
- Credit card numbers
- Social Security Numbers (SSN)
- And more...
