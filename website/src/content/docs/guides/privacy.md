---
title: Privacy Scrubbing
description: Scrub sensitive PII data locally using Regex & LLMs.
---

## Overview

Loclean provides privacy-first data cleaning by scrubbing sensitive PII (Personally Identifiable Information) data locally. No data leaves your machine.

> **See also:** [Quick Start](/loclean/getting-started/quick-start/) | [Structured Extraction](/loclean/guides/extraction/) | [API Reference](/loclean/reference/api/)

## Features

- **Local Processing**: All processing happens on your machine using local LLMs
- **Hybrid Approach**: Combines Regex patterns and LLM-based detection for accuracy
- **Zero Cost**: No cloud API calls, completely free to use
- **Privacy Guaranteed**: Your sensitive data never leaves your infrastructure

## Basic Usage

### Scrub Text

```python
import loclean

# Text with PII
text = "Contact John Doe at john@example.com or call 555-1234"

# Scrub all PII (default: mask mode)
cleaned = loclean.scrub(text)
print(cleaned)
```

**Output:**
```
Contact [REDACTED] at [REDACTED] or call [REDACTED]
```

### Scrub DataFrame

```python
import polars as pl
import loclean

df = pl.DataFrame({
    "text": [
        "Contact John Doe at john@example.com",
        "Call Mary Smith at 555-9876",
        "Email: admin@company.com"
    ]
})

# Scrub PII in DataFrame column
result = loclean.scrub(df, target_col="text")
print(result["text"])
```

**Output:**
```
shape: (3,)
Series: 'text' [str]
[
	"Contact [REDACTED] at [REDACTED]"
	"Call [REDACTED] at [REDACTED]"
	"Email: [REDACTED]"
]
```

## Scrubbing Modes

### Mask Mode (Default)

Replaces PII with `[REDACTED]` or similar placeholders:

```python
cleaned = loclean.scrub(
    "John Doe: john@example.com",
    mode="mask"
)
print(cleaned)
```

**Output:**
```
[REDACTED]: [REDACTED]
```

### Replace Mode

Replaces PII with realistic fake data:

```python
cleaned = loclean.scrub(
    "John Doe: john@example.com",
    mode="replace"
)
print(cleaned)
```

**Output:**
```
Jane Smith: jane.smith@example.net
```

## Selective Scrubbing

Scrub only specific PII types:

```python
# Only scrub emails and phone numbers
cleaned = loclean.scrub(
    "John Doe: john@example.com, 555-1234",
    strategies=["email", "phone"]
)
```

Available strategies:
- `email` - Email addresses
- `phone` - Phone numbers
- `name` - Person names
- `address` - Physical addresses
- `ssn` - Social Security Numbers
- `credit_card` - Credit card numbers
- `ip_address` - IP addresses

## Before/After Examples

### Example 1: Medical Records

**Before:**
```
Patient John Doe, DOB: 01/15/1980, SSN: 123-45-6789, 
diagnosed with condition X. Contact: john@hospital.com
```

**After (Mask):**
```
Patient [REDACTED], DOB: [REDACTED], SSN: [REDACTED], 
diagnosed with condition X. Contact: [REDACTED]
```

**After (Replace):**
```
Patient Jane Smith, DOB: 05/20/1985, SSN: 987-65-4321, 
diagnosed with condition X. Contact: jane.smith@hospital.com
```

### Example 2: Customer Data

**Before:**
```
Customer: Mary Johnson
Email: mary.j@company.com
Phone: (555) 123-4567
Address: 123 Main St, New York, NY 10001
```

**After:**
```
Customer: [REDACTED]
Email: [REDACTED]
Phone: [REDACTED]
Address: [REDACTED]
```

## Locale Support

Generate fake data in different locales:

```python
# Vietnamese locale
cleaned = loclean.scrub(
    "Liên hệ Nguyễn Văn A tại nguyenvana@example.com",
    locale="vi_VN"
)

# English locale (default)
cleaned = loclean.scrub(
    "Contact John Doe at john@example.com",
    locale="en_US"
)
```

## Configuration Options

```python
result = loclean.scrub(
    df,
    target_col="text",
    strategies=["email", "phone"],  # Specific PII types
    mode="mask",                     # or "replace"
    locale="en_US",                  # Locale for fake data
    model_name="phi-3-mini"          # Model selection
)
```

## Supported PII Types

- **Email addresses**: `user@example.com`
- **Phone numbers**: `555-1234`, `+1-555-123-4567`
- **Names**: `John Doe`, `Mary Smith`
- **Addresses**: `123 Main St, New York, NY 10001`
- **Credit card numbers**: `1234-5678-9012-3456`
- **Social Security Numbers**: `123-45-6789`
- **IP addresses**: `192.168.1.1`
- **Dates of birth**: `01/15/1980`
- And more...

## Best Practices

1. **Choose the right mode**: Use `mask` for anonymization, `replace` for realistic test data
2. **Selective scrubbing**: Only scrub what you need to reduce processing time
3. **Batch processing**: Process large datasets in batches for better performance
4. **Verify results**: Always review scrubbed data to ensure completeness
5. **Compliance**: Ensure scrubbing meets your regulatory requirements (HIPAA, GDPR, etc.)

## Privacy Guarantee

- ✅ All processing happens locally
- ✅ No data sent to external APIs
- ✅ Models run on your machine
- ✅ Complete data sovereignty
- ✅ Zero cloud costs

## Related Topics

- [Quick Start Guide](/loclean/getting-started/quick-start/) - Basic usage examples
- [Data Cleaning](/loclean/getting-started/data-cleaning/) - Clean and normalize data
- [Structured Extraction](/loclean/guides/extraction/) - Extract structured data
- [Performance Optimization](/loclean/guides/performance/) - Tips for faster processing
- [API Reference](/loclean/reference/api/) - Complete `scrub()` function documentation
- [Use Cases](/loclean/guides/use-cases/) - Real-world privacy scenarios
