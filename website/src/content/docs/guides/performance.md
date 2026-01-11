---
title: Performance Optimization
description: Tips and best practices for optimizing Loclean performance.
sidebar:
  order: 2
---

## Overview

Loclean is designed for performance, but following these best practices will help you get the most out of it, especially when processing large datasets.

## Batch Processing

The `clean()` and `extract()` functions automatically process data in batches. Adjust batch size based on your data:

```python
# Small batches for complex extractions (more accurate)
result = loclean.extract(
    df,
    schema=ComplexSchema,
    target_col="text",
    batch_size=25  # Smaller batches for complex schemas
)

# Large batches for simple extractions (faster)
result = loclean.clean(
    df,
    target_col="weight",
    instruction="Extract weight",
    batch_size=100  # Larger batches for simple tasks
)
```

**Recommendations:**
- Simple extractions: 50-100 items per batch
- Complex schemas: 25-50 items per batch
- Very large datasets: 100-200 items per batch

## Parallel Processing

Enable parallel processing for datasets with >1000 rows:

```python
result = loclean.clean(
    df,
    target_col="data",
    instruction="Extract value",
    parallel=True,      # Enable parallel processing
    max_workers=None    # Auto-detect optimal worker count
)
```

**When to use:**
- ✅ Large datasets (>1000 rows)
- ✅ Simple extraction tasks
- ✅ Multiple CPU cores available
- ❌ Small datasets (<100 rows) - overhead not worth it
- ❌ Very complex schemas - may cause memory issues

## Model Selection

Choose the right model for your use case:

### For Speed

```python
# Use tinyllama for fastest processing
result = loclean.clean(
    df,
    target_col="data",
    model_name="tinyllama"  # Smallest, fastest model
)
```

**Best for:** Simple extractions, large datasets, real-time processing

### For Balance

```python
# Use phi-3-mini for balanced performance (default)
result = loclean.clean(
    df,
    target_col="data",
    model_name="phi-3-mini"  # Default, good balance
)
```

**Best for:** Most use cases, moderate complexity

### For Quality

```python
# Use larger models for complex extractions
result = loclean.extract(
    df,
    schema=ComplexSchema,
    model_name="qwen3-4b"  # Higher quality, slower
)
```

**Best for:** Complex schemas, high accuracy requirements

## Caching Strategy

Loclean automatically caches inference results. Understanding how caching works:

### Cache Key Generation

Results are cached based on:
- Instruction/schema
- Input text
- Model name

```python
# First call - processes and caches
result1 = loclean.clean(df, target_col="data", instruction="Extract value")

# Second call with same data - uses cache (instant)
result2 = loclean.clean(df, target_col="data", instruction="Extract value")
```

### Cache Location

By default, cache is stored in `~/.cache/loclean`. You can customize:

```python
result = loclean.clean(
    df,
    target_col="data",
    cache_dir=Path("/fast/ssd/cache")  # Use faster storage
)
```

### Cache Management

```python
from loclean.cache import LocleanCache

cache = LocleanCache()
cache.clear()  # Clear all cached results
```

## Memory Optimization

### Process in Chunks

For very large datasets, process in chunks:

```python
import polars as pl

# Process 10,000 rows at a time
chunk_size = 10000
results = []

for i in range(0, len(df), chunk_size):
    chunk = df.slice(i, chunk_size)
    result = loclean.clean(chunk, target_col="data")
    results.append(result)

# Combine results
final_result = pl.concat(results)
```

### Use Lazy Evaluation (Polars)

Polars lazy evaluation is preserved:

```python
import polars as pl

df = pl.scan_csv("large_file.csv")  # Lazy DataFrame

# Clean operation
result = loclean.clean(
    df,
    target_col="data",
    instruction="Extract value"
)

# Still lazy - no computation yet
filtered = result.filter(pl.col("clean_value") > 100)

# Compute only when needed
final = filtered.collect()
```

## GPU Acceleration

If you have a GPU available:

```python
result = loclean.clean(
    df,
    target_col="data",
    instruction="Extract value",
    n_gpu_layers=20  # Use GPU for first 20 layers
)
```

**Benefits:**
- 2-5x faster inference
- Better for large batches

**Requirements:**
- CUDA-compatible GPU
- `llama-cpp-python` compiled with CUDA support

## Best Practices Summary

1. **Start with defaults**: Default settings work well for most cases
2. **Profile first**: Measure performance before optimizing
3. **Use appropriate batch sizes**: Balance speed vs memory
4. **Enable parallel for large datasets**: >1000 rows
5. **Choose the right model**: Speed vs quality trade-off
6. **Leverage caching**: Reuse results when possible
7. **Process in chunks**: For datasets that don't fit in memory
8. **Use GPU if available**: Significant speedup for large batches

## Performance Benchmarks

Typical performance on a modern CPU (8 cores):

- **Simple extraction** (tinyllama): ~100 items/second
- **Complex extraction** (phi-3-mini): ~20 items/second
- **With parallel processing**: 3-5x speedup
- **With GPU**: 2-5x additional speedup

*Note: Actual performance depends on hardware, model size, and data complexity.*

## Related Topics

- [Data Cleaning](/loclean/getting-started/data-cleaning/) - Clean and normalize data
- [Structured Extraction](/loclean/guides/extraction/) - Extract complex data
- [Model Management](/loclean/guides/models/) - Choose the right model
- [Configuration](/loclean/reference/configuration/) - Optimize engine settings
- [API Reference](/loclean/reference/api/) - Function parameters for performance tuning
