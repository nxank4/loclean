---
title: Configuration
description: Configure Loclean engines, models, and caching.
---

## EngineConfig

The `EngineConfig` class provides hierarchical configuration management for inference engines.

### Configuration Priority

1. **Runtime Parameters** (highest priority)
2. **Environment Variables** (`LOCLEAN_*`)
3. **Project Config** (`[tool.loclean]` in `pyproject.toml`)
4. **Defaults** (lowest priority)

### Configuration Fields

```python
from loclean.inference.config import EngineConfig

config = EngineConfig(
    engine="llama-cpp",           # Inference engine: "llama-cpp" (openai, anthropic, gemini coming soon)
    model="phi-3-mini",           # Model identifier
    api_key=None,                 # API key for cloud providers (not used yet - cloud APIs coming soon)
    cache_dir=Path("~/.cache/loclean"),  # Cache directory
    n_ctx=4096,                   # Context window size (512-32768)
    n_gpu_layers=0                # GPU layers (0 = CPU only)
)
```

> **Note:** Cloud API support (OpenAI, Anthropic, Gemini) is planned for future releases. Currently, only `llama-cpp` engine is available. See [Installation](/loclean/getting-started/installation/) for details.

### Environment Variables

Set configuration via environment variables:

```bash
export LOCLEAN_MODEL="tinyllama"
export LOCLEAN_N_CTX=8192
export LOCLEAN_CACHE_DIR="/custom/cache/path"
```

### Project Configuration

Configure in `pyproject.toml`:

```toml
[tool.loclean]
model = "phi-3-mini"
n_ctx = 4096
n_gpu_layers = 0
cache_dir = "~/.cache/loclean"
```

## Model Selection

### Available Models

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| `tinyllama` | 800 MB | 2K | Speed, simple tasks |
| `phi-3-mini` | 2.4 GB | 4K | Balance (default) |
| `gemma-2b` | 1.5 GB | 8K | Balanced performance |
| `qwen3-4b` | 2.5 GB | 8K | Quality, complex schemas |
| `gemma-3-4b` | 2.5 GB | 8K | Larger context needs |
| `deepseek-r1` | 3.0 GB | 16K | Reasoning tasks |

### Choosing a Model

**For Speed:**
```python
result = loclean.clean(df, target_col="data", model_name="tinyllama")
```

**For Balance (Default):**
```python
result = loclean.clean(df, target_col="data")  # Uses phi-3-mini
```

**For Quality:**
```python
result = loclean.extract(df, schema=ComplexSchema, model_name="qwen3-4b")
```

**For Reasoning:**
```python
result = loclean.extract(df, schema=ReasoningSchema, model_name="deepseek-r1")
```

## Context Window

The `n_ctx` parameter controls the maximum context window size:

```python
# Small context (faster, less memory)
result = loclean.clean(df, target_col="data", n_ctx=2048)

# Large context (slower, more memory, handles longer text)
result = loclean.extract(df, schema=Schema, n_ctx=8192)
```

**Recommendations:**
- Simple extractions: 2048-4096
- Complex schemas: 4096-8192
- Very long text: 8192-16384

## GPU Configuration

Enable GPU acceleration:

```python
# Use GPU for first 20 layers
result = loclean.clean(
    df,
    target_col="data",
    n_gpu_layers=20
)
```

**Requirements:**
- CUDA-compatible GPU
- `llama-cpp-python` compiled with CUDA support

**Benefits:**
- 2-5x faster inference
- Better for large batches

## Cache Configuration

### Default Cache Location

Models and inference results are cached in `~/.cache/loclean` by default.

### Custom Cache Directory

```python
from pathlib import Path

result = loclean.clean(
    df,
    target_col="data",
    cache_dir=Path("/fast/ssd/cache")
)
```

### Cache Management

```python
from loclean.cache import LocleanCache

cache = LocleanCache(cache_dir=Path("/custom/cache"))

# Clear all cached results
cache.clear()

# Get cache statistics
stats = cache.get_stats()
```

## Advanced Configuration

### Custom Engine Instance

For advanced use cases, create a custom engine:

```python
from loclean.inference.local.llama_cpp import LlamaCppEngine

engine = LlamaCppEngine(
    model_name="qwen3-4b",
    cache_dir=Path("/custom/cache"),
    n_ctx=8192,
    n_gpu_layers=20
)

# Use with NarwhalsEngine directly
from loclean.engine.narwhals_ops import NarwhalsEngine

result = NarwhalsEngine.process_column(
    df,
    "data",
    engine,
    "Extract value"
)
```

### Factory Pattern

Use the factory to create engines:

```python
from loclean.inference.factory import create_engine
from loclean.inference.config import EngineConfig

config = EngineConfig(
    model="tinyllama",
    n_ctx=2048
)

engine = create_engine(config)
```

## Best Practices

1. **Use defaults for most cases**: Default configuration works well
2. **Override per-call**: Pass parameters to individual function calls
3. **Environment variables for deployment**: Set `LOCLEAN_*` in production
4. **Project config for teams**: Use `pyproject.toml` for shared settings
5. **Monitor cache usage**: Large caches can consume disk space

## Related Topics

- [Installation Guide](/loclean/getting-started/installation/) - Setup instructions
- [Model Management](/loclean/guides/models/) - Download and manage models
- [API Reference](/loclean/reference/api/) - Function parameters
- [Performance Optimization](/loclean/guides/performance/) - Configuration tips
