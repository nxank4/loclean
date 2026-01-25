<p align="center">
  <a href="https://github.com/nxank4/loclean">
    <picture>
      <source srcset="assets/dark-loclean.svg" media="(prefers-color-scheme: dark)">
      <source srcset="assets/light-loclean.svg" media="(prefers-color-scheme: light)">
      <img src="assets/light-loclean.svg" alt="Loclean logo" width="200" height="200">
    </picture>
  </a>
</p>
<p align="center">
  <img src="assets/demo.gif" alt="Loclean Demo">
</p>
<p align="center">The All-in-One Local AI Data Cleaner.</p>

<p align="center">
  <a href="https://www.producthunt.com/products/loclean?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-loclean" target="_blank" rel="noopener noreferrer"><img alt="Loclean - The All-in-One Local AI Data Cleaning Library | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1067540&amp;theme=light&amp;t=1769264214743"></a>
</p>

<h3 align="center">
  <code>pip install loclean</code>
</h3>

<p align="center">
  <a href="https://pypi.org/project/loclean"><img src="https://img.shields.io/pypi/v/loclean?style=flat-square&color=3776ab" alt="PyPI"></a>
  <a href="https://pypi.org/project/loclean"><img src="https://img.shields.io/pypi/pyversions/loclean?style=flat-square&color=3776ab" alt="Python Versions"></a>
  <a href="https://github.com/nxank4/loclean/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/loclean?style=flat-square&color=3776ab" alt="License"></a>
  <a href="https://github.com/nxank4/loclean/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nxank4/loclean/ci.yml?style=flat-square" alt="CI Status"></a>
  <a href="https://nxank4.github.io/loclean"><img src="https://img.shields.io/badge/docs-loclean-blue?style=flat-square" alt="Documentation"></a>
</p>

# Why Loclean?

> **Documentation:** [nxank4.github.io/loclean](https://nxank4.github.io/loclean)

Loclean bridges the gap between data engineering and local AI, designed for production pipelines where privacy and stability are non-negotiable.

## Privacy first and zero cost

Leverage the power of small language models (SLMs) including **Phi-3**, **Qwen**, **Gemma**, **DeepSeek**, **TinyLlama**, and **LFM2.5** running locally via `llama.cpp`. Clean sensitive PII, medical records, or proprietary data without a single byte leaving your infrastructure. See the [available models](#available-models) section for the full list.

## Deterministic outputs

Forget about "hallucinations" or parsing loose text. Loclean uses **GBNF grammars** and **Pydantic V2** to force the LLM to output valid, type-safe JSON. If it breaks the schema, it doesn't pass.

## Structured extraction with Pydantic

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

# Extract from dataframe (default: structured dict for performance)
import polars as pl
df = pl.DataFrame({"description": ["Selling red t-shirt for 50k"]})
result = loclean.extract(df, schema=Product, target_col="description")

# Query with Polars struct (vectorized operations)
result.filter(pl.col("description_extracted").struct.field("price") > 50000)
```

The `extract()` function ensures 100% compliance with your Pydantic schema through:
- **Dynamic GBNF grammar generation**: Automatically converts Pydantic schemas to GBNF grammars
- **JSON repair**: Automatically fixes malformed JSON output from LLMs
- **Retry logic**: Retries with adjusted prompts when validation fails

Loclean also provides `clean()` for general data cleaning and `scrub()` for privacy-preserving PII redaction. Explore the [examples](examples/) and [documentation](https://nxank4.github.io/loclean) to discover more features.

## Backend agnostic (zero copy)

Built on **Narwhals**, Loclean supports **Pandas**, **Polars**, **PyArrow**, **Modin**, **cuDF**, and other backends natively. The library automatically detects your dataframe backend and uses the most efficient operations for each.

* Running Polars? We keep it lazy.
* Running Pandas? We handle it seamlessly.
* **No heavy dependency lock-in.**

For advanced usage patterns, caching strategies, batch processing, parallel execution, and performance optimization tips, check out the [documentation](https://nxank4.github.io/loclean).

# Installation

## Requirements

* Python 3.10, 3.11, 3.12, 3.13, 3.14, or 3.15
* No GPU required (runs on CPU by default)

## Basic installation

**Using pip (recommended):**

```bash
pip install loclean
```

The basic installation includes local inference support (via `llama-cpp-python`).

> **Installation notice:**
> - **Fast (30-60 seconds):** Pre-built wheels are available for most platforms (Linux x86_64, macOS, Windows)
> - **Slow (5-10 minutes):** If you see "Building wheels for collected packages: llama-cpp-python", it's building from source. This is **normal** and only happens when no pre-built wheel is available for your platform. Please be patient - this is not an error!
>
> **To ensure fast installation:**
> ```bash
> pip install --upgrade pip setuptools wheel
> pip install loclean
> ```
> This ensures pip can find and use pre-built wheels when available.

**Using uv (alternative, often faster):**

```bash
uv pip install loclean
```

**Using conda/mamba:**

```bash
conda install -c conda-forge loclean
# or
mamba install -c conda-forge loclean
```

## Optional dependencies

The basic installation includes local inference support. Loclean uses **Narwhals** for backend-agnostic dataframe operations, so if you already have **Pandas**, **Polars**, or **PyArrow** installed, the basic installation is sufficient.

**Install dataframe libraries (if not already present):**

If you don't have any dataframe library installed, or want to ensure you have all supported backends:

```bash
pip install loclean[data]
```

This installs: `pandas>=2.3.3`, `polars>=0.20.0`, `pyarrow>=22.0.0`

**For cloud API support (OpenAI, Anthropic, Gemini):**

Cloud API support is planned for future releases. Currently, only local inference is available:

```bash
pip install loclean[cloud]
```

**For privacy features (Faker integration):**

```bash
pip install loclean[privacy]
```

This installs: `faker>=20.0.0` for fake data generation in privacy scrubbing.

**Install all optional dependencies:**

```bash
pip install loclean[all]
```

This installs `loclean[data]`, `loclean[cloud]`, and `loclean[privacy]`. Useful for production environments where you want all features available.

> **Note for developers:** If you're contributing to Loclean, use the [Development installation](#development-installation) section below (git clone + `uv sync --dev`), not `loclean[all]`.

## Development installation

To contribute or run tests locally:

```bash
# Clone the repository
git clone https://github.com/nxank4/loclean.git
cd loclean

# Install with development dependencies (using uv)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

# Model management

Loclean automatically downloads models on first use, but you can pre-download them using the command line:

```bash
# Download a specific model
loclean model download --name phi-3-mini

# List available models
loclean model list

# Check download status
loclean model status
```

## Available models

- **phi-3-mini**: Microsoft Phi-3 Mini (3.8B, 4K context) - Default, balanced
- **tinyllama**: TinyLlama 1.1B - Smallest, fastest
- **gemma-2b**: Google Gemma 2B Instruct - Balanced performance
- **qwen3-4b**: Qwen3 4B - Higher quality
- **gemma-3-4b**: Gemma 3 4B - Larger context
- **deepseek-r1**: DeepSeek R1 - Reasoning model
- **lfm2.5**: Liquid LFM2.5-1.2B Instruct (1.17B, 32K context) - Best-in-class 1B scale, optimized for agentic tasks and data extraction

Models are cached in `~/.cache/loclean` by default. You can specify a custom cache directory using the `--cache-dir` option.

# Quick start

Loclean is best learned by example. We provide a set of Jupyter notebooks to help you get started:

- **[01-quick-start.ipynb](examples/01-quick-start.ipynb)**: Core features, structured extraction, and privacy scrubbing.
- **[02-data-cleaning.ipynb](examples/02-data-cleaning.ipynb)**: Comprehensive data cleaning strategies.
- **[03-privacy-scrubbing.ipynb](examples/03-privacy-scrubbing.ipynb)**: Deep dive into PII redaction.
- **[04-structured-extraction.ipynb](examples/04-structured-extraction.ipynb)**: Advanced structured extraction patterns.
- **[05-debug-mode.ipynb](examples/05-debug-mode.ipynb)**: Debugging and verbose mode usage.

Check out the **[examples/](examples/)** directory for more details.

# Contributing

We love contributions! Loclean is strictly open-source under the **Apache 2.0 License**.

Please read our **[contributing guide](CONTRIBUTION.md)** for details on how to set up your development environment, run tests, and submit pull requests.

_Built for the data community._

[![Star History Chart](https://api.star-history.com/svg?repos=nxank4/loclean&type=date&legend=top-left)](https://www.star-history.com/#nxank4/loclean&type=date&legend=top-left)
