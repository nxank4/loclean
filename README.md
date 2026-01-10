<p align="center">
  <a href="https://github.com/nxank4/loclean">
    <picture>
      <source srcset="src/assets/dark-loclean.svg" media="(prefers-color-scheme: dark)">
      <source srcset="src/assets/light-loclean.svg" media="(prefers-color-scheme: light)">
      <img src="src/assets/light-loclean.svg" alt="Loclean logo" width="200" height="200">
    </picture>
  </a>
</p>
<p align="center">The All-in-One Local AI Data Cleaner.</p>
<p align="center">
  <a href="https://pypi.org/project/loclean"><img src="https://img.shields.io/pypi/v/loclean?color=blue&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/loclean"><img src="https://img.shields.io/pypi/pyversions/loclean?style=flat-square" alt="Python Versions"></a>
  <a href="https://github.com/nxank4/loclean/actions/workflows/ci.yml"><img src="https://github.com/nxank4/loclean/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/nxank4/loclean/blob/main/LICENSE"><img src="https://img.shields.io/github/license/nxank4/loclean?style=flat-square" alt="License"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
</p>

# Why Loclean?

Loclean bridges the gap between **Data Engineering** and **Local AI**, designed for production pipelines where privacy and stability are non-negotiable.

## Privacy-First & Zero Cost

Leverage the power of Small Language Models (SLMs) like **Phi-3** and **Llama-3** running locally via `llama.cpp`. Clean sensitive PII, medical records, or proprietary data without a single byte leaving your infrastructure.

## Deterministic Outputs

Forget about "hallucinations" or parsing loose text. Loclean uses **GBNF Grammars** and **Pydantic V2** to force the LLM to output valid, type-safe JSON. If it breaks the schema, it doesn't pass.

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
item = loclean.extract("Bán cái áo thun đỏ giá 50k", schema=Product)
print(item.name)  # "áo thun"
print(item.price)  # 50000

# Extract from DataFrame (default: structured dict for performance)
import polars as pl
df = pl.DataFrame({"description": ["Bán áo thun đỏ giá 50k"]})
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

# Installation

## Requirements

* Python 3.10, 3.11, 3.12, or 3.13
* No GPU required (runs on CPU by default)

## Basic Installation

**Using pip:**

```bash
pip install loclean
```

**Using uv (recommended for faster installs):**

```bash
uv pip install loclean
```

**Using conda/mamba:**

```bash
conda install -c conda-forge loclean
# or
mamba install -c conda-forge loclean
```

## Optional Dependencies

The basic installation includes **local inference** support (via `llama-cpp-python`). Loclean uses **Narwhals** for backend-agnostic DataFrame operations, so if you already have **Pandas**, **Polars**, or **PyArrow** installed, the basic installation is sufficient.

**Install DataFrame libraries (if not already present):**

If you don't have any DataFrame library installed, or want to ensure you have all supported backends:

```bash
pip install loclean[data]
```

This installs: `pandas>=2.3.3`, `polars>=0.20.0`, `pyarrow>=22.0.0`

**For Cloud API support (OpenAI, Anthropic, Gemini):**

Cloud API support is planned for future releases. Currently, only local inference is available:

```bash
pip install loclean[cloud]
```

**Install all optional dependencies:**

```bash
pip install loclean[all]
```

This installs both `loclean[data]` and `loclean[cloud]`. Useful for production environments where you want all features available.

> **Note for developers:** If you're contributing to Loclean, use the [Development Installation](#development-installation) section below (git clone + `uv sync --dev`), not `loclean[all]`.

## Development Installation

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

# Quick Start

_in progress..._

# How It Works

_in progress..._

# Roadmap

The development of Loclean is organized into three phases, prioritizing MVP delivery while maintaining a long-term vision.

## Phase 1: The "Smart" Engine (Hybrid Core)

**Goal: Get `loclean.clean()` running fast and accurately.**

* [ ] **Hybrid Router Architecture**: Build `clean(strategy='auto')` function. Automatically run Regex first, LLM second.
* [ ] **Strict Output (Pydantic + GBNF)**: Ensure 100% LLM outputs valid JSON Schema. (Using llama-cpp-python grammar).
* [ ] **Simple Extraction**: Extract basic information from raw text (Unstructured to Structured).

## Phase 2: The "Safe" Layer (Security & Optimization)

**Goal: Convince enterprises to trust and adopt the library.**

* [ ] **Semantic PII Redaction**: Masking sensitive names, phone numbers, emails, and addresses.
* [ ] **SQLite Caching System**: Cache LLM results to avoid redundant costs/time. (As discussed above).
* [ ] **Batch Processing**: Parallel processing (Parallelism) to handle millions of rows without freezing.

## Phase 3: The "Magic" (Advanced Features)

**Goal: Do things that Regex can never do.**

* [ ] **Contextual Imputation**: Fill missing values based on context (e.g., seeing Zipcode 70000 -> Auto-fill City: TP.HCM).
* [ ] **Entity Canonicalization**: Group entities (Fuzzy matching + Semantic matching).
* [ ] **Interactive CLI**: Terminal interface to review AI changes with low confidence.

# Contributing

We love contributions! Loclean is strictly open-source under the **Apache 2.0 License**.

1. **Fork** the repo on GitHub.
2. **Clone** your fork locally.
3. **Create** a new branch (`git checkout -b feature/amazing-feature`).
4. **Commit** your changes.
5. **Push** to your fork and submit a **Pull Request**.

_Built for the Data Community._

[![Star History Chart](https://api.star-history.com/svg?repos=nxank4/loclean&type=date&legend=top-left)](https://www.star-history.com/#nxank4/loclean&type=date&legend=top-left)