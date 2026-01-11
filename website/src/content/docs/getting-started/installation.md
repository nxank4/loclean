---
title: Installation
description: Install Loclean for local-first semantic data cleaning.
sidebar:
  order: 1
---

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- No GPU required (runs on CPU by default)

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
