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

**For DataFrame operations (Pandas, Polars, PyArrow):**

```bash
pip install loclean[data]
```

**For Cloud API support (OpenAI, Anthropic, Gemini):**

```bash
pip install loclean[cloud]
```

**Install everything:**

```bash
pip install loclean[all]
```

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

# How It Works (The Architecture)

_in progress..._

# Roadmap

The development of Loclean is organized into three phases, prioritizing MVP delivery while maintaining a long-term vision.

## Phase 1: The "Smart" Engine (Phần Lõi Hybrid)

**Goal: Get `loclean.clean()` running fast and accurately.**

* [ ] **Hybrid Router Architecture**: Build `clean(strategy='auto')` function. Automatically run Regex first, LLM second.
* [ ] **Strict Output (Pydantic + GBNF)**: Ensure 100% LLM outputs valid JSON Schema. (Using llama-cpp-python grammar).
* [ ] **Simple Extraction**: Extract basic information from raw text (Unstructured to Structured).

## Phase 2: The "Safe" Layer (Bảo mật & Tối ưu)

**Goal: Convince enterprises to trust and adopt the library.**

* [ ] **Semantic PII Redaction**: Masking sensitive names, phone numbers, emails, and addresses.
* [ ] **SQLite Caching System**: Cache LLM results to avoid redundant costs/time. (As discussed above).
* [ ] **Batch Processing**: Parallel processing (Parallelism) to handle millions of rows without freezing.

## Phase 3: The "Magic" (Tính năng nâng cao)

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
