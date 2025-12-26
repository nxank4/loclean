<p align="center">
  <a href="https://pypi.org/project/semantix"><img src="https://img.shields.io/pypi/v/semantix?color=blue&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/semantix"><img src="https://img.shields.io/pypi/pyversions/semantix?style=flat-square" alt="Python Versions"></a>
  <a href="https://github.com/nxank4/semantix/actions/workflows/ci.yml"><img src="https://github.com/nxank4/semantix/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/nxank4/semantix/blob/main/LICENSE"><img src="https://img.shields.io/github/license/nxank4/semantix?style=flat-square" alt="License"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
</p>

---

# Semantix ⚡🧠

**The All-in-One Local AI Data Cleaner.**

Clean messy tabular data using local AI.
**No API keys required. No GPU required.**

## 🔥 Why Semantix?

Semantix bridges the gap between **Data Engineering** and **Local AI**, designed for production pipelines where privacy and stability are non-negotiable.

### 🔒 Privacy-First & Zero Cost

Leverage the power of Small Language Models (SLMs) like **Phi-3** and **Llama-3** running locally via `llama.cpp`. Clean sensitive PII, medical records, or proprietary data without a single byte leaving your infrastructure.

### 🛡️ Deterministic Outputs

Forget about "hallucinations" or parsing loose text. Semantix uses **GBNF Grammars** and **Pydantic V2** to force the LLM to output valid, type-safe JSON. If it breaks the schema, it doesn't pass.

### ⚡ Backend Agnostic (Zero-Copy)

Built on **Narwhals**, Semantix supports **Pandas**, **Polars**, and **PyArrow** natively.

* Running Polars? We keep it lazy.
* Running Pandas? We handle it seamlessly.
* **No heavy dependency lock-in.**

## 🚀 Installation

### Requirements

* Python 3.10 or higher
* No GPU required (runs on CPU by default)

### Basic Installation

**Using pip:**

```bash
pip install semantix
```

**Using uv (recommended for faster installs):**

```bash
uv pip install semantix
```

**Using conda/mamba:**

```bash
conda install -c conda-forge semantix
# or
mamba install -c conda-forge semantix
```

### Optional Dependencies

**For DataFrame operations (Pandas, Polars, PyArrow):**

```bash
pip install semantix[data]
```

**For Cloud API support (OpenAI, Anthropic, Gemini):**

```bash
pip install semantix[cloud]
```

**Install everything:**

```bash
pip install semantix[all]
```

### Development Installation

To contribute or run tests locally:

```bash
# Clone the repository
git clone https://github.com/nxank4/semantix.git
cd semantix

# Install with development dependencies (using uv)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

## ⚡ Quick Start

_in progress..._

## 🏗️ How It Works (The Architecture)

_in progress..._

## 🗺️ Roadmap

The development of Semantix is focused on three key areas: **Reliability**, **Privacy**, and **Integration**.

### 📍 Phase 1: Core Intelligence (Current Focus)

**Goal: Build a deterministic and smart cleaning engine.**

* [x] **Strict Schema Mode**: Guarantee valid outputs by forcing the LLM to adhere to **Pydantic** models using GBNF grammar (eliminates JSON parsing errors).
* [ ] **Contextual Imputation**: Fill `null` values intelligently by reasoning over surrounding column context (e.g., inferring `State` from `Zip Code`).
* [ ] **Entity Canonicalization**: Map messy variations (e.g., "Apple Inc.", "apple comp", "AAPL") to a single "Golden Record" standard.

### 📍 Phase 2: Privacy & Advanced Extraction

**Goal: Specialized features for enterprise-grade data handling.**

* [ ] **Unstructured Extraction**: Parse free-text fields (Logs, Bios, Reviews) into structured tabular data.
* [ ] **Semantic PII Redaction**: Automatically detect and mask sensitive entities (Names, SSNs, Emails) locally to ensure data privacy.
* [ ] **Semantic Outlier Detection**: Flag values that are _statistically_ normal but _contextually_ impossible (e.g., "Age: 200").

### 📍 Phase 3: Ecosystem & DX

**Goal: Make Semantix a first-class citizen in the Python data stack.**

* [ ] **Native Dataframe Accessors**: Direct integration for **Pandas** and **Polars** (e.g., `df.semantix.clean(...)`) via PyArrow.
* [ ] **Interactive CLI Review**: A "Human-in-the-loop" mode to review and approve low-confidence AI changes via the terminal.
* [ ] **Custom LoRA Adapters**: Support for loading lightweight, domain-specific fine-tunes (e.g., Medical, Legal) without replacing the base model.

## 🤝 Contributing

We love contributions! Semantix is strictly open-source under the **Apache 2.0 License**.

1. **Fork** the repo on GitHub.
2. **Clone** your fork locally.
3. **Create** a new branch (`git checkout -b feature/amazing-feature`).
4. **Commit** your changes.
5. **Push** to your fork and submit a **Pull Request**.

_Built with ❤️ for the Data Community._
