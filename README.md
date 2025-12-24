# Semantix ⚡🧠

**The All-in-One Local AI Data Cleaner.**

Clean messy tabular data using local AI.
**No API keys required. No GPU required.**

## 🔥 Why Semantix?

_in progress..._

## 🚀 Installation

```bash
pip install semantix
```

## ⚡ Quick Start

_in progress..._

## 🏗️ How It Works (The Architecture)

_in progress..._

## 🗺️ Roadmap

The development of Semantix is focused on three key areas: **Reliability**, **Privacy**, and **Integration**.

### 📍 Phase 1: Core Intelligence (Current Focus)

**Goal: Build a deterministic and smart cleaning engine.**

- [ ] **Strict Schema Mode**: Guarantee valid outputs by forcing the LLM to adhere to **Pydantic** models using GBNF grammar (eliminates JSON parsing errors).
- [ ] **Contextual Imputation**: Fill `null` values intelligently by reasoning over surrounding column context (e.g., inferring `State` from `Zip Code`).
- [ ] **Entity Canonicalization**: Map messy variations (e.g., "Apple Inc.", "apple comp", "AAPL") to a single "Golden Record" standard.

### 📍 Phase 2: Privacy & Advanced Extraction

**Goal: Specialized features for enterprise-grade data handling.**

- [ ] **Unstructured Extraction**: Parse free-text fields (Logs, Bios, Reviews) into structured tabular data.
- [ ] **Semantic PII Redaction**: Automatically detect and mask sensitive entities (Names, SSNs, Emails) locally to ensure data privacy.
- [ ] **Semantic Outlier Detection**: Flag values that are *statistically* normal but *contextually* impossible (e.g., "Age: 200").

### 📍 Phase 3: Ecosystem & DX

**Goal: Make Semantix a first-class citizen in the Python data stack.**

- [ ] **Native Dataframe Accessors**: Direct integration for **Pandas** and **Polars** (e.g., `df.semantix.clean(...)`) via PyArrow.
- [ ] **Interactive CLI Review**: A "Human-in-the-loop" mode to review and approve low-confidence AI changes via the terminal.
- [ ] **Custom LoRA Adapters**: Support for loading lightweight, domain-specific fine-tunes (e.g., Medical, Legal) without replacing the base model.

## 🤝 Contributing

We love contributions! Semantix is strictly open-source under the **Apache 2.0 License**.

1. **Fork** the repo on GitHub.
2. **Clone** your fork locally.
3. **Create** a new branch (`git checkout -b feature/amazing-feature`).
4. **Commit** your changes.
5. **Push** to your fork and submit a **Pull Request**.

_Built with ❤️ for the Data Community._
