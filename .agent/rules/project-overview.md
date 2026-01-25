---
trigger: always
---

# Loclean Project Overview

## What is Loclean?

Loclean is a **local-first Python library** for semantic data cleaning using Small Language Models (SLMs). It bridges the gap between data engineering and local AI, designed for production pipelines where privacy and stability are non-negotiable.

## Core Philosophy

- **Local-First & Private**: Default to local execution via `llama-cpp-python`. Cloud APIs are opt-in and not yet implemented.
- **Deterministic**: Outputs must be strictly structured using Pydantic V2 models and GBNF grammars (eliminates JSON parsing errors).
- **Backend Agnostic**: Support Pandas, Polars, PyArrow, Modin, cuDF transparently using `narwhals`. **Strictly NO direct Pandas/Polars imports in core logic.**
- **Production Ready**: Code must be strictly typed, well-documented, and use lazy loading for heavy dependencies.

## Technology Stack

- **Language**: Python 3.10+ (tested on 3.10, 3.11, 3.12, 3.13, 3.14, 3.15)
- **Data Engine**: `narwhals>=2.14.0` (Strictly NO direct Pandas/Polars imports in core logic)
- **Validation**: `pydantic>=2.12.5` (Use `BaseModel` for everything, `ConfigDict` for model config)
- **Local Inference**: `llama-cpp-python>=0.2.23` (with GBNF grammar support)
- **Model Management**: `huggingface_hub>=0.20.0` (for downloading GGUF models)
- **Caching**: SQLite3 (via `LocleanCache` class) for persistent inference result caching
- **Cloud Inference**: `openai` (via `instructor`), `google-generativeai` (Optional/Lazy loaded, not yet implemented)
- **Templating**: `jinja2` (for prompt management via `PromptAdapter` classes)
- **Testing**: `pytest>=9.0.2`, `pytest-mock` (use markers: `@pytest.mark.slow`, `@pytest.mark.cloud`)
- **Code Quality**: `ruff` (linting & formatting), `mypy` (type checking)
- **Package Manager**: `uv` (use `uv run` for commands, `uv sync` for dependencies)

## Project Structure

```
src/loclean/
├── __init__.py          # Public API (clean(), get_engine(), extract(), scrub())
├── cache.py             # LocleanCache (SQLite-based)
├── engine/              # NarwhalsEngine for dataframe ops
├── inference/
│   ├── base.py          # InferenceEngine ABC
│   ├── schemas.py       # ExtractionResult Pydantic model
│   ├── config.py        # EngineConfig, load_config()
│   ├── factory.py       # create_engine() with lazy loading
│   ├── adapters.py      # PromptAdapter (Phi3, Qwen, Llama)
│   ├── manager.py       # LocalInferenceEngine (deprecated - DO NOT MODIFY)
│   └── local/
│       └── llama_cpp.py # LlamaCppEngine implementation
├── extraction/          # Structured extraction with Pydantic schemas
├── privacy/             # PII detection and scrubbing
├── resources/
│   ├── grammars/        # .gbnf files (e.g., json.gbnf)
│   └── templates/       # .j2 files (Jinja2 templates)
└── utils/
    └── resources.py     # load_grammar(), load_template()
```

## Key Architectural Principles

1. **Backend Agnosticism**: Use `narwhals` (`nw`) for all dataframe operations. Never import `pandas` or `polars` directly in core logic.
2. **Lazy Loading**: Heavy libraries (`llama_cpp`, `openai`, `torch`) must be imported inside methods or factory functions, never at module top-level.
3. **Resource Management**: Never use `open("src/...")`. Always use `utils.resources.load_grammar()` and `utils.resources.load_template()` which wrap `importlib.resources`.
4. **Type Safety**: All code must pass `mypy --strict`. Use type hints for everything.
5. **Public API**: Only symbols in `__all__` in `__init__.py` are public. Maintain backward compatibility.

## Available Models

Loclean supports multiple SLMs via the model registry:
- **phi-3-mini**: Microsoft Phi-3 Mini (3.8B, 4K context) - Default, balanced
- **tinyllama**: TinyLlama 1.1B - Smallest, fastest
- **gemma-2b**: Google Gemma 2B Instruct - Balanced performance
- **qwen3-4b**: Qwen3 4B - Higher quality
- **gemma-3-4b**: Gemma 3 4B - Larger context
- **deepseek-r1**: DeepSeek R1 - Reasoning model
- **lfm2.5**: Liquid LFM2.5-1.2B Instruct (1.17B, 32K context) - Best-in-class 1B scale

## Main Features

1. **Structured Extraction** (`extract()`): Extract structured data from unstructured text with guaranteed Pydantic schema compliance
2. **Data Cleaning** (`clean()`): Semantic data cleaning with custom instructions
3. **Privacy Scrubbing** (`scrub()`): PII detection and redaction (mask or replace with fake data)
4. **Model Management**: CLI commands for downloading and managing models

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Fast, isolated tests with mocked dependencies
- **Scenario Tests** (`tests/scenarios/`): End-to-end logic and UX verification
- **Integration Tests** (`tests/integration/`): Component interaction tests (may require model loading)

## Documentation

- **Main README**: `README.md` - User-facing documentation
- **Source README**: `src/loclean/README.md` - Developer documentation for source code
- **Tests README**: `tests/README.md` - Testing guidelines and structure
- **Examples README**: `examples/README.md` - Example notebooks documentation
- **Contributing Guide**: `CONTRIBUTION.md` - Contribution guidelines
- **Online Docs**: https://nxank4.github.io/loclean

## Development Workflow

1. **Setup**: `uv sync --all-extras --dev`
2. **Code**: Follow architecture guidelines and coding standards
3. **Test**: `uv run pytest`
4. **Lint**: `uv run ruff check .` and `uv run ruff format .`
5. **Type Check**: `uv run mypy .`
6. **Commit**: Use Conventional Commits format
7. **PR**: Ensure all CI checks pass

## Important Notes

- **Deprecated Files**: Files marked as `(deprecated)` should NOT be modified unless explicitly requested. They are kept for backward compatibility.
- **Public API**: Changes to `__init__.py` require careful consideration for backward compatibility.
- **Resource Files**: Grammar and template files are read-only at runtime. Use `utils.resources` to load them.
