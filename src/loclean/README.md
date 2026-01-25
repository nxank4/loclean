# Loclean Source

This directory contains the core source code for the `loclean` library.

## Architecture Guidelines

Loclean is designed with a layered architecture to support local-first, privacy-aware data engineering.

## Directory Structure and File Organization

### Root Level Files

- **`__init__.py`**: Defines the public API. Only symbols listed in `__all__` are exported. This is the main entry point for users.
- **`_version.py`**: Contains the library version string. Updated automatically during releases.
- **`cache.py`**: SQLite-based caching implementation (`LocleanCache`) for inference results. Uses SHA256 hashing for cache keys.

### `cli/` - Command-Line Interface

Handles user interactions via terminal commands using `typer`.

- **`__init__.py`**: CLI app initialization and entry point
- **`model.py`**: Model management commands (download, list, status)
- **`model_commands.py`**: Implementation of model CLI commands with Rich progress bars and tables

**Guidelines:**
- All CLI commands should use Rich for output formatting
- Error messages should be user-friendly and actionable
- Use `typer` for argument parsing and validation

### `engine/` - DataFrame Processing

Backend-agnostic dataframe operations using **Narwhals**.

- **`narwhals_ops.py`**: Core dataframe operations wrapper. Supports Pandas, Polars, PyArrow, Modin, cuDF, and other Narwhals-compatible backends.

**Guidelines:**
- **Never import Pandas or Polars directly** in core logic. Always use `narwhals` (`nw`) for operations.
- Use `nw.from_native()` to convert native dataframes to Narwhals-compatible format
- Preserve lazy evaluation for Polars when possible
- Raise clear `ValueError` if backend is unrecognized (no silent fallbacks)

### `extraction/` - Structured Data Extraction

Logic for extracting structured data from unstructured text using Pydantic schemas and GBNF grammars.

- **`extractor.py`**: Main `Extractor` class. Handles schema-to-grammar conversion, JSON repair, retry logic, and validation.
- **`extract_dataframe.py`**: DataFrame-specific extraction logic with batch processing support.
- **`grammar_utils.py`**: Utilities for converting Pydantic schemas to GBNF grammars.
- **`json_repair.py`**: JSON repair utilities for fixing malformed LLM outputs.

**Guidelines:**
- Always use Pydantic V2 models for schemas
- GBNF grammars must be generated dynamically from schemas
- Implement retry logic with adjusted prompts on validation failures
- Cache extraction results using `LocleanCache`

### `inference/` - LLM Inference Engine

Adapters and engines for LLM inference. Supports local inference via `llama-cpp-python` and allows for future cloud API expansion.

- **`base.py`**: Abstract base class `InferenceEngine` that all inference backends must implement.
- **`config.py`**: Configuration management (`EngineConfig`, `load_config()`). Handles priority: function param > env var > pyproject.toml > defaults.
- **`factory.py`**: Factory function `create_engine()` for instantiating inference engines with lazy loading.
- **`adapters.py`**: Prompt adapters (Phi3Adapter, QwenAdapter, LlamaAdapter) for formatting prompts based on model type. Auto-selected via `get_adapter()`.
- **`schemas.py`**: Pydantic models for inference results (`ExtractionResult`).
- **`manager.py`**: (Deprecated) Legacy `LocalInferenceEngine` alias. Do not modify.

#### `inference/local/` - Local Inference Implementation

- **`llama_cpp.py`**: `LlamaCppEngine` implementation. Supports multiple models (Phi-3, Qwen, Gemma, DeepSeek, TinyLlama, LFM2.5) via `_MODEL_REGISTRY`.
- **`downloader.py`**: Model download logic using HuggingFace Hub.
- **`exceptions.py`**: Custom exceptions for local inference errors.

**Guidelines:**
- Use lazy loading for heavy dependencies (`llama_cpp`, `torch`, etc.) - import inside methods, not at module level
- Models are registered in `_MODEL_REGISTRY` with repo, filename, size, and description
- Always use GBNF grammars for constrained JSON output
- Implement proper error handling and logging

### `privacy/` - PII Detection and Scrubbing

Modules for detecting and scrubbing personally identifiable information (PII).

- **`detector.py`**: Main PII detection interface. Combines regex and LLM-based detection.
- **`regex_detector.py`**: Regex-based PII detection patterns.
- **`llm_detector.py`**: LLM-based PII detection using structured extraction.
- **`generator.py`**: Fake data generation using Faker (for replace mode).
- **`scrub.py`**: Scrubbing logic with mask and replace modes.
- **`schemas.py`**: Pydantic models for PII entities and detection results.

**Guidelines:**
- Support multiple locales for PII detection (default: `vi_VN`)
- Use both regex and LLM detection for comprehensive coverage
- Implement selective scrubbing strategies
- Faker integration is optional (requires `loclean[privacy]`)

### `resources/` - Static Resources

Static files shipped with the package (GBNF grammars, Jinja2 templates).

- **`grammars/`**: GBNF grammar files (`.gbnf`) for JSON, email, lists, PII detection, etc.
- **`templates/`**: Jinja2 templates (`.j2`) for prompt formatting (cleaning instructions, model-specific formats).

**Guidelines:**
- **Never use `open("src/...")`**. Always use `utils.resources.load_grammar()` and `utils.resources.load_template()` which wrap `importlib.resources` for zip-safe distribution.
- Resources are read-only at runtime
- Add new grammars/templates as needed, but keep them focused and reusable

### `utils/` - Utility Functions

General utilities used across the library.

- **`resources.py`**: Functions for loading grammars and templates (`load_grammar()`, `load_template()`). Handles both installed package and development fallback paths.
- **`logging.py`**: Logging configuration with Rich support (`configure_module_logger()`, `setup_rich_logging()`).
- **`rich_output.py`**: Rich terminal output utilities (progress bars, tables, panels, error summaries).

**Guidelines:**
- Use `logging` module, not `print()` statements
- Rich output should be optional and gracefully degrade if Rich is unavailable
- Utility functions should be pure and well-tested

## Development Guidelines

### Code Quality Standards

1. **Type Hints**: All functions and methods must have complete type hints. Use `TYPE_CHECKING` for forward references.
2. **Type Checking**: Code must pass `mypy --strict`. Run `uv run mypy .` before committing.
3. **Linting**: Code must pass `ruff check .`. Run `uv run ruff check .` and `uv run ruff format .` before committing.
4. **Testing**: All new code must have corresponding unit tests. Run `uv run pytest` before committing.

### Import Guidelines

- **Lazy Loading**: Heavy dependencies (`llama_cpp`, `torch`, `openai`) must be imported inside methods or factory functions, never at module top-level.
- **Backend Agnosticism**: Never import `pandas` or `polars` directly in core logic. Use `narwhals` (`nw`) instead.
- **Optional Dependencies**: Wrap optional imports in `try/except ImportError` blocks.

### Resource Management

- **Static Files**: Always use `utils.resources.load_grammar()` and `utils.resources.load_template()` to load resources.
- **Caching**: Use `LocleanCache` for inference result caching. Cache keys use format: `v3::{instruction}::{text}`.

### Public API

- **`__init__.py`**: Only symbols listed in `__all__` are considered public API.
- **Backward Compatibility**: When refactoring, keep old aliases with `DeprecationWarning` (see `inference/manager.py` for example).
- **Breaking Changes**: Document breaking changes clearly and provide migration paths.

### Error Handling

- **User-Friendly Messages**: Error messages should be clear and actionable.
- **Logging**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR).
- **Rich Error Output**: Use `rich_output` utilities for formatted error summaries when appropriate.

### Testing Requirements

- **Unit Tests**: Fast, isolated tests that mock external dependencies.
- **Integration Tests**: Tests that verify component interactions (may require model loading).
- **Scenario Tests**: High-level tests for UX/DX verification.

### Documentation

- **Docstrings**: All public functions and classes must have docstrings explaining arguments, return values, and purpose.
- **Comments**: Only comment to explain "why" (complex logic), not "what" (obvious code).
- **Type Hints**: Serve as inline documentation - make them comprehensive.

## Local-First Principle

Default implementations should always prioritize local execution (CPU/GPU) without external API calls. Cloud APIs are opt-in and not yet implemented.

## Backward Compatibility

When refactoring:
- Keep old aliases with `DeprecationWarning` using `warnings.warn()` with `stacklevel=2`
- Maintain public API contracts in `__init__.py`
- Document migration paths for breaking changes
