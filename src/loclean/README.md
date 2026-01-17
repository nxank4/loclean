# Loclean Source

This directory contains the core source code for the `loclean` library.

## Architecture Guidelines

Loclean is designed with a layered architecture to support local-first, privacy-aware data engineering.

### Directory Structure

*   **`cli/`**: The command-line interface implementation (powered by `typer`). Handles user input from the terminal and delegates to core logic.
*   **`engine/`**: The dataframe processing engine. We use **Narwhals** to provide backend-agnostic operations, supporting Pandas, Polars, and PyArrow seamlessly.
*   **`extraction/`**: Logic for structured data extraction. Contains the `Extractor` class and logic for converting Pydantic schemas to GBNF grammars.
*   **`inference/`**: Adapters for LLM inference. The primary local backend is `llama-cpp-python` (`local/`), but the interface allows for future expansion.
*   **`privacy/`**: Modules dedicated to PII detection and scrubbing (scrubbing logic, regex patterns, Faker integration).
*   **`resources/`**: Static resources, mappings, or default configurations shipped with the package.
*   **`utils/`**: General utility functions used across the library.

## Development Notes

*   **Public API**: The public API is explicitly defined in `__init__.py`. Only symbols listed in `__all__` are considered public.
*   **Type Hinting**: We enforce strict type checking. All valid code must pass `mypy --strict`.
*   **Local-First**: Default implementations should always prioritize local execution (CPU/GPU) without external API calls.
