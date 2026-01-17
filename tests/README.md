# Loclean Test Suite

This directory contains the test suite for the Loclean library. The tests are structured to ensure high code quality, correct functionality across different backends (Pandas, Polars), and privacy compliance.

## Directory Structure

*   **`unit/`**: Contains fast, isolated unit tests that verify individual components (e.g., specific regex patterns in `privacy`, schema generation in `extraction`). These tests mock external dependencies like LLMs.
*   **`integration/`**: Contains broader tests that verify how components work together. Some may require a realistic environment or actual model loading.
*   **`conftest.py`**: Pytest configuration and shared fixtures.

## Running Tests

We use `pytest` for testing and `uv` for dependency management.

### Run All Tests
```bash
uv run pytest
```

### Run Only Unit Tests (Fast)
```bash
uv run pytest tests/unit
```

### Run with Coverage
```bash
uv run pytest --cov=src/loclean --cov-report=term-missing
```

## Test Configuration

Configuration is managed in `pyproject.toml` under `[tool.pytest.ini_options]`. We enforce strict markers and coverage requirements.

### Markers
*   `slow`: Marks tests that take longer to execute.
*   `cloud`: Marks tests that require external cloud API access.

To skip slow tests:
```bash
uv run pytest -m "not slow"
```
