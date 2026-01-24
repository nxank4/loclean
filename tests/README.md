# Loclean Test Suite

This directory contains the test suite for the Loclean library. The tests are structured to ensure high code quality, correct functionality across different backends (Pandas, Polars), and privacy compliance.

## Directory Structure

*   **`unit/`**: Contains fast, isolated unit tests that verify individual components (e.g., specific regex patterns in `privacy`, schema generation in `extraction`). These tests mock external dependencies like LLMs.
*   **`scenarios/`**: High-level tests verifying end-to-end logic, UI/UX (terminal output), and Developer Experience (DX). These ensure the library feels polished for end-users.
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

### Run Scenario Tests (Logic & UX)
```bash
uv run pytest tests/scenarios
```

### Run with Coverage
```bash
uv run pytest --cov=src/loclean --cov-report=term-missing
```

## Scenario Testing Guidelines
Scenario tests in `tests/scenarios/` are designed to verify the library from a user's perspective.
- **Logic**: Use `test_e2e_flows.py` for complex pipelines (e.g., scrub + clean).
- **UX/UI**: Use `test_ux_interface.py` to assert on Rich terminal output (panels, tables, colors).
- **DX**: Use `test_error_experience.py` to ensure error messages are helpful and formatted correctly.

When adding new features, consider adding a scenario test to ensure the "feel" of the library remains high-quality.

## Test Configuration

Configuration is managed in `pyproject.toml` under `[tool.pytest.ini_options]`. We enforce strict markers and coverage requirements.

### Markers
*   `slow`: Marks tests that take longer to execute.
*   `cloud`: Marks tests that require external cloud API access.

To skip slow tests:
```bash
uv run pytest -m "not slow"
```
