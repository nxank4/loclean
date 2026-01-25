# Loclean Test Suite

This directory contains the test suite for the Loclean library. The tests are structured to ensure high code quality, correct functionality across different backends (Pandas, Polars, PyArrow, Modin, cuDF), and privacy compliance.

## Directory Structure

### `unit/` - Unit Tests

Fast, isolated unit tests that verify individual components. These tests mock external dependencies like LLMs and should run in < 1 second each.

**Structure mirrors source code:**
- **`cli/`**: Tests for CLI commands and user interactions
  - `test_cli_init.py`: CLI app initialization and command registration
  - `test_model.py`: Model management commands
  - `test_model_commands.py`: Model command implementations
- **`engine/`**: Tests for dataframe operations
  - `test_narwhals_ops.py`: Narwhals wrapper operations
- **`extraction/`**: Tests for structured extraction
  - `test_extractor.py`: Main extractor logic, schema-to-grammar conversion, retry logic
  - `test_extract_dataframe.py`: DataFrame extraction with batch processing
  - `test_grammar_utils.py`: GBNF grammar generation from Pydantic schemas
  - `test_json_repair.py`: JSON repair utilities
- **`inference/`**: Tests for inference engines
  - **`local/`**: Local inference tests
    - `test_llama_cpp.py`: LlamaCppEngine implementation, model registry, GBNF grammar usage
    - `test_downloader.py`: Model download logic
    - `test_exceptions.py`: Custom exception handling
  - `test_base.py`: InferenceEngine abstract base class
  - `test_config.py`: Configuration management and priority handling
  - `test_factory.py`: Engine factory and lazy loading
  - `test_adapters.py`: Prompt adapters (Phi3, Qwen, Llama)
  - `test_schemas.py`: Pydantic models for inference results
  - `test_manager.py`: (Deprecated) Legacy manager tests
- **`privacy/`**: Tests for PII detection and scrubbing
  - `test_detector.py`: Main PII detection interface
  - `test_regex_detector.py`: Regex-based PII patterns
  - `test_llm_detector.py`: LLM-based PII detection
  - `test_generator.py`: Fake data generation with Faker
  - `test_scrub.py`: Scrubbing logic (mask/replace modes)
  - `test_schemas.py`: PII entity and detection result models
  - `test_detector_functions.py`: Public detector functions
- **`utils/`**: Tests for utility functions
  - `test_resources.py`: Resource loading (grammars, templates)
  - `test_logging.py`: Logging configuration and Rich support
  - `test_rich_output.py`: Rich terminal output utilities
- **`test_cache.py`**: SQLite cache implementation tests
- **`test_public_api.py`**: Public API contract tests

**Guidelines for Unit Tests:**
- Mock all external dependencies (LLMs, file I/O, network calls)
- Tests should be fast (< 1 second each)
- Use `pytest.fixture` for shared test data
- Test both success and error paths
- Use descriptive test names: `test_function_name_scenario_expected_result`

### `scenarios/` - Scenario Tests

High-level tests verifying end-to-end logic, UI/UX (terminal output), and developer experience (DX). These ensure the library feels polished for end-users.

- **`test_e2e_flows.py`**: Complex end-to-end pipelines (e.g., scrub + clean, extract + validate)
- **`test_ux_interface.py`**: Rich terminal output verification (panels, tables, colors, progress bars)
- **`test_error_experience.py`**: Error message formatting and helpfulness

**Guidelines for Scenario Tests:**
- Test from a user's perspective
- Verify Rich output formatting and colors
- Ensure error messages are helpful and actionable
- Test complex workflows that span multiple modules

### `integration/` - Integration Tests

Broader tests that verify how components work together. Some may require a realistic environment or actual model loading.

- **`test_core.py`**: Core functionality integration tests
- **`test_reasoning.py`**: Reasoning model integration tests

**Guidelines for Integration Tests:**
- May require actual model loading (mark with `@pytest.mark.slow`)
- Test component interactions without heavy mocking
- Verify real-world usage scenarios

### `conftest.py` - Pytest Configuration

Shared pytest fixtures and configuration used across all test modules.

**Common Fixtures:**
- `tmp_cache_dir`: Temporary cache directory for tests
- `mock_engine`: Mock inference engine for unit tests
- `sample_dataframe`: Sample dataframe fixtures for different backends

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

### Run Integration Tests
```bash
uv run pytest tests/integration
```

### Run with Coverage
```bash
uv run pytest --cov=src/loclean --cov-report=term-missing
```

### Run Specific Test File
```bash
uv run pytest tests/unit/extraction/test_extractor.py
```

### Run Specific Test Function
```bash
uv run pytest tests/unit/extraction/test_extractor.py::test_extract_with_retry_logic
```

## Test Markers

Configuration is managed in `pyproject.toml` under `[tool.pytest.ini_options]`. We enforce strict markers and coverage requirements.

### Available Markers

- **`slow`**: Marks tests that take longer to execute (e.g., model loading, integration tests)
- **`cloud`**: Marks tests that require external cloud API access (not yet implemented)

### Using Markers

To skip slow tests:
```bash
uv run pytest -m "not slow"
```

To run only fast tests:
```bash
uv run pytest -m "not slow and not cloud"
```

## Writing Tests

### Test Structure

Follow this pattern:
```python
def test_function_name_scenario_expected_result():
    """Test description explaining what is being tested."""
    # Arrange: Set up test data and mocks
    # Act: Execute the code under test
    # Assert: Verify the expected outcome
```

### Mocking Guidelines

- **Mock LLM calls**: Always mock `InferenceEngine.extract()` or `LlamaCppEngine` calls
- **Mock file I/O**: Use `unittest.mock.patch` for file operations
- **Mock network calls**: Mock HuggingFace Hub downloads
- **Use fixtures**: Share common test data via `pytest.fixture`

### Example Test

```python
from unittest.mock import Mock, patch
import pytest
from loclean.extraction.extractor import Extractor

def test_extract_with_valid_schema_returns_pydantic_model():
    """Test that extract() returns a valid Pydantic model."""
    # Arrange
    mock_engine = Mock()
    mock_engine.extract.return_value = '{"name": "test", "price": 100}'
    extractor = Extractor(engine=mock_engine)
    
    # Act
    result = extractor.extract("test text", schema=Product)
    
    # Assert
    assert isinstance(result, Product)
    assert result.name == "test"
    assert result.price == 100
```

## Scenario Testing Guidelines

Scenario tests in `tests/scenarios/` are designed to verify the library from a user's perspective.

- **Logic**: Use `test_e2e_flows.py` for complex pipelines (e.g., scrub + clean, extract + validate)
- **UX/UI**: Use `test_ux_interface.py` to assert on Rich terminal output (panels, tables, colors, progress bars)
- **DX**: Use `test_error_experience.py` to ensure error messages are helpful and formatted correctly

When adding new features, consider adding a scenario test to ensure the "feel" of the library remains high-quality.

## Test Coverage Requirements

- Aim for > 80% code coverage
- Critical paths (extraction, inference, privacy) should have > 90% coverage
- Use `--cov-report=term-missing` to identify untested lines

## Pre-Commit Checklist

Before committing test changes:

1. ✅ All tests pass: `uv run pytest`
2. ✅ Linting passes: `uv run ruff check tests/`
3. ✅ Type checking passes: `uv run mypy tests/` (if applicable)
4. ✅ Coverage maintained or improved
5. ✅ New tests follow naming conventions
6. ✅ Tests are fast (< 1s for unit tests)
7. ✅ Slow tests are marked with `@pytest.mark.slow`

## Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach recommended)
2. **Test both success and error paths**
3. **Add unit tests** for individual components
4. **Add scenario tests** for user-facing features
5. **Add integration tests** for complex workflows
6. **Update this README** if adding new test categories or patterns

## Common Issues

### Tests Failing Due to Cache

If tests fail due to cached results:
```bash
# Clear pytest cache
rm -rf .pytest_cache
```

### Tests Failing Due to Model Downloads

Mock model downloads in unit tests. Integration tests that require models should be marked with `@pytest.mark.slow`.

### Type Checking Issues

If mypy complains about test code:
- Use `# type: ignore[error-code]` sparingly and document why
- Ensure test fixtures have proper type hints
