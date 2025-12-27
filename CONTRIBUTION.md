# Contributing to Loclean

Thank you for your interest in contributing! We use modern Python tooling to ensure code quality.

## 1. Prerequisites

You need to have **[uv](https://github.com/astral-sh/uv)** installed. It manages Python versions and dependencies for us.

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Setup Development Environment

```bash
git clone https://github.com/nxank4/loclean.git
cd loclean

# This will create a virtualenv and install all dev dependencies
uv sync --all-extras
```

## 3. Running Checks (Before you commit)

Please run these commands to ensure your code passes CI:

```bash
# Run Unit Tests
uv run pytest

# Run Linter & Formatter
uv run ruff check .
uv run ruff format .

# Run Type Checker
uv run mypy .
```

## 4. Pull Request Guidelines

We use Conventional Commits. Please title your PRs like: `feat: add new adapter` or `fix: regex pattern`.

Please ensure your PR updates `schemas.py` if output structure changes.

## 5. Code Style

- Use type hints for all functions
- Follow PEP 8 style guide
- Run `ruff check .` and `mypy .` before committing
