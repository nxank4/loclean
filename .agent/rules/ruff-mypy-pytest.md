---
trigger: glob
---

## 1. Tooling & Environment
- **Prioritize `uv`**: Always use `uv` commands (e.g., `uv run`, `uv pip`, `uv venv`) instead of standard `python` or `pip` commands.

## 2. Quality Assurance & CI/CD
- **Linting & Formatting**: Use **Ruff** for linting and formatting.
- **Type Checking**: Use **Mypy** for static type checking.
- **Conflict Resolution**: Be aware that Mypy changes can sometimes conflict with Ruff rules. **Always double-check Ruff after applying Mypy fixes** to ensure the code remains compliant and to prevent CI/CD failures.
- **Testing**: Run **Pytest** to verify functionality.
- **Optimization**: Run these checks locally before pushing to keep CI/CD pipelines fast and green.

## 3. Git Workflow & Version Control
- **Branching**: Never commit directly to `main`/`master`. Create feature branches (e.g., `feat/user-auth`).
- **Atomic Commits**: Commit changes in small, logical chunks.
- **Pull Requests**: Finalize tasks by preparing a PR description.

## 4. Code Quality
- **Optimization First**: Produce code optimized for performance and readability.
- **Clean Implementation**: Refactor logic for clarity.
- **Comment Policy**: Remove trash comments. Only explain "why" or "how" for complex logic. No "edited by" comments.

## 5. Documentation & Organization
- **Structure**: Maintain a logical folder structure.
- **Docstrings**: Required for all public functions and classes.