"""Stdin/stdout orchestration runner for workflow automation.

This module provides a lightweight execution script that:

1. Reads a JSON payload from **stdin**.
2. Validates it against :class:`PipelinePayload`.
3. Instantiates the inference engine via :func:`load_config`.
4. Loads the data into a Narwhals-compatible DataFrame and executes
   :func:`loclean.clean`.
5. Writes the augmented result as JSON to **stdout**.

Exit codes are deterministic so that DAG managers (n8n, Airflow) can
branch downstream tasks accordingly.

Usage::

    echo '{"data": [...], "target_col": "price"}' \\
        | python -m loclean.orchestration.runner
"""

from __future__ import annotations

import json
import sys
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, ValidationError

from loclean import clean
from loclean.inference.config import load_config

EXIT_OK: int = 0
EXIT_INVALID_PAYLOAD: int = 1
EXIT_PIPELINE_ERROR: int = 2
EXIT_SERIALIZATION_ERROR: int = 3


class PipelinePayload(BaseModel):
    """Schema for the incoming JSON payload.

    Attributes:
        data: Row-oriented records (list of dicts).
        target_col: Column to pass to :func:`loclean.clean`.
        instruction: Cleaning instruction forwarded to the LLM.
        engine_config: Key-value overrides passed to :func:`load_config`.
        batch_size: Number of unique values per processing batch.
    """

    data: list[dict[str, Any]]
    target_col: str
    instruction: str = Field(
        default="Extract the numeric value and unit as-is.",
    )
    engine_config: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=50, ge=1)


def run_pipeline(payload: PipelinePayload) -> dict[str, Any]:
    """Execute the cleaning pipeline for a validated payload.

    Args:
        payload: Validated pipeline payload.

    Returns:
        Dictionary with ``status``, ``data`` (list of row dicts), and
        ``row_count``.

    Raises:
        ValueError: If *target_col* is missing from the data.
        RuntimeError: On unexpected engine or cleaning failures.
    """
    config = load_config(**payload.engine_config)

    df = pl.DataFrame(payload.data)

    result_df = clean(
        df,
        payload.target_col,
        payload.instruction,
        model=config.model,
        host=config.host,
        verbose=config.verbose,
        batch_size=payload.batch_size,
    )

    rows: list[dict[str, Any]] = result_df.to_dicts()

    return {
        "status": "ok",
        "data": rows,
        "row_count": len(rows),
    }


def _error_response(code: int, message: str) -> dict[str, Any]:
    """Build a structured error response dict.

    Args:
        code: Exit code constant.
        message: Human-readable error description.

    Returns:
        Error dict with ``status``, ``code``, and ``message``.
    """
    return {"status": "error", "code": code, "message": message}


def main() -> None:
    """Entry point: read stdin → clean → write stdout."""
    try:
        raw = sys.stdin.read()
    except Exception as exc:
        result = _error_response(EXIT_INVALID_PAYLOAD, f"stdin read error: {exc}")
        sys.stdout.write(json.dumps(result))
        sys.exit(EXIT_INVALID_PAYLOAD)
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        result = _error_response(EXIT_INVALID_PAYLOAD, f"Invalid JSON: {exc}")
        sys.stdout.write(json.dumps(result))
        sys.exit(EXIT_INVALID_PAYLOAD)
        return

    try:
        payload = PipelinePayload(**data)
    except ValidationError as exc:
        result = _error_response(
            EXIT_INVALID_PAYLOAD, f"Payload validation failed: {exc}"
        )
        sys.stdout.write(json.dumps(result))
        sys.exit(EXIT_INVALID_PAYLOAD)
        return

    try:
        output = run_pipeline(payload)
    except Exception as exc:
        result = _error_response(EXIT_PIPELINE_ERROR, f"Pipeline error: {exc}")
        sys.stdout.write(json.dumps(result))
        sys.exit(EXIT_PIPELINE_ERROR)
        return

    try:
        sys.stdout.write(json.dumps(output))
    except (TypeError, ValueError) as exc:
        result = _error_response(
            EXIT_SERIALIZATION_ERROR, f"Serialization error: {exc}"
        )
        sys.stdout.write(json.dumps(result))
        sys.exit(EXIT_SERIALIZATION_ERROR)
        return

    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
