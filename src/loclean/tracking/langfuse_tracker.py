"""Langfuse experiment tracker implementation."""

from __future__ import annotations

import uuid
from typing import Any

from loclean.tracking.base import BaseTracker

try:
    from langfuse import Langfuse  # type: ignore[import-untyped]

    _HAS_LANGFUSE = True
except ImportError:
    _HAS_LANGFUSE = False


class LangfuseTracker(BaseTracker):
    """Track evaluation runs in Langfuse.

    Requires the ``langfuse`` package (``pip install loclean[tracking]``).
    Credentials are read from the standard ``LANGFUSE_PUBLIC_KEY``,
    ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST`` environment variables.

    Args:
        **kwargs: Forwarded to ``langfuse.Langfuse()``.

    Raises:
        ImportError: If ``langfuse`` is not installed.
    """

    def __init__(self, **kwargs: Any) -> None:
        if not _HAS_LANGFUSE:
            raise ImportError(
                "langfuse is required for LangfuseTracker. "
                "Install it with: pip install loclean[tracking]"
            )
        self._client: Any = Langfuse(**kwargs)
        self._runs: dict[str, Any] = {}

    def start_run(self, name: str, metadata: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        trace = self._client.trace(
            id=run_id,
            name=name,
            metadata=metadata,
        )
        self._runs[run_id] = trace
        return run_id

    def log_step(
        self,
        run_id: str,
        step_name: str,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any],
    ) -> None:
        trace = self._runs.get(run_id)
        if trace is None:
            return
        trace.span(
            name=step_name,
            input=input_text,
            output=output_text,
            metadata=metadata,
        )

    def log_score(
        self,
        run_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        self._client.score(
            trace_id=run_id,
            name=name,
            value=value,
            comment=comment,
        )

    def end_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        self._client.flush()
