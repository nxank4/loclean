"""Abstract base for experiment trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    """Interface that all tracker back-ends must implement.

    Each tracker manages *runs* (identified by a string ``run_id``).
    Within a run, callers can log individual steps and numeric scores.
    """

    @abstractmethod
    def start_run(self, name: str, metadata: dict[str, Any]) -> str:
        """Begin a new tracked run.

        Args:
            name: Human-readable run name.
            metadata: Arbitrary metadata (model, host, params, …).

        Returns:
            A unique ``run_id`` string for subsequent calls.
        """
        ...

    @abstractmethod
    def log_step(
        self,
        run_id: str,
        step_name: str,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Record a single evaluation step within a run.

        Args:
            run_id: Identifier returned by ``start_run``.
            step_name: Short label for this step.
            input_text: Raw input fed to the loclean function.
            output_text: Output produced by the function.
            metadata: Extra context (expected output, metric name, …).
        """
        ...

    @abstractmethod
    def log_score(
        self,
        run_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Attach a numeric score to a run.

        Args:
            run_id: Identifier returned by ``start_run``.
            name: Score name (e.g. ``"exact_match"``).
            value: Numeric value in ``[0, 1]``.
            comment: Optional human-readable detail.
        """
        ...

    @abstractmethod
    def end_run(self, run_id: str) -> None:
        """Finalise and flush a run.

        Args:
            run_id: Identifier returned by ``start_run``.
        """
        ...
