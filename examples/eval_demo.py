"""Demo: evaluate loclean outputs and optionally log traces to Langfuse.

Run with:
    uv run python examples/eval_demo.py

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
environment variables to enable Langfuse tracking.
"""

from __future__ import annotations

import os

from rich.console import Console
from rich.table import Table

from loclean.eval import Evaluator, TestCase

console = Console()


def _build_cases() -> list[TestCase]:
    """Return a handful of test cases across tasks."""
    return [
        TestCase(
            input="Contact John Doe at john@example.com",
            expected_output="Contact [PERSON] at [EMAIL]",
            task="scrub",
            metric="exact_match",
        ),
        TestCase(
            input="Email jane@corp.io or call 555-1234",
            expected_output="",
            task="scrub",
            metric="pii_masking_recall",
            metadata={"pii_tokens": ["jane@corp.io", "555-1234"]},
        ),
    ]


def _get_tracker():  # type: ignore[no-untyped-def]
    """Try to build a LangfuseTracker if credentials are available."""
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from loclean.tracking import LangfuseTracker

            return LangfuseTracker()
        except ImportError:
            console.print("[yellow]langfuse not installed — skipping tracking[/yellow]")
    return None


def main() -> None:
    cases = _build_cases()
    tracker = _get_tracker()

    evaluator = Evaluator(tracker=tracker)
    summary = evaluator.run(cases)

    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")
    table.add_column("Details")

    for r in summary.results:
        table.add_row(
            r.test_case.task,
            r.test_case.metric,
            f"{r.score:.2f}",
            "✅" if r.passed else "❌",
            r.details,
        )

    console.print(table)
    console.print(
        f"\n[bold]Mean Score:[/bold] {summary.mean_score:.2%}  "
        f"[bold]Pass Rate:[/bold] {summary.pass_rate:.2%}"
    )


if __name__ == "__main__":
    main()
