"""Unit tests for loclean.tracking module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLangfuseTracker:
    """Verify LangfuseTracker calls the Langfuse SDK correctly."""

    @patch("loclean.tracking.langfuse_tracker.Langfuse")
    def test_start_run(self, MockLangfuse: MagicMock) -> None:
        from loclean.tracking.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        run_id = tracker.start_run("test", {"k": "v"})

        assert isinstance(run_id, str)
        assert len(run_id) == 32  # hex uuid
        MockLangfuse.return_value.trace.assert_called_once()

    @patch("loclean.tracking.langfuse_tracker.Langfuse")
    def test_log_step(self, MockLangfuse: MagicMock) -> None:
        from loclean.tracking.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        run_id = tracker.start_run("run", {})
        tracker.log_step(run_id, "s1", "in", "out", {"m": 1})

        MockLangfuse.return_value.trace.return_value.span.assert_called_once()

    @patch("loclean.tracking.langfuse_tracker.Langfuse")
    def test_log_score(self, MockLangfuse: MagicMock) -> None:
        from loclean.tracking.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        run_id = tracker.start_run("run", {})
        tracker.log_score(run_id, "accuracy", 0.95, "good")

        MockLangfuse.return_value.score.assert_called_once_with(
            trace_id=run_id,
            name="accuracy",
            value=0.95,
            comment="good",
        )

    @patch("loclean.tracking.langfuse_tracker.Langfuse")
    def test_end_run_flushes(self, MockLangfuse: MagicMock) -> None:
        from loclean.tracking.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        run_id = tracker.start_run("run", {})
        tracker.end_run(run_id)

        MockLangfuse.return_value.flush.assert_called_once()
        assert run_id not in tracker._runs

    @patch("loclean.tracking.langfuse_tracker.Langfuse")
    def test_log_step_unknown_run_ignored(self, MockLangfuse: MagicMock) -> None:
        from loclean.tracking.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        tracker.log_step("bad-id", "s1", "in", "out", {})
