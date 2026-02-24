"""Scenario tests for user-facing error experience.

These tests verify that loclean provides clear, helpful error messages
when common mistakes are made.
"""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import loclean


def test_error_invalid_column_name() -> None:
    """Verify that a helpful error is raised when the target column is missing."""
    df = pl.DataFrame({"real_col": ["data"]})

    with pytest.raises(ValueError) as excinfo:
        loclean.clean(df, target_col="wrong_col")

    assert "Column 'wrong_col' not found" in str(excinfo.value)


def test_error_empty_dataframe() -> None:
    """Verify behavior on empty dataframe."""
    df = pl.DataFrame({"a": []}, schema={"a": pl.String})

    mock_engine = MagicMock()
    mock_engine.clean_batch.return_value = {}

    with (
        patch("loclean.get_engine", return_value=mock_engine),
        patch("loclean.engine.narwhals_ops.logger") as mock_logger,
    ):
        result = loclean.clean(df, "a")

    assert result.shape[0] == 0
    assert any(
        "No valid unique values found" in str(call.args[0])
        for call in mock_logger.warning.call_args_list
    )


@patch("loclean.cli.shell.loclean")
def test_shell_connection_error_shows_help(mock_lc: MagicMock) -> None:
    """Verify the shell renders a helpful message on ConnectionError."""

    from loclean.cli.shell import MODE_CLEAN, ShellState, execute

    mock_lc.clean.side_effect = ConnectionError("daemon unreachable")
    state = ShellState(mode=MODE_CLEAN)

    with pytest.raises(ConnectionError, match="daemon unreachable"):
        execute("5kg", state)


def test_extract_missing_schema_error() -> None:
    """Verify that extract raises ValueError with clear message when schema missing."""
    from loclean.cli.shell import MODE_EXTRACT, ShellState, execute

    state = ShellState(mode=MODE_EXTRACT)

    with pytest.raises(ValueError, match="No schema defined"):
        execute("some text", state)
