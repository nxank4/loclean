"""Tests for loclean.cli.shell module."""

import io
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from rich.console import Console

from loclean.cli.shell import (
    MODE_CLEAN,
    MODE_EXTRACT,
    MODE_SCRUB,
    ShellState,
    execute,
    handle_command,
    parse_schema,
    render,
)


def _make_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


class TestShellState:
    """Tests for ShellState defaults."""

    def test_defaults(self) -> None:
        state = ShellState()
        assert state.mode == MODE_CLEAN
        assert state.instruction is None
        assert state.schema is None
        assert state.model is None


class TestParseSchema:
    """Tests for parse_schema."""

    def test_simple_schema(self) -> None:
        schema = parse_schema(["name:str", "price:int"])
        assert "name" in schema.model_fields
        assert "price" in schema.model_fields

    def test_all_types(self) -> None:
        schema = parse_schema(["a:str", "b:int", "c:float", "d:bool"])
        assert len(schema.model_fields) == 4

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid schema field"):
            parse_schema(["badfield"])

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown type"):
            parse_schema(["name:list"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            parse_schema([])


class TestHandleCommand:
    """Tests for slash command handling."""

    def test_quit(self) -> None:
        state = ShellState()
        assert handle_command("/quit", state, _make_console()) is True

    def test_exit(self) -> None:
        state = ShellState()
        assert handle_command("/exit", state, _make_console()) is True

    def test_help(self) -> None:
        state = ShellState()
        assert handle_command("/help", state, _make_console()) is False

    def test_mode_clean(self) -> None:
        state = ShellState(mode=MODE_EXTRACT)
        handle_command("/mode clean", state, _make_console())
        assert state.mode == MODE_CLEAN

    def test_mode_extract(self) -> None:
        state = ShellState()
        handle_command("/mode extract", state, _make_console())
        assert state.mode == MODE_EXTRACT

    def test_mode_scrub(self) -> None:
        state = ShellState()
        handle_command("/mode scrub", state, _make_console())
        assert state.mode == MODE_SCRUB

    def test_mode_invalid(self) -> None:
        state = ShellState()
        handle_command("/mode invalid", state, _make_console())
        assert state.mode == MODE_CLEAN

    def test_schema(self) -> None:
        state = ShellState()
        handle_command("/schema name:str price:int", state, _make_console())
        assert state.schema is not None
        assert "name" in state.schema.model_fields

    def test_schema_invalid(self) -> None:
        state = ShellState()
        handle_command("/schema badfield", state, _make_console())
        assert state.schema is None

    def test_instruction_set(self) -> None:
        state = ShellState()
        handle_command("/instruction Convert to kg", state, _make_console())
        assert state.instruction == "Convert to kg"

    def test_instruction_clear(self) -> None:
        state = ShellState(instruction="old")
        handle_command("/instruction", state, _make_console())
        assert state.instruction is None

    def test_model_set(self) -> None:
        state = ShellState()
        handle_command("/model llama3", state, _make_console())
        assert state.model == "llama3"

    def test_model_empty(self) -> None:
        state = ShellState()
        handle_command("/model", state, _make_console())
        assert state.model is None

    def test_unknown_command(self) -> None:
        state = ShellState()
        result = handle_command("/foobar", state, _make_console())
        assert result is False


class TestExecute:
    """Tests for execute routing."""

    @patch("loclean.cli.shell.loclean")
    def test_clean_mode(self, mock_loclean: MagicMock) -> None:
        mock_loclean.clean.return_value = "cleaned"
        state = ShellState(mode=MODE_CLEAN)
        result = execute("5kg", state)
        mock_loclean.clean.assert_called_once()
        assert result == "cleaned"

    @patch("loclean.cli.shell.loclean")
    def test_extract_mode(self, mock_loclean: MagicMock) -> None:
        mock_loclean.extract.return_value = {"name": "shirt"}
        state = ShellState(mode=MODE_EXTRACT)
        state.schema = parse_schema(["name:str"])
        result = execute("red shirt", state)
        mock_loclean.extract.assert_called_once()
        assert result == {"name": "shirt"}

    def test_extract_mode_no_schema_raises(self) -> None:
        state = ShellState(mode=MODE_EXTRACT)
        with pytest.raises(ValueError, match="No schema defined"):
            execute("some text", state)

    @patch("loclean.cli.shell.loclean")
    def test_scrub_mode(self, mock_loclean: MagicMock) -> None:
        mock_loclean.scrub.return_value = "Contact [PERSON] at [EMAIL]"
        state = ShellState(mode=MODE_SCRUB)
        result = execute("Contact John at john@example.com", state)
        mock_loclean.scrub.assert_called_once()
        assert "[PERSON]" in result

    @patch("loclean.cli.shell.loclean")
    def test_model_override_forwarded(self, mock_loclean: MagicMock) -> None:
        mock_loclean.clean.return_value = "result"
        state = ShellState(mode=MODE_CLEAN, model="llama3")
        execute("test", state)
        _, kwargs = mock_loclean.clean.call_args
        assert kwargs["model"] == "llama3"


class TestRender:
    """Tests for render functions (smoke tests)."""

    def test_render_clean(self) -> None:
        import polars as pl

        df = pl.DataFrame(
            {
                "input": ["5kg"],
                "clean_value": ["5"],
                "clean_unit": ["kg"],
                "clean_reasoning": ["parsed"],
            }
        )
        render(df, MODE_CLEAN, _make_console())

    def test_render_extract_dict(self) -> None:
        render({"name": "shirt", "price": 50}, MODE_EXTRACT, _make_console())

    def test_render_extract_pydantic(self) -> None:
        class Item(BaseModel):
            name: str
            price: int

        render(Item(name="shirt", price=50), MODE_EXTRACT, _make_console())

    def test_render_scrub(self) -> None:
        render("Contact [PERSON] at [EMAIL]", MODE_SCRUB, _make_console())
