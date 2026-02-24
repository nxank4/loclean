"""Interactive shell for loclean.

Provides a continuous REPL that processes user text through the local
inference engine, with styled Rich output and multiline Prompt Toolkit input.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from pydantic import BaseModel, ValidationError, create_model
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import loclean
from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)

MODE_CLEAN = "clean"
MODE_EXTRACT = "extract"
MODE_SCRUB = "scrub"
VALID_MODES = {MODE_CLEAN, MODE_EXTRACT, MODE_SCRUB}

_TYPE_MAP: Dict[str, Type[Any]] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

HELP_TEXT = """\
[bold cyan]Loclean Interactive Shell[/bold cyan]

[bold]Slash Commands[/bold]
  /mode clean|extract|scrub   Switch operational mode
  /schema Field:type ...      Define inline Pydantic schema (extract mode)
                               Types: str, int, float, bool
  /instruction <text>         Set custom instruction
  /model <name>               Switch Ollama model
  /help                       Show this help
  /quit                       Exit the shell

[bold]Input[/bold]
  Type any text and press [bold]Enter[/bold] to process it.
  Press [bold]Ctrl+D[/bold] to exit.
"""


class ShellState:
    """Mutable session state for the interactive shell."""

    __slots__ = ("mode", "instruction", "schema", "model", "host", "verbose")

    def __init__(
        self,
        mode: str = MODE_CLEAN,
        instruction: str | None = None,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        host: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.mode = mode
        self.instruction = instruction
        self.schema = schema
        self.model = model
        self.host = host
        self.verbose = verbose


def _build_prompt(state: ShellState) -> HTML:
    """Build an HTML-formatted prompt string reflecting current mode."""
    mode_colors = {
        MODE_CLEAN: "ansigreen",
        MODE_EXTRACT: "ansicyan",
        MODE_SCRUB: "ansiyellow",
    }
    color = mode_colors.get(state.mode, "ansiwhite")
    return HTML(f"<{color}>[{state.mode}]</{color}> › ")


def parse_schema(tokens: List[str]) -> type[BaseModel]:
    """Parse ``Field:type`` tokens into a dynamic Pydantic model.

    Args:
        tokens: List of ``"name:type"`` strings (e.g. ``["name:str", "price:int"]``).

    Returns:
        A dynamically-created Pydantic BaseModel subclass.

    Raises:
        ValueError: For malformed tokens or unknown types.
    """
    fields: Dict[str, Any] = {}
    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid schema field '{token}'. "
                "Expected format: FieldName:type (e.g. name:str)"
            )
        name, type_str = token.split(":", 1)
        name = name.strip()
        type_str = type_str.strip().lower()
        if type_str not in _TYPE_MAP:
            raise ValueError(
                f"Unknown type '{type_str}' for field '{name}'. "
                f"Supported: {', '.join(_TYPE_MAP)}"
            )
        fields[name] = (_TYPE_MAP[type_str], ...)

    if not fields:
        raise ValueError("Schema must have at least one field.")

    model: type[BaseModel] = create_model("DynamicSchema", **fields)  # type: ignore[call-overload]
    return model


def handle_command(line: str, state: ShellState, console: Console) -> bool:
    """Process a slash command, mutating *state* as needed.

    Args:
        line: The raw input line starting with ``/``.
        state: Mutable session state.
        console: Rich console for feedback.

    Returns:
        ``True`` if the shell should exit, ``False`` to continue.
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        return True

    if cmd == "/help":
        console.print(HELP_TEXT)
        return False

    if cmd == "/mode":
        mode = arg.strip().lower()
        if mode not in VALID_MODES:
            console.print(
                f"[red]Unknown mode '{mode}'. "
                f"Choose from: {', '.join(sorted(VALID_MODES))}[/red]"
            )
        else:
            state.mode = mode
            console.print(f"[green]✓[/green] Mode set to [bold]{mode}[/bold]")
        return False

    if cmd == "/schema":
        tokens = arg.strip().split()
        if not tokens:
            console.print("[red]Usage: /schema Field:type ...[/red]")
            return False
        try:
            state.schema = parse_schema(tokens)
            field_names = list(state.schema.model_fields.keys())
            console.print(
                f"[green]✓[/green] Schema set: "
                f"[bold cyan]{', '.join(field_names)}[/bold cyan]"
            )
        except ValueError as exc:
            console.print(f"[red]✗[/red] {exc}")
        return False

    if cmd == "/instruction":
        if not arg.strip():
            state.instruction = None
            console.print("[green]✓[/green] Instruction cleared.")
        else:
            state.instruction = arg.strip()
            console.print(
                f"[green]✓[/green] Instruction set: [dim]{state.instruction[:80]}[/dim]"
            )
        return False

    if cmd == "/model":
        name = arg.strip()
        if not name:
            console.print("[red]Usage: /model <name>[/red]")
        else:
            state.model = name
            console.print(
                f"[green]✓[/green] Model set to [bold cyan]{name}[/bold cyan]"
            )
        return False

    console.print(f"[red]Unknown command '{cmd}'. Type /help for usage.[/red]")
    return False


def execute(text: str, state: ShellState) -> Any:
    """Route *text* to the appropriate loclean function.

    Args:
        text: Raw user input to process.
        state: Current session state (mode, schema, instruction, model).

    Returns:
        Result from the loclean API call.

    Raises:
        ConnectionError: Ollama daemon unreachable.
        RuntimeError: Model pull failure.
        ValidationError: Pydantic schema validation failure.
        ValueError: Missing schema in extract mode, or extraction failure.
    """
    engine_kwargs: Dict[str, Any] = {}
    if state.model is not None:
        engine_kwargs["model"] = state.model
    if state.host is not None:
        engine_kwargs["host"] = state.host
    if state.verbose:
        engine_kwargs["verbose"] = True

    if state.mode == MODE_CLEAN:
        import polars as pl

        df = pl.DataFrame({"input": [text]})
        clean_kwargs: Dict[str, Any] = {**engine_kwargs}
        if state.instruction is not None:
            clean_kwargs["instruction"] = state.instruction
        return loclean.clean(
            df,
            "input",
            **clean_kwargs,
        )

    if state.mode == MODE_EXTRACT:
        if state.schema is None:
            raise ValueError(
                "No schema defined. Use /schema to define one first. "
                "Example: /schema name:str price:int"
            )
        return loclean.extract(
            text,
            state.schema,
            instruction=state.instruction,
            **engine_kwargs,
        )

    if state.mode == MODE_SCRUB:
        return loclean.scrub(
            text,
            **engine_kwargs,
        )

    raise ValueError(f"Unknown mode: {state.mode}")


def render(result: Any, mode: str, console: Console) -> None:
    """Render the result to the terminal using Rich.

    Args:
        result: Return value from ``execute()``.
        mode: Active operational mode.
        console: Rich console instance.
    """
    if mode == MODE_CLEAN:
        _render_clean(result, console)
    elif mode == MODE_EXTRACT:
        _render_extract(result, console)
    elif mode == MODE_SCRUB:
        _render_scrub(result, console)
    else:
        console.print(result)


def _render_clean(result: Any, console: Console) -> None:
    """Render clean-mode result as a Rich table."""
    table = Table(
        title="Extraction Result",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Input", style="dim")
    table.add_column("Value", style="bold green")
    table.add_column("Unit", style="cyan")
    table.add_column("Reasoning", style="dim italic")

    try:
        import polars as pl

        if isinstance(result, pl.DataFrame):
            for row in result.iter_rows(named=True):
                table.add_row(
                    str(row.get("input", "")),
                    str(row.get("clean_value", "")),
                    str(row.get("clean_unit", "")),
                    str(row.get("clean_reasoning", "")),
                )
    except Exception:
        table.add_row(str(result), "", "", "")

    console.print(table)


def _render_extract(result: Any, console: Console) -> None:
    """Render extract-mode result as a Rich panel with key-value pairs."""
    if isinstance(result, BaseModel):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        console.print(result)
        return

    lines: List[str] = []
    for key, value in data.items():
        lines.append(f"[bold cyan]{key}[/bold cyan]: {value}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Extracted Data[/bold]",
            border_style="green",
        )
    )


def _render_scrub(result: Any, console: Console) -> None:
    """Render scrub-mode result as side-by-side original vs scrubbed."""
    if isinstance(result, str):
        original = Text("(see input above)", style="dim")
        scrubbed = Text(result, style="bold green")
        console.print(
            Columns(
                [
                    Panel(original, title="Original", border_style="red"),
                    Panel(scrubbed, title="Scrubbed", border_style="green"),
                ],
                equal=True,
            )
        )
    else:
        console.print(result)


def run_shell(
    model: Optional[str] = None,
    host: Optional[str] = None,
    mode: str = MODE_CLEAN,
    verbose: bool = False,
) -> None:
    """Start the interactive loclean shell.

    Args:
        model: Ollama model tag override.
        host: Ollama server URL override.
        mode: Initial operational mode.
        verbose: Enable detailed logging.
    """
    console = Console()
    state = ShellState(
        mode=mode,
        model=model,
        host=host,
        verbose=verbose,
    )

    console.print(
        Panel(
            "[bold white]Loclean Interactive Shell[/bold white]\n"
            "[dim]Type text to process, or /help for commands.[/dim]",
            border_style="bright_blue",
        )
    )

    session: PromptSession[str] = PromptSession()

    while True:
        try:
            text = session.prompt(_build_prompt(state))
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        text = text.strip()
        if not text:
            continue

        if text.startswith("/"):
            should_exit = handle_command(text, state, console)
            if should_exit:
                console.print("[dim]Goodbye![/dim]")
                break
            continue

        try:
            result = execute(text, state)
            render(result, state.mode, console)
        except ConnectionError as exc:
            console.print(
                f"[red]Connection Error:[/red] {exc}\n"
                "[dim]Is Ollama installed? Run: "
                "curl -fsSL https://ollama.com/install.sh | sh[/dim]"
            )
        except ValidationError as exc:
            console.print(
                f"[red]Schema Validation Error:[/red]\n{exc}\n"
                "[dim]Try adjusting your /schema or /instruction.[/dim]"
            )
        except json.JSONDecodeError as exc:
            console.print(
                f"[red]JSON Parse Error:[/red] {exc}\n"
                "[dim]The model returned malformed output. "
                "Try a different model or instruction.[/dim]"
            )
        except ValueError as exc:
            console.print(f"[yellow]⚠[/yellow] {exc}")
        except Exception as exc:
            console.print(f"[red]Unexpected Error:[/red] {type(exc).__name__}: {exc}")
