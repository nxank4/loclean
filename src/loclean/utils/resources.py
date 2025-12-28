"""Resource loading utilities.

This module provides safe resource loading using importlib.resources
for compatibility with zip-apps and PyInstaller binaries.
"""

import importlib.resources
from pathlib import Path


def load_grammar(filename: str) -> str:
    """
    Load a GBNF grammar file from the resources/grammars directory.

    Args:
        filename: Name of the grammar file (e.g., "json.gbnf")

    Returns:
        Grammar content as string

    Raises:
        FileNotFoundError: If the grammar file doesn't exist

    Example:
        >>> grammar_str = load_grammar("json.gbnf")
        >>> "root" in grammar_str
        True
    """
    try:
        package = importlib.resources.files("loclean.resources.grammars")
        grammar_file = package / filename

        if not grammar_file.is_file():
            raise FileNotFoundError(
                f"Grammar file '{filename}' not found in resources/grammars"
            )

        return grammar_file.read_text(encoding="utf-8")
    except ModuleNotFoundError:
        source_path = (
            Path(__file__).parent.parent.parent / "resources" / "grammars" / filename
        )
        if source_path.exists():
            return source_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Grammar file '{filename}' not found") from None


def load_template(filename: str) -> str:
    """
    Load a Jinja2 template file from the resources/templates directory.

    Args:
        filename: Name of the template file (e.g., "phi3.j2")

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If the template file doesn't exist

    Example:
        >>> template_str = load_template("phi3.j2")
        >>> "{{ instruction }}" in template_str
        True
    """
    try:
        package = importlib.resources.files("loclean.resources.templates")
        template_file = package / filename

        if not template_file.is_file():
            raise FileNotFoundError(
                f"Template file '{filename}' not found in resources/templates"
            )

        return template_file.read_text(encoding="utf-8")
    except ModuleNotFoundError:
        source_path = (
            Path(__file__).parent.parent.parent / "resources" / "templates" / filename
        )
        if source_path.exists():
            return source_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Template file '{filename}' not found") from None


def list_grammars() -> list[str]:
    """
    List all available grammar files.

    Returns:
        List of grammar filenames
    """
    try:
        package = importlib.resources.files("loclean.resources.grammars")
        return [
            f.name
            for f in package.iterdir()
            if f.is_file() and f.name.endswith(".gbnf")
        ]
    except ModuleNotFoundError:
        source_path = Path(__file__).parent.parent.parent / "resources" / "grammars"
        if source_path.exists():
            return [f.name for f in source_path.iterdir() if f.suffix == ".gbnf"]
        return []


def list_templates() -> list[str]:
    """
    List all available template files.

    Returns:
        List of template filenames
    """
    try:
        package = importlib.resources.files("loclean.resources.templates")
        return [
            f.name for f in package.iterdir() if f.is_file() and f.name.endswith(".j2")
        ]
    except ModuleNotFoundError:
        source_path = Path(__file__).parent.parent.parent / "resources" / "templates"
        if source_path.exists():
            return [f.name for f in source_path.iterdir() if f.suffix == ".j2"]
        return []
