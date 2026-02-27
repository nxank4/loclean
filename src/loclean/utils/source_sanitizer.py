"""Deterministic source-code sanitizer for LLM-generated Python.

Small models (phi3, etc.) frequently produce output with markdown
fences, prose preambles, non-ASCII operators, and invalid numeric
literals.  This module fixes those issues mechanically — no LLM calls
required — before the code reaches ``compile_sandboxed``.
"""

from __future__ import annotations

import re


def sanitize_source(source: str) -> str:
    """Clean up common LLM output artifacts from Python source code.

    Applies a sequence of deterministic transformations:

    1. Strip markdown code fences (````python`` / `````)
    2. Remove prose before the first ``import`` / ``def`` / ``from``
    3. Remove trailing prose after the last function body
    4. Replace non-ASCII mathematical operators
    5. Fix invalid numeric literals
    6. Strip stray inline backticks

    Args:
        source: Raw LLM-generated Python source.

    Returns:
        Cleaned source code ready for ``compile_sandboxed``.
    """
    source = _strip_markdown_fences(source)
    source = _strip_prose(source)
    source = _fix_unicode_operators(source)
    source = _fix_numeric_literals(source)
    source = _strip_backticks(source)
    return source


def _strip_markdown_fences(source: str) -> str:
    """Remove markdown code fences wrapping the code block."""
    lines = source.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _strip_prose(source: str) -> str:
    """Remove explanatory text before/after the actual code.

    Keeps lines starting from the first ``import``, ``from``, or
    ``def`` statement through the end of the last indented block.
    """
    lines = source.split("\n")

    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ")):
            start_idx = i
            break

    end_idx = len(lines)
    for i in range(len(lines) - 1, start_idx - 1, -1):
        stripped = lines[i].strip()
        if stripped and not _is_prose_line(stripped):
            end_idx = i + 1
            break

    return "\n".join(lines[start_idx:end_idx])


def _is_prose_line(line: str) -> bool:
    """Heuristic: a line is 'prose' if it looks like natural language."""
    if not line:
        return False
    if line.startswith(("#", "import ", "from ", "def ", "class ", "return ")):
        return False
    if line[0] in (" ", "\t", "@"):
        return False
    words = line.split()
    if len(words) >= 4 and not any(c in line for c in ("=", "(", ")", "[", "]", ":")):
        return True
    return False


_UNICODE_MAP: dict[str, str] = {
    "\u00d7": "*",  # ×
    "\u00f7": "/",  # ÷
    "\u2212": "-",  # −  (minus sign)
    "\u2013": "-",  # –  (en dash)
    "\u2014": "-",  # —  (em dash)
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
}


def _fix_unicode_operators(source: str) -> str:
    """Replace non-ASCII mathematical and typographic characters."""
    for char, replacement in _UNICODE_MAP.items():
        source = source.replace(char, replacement)
    return source


def _fix_numeric_literals(source: str) -> str:
    """Fix invalid numeric literals commonly produced by small models.

    Patterns handled:
    - ``0b2``, ``0b3`` etc. (invalid binary digits) → decimal
    - Trailing currency/unit symbols on numbers (``100$``, ``50€``)
    """
    source = re.sub(
        r"\b0b([2-9]\d*)\b",
        lambda m: m.group(1),
        source,
    )
    source = re.sub(
        r"(\d+\.?\d*)[€$£%]+",
        r"\1",
        source,
    )
    return source


def _strip_backticks(source: str) -> str:
    """Remove stray inline backticks wrapping expressions."""
    source = re.sub(r"`([^`\n]+)`", r"\1", source)
    return source
