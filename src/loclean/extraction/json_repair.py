"""JSON repair utilities for fixing malformed JSON output from LLMs.

This module provides a wrapper around the json-repair library to automatically
fix common JSON formatting issues such as missing brackets, trailing commas,
and unclosed strings.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import json_repair

    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    logger.warning(
        "json-repair not available. JSON repair functionality will be limited. "
        "Install with: pip install json-repair"
    )


def repair_json(text: str) -> str:
    """
    Attempt to repair malformed JSON text.

    This function uses the json-repair library to fix common JSON issues:
    - Missing brackets or braces
    - Trailing commas
    - Unclosed strings
    - Invalid escape sequences
    - Other common formatting errors

    If json-repair is not available or repair fails, the original text is returned.
    Pydantic validation will handle any remaining errors.

    Args:
        text: Potentially malformed JSON string to repair.

    Returns:
        Repaired JSON string, or original text if repair fails or json-repair
        is not available.

    Example:
        >>> repair_json('{"name": "test",}')  # Trailing comma
        '{"name": "test"}'
        >>> repair_json('{"name": "test"')  # Missing closing brace
        '{"name": "test"}'
    """
    if not HAS_JSON_REPAIR:
        logger.debug("json-repair not available, returning original text")
        return text

    try:
        repaired = json_repair.repair_json(text)
        if repaired != text:
            logger.debug("Successfully repaired JSON")
        return repaired
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}. Returning original text.")
        return text
