"""Grammar utilities for converting Pydantic schemas to GBNF grammars.

This module leverages llama-cpp-python's built-in JSON Schema support to convert
Pydantic models to GBNF grammars, avoiding the need for manual parsing.
"""

import functools
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def get_grammar_from_schema(schema: type[BaseModel]) -> "LlamaGrammar":
    """
    Convert a Pydantic schema to a GBNF grammar using JSON Schema.

    This function leverages Pydantic's `model_json_schema()` to generate a JSON
    Schema, then uses llama-cpp-python's `LlamaGrammar.from_json_schema()` to
    convert it to GBNF. This approach automatically supports all Pydantic features
    including nested models, Optional types, List types, Union types, Literal types,
    Annotated types, and more.

    The result is cached using LRU cache to avoid recompiling the same grammar
    multiple times.

    Args:
        schema: Pydantic BaseModel class to convert to GBNF grammar.

    Returns:
        LlamaGrammar instance that enforces the schema structure.

    Raises:
        ValueError: If schema is not a Pydantic BaseModel subclass.
        ImportError: If llama_cpp is not available.

    Example:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: int
        >>> grammar = get_grammar_from_schema(Product)
        >>> # Grammar can now be used with LlamaCppEngine
    """
    if not issubclass(schema, BaseModel):
        raise ValueError(
            f"Schema must be a Pydantic BaseModel subclass, got {type(schema)}"
        )

    try:
        from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is required for grammar generation. "
            "Install it with: pip install llama-cpp-python"
        ) from e

    # Get JSON Schema from Pydantic
    json_schema = schema.model_json_schema()

    # Convert JSON Schema to GBNF grammar using llama-cpp-python
    # LlamaGrammar.from_json_schema expects a JSON string, not a dict
    import json
    json_schema_str = json.dumps(json_schema)
    grammar = LlamaGrammar.from_json_schema(json_schema_str)

    logger.debug(f"Generated GBNF grammar for schema: {schema.__name__}")
    return grammar


def _get_schema_key(schema: type[BaseModel]) -> str:
    """
    Generate a cache key for a schema.

    Used internally by LRU cache to identify unique schemas.
    The key is based on schema name and sorted field names.

    Args:
        schema: Pydantic BaseModel class.

    Returns:
        String key representing the schema.
    """
    field_names = sorted(schema.model_fields.keys())
    return f"{schema.__name__}::{','.join(field_names)}"
