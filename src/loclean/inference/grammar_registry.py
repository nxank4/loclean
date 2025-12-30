"""Grammar registry for managing GBNF grammars with caching.

This module provides a centralized registry for GBNF grammars used in structured
output generation. It supports both preset string-based grammars and dynamic
Pydantic model-based grammars, with LRU caching for performance optimization.
"""

import json
from functools import lru_cache
from typing import Type, Union

from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]
from pydantic import BaseModel


class GrammarRegistry:
    """
    Registry for managing GBNF grammars with LRU caching.

    This class provides a centralized way to get compiled LlamaGrammar objects
    from either preset string keys or Pydantic model classes. Compiled grammars
    are cached to avoid redundant compilation overhead.

    Attributes:
        PRESETS: Dictionary mapping preset names to GBNF grammar strings.
                 These are hand-written grammars for specific use cases that
                 may be difficult to generate from JSON Schema.

    Examples:
        >>> # Use preset grammar
        >>> grammar = GrammarRegistry.get("json")
        >>> isinstance(grammar, LlamaGrammar)
        True

        >>> # Use Pydantic model
        >>> from loclean.inference.schemas import ExtractionResult
        >>> grammar = GrammarRegistry.get(ExtractionResult)
        >>> isinstance(grammar, LlamaGrammar)
        True
    """

    # Preset grammars for common use cases
    # These are hand-written GBNF strings for cases where JSON Schema
    # conversion might be insufficient or overly complex
    PRESETS: dict[str, str] = {
        "json": (
            "root   ::= object\n"
            'object ::= "{" ws "\\"reasoning\\"" ws ":" ws string "," ws '
            '"\\"value\\"" ws ":" ws number "," ws "\\"unit\\"" ws ":" ws '
            'string ws "}"\n'
            'number ::= ("-"? ([0-9]+ ("." [0-9]+)?))\n'
            'string ::= "\\"" ([^"]*) "\\""\n'
            "ws     ::= [ \\t\\n]*\n"
        ),
        "list_str": (
            "root   ::= array\n"
            'array  ::= "[" ws ( string ( "," ws string )* )? "]" ws\n'
            'string ::= "\\"" ([^"]*) "\\""\n'
            "ws     ::= [ \\t\\n]*\n"
        ),
        "email": (
            "root   ::= string\n"
            'string ::= "\\"" email_pattern "\\""\n'
            'email_pattern ::= [a-zA-Z0-9._%+-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,}\n'
            "ws     ::= [ \\t\\n]*\n"
        ),
    }

    @classmethod
    @lru_cache(maxsize=32)
    def get(cls, schema: Union[str, Type[BaseModel]]) -> LlamaGrammar:
        """
        Get a compiled LlamaGrammar object for the given schema.

        This method supports two input types:
        1. String preset key (e.g., "json", "email", "list_str")
        2. Pydantic BaseModel subclass (dynamic grammar generation)

        Compiled grammars are cached using LRU cache to avoid redundant
        compilation overhead when processing batches.

        Args:
            schema: Either a preset string key or a Pydantic BaseModel class.

        Returns:
            Compiled LlamaGrammar object ready for use in inference.

        Raises:
            ValueError: If schema is a string that doesn't match any preset.
            TypeError: If schema is neither a string nor a Pydantic BaseModel.

        Examples:
            >>> # Use preset
            >>> grammar = GrammarRegistry.get("json")

            >>> # Use Pydantic model
            >>> from loclean.inference.schemas import ExtractionResult
            >>> grammar = GrammarRegistry.get(ExtractionResult)
        """
        # Case 1: String preset key
        if isinstance(schema, str):
            if schema not in cls.PRESETS:
                raise ValueError(
                    f"Unknown grammar preset: {schema}. "
                    f"Available presets: {list(cls.PRESETS.keys())}"
                )
            return LlamaGrammar.from_string(cls.PRESETS[schema])

        # Case 2: Pydantic BaseModel class
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # llama-cpp-python handles JSON Schema -> GBNF conversion internally
            # Convert dict to JSON string as required by from_json_schema
            json_schema_dict = schema.model_json_schema()
            json_schema_str = json.dumps(json_schema_dict)
            return LlamaGrammar.from_json_schema(json_schema_str)

        raise TypeError(
            f"Schema must be a preset string or Pydantic BaseModel class, "
            f"got {type(schema).__name__}"
        )

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the LRU cache for compiled grammars.

        Useful for testing or when memory needs to be freed.
        """
        cls.get.cache_clear()
