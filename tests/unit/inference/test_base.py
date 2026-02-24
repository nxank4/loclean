"""Test cases for InferenceEngine abstract base class."""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from loclean.inference.base import InferenceEngine


class TestInferenceEngine:
    """Test cases for InferenceEngine abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that InferenceEngine cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            InferenceEngine()  # type: ignore[abstract]

        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower()

    def test_subclass_without_implementation_raises_error(self) -> None:
        """Subclass missing abstract methods cannot be instantiated."""

        class IncompleteEngine(InferenceEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]

    def test_subclass_missing_generate_raises_error(self) -> None:
        """Subclass implementing only clean_batch still cannot be instantiated."""

        class PartialEngine(InferenceEngine):
            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {}

        with pytest.raises(TypeError):
            PartialEngine()  # type: ignore[abstract]

    def test_subclass_with_full_implementation(self) -> None:
        """Subclass implementing both abstract methods can be instantiated."""

        class MockEngine(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return '{"value": 1}'

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {
                    item: {"reasoning": "test", "value": 1.0, "unit": "kg"}
                    for item in items
                }

        engine = MockEngine()
        assert isinstance(engine, InferenceEngine)

    def test_generate_method_signature(self) -> None:
        """generate() accepts prompt and optional schema."""

        class MockEngine(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return '{"result": "test"}'

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {}

        engine = MockEngine()
        result = engine.generate("test prompt")
        assert isinstance(result, str)

    def test_clean_batch_return_type(self) -> None:
        """clean_batch() returns Dict[str, Optional[Dict]]."""

        class MockEngine(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return ""

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {
                    "test": {"reasoning": "r", "value": 5.5, "unit": "kg"},
                    "failed": None,
                }

        engine = MockEngine()
        result = engine.clean_batch(["test", "failed"], "Extract")
        assert isinstance(result, dict)
        assert result["test"] == {"reasoning": "r", "value": 5.5, "unit": "kg"}
        assert result["failed"] is None

    def test_isinstance_check(self) -> None:
        """Subclasses pass isinstance check."""

        class MockEngine(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return ""

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {}

        engine = MockEngine()
        assert isinstance(engine, InferenceEngine)
        assert isinstance(engine, MockEngine)

    def test_multiple_subclasses_coexist(self) -> None:
        """Multiple subclasses can coexist."""

        class EngineA(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return "A"

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {"a": {"reasoning": "A", "value": 1.0, "unit": "kg"}}

        class EngineB(InferenceEngine):
            def generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> str:
                return "B"

            def clean_batch(
                self, items: List[str], instruction: str
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                return {"b": {"reasoning": "B", "value": 2.0, "unit": "lb"}}

        engine_a = EngineA()
        engine_b = EngineB()

        assert isinstance(engine_a, InferenceEngine)
        assert isinstance(engine_b, InferenceEngine)
        assert not isinstance(engine_a, EngineB)
