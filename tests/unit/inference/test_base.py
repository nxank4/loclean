"""Test cases for InferenceEngine abstract base class."""

from typing import Any, Dict, List, Optional

import pytest

from loclean.inference.base import InferenceEngine


class TestInferenceEngine:
    """Test cases for InferenceEngine abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that InferenceEngine cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            InferenceEngine()  # type: ignore[abstract]

        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower() or "clean_batch" in error_msg.lower()

    def test_subclass_without_implementation_raises_error(self) -> None:
        """Test that subclass without clean_batch implementation
        cannot be instantiated."""

        class IncompleteEngine(InferenceEngine):
            """Subclass that doesn't implement clean_batch."""

            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteEngine()  # type: ignore[abstract]

        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower() or "clean_batch" in error_msg.lower()

    def test_subclass_with_implementation_can_be_instantiated(self) -> None:
        """Test that subclass with clean_batch implementation can be instantiated."""

        class MockEngine(InferenceEngine):
            """Mock engine that implements clean_batch."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation of clean_batch."""
                return {
                    item: {"reasoning": "test", "value": 1.0, "unit": "kg"}
                    for item in items
                }

        # Should not raise TypeError
        engine = MockEngine()
        assert isinstance(engine, InferenceEngine)
        assert isinstance(engine, MockEngine)

    def test_clean_batch_signature(self) -> None:
        """Test that clean_batch has correct signature."""

        class MockEngine(InferenceEngine):
            """Mock engine for testing signature."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation."""
                return {}

        engine = MockEngine()
        # Verify method exists and is callable
        assert hasattr(engine, "clean_batch")
        assert callable(engine.clean_batch)

    def test_clean_batch_return_type(self) -> None:
        """Test that clean_batch returns correct type."""

        class MockEngine(InferenceEngine):
            """Mock engine for testing return type."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation returning correct type."""
                return {
                    "test": {"reasoning": "test reasoning", "value": 5.5, "unit": "kg"},
                    "failed": None,
                }

        engine = MockEngine()
        result = engine.clean_batch(["test", "failed"], "Extract value and unit")
        assert isinstance(result, dict)
        assert "test" in result
        assert "failed" in result
        assert result["test"] == {
            "reasoning": "test reasoning",
            "value": 5.5,
            "unit": "kg",
        }
        assert result["failed"] is None

    def test_clean_batch_with_empty_items(self) -> None:
        """Test that clean_batch handles empty items list."""

        class MockEngine(InferenceEngine):
            """Mock engine for testing empty items."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation."""
                return {}

        engine = MockEngine()
        result = engine.clean_batch([], "Extract value and unit")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_clean_batch_with_multiple_items(self) -> None:
        """Test that clean_batch processes multiple items correctly."""

        class MockEngine(InferenceEngine):
            """Mock engine for testing multiple items."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation."""
                return {
                    item: {
                        "reasoning": f"Extracted from {item}",
                        "value": float(len(item)),
                        "unit": "kg",
                    }
                    for item in items
                }

        engine = MockEngine()
        items = ["10kg", "5.5lb", "20g"]
        result = engine.clean_batch(items, "Extract value and unit")
        assert len(result) == 3
        for item in items:
            assert item in result
            item_result = result[item]
            assert item_result is not None
            if item_result is not None:
                assert "reasoning" in item_result
                assert "value" in item_result
                assert "unit" in item_result

    def test_clean_batch_with_none_results(self) -> None:
        """Test that clean_batch can return None for failed extractions."""

        class MockEngine(InferenceEngine):
            """Mock engine that returns None for some items."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation returning None for invalid items."""
                result: Dict[str, Optional[Dict[str, Any]]] = {}
                for item in items:
                    if "invalid" in item.lower():
                        result[item] = None
                    else:
                        result[item] = {"reasoning": "ok", "value": 1.0, "unit": "kg"}
                return result

        engine = MockEngine()
        items = ["10kg", "invalid_input", "5.5lb"]
        result = engine.clean_batch(items, "Extract value and unit")
        assert result["10kg"] is not None
        assert result["invalid_input"] is None
        assert result["5.5lb"] is not None

    def test_clean_batch_instruction_parameter(self) -> None:
        """Test that clean_batch receives and can use instruction parameter."""

        class MockEngine(InferenceEngine):
            """Mock engine that uses instruction parameter."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation that uses instruction."""
                return {
                    item: {
                        "reasoning": f"Following instruction: {instruction}",
                        "value": 1.0,
                        "unit": "kg",
                    }
                    for item in items
                }

        engine = MockEngine()
        instruction = "Convert to kilograms"
        result = engine.clean_batch(["10kg"], instruction)
        assert result["10kg"] is not None
        assert (
            "Following instruction: Convert to kilograms" in result["10kg"]["reasoning"]
        )

    def test_isinstance_check(self) -> None:
        """Test that subclasses pass isinstance check."""

        class MockEngine(InferenceEngine):
            """Mock engine for isinstance testing."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation."""
                return {}

        engine = MockEngine()
        assert isinstance(engine, InferenceEngine)
        assert isinstance(engine, MockEngine)

    def test_multiple_subclasses(self) -> None:
        """Test that multiple subclasses can coexist."""

        class EngineA(InferenceEngine):
            """First mock engine."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation A."""
                return {"a": {"reasoning": "A", "value": 1.0, "unit": "kg"}}

        class EngineB(InferenceEngine):
            """Second mock engine."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation B."""
                return {"b": {"reasoning": "B", "value": 2.0, "unit": "lb"}}

        engine_a = EngineA()
        engine_b = EngineB()

        assert isinstance(engine_a, InferenceEngine)
        assert isinstance(engine_b, InferenceEngine)
        assert not isinstance(engine_a, EngineB)
        assert not isinstance(engine_b, EngineA)

    def test_clean_batch_result_structure(self) -> None:
        """Test that clean_batch returns results with expected structure."""

        class MockEngine(InferenceEngine):
            """Mock engine for testing result structure."""

            def clean_batch(
                self,
                items: List[str],
                instruction: str,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
                """Mock implementation with proper structure."""
                return {
                    "10kg": {
                        "reasoning": "Extracted 10kg",
                        "value": 10.0,
                        "unit": "kg",
                    }
                }

        engine = MockEngine()
        result = engine.clean_batch(["10kg"], "Extract value and unit")
        assert "10kg" in result
        assert result["10kg"] is not None
        assert result["10kg"]["reasoning"] == "Extracted 10kg"
        assert result["10kg"]["value"] == 10.0
        assert result["10kg"]["unit"] == "kg"
