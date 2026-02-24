"""Test cases for LLM-based PII detector."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from loclean.cache import LocleanCache
from loclean.privacy.llm_detector import LLMDetector


@pytest.fixture
def mock_inference_engine() -> MagicMock:
    """Create a mock inference engine with generate method."""
    engine = MagicMock()
    return engine


@pytest.fixture
def mock_cache(tmp_path: Path) -> LocleanCache:
    """Create a temporary cache for testing."""
    return LocleanCache(cache_dir=tmp_path)


class TestLLMDetectorInitialization:
    """Test cases for LLMDetector initialization."""

    def test_initialization_with_inference_engine(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test initialization with inference engine."""
        detector = LLMDetector(inference_engine=mock_inference_engine)

        assert detector.inference_engine is mock_inference_engine
        assert detector.cache is not None
        assert isinstance(detector.cache, LocleanCache)

    def test_initialization_with_custom_cache(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test initialization with custom cache."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        assert detector.cache is mock_cache

    def test_initialization_without_cache_default_locleancache(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test initialization without cache (default LocleanCache)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=None)

        assert detector.cache is not None
        assert isinstance(detector.cache, LocleanCache)


class TestDetectBatch:
    """Test cases for detect_batch method."""

    def test_detection_with_llm_strategies_person(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with LLM strategies (person)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Found person name",
            }
        )

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].type == "person"

    def test_detection_with_llm_strategies_address(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with LLM strategies (address)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [
                    {"type": "address", "value": "123 Main St", "start": 0, "end": 11}
                ],
                "reasoning": "Found address",
            }
        )

        results = detector.detect_batch(["123 Main St"], ["address"])

        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].type == "address"

    def test_detection_with_non_llm_strategies_returns_empty(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with non-LLM strategies (returns empty)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        results = detector.detect_batch(["test@example.com"], ["email", "phone"])

        assert len(results) == 1
        assert len(results[0].entities) == 0
        mock_inference_engine.generate.assert_not_called()

    def test_detection_with_mixed_strategies(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with mixed strategies."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Found person",
            }
        )

        results = detector.detect_batch(["Contact John"], ["person", "email"])

        assert len(results) == 1

    def test_cache_hit_scenario_all_items_cached(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache hit scenario (all items cached)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        cached_data = {
            "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
            "reasoning": "Cached result",
        }
        mock_cache.set_batch(
            ["Contact John"], "Detect person", {"Contact John": cached_data}
        )

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        assert results[0].reasoning == "Cached result"
        mock_inference_engine.generate.assert_not_called()

    def test_cache_miss_scenario_all_items_need_inference(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache miss scenario (all items need inference)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Detected",
            }
        )

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        mock_inference_engine.generate.assert_called()

    def test_partial_cache_hits(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test partial cache hits."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        cached_data = {
            "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
            "reasoning": "Cached",
        }
        mock_cache.set_batch(
            ["Contact John"], "Detect person", {"Contact John": cached_data}
        )

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [
                    {"type": "person", "value": "Alice", "start": 0, "end": 5}
                ],
                "reasoning": "Detected",
            }
        )

        results = detector.detect_batch(["Contact John", "Contact Alice"], ["person"])

        assert len(results) == 2
        assert results[0].reasoning == "Cached"
        assert results[1].reasoning == "Detected"
        assert mock_inference_engine.generate.call_count == 1

    def test_cache_result_parsing_invalid_json_fallback_to_empty(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache result parsing (invalid JSON - fallback to empty)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_cache.set_batch(
            ["Contact John"], "Detect person", {"Contact John": {"invalid": "data"}}
        )

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_batch_processing_with_multiple_items(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test batch processing with multiple items."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.side_effect = [
            json.dumps(
                {
                    "entities": [
                        {"type": "person", "value": "John", "start": 0, "end": 4}
                    ],
                    "reasoning": "Found John",
                }
            ),
            json.dumps(
                {
                    "entities": [
                        {"type": "person", "value": "Alice", "start": 0, "end": 5}
                    ],
                    "reasoning": "Found Alice",
                }
            ),
        ]

        results = detector.detect_batch(["Contact John", "Contact Alice"], ["person"])

        assert len(results) == 2
        assert results[0].entities[0].value == "John"
        assert results[1].entities[0].value == "Alice"

    def test_empty_items_list(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test empty items list."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        results = detector.detect_batch([], ["person"])

        assert len(results) == 0

    @patch("loclean.utils.rich_output.log_cache_stats")
    def test_rich_cache_statistics_logging(
        self,
        mock_log_cache: Mock,
        mock_inference_engine: MagicMock,
        mock_cache: LocleanCache,
    ) -> None:
        """Test Rich cache statistics logging."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Detected",
            }
        )

        detector.detect_batch(["Contact John"], ["person"])

        mock_log_cache.assert_called_once()


class TestDetectWithLLM:
    """Test cases for _detect_with_llm method."""

    def test_llm_detection_with_valid_output(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with valid output."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Found person",
            }
        )

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].type == "person"

    def test_llm_detection_with_json_decode_error(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with JSON decode error."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = "invalid json"

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_llm_detection_with_inference_exception(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with inference exception."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.side_effect = Exception("Inference failed")

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_llm_detection_with_empty_output(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with empty output."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = ""

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_generate_called_with_piidetectionresult_schema(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test generate is called with PIIDetectionResult schema."""
        from loclean.privacy.schemas import PIIDetectionResult

        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {
                "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
                "reasoning": "Found",
            }
        )

        detector._detect_with_llm(["Contact John"], ["person"])

        call_kwargs = mock_inference_engine.generate.call_args[1]
        assert call_kwargs["schema"] is PIIDetectionResult
