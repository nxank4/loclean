"""Test cases for LLM-based PII detector."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from loclean.cache import LocleanCache
from loclean.privacy.llm_detector import LLMDetector


@pytest.fixture
def mock_inference_engine() -> MagicMock:
    """Create a mock inference engine with LLM access."""
    engine = MagicMock()
    engine.llm = MagicMock()
    engine.adapter = MagicMock()
    engine.adapter.format.return_value = "formatted prompt"
    engine.adapter.get_stop_tokens.return_value = ["</s>", "\n"]
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

    @patch("loclean.privacy.llm_detector.load_template")
    @patch("loclean.utils.resources.load_grammar")
    def test_grammar_and_template_loading(
        self,
        mock_load_grammar: Mock,
        mock_load_template: Mock,
        mock_inference_engine: MagicMock,
    ) -> None:
        """Test grammar and template loading."""
        mock_load_grammar.return_value = "grammar content"
        mock_load_template.return_value = "template content"

        detector = LLMDetector(inference_engine=mock_inference_engine)

        mock_load_grammar.assert_called_once_with("pii_detection.gbnf")
        mock_load_template.assert_called_once_with("pii_detection.j2")
        assert detector.grammar_str == "grammar content"
        assert detector.template_str == "template content"


class TestDetectBatch:
    """Test cases for detect_batch method."""

    def test_detection_with_llm_strategies_person(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with LLM strategies (person)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        # Mock LLM output
        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found person name",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].type == "person"

    def test_detection_with_llm_strategies_address(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with LLM strategies (address)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "address",
                                    "value": "123 Main St",
                                    "start": 0,
                                    "end": 11,
                                }
                            ],
                            "reasoning": "Found address",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

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
        mock_inference_engine.llm.create_completion.assert_not_called()

    def test_detection_with_mixed_strategies(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test detection with mixed strategies."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found person",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector.detect_batch(["Contact John"], ["person", "email"])

        # Should only process person (LLM strategy), not email
        assert len(results) == 1

    def test_cache_hit_scenario_all_items_cached(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache hit scenario (all items cached)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        # Pre-populate cache
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
        mock_inference_engine.llm.create_completion.assert_not_called()

    def test_cache_miss_scenario_all_items_need_inference(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache miss scenario (all items need inference)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Detected",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        mock_inference_engine.llm.create_completion.assert_called()

    def test_partial_cache_hits(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test partial cache hits."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        # Cache one item
        cached_data = {
            "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
            "reasoning": "Cached",
        }
        mock_cache.set_batch(
            ["Contact John"], "Detect person", {"Contact John": cached_data}
        )

        # Mock LLM for second item
        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "Alice",
                                    "start": 0,
                                    "end": 5,
                                }
                            ],
                            "reasoning": "Detected",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector.detect_batch(["Contact John", "Contact Alice"], ["person"])

        assert len(results) == 2
        assert results[0].reasoning == "Cached"
        assert results[1].reasoning == "Detected"
        # Should only call LLM once (for Alice)
        assert mock_inference_engine.llm.create_completion.call_count == 1

    def test_cache_result_parsing_valid_json(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache result parsing (valid JSON)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        cached_data = {
            "entities": [{"type": "person", "value": "John", "start": 0, "end": 4}],
            "reasoning": "Valid cached",
        }
        mock_cache.set_batch(
            ["Contact John"], "Detect person", {"Contact John": cached_data}
        )

        results = detector.detect_batch(["Contact John"], ["person"])

        assert len(results) == 1
        assert results[0].reasoning == "Valid cached"
        assert len(results[0].entities) == 1

    def test_cache_result_parsing_invalid_json_fallback_to_empty(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test cache result parsing (invalid JSON - fallback to empty)."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        # Set invalid cached data
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

        mock_outputs = [
            {
                "choices": [
                    {
                        "text": json.dumps(
                            {
                                "entities": [
                                    {
                                        "type": "person",
                                        "value": "John",
                                        "start": 0,
                                        "end": 4,
                                    }
                                ],
                                "reasoning": "Found John",
                            }
                        )
                    }
                ]
            },
            {
                "choices": [
                    {
                        "text": json.dumps(
                            {
                                "entities": [
                                    {
                                        "type": "person",
                                        "value": "Alice",
                                        "start": 0,
                                        "end": 5,
                                    }
                                ],
                                "reasoning": "Found Alice",
                            }
                        )
                    }
                ]
            },
        ]
        mock_inference_engine.llm.create_completion.side_effect = mock_outputs

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

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Detected",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        detector.detect_batch(["Contact John"], ["person"])

        mock_log_cache.assert_called_once()


class TestDetectWithLLM:
    """Test cases for _detect_with_llm method."""

    def test_llm_detection_with_valid_output(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with valid output."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found person",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].type == "person"

    def test_llm_detection_with_json_decode_error(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with JSON decode error."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {"choices": [{"text": "invalid json"}]}
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_llm_detection_with_inference_exception(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with inference exception."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.llm.create_completion.side_effect = Exception(
            "Inference failed"
        )

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_llm_detection_with_empty_output(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with empty output."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {"choices": [{"text": ""}]}
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_llm_detection_with_dict_output_format(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with dict output format."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_llm_detection_with_iterator_output_format(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test LLM detection with iterator output format."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        # Simulate iterator output
        mock_output = iter(
            [
                {
                    "choices": [
                        {
                            "text": json.dumps(
                                {
                                    "entities": [
                                        {
                                            "type": "person",
                                            "value": "John",
                                            "start": 0,
                                            "end": 4,
                                        }
                                    ],
                                    "reasoning": "Found",
                                }
                            )
                        }
                    ]
                }
            ]
        )
        mock_inference_engine.llm.create_completion.return_value = mock_output

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_fallback_when_inference_engine_lacks_llm_access(
        self, mock_cache: LocleanCache
    ) -> None:
        """Test fallback when inference engine lacks LLM access."""
        # Create engine without llm attribute
        engine = MagicMock()
        del engine.llm

        detector = LLMDetector(inference_engine=engine, cache=mock_cache)

        results = detector._detect_with_llm(["Contact John"], ["person"])

        assert len(results) == 1
        assert len(results[0].entities) == 0

    @patch("llama_cpp.LlamaGrammar")
    def test_grammar_generation_from_piidetectionresult_schema(
        self,
        mock_grammar: Mock,
        mock_inference_engine: MagicMock,
        mock_cache: LocleanCache,
    ) -> None:
        """Test grammar generation from PIIDetectionResult schema."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_grammar_instance = MagicMock()
        mock_grammar.from_json_schema.return_value = mock_grammar_instance

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        detector._detect_with_llm(["Contact John"], ["person"])

        mock_grammar.from_json_schema.assert_called_once()

    def test_prompt_formatting_with_adapter(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test prompt formatting with adapter."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        detector._detect_with_llm(["Contact John"], ["person"])

        mock_inference_engine.adapter.format.assert_called()
        mock_inference_engine.adapter.get_stop_tokens.assert_called()

    def test_stop_tokens_usage(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test stop tokens usage."""
        detector = LLMDetector(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_output = {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "entities": [
                                {
                                    "type": "person",
                                    "value": "John",
                                    "start": 0,
                                    "end": 4,
                                }
                            ],
                            "reasoning": "Found",
                        }
                    )
                }
            ]
        }
        mock_inference_engine.llm.create_completion.return_value = mock_output

        detector._detect_with_llm(["Contact John"], ["person"])

        call_kwargs = mock_inference_engine.llm.create_completion.call_args[1]
        assert "stop" in call_kwargs
        assert call_kwargs["stop"] == ["</s>", "\n"]
