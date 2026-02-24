"""Test cases for Extractor class."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from loclean.cache import LocleanCache
from loclean.extraction.extractor import Extractor


class Product(BaseModel):
    """Test schema for extraction."""

    name: str
    price: int
    color: str


class SimpleSchema(BaseModel):
    """Simple test schema."""

    value: str


@pytest.fixture
def mock_inference_engine() -> MagicMock:
    """Create a mock inference engine with generate method."""
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def mock_cache(tmp_path: Path) -> LocleanCache:
    """Create a temporary cache for testing."""
    return LocleanCache(cache_dir=tmp_path)


class TestExtractorInitialization:
    """Test cases for Extractor initialization."""

    def test_initialization_with_inference_engine(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test initialization with inference engine."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        assert extractor.inference_engine is mock_inference_engine
        assert extractor.cache is None
        assert extractor.max_retries == 3

    def test_initialization_with_cache(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test initialization with cache."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        assert extractor.cache is mock_cache

    def test_initialization_without_cache(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test initialization without cache."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=None)

        assert extractor.cache is None

    def test_max_retries_configuration(self, mock_inference_engine: MagicMock) -> None:
        """Test max_retries configuration."""
        extractor = Extractor(inference_engine=mock_inference_engine, max_retries=5)

        assert extractor.max_retries == 5


class TestExtract:
    """Test cases for extract method."""

    def test_successful_extraction_with_valid_schema(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test successful extraction with valid schema."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {"name": "t-shirt", "price": 50000, "color": "red"}
        )

        result = extractor.extract("Selling red t-shirt for 50k", Product)

        assert isinstance(result, Product)
        assert result.name == "t-shirt"
        assert result.price == 50000
        assert result.color == "red"

    def test_extraction_with_cache_hit(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction with cache hit."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        cached_data = {"name": "t-shirt", "price": 50000, "color": "red"}
        instruction = extractor._build_instruction(Product, None)
        cache_key = extractor._get_cache_key(
            "Selling red t-shirt for 50k", Product, instruction
        )
        mock_cache.set_batch(
            ["Selling red t-shirt for 50k"],
            cache_key,
            {"Selling red t-shirt for 50k": cached_data},
        )

        result = extractor.extract("Selling red t-shirt for 50k", Product)

        assert isinstance(result, Product)
        assert result.name == "t-shirt"
        mock_inference_engine.generate.assert_not_called()

    def test_extraction_with_cache_miss(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction with cache miss."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {"name": "t-shirt", "price": 50000, "color": "red"}
        )

        result = extractor.extract("Selling red t-shirt for 50k", Product)

        assert isinstance(result, Product)
        mock_inference_engine.generate.assert_called()

    def test_extraction_with_custom_instruction(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction with custom instruction."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {"name": "t-shirt", "price": 50000, "color": "red"}
        )

        result = extractor.extract(
            "Selling red t-shirt for 50k", Product, instruction="Extract product info"
        )

        assert isinstance(result, Product)
        call_args = mock_inference_engine.generate.call_args
        assert "Extract product info" in call_args[0][0]

    def test_extraction_with_invalid_schema_not_basemodel(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test extraction with invalid schema (not BaseModel)."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        class NotBaseModel:
            pass

        with pytest.raises(ValueError, match="Schema must be a Pydantic BaseModel"):
            extractor.extract("test", NotBaseModel)  # type: ignore[arg-type]

    def test_extraction_failure_after_max_retries(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction failure after max_retries."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=2
        )

        mock_inference_engine.generate.return_value = json.dumps({"invalid": "data"})

        with pytest.raises(ValueError, match="Failed to extract"):
            extractor.extract("test", Product)

    def test_extraction_with_json_repair_needed(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction with JSON repair needed."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = (
            '{"name": "t-shirt", "price": 50000, "color": "red",}'
        )

        result = extractor.extract("Selling red t-shirt for 50k", Product)

        assert isinstance(result, Product)
        assert result.name == "t-shirt"

    def test_extraction_with_validation_error_triggers_retry(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test extraction with validation error (triggers retry)."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=3
        )

        mock_inference_engine.generate.side_effect = [
            json.dumps({"name": "t-shirt"}),  # Missing fields
            json.dumps({"name": "t-shirt", "price": 50000, "color": "red"}),
        ]

        result = extractor.extract("Selling red t-shirt for 50k", Product)

        assert isinstance(result, Product)
        assert mock_inference_engine.generate.call_count == 2


class TestExtractBatch:
    """Test cases for extract_batch method."""

    def test_batch_extraction_with_multiple_items(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test batch extraction with multiple items."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.side_effect = [
            json.dumps({"name": "t-shirt", "price": 50000, "color": "red"}),
            json.dumps({"name": "jeans", "price": 80000, "color": "blue"}),
        ]

        results = extractor.extract_batch(
            ["Selling red t-shirt for 50k", "Selling blue jeans for 80k"], Product
        )

        assert len(results) == 2
        assert results["Selling red t-shirt for 50k"] is not None
        assert results["Selling blue jeans for 80k"] is not None

    def test_batch_extraction_with_cache_hits(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test batch extraction with cache hits."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        instruction = extractor._build_instruction(Product, None)
        cached_data = {"name": "t-shirt", "price": 50000, "color": "red"}
        cache_key = extractor._get_cache_key("", Product, instruction)
        mock_cache.set_batch(
            ["Selling red t-shirt for 50k"],
            cache_key,
            {"Selling red t-shirt for 50k": cached_data},
        )

        results = extractor.extract_batch(["Selling red t-shirt for 50k"], Product)

        assert len(results) == 1
        assert results["Selling red t-shirt for 50k"] is not None
        mock_inference_engine.generate.assert_not_called()

    def test_batch_extraction_with_mixed_results(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test batch extraction with mixed results (some success, some failure)."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=1
        )

        mock_inference_engine.generate.side_effect = [
            json.dumps({"name": "t-shirt", "price": 50000, "color": "red"}),
            json.dumps({"invalid": "data"}),
        ]

        results = extractor.extract_batch(
            ["Selling red t-shirt for 50k", "Invalid text"], Product
        )

        assert len(results) == 2
        assert results["Selling red t-shirt for 50k"] is not None
        assert results["Invalid text"] is None

    def test_batch_extraction_with_empty_list(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test batch extraction with empty list."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        results = extractor.extract_batch([], Product)

        assert len(results) == 0

    @patch("loclean.utils.rich_output.log_processing_summary")
    def test_rich_processing_summary_logging(
        self,
        mock_log_summary: Mock,
        mock_inference_engine: MagicMock,
        mock_cache: LocleanCache,
    ) -> None:
        """Test Rich processing summary logging."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps(
            {"name": "t-shirt", "price": 50000, "color": "red"}
        )

        extractor.extract_batch(["Selling red t-shirt for 50k"], Product)

        mock_log_summary.assert_called_once()

    @patch("loclean.utils.rich_output.log_error_summary")
    def test_rich_error_summary_logging_when_errors_gt_3(
        self,
        mock_log_errors: Mock,
        mock_inference_engine: MagicMock,
        mock_cache: LocleanCache,
    ) -> None:
        """Test Rich error summary logging (when errors > 3)."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=0
        )

        from typing import Any

        def mock_extract(*args: object, **kwargs: object) -> None:
            errors: list[dict[str, Any]] = [
                {"type": "missing", "loc": ("name",), "msg": "Field required"}
            ]
            raise ValidationError.from_exception_data("Product", errors)  # type: ignore[arg-type]

        with patch.object(extractor, "_extract_with_retry", side_effect=mock_extract):
            items = [f"Invalid text {i}" for i in range(4)]
            extractor.extract_batch(items, Product)

        mock_log_errors.assert_called_once()


class TestExtractWithRetry:
    """Test cases for _extract_with_retry method."""

    def test_successful_extraction_on_first_attempt(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test successful extraction on first attempt."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps({"value": "test"})

        result = extractor._extract_with_retry("test", SimpleSchema, "Extract value", 0)

        assert isinstance(result, SimpleSchema)
        assert result.value == "test"

    def test_retry_on_validation_error(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test retry on validation error."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=2
        )

        mock_inference_engine.generate.side_effect = [
            json.dumps({"invalid": "data"}),
            json.dumps({"value": "test"}),
        ]

        result = extractor._extract_with_retry("test", SimpleSchema, "Extract value", 0)

        assert isinstance(result, SimpleSchema)
        assert mock_inference_engine.generate.call_count == 2

    def test_max_retries_exceeded_returns_none(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test max retries exceeded (returns None)."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=2
        )

        mock_inference_engine.generate.return_value = json.dumps({"invalid": "data"})

        result = extractor._extract_with_retry("test", SimpleSchema, "Extract value", 2)

        assert result is None

    def test_generate_called_with_schema(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test that generate is called with the schema parameter."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps({"value": "test"})

        extractor._extract_with_retry("test", SimpleSchema, "Extract value", 0)

        call_args = mock_inference_engine.generate.call_args
        assert call_args[1]["schema"] is SimpleSchema

    def test_exception_handling_during_extraction(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test exception handling during extraction."""
        extractor = Extractor(
            inference_engine=mock_inference_engine, cache=mock_cache, max_retries=2
        )

        mock_inference_engine.generate.side_effect = Exception("LLM error")

        result = extractor._extract_with_retry("test", SimpleSchema, "Extract value", 0)

        assert result is None


class TestParseAndValidate:
    """Test cases for _parse_and_validate method."""

    def test_parsing_valid_json_string(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test parsing valid JSON string."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        result = extractor._parse_and_validate(
            json.dumps({"value": "test"}), SimpleSchema, "test", "Extract", 0
        )

        assert isinstance(result, SimpleSchema)
        assert result.value == "test"

    def test_parsing_json_dict_already_parsed(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test parsing JSON dict (already parsed)."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        result = extractor._parse_and_validate(
            {"value": "test"},  # type: ignore[arg-type]
            SimpleSchema,
            "test",
            "Extract",
            0,
        )

        assert isinstance(result, SimpleSchema)
        assert result.value == "test"

    @patch("loclean.extraction.extractor.repair_json")
    def test_json_repair_on_malformed_json(
        self,
        mock_repair: Mock,
        mock_inference_engine: MagicMock,
        mock_cache: LocleanCache,
    ) -> None:
        """Test JSON repair on malformed JSON."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_repair.return_value = json.dumps({"value": "test"})

        result = extractor._parse_and_validate(
            '{"value": "test",}', SimpleSchema, "test", "Extract", 0
        )

        assert isinstance(result, SimpleSchema)
        mock_repair.assert_called_once()

    def test_validation_success(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test validation success."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        result = extractor._parse_and_validate(
            json.dumps({"value": "test"}), SimpleSchema, "test", "Extract", 0
        )

        assert isinstance(result, SimpleSchema)

    def test_validation_error_triggers_retry(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test validation error (triggers retry)."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        with pytest.raises(ValidationError):
            extractor._parse_and_validate(
                json.dumps({"invalid": "data"}), SimpleSchema, "test", "Extract", 0
            )

    def test_json_decode_error_handling(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test JSON decode error handling."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        with patch(
            "loclean.extraction.extractor.repair_json", return_value="still invalid"
        ):
            with pytest.raises(ValueError, match="Invalid JSON"):
                extractor._parse_and_validate(
                    "invalid json", SimpleSchema, "test", "Extract", 0
                )


class TestBuildInstruction:
    """Test cases for _build_instruction method."""

    def test_custom_instruction_usage(self, mock_inference_engine: MagicMock) -> None:
        """Test custom instruction usage."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        instruction = extractor._build_instruction(SimpleSchema, "Custom instruction")

        assert instruction == "Custom instruction"

    def test_auto_generated_instruction_from_schema(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test auto-generated instruction from schema."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        instruction = extractor._build_instruction(SimpleSchema, None)

        assert "Extract structured information" in instruction
        assert "SimpleSchema" in instruction or "value" in instruction

    def test_instruction_with_schema_fields(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test instruction with schema fields."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        instruction = extractor._build_instruction(Product, None)

        assert "Extract structured information" in instruction
        assert (
            "Product" in instruction or "name" in instruction or "price" in instruction
        )


class TestGetCacheKey:
    """Test cases for _get_cache_key method."""

    def test_cache_key_generation_consistency(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test cache key generation consistency."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        key1 = extractor._get_cache_key("test", SimpleSchema, "instruction")
        key2 = extractor._get_cache_key("test", SimpleSchema, "instruction")

        assert key1 == key2

    def test_cache_key_includes_schema_and_instruction(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test cache key includes schema and instruction."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        key = extractor._get_cache_key("test", SimpleSchema, "Extract value")

        assert "SimpleSchema" in key
        assert "Extract value" in key

    def test_cache_key_uniqueness_for_different_schemas(
        self, mock_inference_engine: MagicMock
    ) -> None:
        """Test cache key uniqueness for different schemas."""
        extractor = Extractor(inference_engine=mock_inference_engine)

        key1 = extractor._get_cache_key("test", SimpleSchema, "instruction")
        key2 = extractor._get_cache_key("test", Product, "instruction")

        assert key1 != key2


class TestRetryExtraction:
    """Test cases for _retry_extraction method."""

    def test_retry_with_adjusted_prompt(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test retry with adjusted prompt."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        mock_inference_engine.generate.return_value = json.dumps({"value": "test"})

        result = extractor._retry_extraction("test", SimpleSchema, "Extract", 0)

        assert isinstance(result, SimpleSchema)
        call_args = mock_inference_engine.generate.call_args[0][0]
        assert "IMPORTANT" in call_args or "strictly match" in call_args

    def test_retry_count_increment(
        self, mock_inference_engine: MagicMock, mock_cache: LocleanCache
    ) -> None:
        """Test retry count increment."""
        extractor = Extractor(inference_engine=mock_inference_engine, cache=mock_cache)

        with patch.object(extractor, "_extract_with_retry") as mock_retry:
            mock_retry.return_value = SimpleSchema(value="test")
            extractor._retry_extraction("test", SimpleSchema, "Extract", 0)

            mock_retry.assert_called_once()
            call_args = mock_retry.call_args[0]
            assert call_args[3] == 1  # retry_count should be incremented
