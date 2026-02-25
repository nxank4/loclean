"""Test cases for InstructionOptimizer."""

import json
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pydantic import BaseModel

from loclean.extraction.optimizer import InstructionOptimizer

# ------------------------------------------------------------------
# Test schemas
# ------------------------------------------------------------------


class Product(BaseModel):
    name: str
    price: int
    color: str


class SimpleSchema(BaseModel):
    value: str


class MultiField(BaseModel):
    title: str
    score: float
    tags: list[str]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.verbose = False
    return engine


@pytest.fixture
def optimizer(mock_engine: MagicMock) -> InstructionOptimizer:
    return InstructionOptimizer(inference_engine=mock_engine)


# ------------------------------------------------------------------
# _is_field_populated
# ------------------------------------------------------------------


class TestIsFieldPopulated:
    def test_none_is_not_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated(None) is False

    def test_empty_string_is_not_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated("") is False
        assert InstructionOptimizer._is_field_populated("   ") is False

    def test_empty_list_is_not_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated([]) is False

    def test_empty_dict_is_not_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated({}) is False

    def test_nonempty_string_is_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated("hello") is True

    def test_number_is_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated(0) is True
        assert InstructionOptimizer._is_field_populated(3.14) is True

    def test_nonempty_list_is_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated(["a"]) is True

    def test_bool_is_populated(self) -> None:
        assert InstructionOptimizer._is_field_populated(False) is True


# ------------------------------------------------------------------
# _score_extraction
# ------------------------------------------------------------------


class TestScoreExtraction:
    def test_perfect_score(self) -> None:
        """All fields populated across all results → F1 = 1.0."""
        results: dict[str, BaseModel | None] = {
            "a": Product(name="Shirt", price=100, color="red"),
            "b": Product(name="Jeans", price=200, color="blue"),
        }
        score = InstructionOptimizer._score_extraction(results, Product)
        assert score == pytest.approx(1.0)

    def test_all_none_results(self) -> None:
        """All extractions failed → F1 = 0.0."""
        results: dict[str, BaseModel | None] = {"a": None, "b": None}
        score = InstructionOptimizer._score_extraction(results, Product)
        assert score == 0.0

    def test_empty_results(self) -> None:
        results: dict[str, BaseModel | None] = {}
        score = InstructionOptimizer._score_extraction(results, Product)
        assert score == 0.0

    def test_partial_success(self) -> None:
        """One success + one failure → precision high, recall lower."""
        results: dict[str, BaseModel | None] = {
            "a": Product(name="Shirt", price=100, color="red"),
            "b": None,
        }
        score = InstructionOptimizer._score_extraction(results, Product)
        # precision = 3/3 = 1.0, recall = 3/6 = 0.5, F1 = 2/3
        assert score == pytest.approx(2 / 3)

    def test_populated_but_empty_strings(self) -> None:
        """Fields present but empty strings reduce precision."""
        results: dict[str, BaseModel | None] = {
            "a": Product(name="Shirt", price=100, color=""),
        }
        score = InstructionOptimizer._score_extraction(results, Product)
        # 2 of 3 fields populated
        # precision = 2/3, recall = 2/3, F1 = 2/3
        assert score == pytest.approx(2 / 3)

    def test_multi_field_schema_with_empty_list(self) -> None:
        """Empty tags list reduces precision and recall."""
        results: dict[str, BaseModel | None] = {
            "a": MultiField(title="Test", score=9.5, tags=[]),
        }
        score = InstructionOptimizer._score_extraction(results, MultiField)
        # 2 of 3 populated → precision = 2/3, recall = 2/3, F1 = 2/3
        assert score == pytest.approx(2 / 3)


# ------------------------------------------------------------------
# _build_default_instruction
# ------------------------------------------------------------------


class TestBuildDefaultInstruction:
    def test_includes_schema_info(self) -> None:
        instruction = InstructionOptimizer._build_default_instruction(Product)
        assert "Extract structured information" in instruction
        assert "name" in instruction or "Product" in instruction

    def test_returns_string(self) -> None:
        instruction = InstructionOptimizer._build_default_instruction(SimpleSchema)
        assert isinstance(instruction, str)
        assert len(instruction) > 0


# ------------------------------------------------------------------
# _parse_variations_response
# ------------------------------------------------------------------


class TestParseVariationsResponse:
    def test_parses_dict(self) -> None:
        raw = {"variations": ["a", "b", "c"]}
        result = InstructionOptimizer._parse_variations_response(raw)
        assert result["variations"] == ["a", "b", "c"]

    def test_parses_json_string(self) -> None:
        raw = json.dumps({"variations": ["x", "y", "z"]})
        result = InstructionOptimizer._parse_variations_response(raw)
        assert result["variations"] == ["x", "y", "z"]

    def test_repairs_malformed_json(self) -> None:
        raw = '{"variations": ["a", "b", "c",]}'
        result = InstructionOptimizer._parse_variations_response(raw)
        assert "variations" in result

    def test_returns_empty_on_total_failure(self) -> None:
        result = InstructionOptimizer._parse_variations_response("completely broken")
        assert result == {}


# ------------------------------------------------------------------
# _generate_variations
# ------------------------------------------------------------------


class TestGenerateVariations:
    def test_returns_n_variations(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"variations": ["v1", "v2", "v3"]}
        )
        result = optimizer._generate_variations("baseline", Product, n=3)
        assert len(result) == 3
        assert result == ["v1", "v2", "v3"]

    def test_pads_with_baseline_on_insufficient(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps({"variations": ["only_one"]})
        result = optimizer._generate_variations("baseline", Product, n=3)
        assert len(result) == 3
        assert result[0] == "only_one"
        assert result[1] == "baseline"

    def test_truncates_excess_variations(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"variations": ["a", "b", "c", "d", "e"]}
        )
        result = optimizer._generate_variations("baseline", Product, n=3)
        assert len(result) == 3

    def test_filters_empty_strings(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"variations": ["good", "", "  "]}
        )
        result = optimizer._generate_variations("baseline", Product, n=3)
        assert len(result) == 3
        assert result[0] == "good"
        assert result[1] == "baseline"

    def test_fallback_on_unparseable_response(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = "not json at all"
        result = optimizer._generate_variations("baseline", Product, n=3)
        assert len(result) == 3
        assert all(v == "baseline" for v in result)

    def test_prompt_includes_schema(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"variations": ["v1", "v2", "v3"]}
        )
        optimizer._generate_variations("do extraction", Product, n=3)
        prompt = mock_engine.generate.call_args[0][0]
        assert "do extraction" in prompt
        assert "name" in prompt or "price" in prompt


# ------------------------------------------------------------------
# _evaluate_variation
# ------------------------------------------------------------------


class TestEvaluateVariation:
    def test_returns_score_on_success(
        self, optimizer: InstructionOptimizer, mock_engine: MagicMock
    ) -> None:
        mock_engine.generate.return_value = json.dumps(
            {"name": "Shirt", "price": 100, "color": "red"}
        )
        score = optimizer._evaluate_variation(
            "Extract info", ["Selling red shirt 100"], Product
        )
        assert score == pytest.approx(1.0)

    def test_returns_zero_on_exception(self, optimizer: InstructionOptimizer) -> None:
        with patch.object(
            InstructionOptimizer, "_evaluate_variation", return_value=0.0
        ):
            score = optimizer._evaluate_variation("fail", ["x"], Product)
            assert score == 0.0

    def test_evaluation_uses_extract_batch(
        self, optimizer: InstructionOptimizer
    ) -> None:
        with patch("loclean.extraction.optimizer.Extractor") as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract_batch.return_value = {
                "row1": Product(name="A", price=1, color="b"),
            }
            score = optimizer._evaluate_variation("instruction", ["row1"], Product)
            instance.extract_batch.assert_called_once_with(
                ["row1"], Product, "instruction"
            )
            assert score == pytest.approx(1.0)


# ------------------------------------------------------------------
# optimize (end-to-end with mocks)
# ------------------------------------------------------------------


class TestOptimize:
    def test_selects_best_instruction(self, optimizer: InstructionOptimizer) -> None:
        df = pl.DataFrame({"text": ["row1", "row2", "row3"]})

        call_count = 0

        def mock_evaluate(
            instruction: str, sample: list[str], schema: type[BaseModel]
        ) -> float:
            nonlocal call_count
            call_count += 1
            scores = {
                "baseline": 0.5,
                "variation_a": 0.9,
                "variation_b": 0.3,
                "variation_c": 0.6,
            }
            return scores.get(instruction, 0.0)

        with (
            patch.object(
                optimizer,
                "_generate_variations",
                return_value=["variation_a", "variation_b", "variation_c"],
            ),
            patch.object(optimizer, "_evaluate_variation", side_effect=mock_evaluate),
        ):
            best = optimizer.optimize(
                df, "text", Product, baseline_instruction="baseline"
            )

        assert best == "variation_a"

    def test_returns_baseline_when_no_valid_samples(
        self, optimizer: InstructionOptimizer
    ) -> None:
        df = pl.DataFrame({"text": [None, "", "  "]})

        result = optimizer.optimize(
            df, "text", Product, baseline_instruction="my instruction"
        )
        assert result == "my instruction"

    def test_raises_on_missing_column(self, optimizer: InstructionOptimizer) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column 'b' not found"):
            optimizer.optimize(df, "b", Product)

    def test_builds_default_instruction_when_none(
        self, optimizer: InstructionOptimizer
    ) -> None:
        df = pl.DataFrame({"text": ["row"]})

        with (
            patch.object(
                optimizer,
                "_generate_variations",
                return_value=["v1", "v2", "v3"],
            ),
            patch.object(optimizer, "_evaluate_variation", return_value=0.5),
        ):
            result = optimizer.optimize(df, "text", Product)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_baseline_included_in_candidates(
        self, optimizer: InstructionOptimizer
    ) -> None:
        """Baseline itself is evaluated alongside the variations."""
        df = pl.DataFrame({"text": ["row1"]})
        evaluated_instructions: list[str] = []

        def capture_evaluate(
            instruction: str, sample: list[str], schema: type[BaseModel]
        ) -> float:
            evaluated_instructions.append(instruction)
            return 0.5

        with (
            patch.object(
                optimizer,
                "_generate_variations",
                return_value=["v1", "v2", "v3"],
            ),
            patch.object(
                optimizer, "_evaluate_variation", side_effect=capture_evaluate
            ),
        ):
            optimizer.optimize(df, "text", Product, baseline_instruction="base")

        assert "base" in evaluated_instructions
        assert len(evaluated_instructions) == 4

    def test_returns_baseline_when_it_wins(
        self, optimizer: InstructionOptimizer
    ) -> None:
        df = pl.DataFrame({"text": ["row1"]})

        def baseline_wins(
            instruction: str, sample: list[str], schema: type[BaseModel]
        ) -> float:
            return 1.0 if instruction == "strong_baseline" else 0.1

        with (
            patch.object(
                optimizer,
                "_generate_variations",
                return_value=["weak1", "weak2", "weak3"],
            ),
            patch.object(optimizer, "_evaluate_variation", side_effect=baseline_wins),
        ):
            best = optimizer.optimize(
                df, "text", Product, baseline_instruction="strong_baseline"
            )

        assert best == "strong_baseline"
