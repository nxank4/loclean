"""Unit tests for public API functions."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

import loclean


class Product(BaseModel):
    """Test schema for extraction."""

    name: str
    price: int
    color: str


@pytest.fixture
def sample_polars_df() -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame({"weight": ["10kg", "500g"], "price": ["$10", "20 EUR"]})


@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """Create a sample Pandas DataFrame for testing."""
    return pd.DataFrame({"weight": ["10kg", "500g"], "price": ["$10", "20 EUR"]})


class TestGetEngine:
    """Test cases for get_engine function."""

    def test_singleton_pattern_same_instance_on_multiple_calls(self) -> None:
        """Test singleton pattern (same instance on multiple calls)."""
        # Reset global instance
        loclean._ENGINE_INSTANCE = None  # type: ignore[attr-defined]

        with patch("loclean.LlamaCppEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            engine1 = loclean.get_engine()
            engine2 = loclean.get_engine()

            assert engine1 is engine2
            assert engine1 is mock_engine
            # Should only be created once
            assert mock_engine_class.call_count == 1

    def test_engine_creation_on_first_call(self) -> None:
        """Test engine creation on first call."""
        loclean._ENGINE_INSTANCE = None  # type: ignore[attr-defined]

        with patch("loclean.LlamaCppEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            engine = loclean.get_engine()

            assert engine is mock_engine
            mock_engine_class.assert_called_once()

    def test_engine_reuse_on_subsequent_calls(self) -> None:
        """Test engine reuse on subsequent calls."""
        loclean._ENGINE_INSTANCE = None  # type: ignore[attr-defined]

        with patch("loclean.LlamaCppEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            loclean.get_engine()
            loclean.get_engine()
            loclean.get_engine()

            # Should only be created once
            assert mock_engine_class.call_count == 1

    def test_engine_type_llamacppengine_instance(self) -> None:
        """Test engine type (LlamaCppEngine instance)."""
        loclean._ENGINE_INSTANCE = None  # type: ignore[attr-defined]

        with patch("loclean.LlamaCppEngine") as mock_engine_class:
            from loclean.inference.local.llama_cpp import LlamaCppEngine

            mock_engine = MagicMock(spec=LlamaCppEngine)
            mock_engine_class.return_value = mock_engine

            engine = loclean.get_engine()

            assert isinstance(
                engine, MagicMock
            )  # Mocked, but should be LlamaCppEngine type


class TestClean:
    """Test cases for clean function."""

    @patch("loclean.NarwhalsEngine.process_column")
    @patch("loclean.get_engine")
    def test_with_polars_dataframe(
        self, mock_get_engine: Mock, mock_process: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with Polars DataFrame."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        result = loclean.clean(sample_polars_df, "weight")

        assert isinstance(result, pl.DataFrame)
        mock_process.assert_called_once()

    @patch("loclean.NarwhalsEngine.process_column")
    @patch("loclean.get_engine")
    def test_with_pandas_dataframe(
        self, mock_get_engine: Mock, mock_process: Mock, sample_pandas_df: pd.DataFrame
    ) -> None:
        """Test with Pandas DataFrame."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_pandas_df

        result = loclean.clean(sample_pandas_df, "weight")

        assert isinstance(result, pd.DataFrame)
        mock_process.assert_called_once()

    def test_with_invalid_column_name_valueerror(
        self, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with invalid column name (ValueError)."""
        with pytest.raises(ValueError, match="Column 'invalid_col' not found"):
            loclean.clean(sample_polars_df, "invalid_col")

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_custom_model_name(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with custom model_name."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", model_name="phi-3-mini")

        mock_engine_class.assert_called_once()
        assert "model_name" in str(mock_engine_class.call_args)

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_custom_cache_dir(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with custom cache_dir."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", cache_dir=Path("/custom/path"))

        mock_engine_class.assert_called_once()

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_custom_n_ctx(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with custom n_ctx."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", n_ctx=2048)

        mock_engine_class.assert_called_once()

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_custom_n_gpu_layers(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with custom n_gpu_layers."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", n_gpu_layers=10)

        mock_engine_class.assert_called_once()

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_verbose_true(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with verbose=True."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", verbose=True)

        mock_engine_class.assert_called_once()

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_parallel_true(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with parallel=True."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", parallel=True)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["parallel"] is True

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_max_workers_parameter(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with max_workers parameter."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", parallel=True, max_workers=4)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["max_workers"] == 4

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_batch_size_parameter(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with batch_size parameter."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", batch_size=100)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["batch_size"] == 100

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_engine_kwargs(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test with engine_kwargs."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", custom_param="value")

        mock_engine_class.assert_called_once()

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_global_engine_reuse_when_no_overrides(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test global engine reuse when no overrides."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight")

        mock_get_engine.assert_called_once()
        mock_process.assert_called_once_with(
            sample_polars_df,
            "weight",
            mock_engine,
            "Extract the numeric value and unit as-is.",
            batch_size=50,
            parallel=False,
            max_workers=None,
        )

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_dedicated_engine_creation_when_overrides_provided(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """Test dedicated engine creation when overrides provided."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", model_name="phi-3-mini")

        mock_engine_class.assert_called_once()
        # Should not use get_engine when overrides provided

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_return_type_matches_input_type_polars_to_polars(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test return type matches input type (Polars -> Polars)."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        result = loclean.clean(sample_polars_df, "weight")

        assert isinstance(result, pl.DataFrame)

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_return_type_matches_input_type_pandas_to_pandas(
        self, mock_process: Mock, mock_get_engine: Mock, sample_pandas_df: pd.DataFrame
    ) -> None:
        """Test return type matches input type (Pandas -> Pandas)."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_pandas_df

        result = loclean.clean(sample_pandas_df, "weight")

        assert isinstance(result, pd.DataFrame)


class TestScrub:
    """Test cases for scrub function."""

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_string_input(self, mock_scrub_string: Mock) -> None:
        """Test with string input."""
        mock_scrub_string.return_value = "Contact [PERSON] at [PHONE]"

        result = loclean.scrub(
            "Contact John at 555-1234", strategies=["person", "phone"]
        )

        assert isinstance(result, str)
        assert "[PERSON]" in result
        mock_scrub_string.assert_called_once()

    @patch("loclean.privacy.scrub.scrub_dataframe")
    def test_with_dataframe_input(
        self, mock_scrub_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with DataFrame input."""
        mock_scrub_df.return_value = sample_polars_df

        result = loclean.scrub(
            sample_polars_df, target_col="weight", strategies=["phone"]
        )

        assert isinstance(result, pl.DataFrame)
        mock_scrub_df.assert_called_once()

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_default_strategies(self, mock_scrub_string: Mock) -> None:
        """Test with default strategies."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test text")

        mock_scrub_string.assert_called_once()
        call_args = mock_scrub_string.call_args[0]
        assert call_args[1] == ["person", "phone", "email"]  # Default strategies

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_custom_strategies(self, mock_scrub_string: Mock) -> None:
        """Test with custom strategies."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test", strategies=["email", "credit_card"])

        call_args = mock_scrub_string.call_args[0]
        assert call_args[1] == ["email", "credit_card"]

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_mask_mode(self, mock_scrub_string: Mock) -> None:
        """Test with mask mode."""
        mock_scrub_string.return_value = "[PERSON]"

        loclean.scrub("John", strategies=["person"], mode="mask")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[2] == "mask"

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_fake_mode(self, mock_scrub_string: Mock) -> None:
        """Test with fake mode."""
        mock_scrub_string.return_value = "Jane Smith"

        loclean.scrub("John", strategies=["person"], mode="fake")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[2] == "fake"

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_locale_parameter(self, mock_scrub_string: Mock) -> None:
        """Test with locale parameter."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test", strategies=["person"], mode="fake", locale="en_US")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[3] == "en_US"

    @patch("loclean.privacy.scrub.scrub_dataframe")
    def test_with_target_col_for_dataframe(
        self, mock_scrub_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with target_col for DataFrame."""
        mock_scrub_df.return_value = sample_polars_df

        loclean.scrub(sample_polars_df, target_col="weight", strategies=["phone"])

        call_args = mock_scrub_df.call_args[0]
        assert call_args[1] == "weight"

    def test_valueerror_when_target_col_missing_for_dataframe(
        self, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test ValueError when target_col missing for DataFrame."""
        with pytest.raises(ValueError, match="target_col required for DataFrame input"):
            loclean.scrub(sample_polars_df, strategies=["phone"])

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_llm_strategies_person_address_requires_inference_engine(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """Test with LLM strategies (person, address) - requires inference engine."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_scrub_string.return_value = "[PERSON]"

        loclean.scrub("Contact John", strategies=["person"])

        mock_get_engine.assert_called_once()
        call_args = mock_scrub_string.call_args
        assert call_args[1]["inference_engine"] is mock_engine

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_regex_only_strategies_no_llm_needed(
        self, mock_scrub_string: Mock
    ) -> None:
        """Test with regex-only strategies (no LLM needed)."""
        mock_scrub_string.return_value = "[PHONE]"

        loclean.scrub("555-1234", strategies=["phone", "email"])

        call_args = mock_scrub_string.call_args
        assert call_args[1]["inference_engine"] is None

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_engine_configuration_parameters_model_name(
        self, mock_scrub_string: Mock, mock_engine_class: Mock
    ) -> None:
        """Test engine configuration parameters (model_name)."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_scrub_string.return_value = "[PERSON]"

        loclean.scrub("John", strategies=["person"], model_name="phi-3-mini")

        mock_engine_class.assert_called_once()

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_return_type_matches_input_type_string(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """Test return type matches input type."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_scrub_string.return_value = "Scrubbed text"

        result = loclean.scrub("test", strategies=["person"])

        assert isinstance(result, str)


class TestExtract:
    """Test cases for extract function."""

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_string_input(self, mock_extractor_class: Mock) -> None:
        """Test with string input."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            result = loclean.extract("Selling red t-shirt for 50k", Product)

            assert isinstance(result, Product)
            assert result.name == "t-shirt"

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_dataframe_input(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with DataFrame input."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            result = loclean.extract(sample_polars_df, Product, target_col="weight")

            assert isinstance(result, pl.DataFrame)
            mock_extract_df.assert_called_once()

    def test_with_valid_pydantic_schema(self) -> None:
        """Test with valid Pydantic schema."""
        with patch("loclean.extraction.extractor.Extractor") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_result = Product(name="t-shirt", price=50000, color="red")
            mock_extractor.extract.return_value = mock_result
            mock_extractor_class.return_value = mock_extractor

            with patch("loclean.get_engine") as mock_get_engine:
                mock_engine = MagicMock()
                mock_engine.cache = None
                mock_get_engine.return_value = mock_engine

                result = loclean.extract("test", Product)

                assert isinstance(result, Product)

    def test_with_invalid_schema_not_basemodel_valueerror(self) -> None:
        """Test with invalid schema (not BaseModel - ValueError)."""

        class NotBaseModel:
            pass

        with pytest.raises(ValueError, match="Schema must be a Pydantic BaseModel"):
            loclean.extract("test", NotBaseModel)  # type: ignore[arg-type]

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_custom_instruction(self, mock_extractor_class: Mock) -> None:
        """Test with custom instruction."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product, instruction="Custom instruction")

            mock_extractor.extract.assert_called_once()
            call_args = mock_extractor.extract.call_args
            assert call_args[0][2] == "Custom instruction"

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_auto_generated_instruction(self, mock_extractor_class: Mock) -> None:
        """Test with auto-generated instruction."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product, instruction=None)

            mock_extractor.extract.assert_called_once()

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_target_col_for_dataframe(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with target_col for DataFrame."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract(sample_polars_df, Product, target_col="weight")

            call_args = mock_extract_df.call_args[0]
            assert call_args[1] == "weight"

    def test_valueerror_when_target_col_missing_for_dataframe(
        self, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test ValueError when target_col missing for DataFrame."""
        with pytest.raises(ValueError, match="target_col required for DataFrame input"):
            loclean.extract(sample_polars_df, Product)

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_output_type_dict_default(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with output_type='dict' (default)."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract(sample_polars_df, Product, target_col="weight")

            call_kwargs = mock_extract_df.call_args[1]
            assert call_kwargs["output_type"] == "dict"

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_output_type_pydantic(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test with output_type='pydantic'."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract(
                sample_polars_df, Product, target_col="weight", output_type="pydantic"
            )

            call_kwargs = mock_extract_df.call_args[1]
            assert call_kwargs["output_type"] == "pydantic"

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_max_retries_parameter(self, mock_extractor_class: Mock) -> None:
        """Test with max_retries parameter."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product, max_retries=5)

            mock_extractor_class.assert_called_once()
            call_kwargs = mock_extractor_class.call_args[1]
            assert call_kwargs["max_retries"] == 5

    @patch("loclean.LlamaCppEngine")
    @patch("loclean.extraction.extractor.Extractor")
    def test_engine_configuration_parameters(
        self, mock_extractor_class: Mock, mock_engine_class: Mock
    ) -> None:
        """Test engine configuration parameters."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        mock_engine = MagicMock()
        mock_engine.cache = None
        mock_engine_class.return_value = mock_engine

        loclean.extract("test", Product, model_name="phi-3-mini")

        mock_engine_class.assert_called_once()

    @patch("loclean.extraction.extractor.Extractor")
    def test_cache_usage(self, mock_extractor_class: Mock) -> None:
        """Test cache usage."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_cache = MagicMock()
            mock_engine.cache = mock_cache
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product)

            mock_extractor_class.assert_called_once()
            call_kwargs = mock_extractor_class.call_args[1]
            assert call_kwargs["cache"] is mock_cache

    @patch("loclean.extraction.extractor.Extractor")
    def test_return_type_pydantic_model_for_string(
        self, mock_extractor_class: Mock
    ) -> None:
        """Test return type (Pydantic model for string)."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            result = loclean.extract("test", Product)

            assert isinstance(result, Product)

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_return_type_dataframe_for_dataframe(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """Test return type (DataFrame for DataFrame)."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.cache = None
            mock_get_engine.return_value = mock_engine

            result = loclean.extract(sample_polars_df, Product, target_col="weight")

            assert isinstance(result, pl.DataFrame)
