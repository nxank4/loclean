"""Unit tests for public API functions."""

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
        """Calling get_engine() without args returns the same instance."""
        loclean._ENGINE_INSTANCE = None

        with patch("loclean.OllamaEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_cls.return_value = mock_engine

            engine1 = loclean.get_engine()
            engine2 = loclean.get_engine()

            assert engine1 is engine2
            assert engine1 is mock_engine
            assert mock_cls.call_count == 1

    def test_engine_creation_on_first_call(self) -> None:
        """First call creates the engine."""
        loclean._ENGINE_INSTANCE = None

        with patch("loclean.OllamaEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_cls.return_value = mock_engine

            engine = loclean.get_engine()

            assert engine is mock_engine
            mock_cls.assert_called_once()

    def test_custom_params_create_new_instance(self) -> None:
        """Passing model/host/verbose creates a new (non-singleton) instance."""
        loclean._ENGINE_INSTANCE = None

        with patch("loclean.OllamaEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_cls.return_value = mock_engine

            engine = loclean.get_engine(model="llama3")

            assert engine is mock_engine
            mock_cls.assert_called_once_with(model="llama3")


class TestLocleanClass:
    """Test cases for the Loclean class."""

    @patch("loclean.OllamaEngine")
    def test_init_creates_engine(self, mock_cls: Mock) -> None:
        """Loclean() creates an OllamaEngine."""
        mock_cls.return_value = MagicMock()
        client = loclean.Loclean(model="phi3")
        mock_cls.assert_called_once_with(
            model="phi3", host="http://localhost:11434", verbose=False
        )
        assert client.engine is mock_cls.return_value

    @patch("loclean.OllamaEngine")
    def test_extract_delegates_to_extractor(self, mock_cls: Mock) -> None:
        """Loclean.extract() instantiates Extractor and calls extract."""
        mock_cls.return_value = MagicMock()
        client = loclean.Loclean(model="phi3")

        with patch("loclean.extraction.extractor.Extractor") as mock_ext_cls:
            mock_ext = MagicMock()
            mock_result = Product(name="t-shirt", price=50000, color="red")
            mock_ext.extract.return_value = mock_result
            mock_ext_cls.return_value = mock_ext

            result = client.extract("Selling red t-shirt for 50k", Product)

            assert isinstance(result, Product)
            assert result.name == "t-shirt"

    @patch("loclean.OllamaEngine")
    def test_extract_rejects_non_basemodel_schema(self, mock_cls: Mock) -> None:
        """Loclean.extract() raises ValueError for non-BaseModel schema."""
        mock_cls.return_value = MagicMock()
        client = loclean.Loclean()

        class NotAModel:
            pass

        with pytest.raises(ValueError, match="Schema must be a Pydantic BaseModel"):
            client.extract("test", NotAModel)  # type: ignore[arg-type]


class TestClean:
    """Test cases for clean function."""

    @patch("loclean.NarwhalsEngine.process_column")
    @patch("loclean.get_engine")
    def test_with_polars_dataframe(
        self, mock_get_engine: Mock, mock_process: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """clean() works with Polars DataFrame."""
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
        """clean() works with Pandas DataFrame."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_pandas_df

        result = loclean.clean(sample_pandas_df, "weight")

        assert isinstance(result, pd.DataFrame)
        mock_process.assert_called_once()

    def test_with_invalid_column_name_valueerror(
        self, sample_polars_df: pl.DataFrame
    ) -> None:
        """clean() raises ValueError for non-existent column."""
        with pytest.raises(ValueError, match="Column 'invalid_col' not found"):
            loclean.clean(sample_polars_df, "invalid_col")

    @patch("loclean.OllamaEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_custom_model(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """clean() creates a dedicated engine when model= is passed."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", model="llama3")

        mock_engine_class.assert_called_once()
        assert "model" in str(mock_engine_class.call_args)

    @patch("loclean.OllamaEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_with_verbose_true(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """clean() passes verbose to the engine."""
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
        """clean() forwards parallel= to process_column."""
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
        """clean() forwards max_workers= to process_column."""
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
        """clean() forwards batch_size= to process_column."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", batch_size=100)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["batch_size"] == 100

    @patch("loclean.get_engine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_global_engine_reuse_when_no_overrides(
        self, mock_process: Mock, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """clean() reuses global engine when no overrides given."""
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

    @patch("loclean.OllamaEngine")
    @patch("loclean.NarwhalsEngine.process_column")
    def test_dedicated_engine_creation_when_overrides_provided(
        self,
        mock_process: Mock,
        mock_engine_class: Mock,
        sample_polars_df: pl.DataFrame,
    ) -> None:
        """clean() creates a fresh engine when model= is passed."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_process.return_value = sample_polars_df

        loclean.clean(sample_polars_df, "weight", model="llama3")

        mock_engine_class.assert_called_once()


class TestScrub:
    """Test cases for scrub function."""

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_string_input(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() works with string input."""
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
        """scrub() works with DataFrame input."""
        mock_scrub_df.return_value = sample_polars_df

        result = loclean.scrub(
            sample_polars_df, target_col="weight", strategies=["phone"]
        )

        assert isinstance(result, pl.DataFrame)
        mock_scrub_df.assert_called_once()

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_default_strategies(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() defaults to ["person", "phone", "email"]."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test text")

        mock_scrub_string.assert_called_once()
        call_args = mock_scrub_string.call_args[0]
        assert call_args[1] == ["person", "phone", "email"]

    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_custom_strategies(self, mock_scrub_string: Mock) -> None:
        """scrub() passes custom strategies through."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test", strategies=["email", "credit_card"])

        call_args = mock_scrub_string.call_args[0]
        assert call_args[1] == ["email", "credit_card"]

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_mask_mode(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() passes mode='mask'."""
        mock_scrub_string.return_value = "[PERSON]"

        loclean.scrub("John", strategies=["person"], mode="mask")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[2] == "mask"

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_fake_mode(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() passes mode='fake'."""
        mock_scrub_string.return_value = "Jane Smith"

        loclean.scrub("John", strategies=["person"], mode="fake")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[2] == "fake"

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_locale_parameter(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() passes locale through."""
        mock_scrub_string.return_value = "Scrubbed"

        loclean.scrub("test", strategies=["person"], mode="fake", locale="en_US")

        call_args = mock_scrub_string.call_args[0]
        assert call_args[3] == "en_US"

    @patch("loclean.privacy.scrub.scrub_dataframe")
    def test_with_target_col_for_dataframe(
        self, mock_scrub_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """scrub() passes target_col to scrub_dataframe."""
        mock_scrub_df.return_value = sample_polars_df

        loclean.scrub(sample_polars_df, target_col="weight", strategies=["phone"])

        call_args = mock_scrub_df.call_args[0]
        assert call_args[1] == "weight"

    def test_valueerror_when_target_col_missing_for_dataframe(
        self, sample_polars_df: pl.DataFrame
    ) -> None:
        """scrub() raises ValueError when target_col is missing for DataFrame."""
        with pytest.raises(ValueError, match="target_col required for DataFrame input"):
            loclean.scrub(sample_polars_df, strategies=["phone"])

    @patch("loclean.get_engine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_with_llm_strategies_requires_inference_engine(
        self, mock_scrub_string: Mock, mock_get_engine: Mock
    ) -> None:
        """scrub() creates engine for LLM-based strategies (person, address)."""
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
        """scrub() does not create engine for regex-only strategies."""
        mock_scrub_string.return_value = "[PHONE]"

        loclean.scrub("555-1234", strategies=["phone", "email"])

        call_args = mock_scrub_string.call_args
        assert call_args[1]["inference_engine"] is None

    @patch("loclean.OllamaEngine")
    @patch("loclean.privacy.scrub.scrub_string")
    def test_engine_configuration_parameters_model(
        self, mock_scrub_string: Mock, mock_engine_class: Mock
    ) -> None:
        """scrub() creates dedicated engine when model= is passed."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_scrub_string.return_value = "[PERSON]"

        loclean.scrub("John", strategies=["person"], model="llama3")

        mock_engine_class.assert_called_once()


class TestExtract:
    """Test cases for extract function."""

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_string_input(self, mock_extractor_class: Mock) -> None:
        """extract() works with string input."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            result = loclean.extract("Selling red t-shirt for 50k", Product)

            assert isinstance(result, Product)
            assert result.name == "t-shirt"

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_dataframe_input(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """extract() works with DataFrame input."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            result = loclean.extract(sample_polars_df, Product, target_col="weight")

            assert isinstance(result, pl.DataFrame)
            mock_extract_df.assert_called_once()

    def test_with_invalid_schema_not_basemodel_valueerror(self) -> None:
        """extract() raises ValueError for non-BaseModel schema."""

        class NotBaseModel:
            pass

        with pytest.raises(ValueError, match="Schema must be a Pydantic BaseModel"):
            loclean.extract("test", NotBaseModel)  # type: ignore[arg-type]

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_custom_instruction(self, mock_extractor_class: Mock) -> None:
        """extract() passes custom instruction to Extractor."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product, instruction="Custom instruction")

            mock_extractor.extract.assert_called_once()
            call_args = mock_extractor.extract.call_args
            assert call_args[0][2] == "Custom instruction"

    @patch("loclean.extraction.extractor.Extractor")
    def test_with_max_retries_parameter(self, mock_extractor_class: Mock) -> None:
        """extract() passes max_retries to Extractor."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            loclean.extract("test", Product, max_retries=5)

            mock_extractor_class.assert_called_once()
            call_kwargs = mock_extractor_class.call_args[1]
            assert call_kwargs["max_retries"] == 5

    @patch("loclean.OllamaEngine")
    @patch("loclean.extraction.extractor.Extractor")
    def test_engine_configuration_parameters(
        self, mock_extractor_class: Mock, mock_engine_class: Mock
    ) -> None:
        """extract() creates dedicated engine when model= is passed."""
        mock_extractor = MagicMock()
        mock_result = Product(name="t-shirt", price=50000, color="red")
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        loclean.extract("test", Product, model="llama3")

        mock_engine_class.assert_called_once()

    @patch("loclean.get_engine")
    def test_valueerror_when_target_col_missing_for_dataframe(
        self, mock_get_engine: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """extract() raises ValueError when target_col missing for DataFrame."""
        with pytest.raises(ValueError, match="target_col required for DataFrame input"):
            loclean.extract(sample_polars_df, Product)

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_output_type_dict_default(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """extract() defaults to output_type='dict'."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            loclean.extract(sample_polars_df, Product, target_col="weight")

            call_kwargs = mock_extract_df.call_args[1]
            assert call_kwargs["output_type"] == "dict"

    @patch("loclean.extraction.extract_dataframe.extract_dataframe")
    def test_with_output_type_pydantic(
        self, mock_extract_df: Mock, sample_polars_df: pl.DataFrame
    ) -> None:
        """extract() forwards output_type='pydantic'."""
        mock_extract_df.return_value = sample_polars_df

        with patch("loclean.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            loclean.extract(
                sample_polars_df, Product, target_col="weight", output_type="pydantic"
            )

            call_kwargs = mock_extract_df.call_args[1]
            assert call_kwargs["output_type"] == "pydantic"
