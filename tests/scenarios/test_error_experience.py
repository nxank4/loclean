from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import loclean
from loclean.inference.local.exceptions import ModelNotFoundError


def test_error_invalid_column_name() -> None:
    """Verify that a helpful error is raised when the target column is missing."""
    df = pl.DataFrame({"real_col": ["data"]})

    with pytest.raises(ValueError) as excinfo:
        loclean.clean(df, target_col="wrong_col")

    assert "Column 'wrong_col' not found" in str(excinfo.value)


def test_error_model_not_found_message() -> None:
    """Verify that ModelNotFoundError provides a helpful message (conceptually)."""
    # We mock download_model to raise our specific exception
    # ModelNotFoundError(message, model_name, repo_id, filename)
    err = ModelNotFoundError("Not found", "non-existent", "repo", "file.gguf")
    with patch("loclean.inference.local.llama_cpp.download_model", side_effect=err):
        with pytest.raises(ModelNotFoundError) as excinfo:
            loclean.clean(pl.DataFrame({"a": ["b"]}), "a", model_name="non-existent")

    assert "Not found" in str(excinfo.value)


def test_json_repair_recovery() -> None:
    """Verify that if LLM returns bad JSON, we use json-repair to recover."""
    from loclean.extraction.extractor import Extractor

    mock_engine = MagicMock()
    mock_engine.verbose = False
    mock_engine.adapter.format.return_value = "prompt"
    mock_engine.adapter.get_stop_tokens.return_value = []

    # Return broken JSON that needs repair (missing closing brace)
    mock_engine.llm.create_completion.return_value = {
        "choices": [{"text": '{"value": 10.0, "unit": "kg", "reasoning": "test"'}]
    }

    extractor = Extractor(inference_engine=mock_engine)

    from pydantic import BaseModel

    class SimpleSchema(BaseModel):
        value: float
        unit: str
        reasoning: str

    # Should work thanks to json-repair
    # We use Any here to satisfy mypy for the extracted result attributes
    result: Any = extractor.extract("some text", SimpleSchema)

    assert result is not None
    assert result.value == 10.0
    assert result.unit == "kg"


def test_error_empty_dataframe() -> None:
    """Verify behavior on empty dataframe."""
    df = pl.DataFrame({"a": []}, schema={"a": pl.String})

    # Should probably return empty df or log warning
    with patch("loclean.engine.narwhals_ops.logger") as mock_logger:
        result = loclean.clean(df, "a")

    assert result.shape[0] == 0
    assert any(
        "No valid unique values found" in str(call.args[0])
        for call in mock_logger.warning.call_args_list
    )
