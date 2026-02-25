"""Test cases for the orchestration runner."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pydantic import ValidationError

from loclean.orchestration.runner import (
    EXIT_INVALID_PAYLOAD,
    EXIT_OK,
    EXIT_PIPELINE_ERROR,
    PipelinePayload,
    _error_response,
    main,
    run_pipeline,
)

# ------------------------------------------------------------------
# PipelinePayload
# ------------------------------------------------------------------


class TestPipelinePayload:
    def test_valid_minimal(self) -> None:
        p = PipelinePayload(
            data=[{"price": "10 kg"}],
            target_col="price",
        )
        assert p.target_col == "price"
        assert p.instruction == "Extract the numeric value and unit as-is."
        assert p.engine_config == {}
        assert p.batch_size == 50

    def test_valid_full(self) -> None:
        p = PipelinePayload(
            data=[{"x": "1"}],
            target_col="x",
            instruction="Custom instruction",
            engine_config={"model": "llama3"},
            batch_size=10,
        )
        assert p.instruction == "Custom instruction"
        assert p.engine_config["model"] == "llama3"
        assert p.batch_size == 10

    def test_missing_data_raises(self) -> None:
        with pytest.raises(ValidationError):
            PipelinePayload(target_col="x")  # type: ignore[call-arg]

    def test_missing_target_col_raises(self) -> None:
        with pytest.raises(ValidationError):
            PipelinePayload(data=[{"a": 1}])  # type: ignore[call-arg]

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PipelinePayload(
                data=[{"a": 1}],
                target_col="a",
                batch_size=0,
            )


# ------------------------------------------------------------------
# _error_response
# ------------------------------------------------------------------


class TestErrorResponse:
    def test_structure(self) -> None:
        r = _error_response(1, "bad input")
        assert r == {"status": "error", "code": 1, "message": "bad input"}


# ------------------------------------------------------------------
# run_pipeline
# ------------------------------------------------------------------


class TestRunPipeline:
    def test_happy_path(self) -> None:
        payload = PipelinePayload(
            data=[{"price": "10 kg"}, {"price": "20 lbs"}],
            target_col="price",
        )

        fake_result = pl.DataFrame(
            {
                "price": ["10 kg", "20 lbs"],
                "clean_value": [10.0, 20.0],
                "clean_unit": ["kg", "lbs"],
                "clean_reasoning": ["numeric", "numeric"],
            }
        )

        with patch("loclean.orchestration.runner.clean", return_value=fake_result):
            result = run_pipeline(payload)

        assert result["status"] == "ok"
        assert result["row_count"] == 2
        assert len(result["data"]) == 2
        assert result["data"][0]["clean_value"] == 10.0
        assert result["data"][1]["clean_unit"] == "lbs"

    def test_missing_column_raises(self) -> None:
        payload = PipelinePayload(
            data=[{"a": 1}],
            target_col="missing_col",
        )

        with pytest.raises(ValueError, match="Column 'missing_col' not found"):
            run_pipeline(payload)

    def test_engine_config_forwarded(self) -> None:
        payload = PipelinePayload(
            data=[{"x": "1"}],
            target_col="x",
            engine_config={"model": "llama3", "host": "http://custom:11434"},
        )

        fake_result = pl.DataFrame(
            {
                "x": ["1"],
                "clean_value": [1.0],
                "clean_unit": [""],
                "clean_reasoning": [""],
            }
        )

        with (
            patch("loclean.orchestration.runner.clean", return_value=fake_result),
            patch("loclean.orchestration.runner.load_config") as mock_load,
        ):
            mock_load.return_value = MagicMock(
                model="llama3", host="http://custom:11434", verbose=False
            )
            run_pipeline(payload)

            mock_load.assert_called_once_with(
                model="llama3", host="http://custom:11434"
            )


# ------------------------------------------------------------------
# main (stdin/stdout integration)
# ------------------------------------------------------------------


def _run_main(stdin_data: str) -> tuple[str, int]:
    """Helper: run main() with given stdin, capture stdout + exit code."""
    exit_code = EXIT_OK
    stdout_buf = StringIO()

    def fake_exit(code: int) -> None:
        nonlocal exit_code
        exit_code = code

    with (
        patch("sys.stdin", StringIO(stdin_data)),
        patch("sys.stdout", stdout_buf),
        patch("sys.exit", side_effect=fake_exit),
    ):
        try:
            main()
        except SystemExit:
            pass

    return stdout_buf.getvalue(), exit_code


class TestMain:
    def test_invalid_json_exits_1(self) -> None:
        output, code = _run_main("not json {{{")
        assert code == EXIT_INVALID_PAYLOAD
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert parsed["code"] == EXIT_INVALID_PAYLOAD

    def test_invalid_payload_schema_exits_1(self) -> None:
        output, code = _run_main(json.dumps({"wrong_key": 123}))
        assert code == EXIT_INVALID_PAYLOAD
        parsed = json.loads(output)
        assert "validation" in parsed["message"].lower()

    def test_pipeline_error_exits_2(self) -> None:
        payload = {"data": [{"x": "1"}], "target_col": "x"}
        with patch(
            "loclean.orchestration.runner.run_pipeline",
            side_effect=RuntimeError("engine down"),
        ):
            output, code = _run_main(json.dumps(payload))

        assert code == EXIT_PIPELINE_ERROR
        parsed = json.loads(output)
        assert parsed["code"] == EXIT_PIPELINE_ERROR
        assert "engine down" in parsed["message"]

    def test_successful_round_trip(self) -> None:
        payload = {"data": [{"col": "5 kg"}], "target_col": "col"}
        expected_output = {
            "status": "ok",
            "data": [
                {
                    "col": "5 kg",
                    "clean_value": 5.0,
                    "clean_unit": "kg",
                    "clean_reasoning": "parsed",
                }
            ],
            "row_count": 1,
        }

        with patch(
            "loclean.orchestration.runner.run_pipeline",
            return_value=expected_output,
        ):
            output, code = _run_main(json.dumps(payload))

        assert code == EXIT_OK
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["row_count"] == 1
