"""Tests for _resolve_engine helper and Loclean class wrapper methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import loclean

# ------------------------------------------------------------------
# _resolve_engine
# ------------------------------------------------------------------


class TestResolveEngine:
    @patch("loclean.get_engine")
    def test_defaults_return_singleton(self, mock_get: MagicMock) -> None:
        sentinel = MagicMock()
        mock_get.return_value = sentinel
        result = loclean._resolve_engine()
        mock_get.assert_called_once()
        assert result is sentinel

    @patch("loclean.OllamaEngine")
    def test_custom_args_create_new_client(self, mock_cls: MagicMock) -> None:
        sentinel = MagicMock()
        mock_cls.return_value = sentinel
        result = loclean._resolve_engine(model="llama3", verbose=True)
        mock_cls.assert_called_once_with(model="llama3", verbose=True)
        assert result is sentinel

    @patch("loclean.OllamaEngine")
    def test_extra_kwargs_forwarded(self, mock_cls: MagicMock) -> None:
        loclean._resolve_engine(timeout=30)
        mock_cls.assert_called_once_with(timeout=30)


# ------------------------------------------------------------------
# Loclean wrapper methods
# ------------------------------------------------------------------


SAMPLE_DF = pl.DataFrame({"a": [1, 2], "b": [3, 4]})


class TestLocleanWrappers:
    @pytest.fixture
    def client(self) -> MagicMock:
        with patch("loclean.OllamaEngine") as mock_cls:
            engine = MagicMock()
            engine.verbose = False
            mock_cls.return_value = engine
            inst = loclean.Loclean(model="test")
            inst.engine = engine
            return inst

    @patch("loclean.NarwhalsEngine.process_column")
    def test_clean_delegates(self, mock_proc: MagicMock, client: MagicMock) -> None:
        mock_proc.return_value = SAMPLE_DF
        client.clean(SAMPLE_DF, "a")
        mock_proc.assert_called_once()
        _, kwargs = mock_proc.call_args
        assert kwargs.get("batch_size") == 50

    @patch("loclean.extraction.resolver.EntityResolver.resolve")
    def test_resolve_entities_delegates(
        self, mock_resolve: MagicMock, client: MagicMock
    ) -> None:
        mock_resolve.return_value = SAMPLE_DF
        client.resolve_entities(SAMPLE_DF, "a")
        mock_resolve.assert_called_once()

    @patch("loclean.extraction.oversampler.SemanticOversampler.oversample")
    def test_oversample_delegates(self, mock_os: MagicMock, client: MagicMock) -> None:
        mock_os.return_value = SAMPLE_DF
        from pydantic import BaseModel

        class DummySchema(BaseModel):
            a: int
            b: int

        client.oversample(SAMPLE_DF, "a", 1, 5, DummySchema)
        mock_os.assert_called_once()

    @patch("loclean.extraction.shredder.RelationalShredder.shred")
    def test_shred_delegates(self, mock_shred: MagicMock, client: MagicMock) -> None:
        mock_shred.return_value = {"t1": SAMPLE_DF}
        client.shred_to_relations(SAMPLE_DF, "a")
        mock_shred.assert_called_once()

    @patch("loclean.extraction.feature_discovery.FeatureDiscovery.discover")
    def test_discover_features_delegates(
        self, mock_disc: MagicMock, client: MagicMock
    ) -> None:
        mock_disc.return_value = SAMPLE_DF
        client.discover_features(SAMPLE_DF, "a")
        mock_disc.assert_called_once()

    @patch("loclean.validation.quality_gate.QualityGate.evaluate")
    def test_validate_quality_delegates(
        self, mock_eval: MagicMock, client: MagicMock
    ) -> None:
        mock_eval.return_value = {"total_rows": 2, "passed_rows": 2}
        client.validate_quality(SAMPLE_DF, ["rule1"])
        mock_eval.assert_called_once()
