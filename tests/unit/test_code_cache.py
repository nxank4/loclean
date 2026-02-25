"""Tests for code cache integration and hash key generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from loclean.cache import LocleanCache
from loclean.extraction.feature_discovery import FeatureDiscovery
from loclean.extraction.shredder import RelationalShredder, _RelationalSchema, _TableDef
from loclean.utils.cache_keys import compute_code_key

# ------------------------------------------------------------------
# compute_code_key
# ------------------------------------------------------------------


class TestComputeCodeKey:
    def test_deterministic(self) -> None:
        k1 = compute_code_key(
            columns=["a", "b"],
            dtypes=["int", "float"],
            target_col="t",
            module_prefix="test",
        )
        k2 = compute_code_key(
            columns=["a", "b"],
            dtypes=["int", "float"],
            target_col="t",
            module_prefix="test",
        )
        assert k1 == k2

    def test_order_invariant(self) -> None:
        k1 = compute_code_key(
            columns=["b", "a"],
            dtypes=["float", "int"],
            target_col="t",
            module_prefix="test",
        )
        k2 = compute_code_key(
            columns=["a", "b"],
            dtypes=["int", "float"],
            target_col="t",
            module_prefix="test",
        )
        assert k1 == k2

    def test_differs_by_module(self) -> None:
        k1 = compute_code_key(
            columns=["a"],
            dtypes=["int"],
            target_col="t",
            module_prefix="feature_discovery",
        )
        k2 = compute_code_key(
            columns=["a"],
            dtypes=["int"],
            target_col="t",
            module_prefix="shredder",
        )
        assert k1 != k2

    def test_differs_by_target(self) -> None:
        k1 = compute_code_key(
            columns=["a"],
            dtypes=["int"],
            target_col="x",
            module_prefix="test",
        )
        k2 = compute_code_key(
            columns=["a"],
            dtypes=["int"],
            target_col="y",
            module_prefix="test",
        )
        assert k1 != k2

    def test_returns_hex_sha256(self) -> None:
        k = compute_code_key(
            columns=["a"],
            dtypes=["int"],
            target_col="t",
            module_prefix="test",
        )
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ------------------------------------------------------------------
# LocleanCache.get_code / .set_code
# ------------------------------------------------------------------


class TestCodeCacheRoundtrip:
    def test_miss_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocleanCache(cache_dir=Path(tmp))
            assert cache.get_code("unknown") is None
            cache.close()

    def test_set_then_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocleanCache(cache_dir=Path(tmp))
            cache.set_code("abc123", "def f(): return 1")
            assert cache.get_code("abc123") == "def f(): return 1"
            cache.close()

    def test_upsert_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocleanCache(cache_dir=Path(tmp))
            cache.set_code("key", "v1")
            cache.set_code("key", "v2")
            assert cache.get_code("key") == "v2"
            cache.close()


# ------------------------------------------------------------------
# FeatureDiscovery — cache integration & graceful fallback
# ------------------------------------------------------------------

VALID_FEATURE_SOURCE = """
def generate_features(row: dict) -> dict:
    result = {}
    try:
        result["sum_a_b"] = row.get("a", 0) + row.get("b", 0)
    except Exception:
        result["sum_a_b"] = None
    return result
"""

SAMPLE_DF = pl.DataFrame(
    {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "target": [0, 1, 0]}
)


class TestDiscoverCacheHit:
    def test_skips_llm_on_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocleanCache(cache_dir=Path(tmp))
            engine = MagicMock()
            engine.verbose = False

            key = compute_code_key(
                columns=["a", "b", "target"],
                dtypes=[str(SAMPLE_DF[c].dtype) for c in SAMPLE_DF.columns],
                target_col="target",
                module_prefix="feature_discovery",
            )
            cache.set_code(key, VALID_FEATURE_SOURCE)

            fd = FeatureDiscovery(
                inference_engine=engine,
                n_features=1,
                cache=cache,
            )
            result = fd.discover(SAMPLE_DF, "target")

            engine.generate.assert_not_called()
            assert "sum_a_b" in result.columns
            cache.close()


class TestDiscoverCacheMissStores:
    def test_stores_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocleanCache(cache_dir=Path(tmp))
            engine = MagicMock()
            engine.verbose = False
            engine.generate.return_value = VALID_FEATURE_SOURCE

            fd = FeatureDiscovery(
                inference_engine=engine,
                n_features=1,
                cache=cache,
            )
            result = fd.discover(SAMPLE_DF, "target")

            engine.generate.assert_called_once()
            assert "sum_a_b" in result.columns

            key = compute_code_key(
                columns=["a", "b", "target"],
                dtypes=[str(SAMPLE_DF[c].dtype) for c in SAMPLE_DF.columns],
                target_col="target",
                module_prefix="feature_discovery",
            )
            assert cache.get_code(key) is not None
            cache.close()


class TestDiscoverGracefulFallback:
    def test_returns_original_on_exhausted_retries(self) -> None:
        engine = MagicMock()
        engine.verbose = False
        bad_source = "def generate_features(row): raise Exception('bad')"
        engine.generate.return_value = bad_source

        fd = FeatureDiscovery(
            inference_engine=engine,
            n_features=1,
            max_retries=2,
        )
        result = fd.discover(SAMPLE_DF, "target")
        assert list(result.columns) == list(SAMPLE_DF.columns)
        assert len(result) == len(SAMPLE_DF)


# ------------------------------------------------------------------
# RelationalShredder — graceful fallback
# ------------------------------------------------------------------


SAMPLE_LOGS_DF = pl.DataFrame({"log": ["2024-01-01 INFO foo", "2024-01-02 WARN bar"]})

SAMPLE_SCHEMA = _RelationalSchema(
    tables=[
        _TableDef(
            name="events", columns=["ts", "level"], primary_key="ts", foreign_key=None
        ),
        _TableDef(
            name="details", columns=["ts", "msg"], primary_key="ts", foreign_key="ts"
        ),
    ]
)


class TestShredGracefulFallback:
    def test_returns_empty_on_exhausted_retries(self) -> None:
        engine = MagicMock()
        engine.verbose = False
        bad_source = "def extract_relations(log): raise Exception('bad')"
        engine.generate.side_effect = [
            SAMPLE_SCHEMA.model_dump(),
            bad_source,
            bad_source,
            bad_source,
        ]

        s = RelationalShredder(
            inference_engine=engine,
            sample_size=2,
            max_retries=2,
        )
        result = s.shred(SAMPLE_LOGS_DF, "log")
        assert result == {}
