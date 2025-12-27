from typing import Any

import polars as pl
import pytest


@pytest.fixture
def temp_cache_db(tmp_path: Any) -> Any:
    """
    Returns a path to a temporary cache database file.
    This ensures we don't overwrite the user's actual cache.
    """
    return tmp_path / "test_loclean_cache.db"


@pytest.fixture
def messy_df() -> pl.DataFrame:
    """
    Returns a Polars DataFrame with messy data for integration tests.
    """
    return pl.DataFrame({"weight": ["10kg", "500g"], "price": ["$10", "20 EUR"]})
