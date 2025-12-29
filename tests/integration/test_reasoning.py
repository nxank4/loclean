import logging

import polars as pl
import pytest

import loclean

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_smart_no_op() -> None:
    """
    Test that the model uses reasoning to avoid 'Blind Math'.
    Scenario: Input is already in target currency. Should NOT apply conversion rate.
    Old Bug: $50 * 1.1 = $55.
    Expected: $50 -> $50.
    """
    df = pl.DataFrame({"price": ["$50", "100 USD"]})
    instruction = "Convert to USD. Rates: 1 EUR = 1.1 USD."

    clean_df = loclean.clean(df, target_col="price", instruction=instruction)

    # Check $50
    fifty = clean_df.filter(pl.col("price") == "$50")
    val_fifty = fifty.select("clean_value").item()

    logger.info(
        "Reasoning for '$50': %s",
        fifty.select("clean_reasoning").item(),
    )

    assert val_fifty == 50.0, (
        f"Expected 50.0 blocked op, but got {val_fifty}. (Blind math detector)"
    )

    # Check 100 USD
    hundred = clean_df.filter(pl.col("price") == "100 USD")
    val_hundred = hundred.select("clean_value").item()
    assert val_hundred == 100.0


@pytest.mark.slow
def test_complex_reasoning() -> None:
    """
    Test reasoning on mixed logic (Conversion vs No-Op).
    """
    df = pl.DataFrame({"temp": ["32 F", "100 C"]})
    instruction = "Convert to Celsius."

    clean_df = loclean.clean(df, target_col="temp", instruction=instruction)

    # 32 F -> 0 C (Conversion)
    f_row = clean_df.filter(pl.col("temp") == "32 F")
    assert f_row.select("clean_value").item() == 0.0
    unit = str(f_row.select("clean_unit").item()).lower()
    assert "celsius" in unit or "c" == unit

    # 100 C -> 100 C (No-Op)
    c_row = clean_df.filter(pl.col("temp") == "100 C")
    assert c_row.select("clean_value").item() == 100.0


@pytest.mark.slow
def test_reasoning_column_exists() -> None:
    """
    Verify that the 'clean_reasoning' column is exposed to the user.
    """
    df = pl.DataFrame({"val": ["10 units"]})
    clean_df = loclean.clean(df, target_col="val", instruction="Extract value.")

    assert "clean_reasoning" in clean_df.columns
    reasoning = clean_df.select("clean_reasoning").item()
    assert isinstance(reasoning, str)
    assert len(reasoning) > 0
