# Semantix ⚡🧠

**The All-in-One Local AI Data Cleaner.**

Clean messy tabular data using local AI.
**No API keys required. No GPU required. 100x faster than standard LLM loops.**

## 🔥 Why Semantix?

_in progress..._

## 🚀 Installation

```bash
pip install semantix
```

_Note: The first time you run Semantix, it will automatically download the optimized Microsoft Phi-3 Mini model (~2.4GB) to `~/.cache/semantix`. Subsequent runs are instant._

## ⚡ Quick Start

Clean messy weights, distances, or generic units instantly.

```python
import semantix
import polars as pl

# 1. Load messy data
df = pl.DataFrame({
    "raw_weight": ["10kg", "500g", "2 lbs", "10 kgs", "not a weight"]
})

# 2. Clean it! (Default: Extract Value & Unit)
df_clean = semantix.clean(df, target_col="raw_weight")

print(df_clean)
```

**Output:**

```text
┌────────────┬─────────────┬────────────┐
│ raw_weight ┆ clean_value ┆ clean_unit │
│ ---        ┆ ---         ┆ ---        │
│ str        ┆ f64         ┆ str        │
╞════════════╪═════════════╪════════════╡
│ 10kg       ┆ 10.0        ┆ kg         │
│ 500g       ┆ 500.0       ┆ g          │
│ 2 lbs      ┆ 2.0         ┆ lbs        │
│ ...        ┆ ...         ┆ ...        │
└────────────┴─────────────┴────────────┘
```

## 🏗️ How It Works (The Architecture)

Semantix achieves its massive speedup through a **Representative Sampling** architecture:

1.  **⚡ Vectorized Sampling**: We use `Polars` to extract the `unique()` patterns from your specific column. In a dataset of 1M rows, there are often only ~1k unique "messy formats".
2.  **🧠 Local Inference**: We feed _only_ the unique patterns to a local, quantized **Phi-3 Mini** model running on `llama.cpp`.
3.  **🛡️ Structured Decoding**: We use **GBNF Grammars** to force the LLM to output valid JSON `{"value": float, "unit": str}`. It _cannot_ hallucinate conversational filler.
4.  **🔗 Broadcast Join**: The results are mapped back to your original Big Data frame using a high-performance Left Join.

## 🗺️ Roadmap

- [ ] **Schema Enforcement**: Force output to match Pydantic models.
- [ ] **Row-Level Imputation**: Fill `null` values based on other column context.
- [ ] **Entity Resolution**: "Apple Inc." == "Apple Computer, Inc."

_Built with ❤️ for the Data Community._
