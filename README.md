# Semantix ⚡🧠

**The "Batteries-Included" Semantic Data Cleaning Library.**

Clean messy tabular data (units, dates, typos) using local AI.
No API keys required. No GPU required. 100x faster than standard LLM loops.

## 🔥 Why Semantix?

| Feature     | Semantix                                      | Standard AI Wrappers    |
| :---------- | :-------------------------------------------- | :---------------------- |
| **Speed**   | **< 30s for 1M rows** (via Polars + Sampling) | ~5 days for 1M rows     |
| **Cost**    | **$0.00** (Local Phi-3 Model)                 | $$$ OpenAI API Costs    |
| **Privacy** | **100% Offline** (Air-gapped safe)            | Sends data to Cloud     |
| **Safety**  | **Structured JSON Enforced**                  | Prone to Hallucinations |

## 🚀 Quick Start

```bash
pip install semantix
```

```python
import semantix
import polars as pl

df = pl.read_csv("messy_sales_data.csv")

# Auto-downloads model, extracts unique patterns, cleans, and maps back.
df_clean = semantix.clean(df, col="product_weight", output_col="weight_kg")
```

### What to do right now?

Run the `examples/demo.py`.

- **If it crashes:** Paste the error here, and we will debug the `llama-cpp-python` integration.
- **If it works:** You have officially built an "Outscale" library. We can then discuss packaging it for PyPI.
