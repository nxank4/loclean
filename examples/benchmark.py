import time
import logging
import polars as pl
import semantix

# Setup logging to avoid cluttering benchmark output, but keep errors visible
logging.basicConfig(level=logging.ERROR)

def naive_llm_call(text: str) -> None:
    """
    Simulates a very fast API or local LLM call.
    Latency: 0.1 seconds per request.
    """
    time.sleep(0.1)

def run_benchmark():
    ROWS = 100_000
    UNIQUE_PATTERNS = 100 
    
    print(f"🏁 BENCHMARKING: Cleaning {ROWS:,} rows")
    print(f"   (Simulating High-Repetition Data: ~{UNIQUE_PATTERNS} unique patterns)")

    # 1. Generate Data
    # Create a list of ~100 unique patterns repeated to fill 100k rows
    base_patterns = [f"{i} kg" for i in range(UNIQUE_PATTERNS)]
    
    # Repeat patterns to fill ROWS
    import numpy as np
    # Use numpy for efficient repetition or just list mult
    # list mult is slightly less random but fine for this benchmark
    full_data = (base_patterns * (ROWS // len(base_patterns) + 1))[:ROWS]
    
    df = pl.DataFrame({"raw_weight": full_data})
    
    print(f"   Dataset Shape: {df.shape}")

    # 2. Baseline (Naive) - Extrapolated
    print("\n1️⃣ Estimating 'Naive' approach (Standard Loop)...")
    print("   Running naive simulation on 5 samples (0.1s latency/req)...")
    
    naive_samples = df["raw_weight"].head(5).to_list()
    start_naive = time.time()
    for item in naive_samples:
        naive_llm_call(item)
    end_naive = time.time()
    
    avg_naive_time = (end_naive - start_naive) / 5
    projected_naive_time = avg_naive_time * ROWS
    
    print(f"   Avg time per item: {avg_naive_time:.4f}s")
    print(f"   Projected Total Time: {projected_naive_time:.2f}s ({projected_naive_time/3600:.2f} hours)")

    # 3. Semantix (Optimized) - Actual
    print("\n2️⃣ Running Semantix (Vectorized AI)...")
    print("   Initializing engine (this might take a few seconds if loading model)...")
    
    # Ensure cached model is ready or let it load
    # We want to include model loading time in the first run usually, 
    # OR we can warm it up if we only care about processing speed.
    # The prompt implies "actual semantix time" for the cleaning process.
    # user Said: "Initialize semantix.clean. Run it on the entire 100,000 row dataset. Measure actual_semantix_time."
    
    start_semantix = time.time()
    df_clean = semantix.clean(df, "raw_weight")
    end_semantix = time.time()
    
    actual_semantix_time = end_semantix - start_semantix
    
    print(f"   Actual Semantix Time: {actual_semantix_time:.2f}s")
    
    # 4. Reporting
    speedup = projected_naive_time / actual_semantix_time
    
    print("\n" + "="*60)
    print(f"{'METRIC':<20} | {'NAIVE LOOP (Est.)':<20} | {'SEMANTIX (Actual)':<15}")
    print("-" * 60)
    print(f"{'Time (Sec)':<20} | {projected_naive_time:<20.2f} | {actual_semantix_time:<15.2f}")
    print(f"{'Time (Hours)':<20} | {projected_naive_time/3600:<20.2f} | {actual_semantix_time/3600:<15.4f}")
    print("-" * 60)
    print(f"🚀 SPEEDUP FACTOR: {int(speedup)}x FASTER")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()