import time
import logging
import polars as pl
import semantix
import numpy as np

# Setup logging
logging.basicConfig(level=logging.ERROR)

def naive_llm_call(text: str) -> None:
    """Simulates a very fast API or local LLM call (0.1s latency)."""
    time.sleep(0.1)

def run_benchmark():
    ROWS = 10_000 # Reduced slightly for quicker iterative testing in this context, but large enough
    UNIQUE_PATTERNS = 100 
    
    print(f"🏁 BENCHMARKING: Cleaning {ROWS:,} rows")
    print(f"   (Simulating High-Repetition Data: ~{UNIQUE_PATTERNS} unique patterns)")

    # 1. Generate Data
    base_patterns = [f"{i} kg" for i in range(UNIQUE_PATTERNS)]
    # Repeat patterns to fill ROWS
    full_data = (base_patterns * (ROWS // len(base_patterns) + 1))[:ROWS]
    df = pl.DataFrame({"raw_weight": full_data})
    
    print(f"   Dataset: {df.shape} | Columns: {df.columns}")

    # 2. Baseline (Naive) - Extrapolated
    print("\n1️⃣  Estimating 'Naive' Loop (Standard API approach)...")
    print("    Running simulation on 5 samples (0.1s latency/req)...")
    
    naive_samples = df["raw_weight"].head(5).to_list()
    start_naive = time.time()
    for item in naive_samples:
        naive_llm_call(item)
    avg_naive_time = (time.time() - start_naive) / 5
    projected_naive_time = avg_naive_time * ROWS
    
    print(f"    Avg time per item: {avg_naive_time:.4f}s")
    print(f"    Projected Total Time: {projected_naive_time:.2f}s ({projected_naive_time/3600:.2f} hours)")

    # 3. Semantix (First Run)
    print("\n2️⃣  Running Semantix (Run 1: Vectorized + Cached Model)...")
    print("    Processing full dataset...")
    
    # Force a unique instruction to ensure we aren't hitting the disk cache from previous benchmark runs
    # if we want to test "Processing Speed". 
    # BUT, to test "Cache Speedup" we want to run this same thing twice.
    # To act like a "First Run", we use a unique instruction.
    run_id = int(time.time())
    instruction = f"Convert to kg. Run ID {run_id}"

    start_semantix = time.time()
<<<<<<< HEAD
    df_clean = semantix.clean(df, "raw_weight", instruction=instruction)
    actual_semantix_time = time.time() - start_semantix
=======
    semantix.clean(df, "raw_weight")
    end_semantix = time.time()
>>>>>>> origin/main
    
    print(f"    Actual Time: {actual_semantix_time:.2f}s")
    
    # 4. Semantix (Second Run - Cache Hit)
    print("\n3️⃣  Running Semantix (Run 2: Persistent Cache Hit)...")
    print("    Re-running same dataset and instruction...")
    
    start_cache = time.time()
    _ = semantix.clean(df, "raw_weight", instruction=instruction)
    actual_cache_time = time.time() - start_cache
    
    print(f"    Actual Time: {actual_cache_time:.4f}s")

    # 5. Reporting
    vector_speedup = projected_naive_time / actual_semantix_time
    cache_speedup = actual_semantix_time / (actual_cache_time if actual_cache_time > 0 else 0.001)

    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'NAIVE (Est.)':<15} | {'SEMANTIX (Run 1)':<20} | {'CACHE (Run 2)':<15}")
    print("-" * 80)
    print(f"{'Time':<25} | {projected_naive_time:<15.2f}s | {actual_semantix_time:<20.2f}s | {actual_cache_time:<15.4f}s")
    print("-" * 80)
    print(f"🚀 VECTOR SPEEDUP: {int(vector_speedup)}x faster than naive loop")
    print(f"🚀 CACHE SPEEDUP:  {int(cache_speedup)}x faster than first run")
    print("="*80)

if __name__ == "__main__":
    run_benchmark()