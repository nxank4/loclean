"""Loclean performance benchmark.

Demonstrates the speedup from vectorized deduplication and persistent
caching on a large-scale synthetic dataset.  Ollama is started and the
model is pulled automatically if needed — no manual setup required.
"""

import logging
import random
import time
from typing import List

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import loclean

logging.basicConfig(level=logging.INFO)
console = Console()


def generate_complex_messy_data(rows: int, unique_patterns: int) -> List[str]:
    """Generate extremely messy weight data that would be painful for regex."""
    templates = [
        "Product {id}: weight is {val} {unit} (checked)",
        "Approx {val}{unit} - [ID:{id}]",
        "{val} {unit} package",
        "Weight: {val} {unit}",
        "Package weighs nearly {val} {unit} - fragile",
        "ID-{id} | {val}{unit} | Shipment A",
        "The item is {val} {unit}",
    ]

    units = [
        ("kg", 1),
        ("kilograms", 1),
        ("kgs", 1),
        ("kilos", 1),
        ("g", 0.001),
        ("grams", 0.001),
        ("t", 1000),
        ("tonnes", 1000),
        ("tons", 1000),
        ("mg", 0.000001),
        ("milligrams", 0.000001),
    ]

    values = [
        str(round(random.uniform(0.1, 500), 2)) for _ in range(unique_patterns // 2)
    ]
    values += ["five", "twelve", "half", "zero point five", "ten"]

    patterns = []
    for _ in range(unique_patterns):
        tpl = random.choice(templates)
        val = random.choice(values)
        unit_name, _ = random.choice(units)
        patterns.append(
            tpl.format(id=random.randint(1000, 9999), val=val, unit=unit_name)
        )

    return (patterns * (rows // len(patterns) + 1))[:rows]


def run_benchmark() -> None:
    ROWS = 100_000
    UNIQUE_PATTERNS = 30

    console.print(
        Panel(
            f"[bold white]Loclean Performance Benchmark[/bold white]\n"
            f"[dim]Scale: {ROWS:,} rows | "
            f"Unique Messy Patterns: {UNIQUE_PATTERNS}[/dim]",
            title="[bold magenta]Loclean[/bold magenta]",
            subtitle="[italic]Vectorized Deduplication Test[/italic]",
            border_style="bright_blue",
        )
    )

    # 1. Generate data
    with console.status("[bold green]Generating massive messy dataset..."):
        data = generate_complex_messy_data(ROWS, UNIQUE_PATTERNS)
        df = pl.DataFrame({"raw_info": data})

    console.print(f"✅ Dataset ready: [bold]{df.shape}[/bold]")

    # 2. Baseline LLM latency (average of 5 samples)
    console.print(
        "\n[bold cyan]1. Measuring Baseline LLM Speed "
        "(Average of 5 samples)...[/bold cyan]"
    )
    instruction = (
        "Extract numeric weight and normalize to 'kg'. "
        "Strictly follow: 1 ton = 1000kg, 1g = 0.001kg, 1mg = 1e-6kg. "
        "Text numbers: 'half' = 0.5, 'twelve' = 12. "
        "If input is 'twelvemg', value is 0.000012."
    )

    sample_df = df.sample(n=5)
    sample_start = time.time()
    _ = loclean.clean(sample_df, "raw_info", instruction=instruction)
    actual_latency = (time.time() - sample_start) / 5

    projected_naive_time = actual_latency * ROWS
    console.print(
        f"   Measured Avg Latency: [bold yellow]{actual_latency:.2f}s/row[/bold yellow]"
    )
    console.print(
        f"   Projected Naive Loop: [bold red]{projected_naive_time:,.2f}s[/bold red] "
        f"({projected_naive_time / 3600:.2f} hours)"
    )

    # 3. Loclean Run 1 — deduplicated
    console.print(
        "\n[bold cyan]2. Running Loclean (Run 1: Vectorized + Deduplicated)[/bold cyan]"
    )
    benchmark_instruction = f"{instruction} | BenchID:{int(time.time())}"

    start_loclean = time.time()
    result = loclean.clean(
        df, "raw_info", instruction=benchmark_instruction, batch_size=10
    )
    actual_loclean_time = time.time() - start_loclean

    console.print(
        f"   Loclean processed {ROWS:,} rows in: "
        f"[bold green]{actual_loclean_time:.2f}s[/bold green]"
    )

    # 4. Loclean Run 2 — cache hit
    console.print(
        "\n[bold cyan]3. Running Loclean (Run 2: Persistent Cache Hit)[/bold cyan]"
    )
    start_cache = time.time()
    _ = loclean.clean(df, "raw_info", instruction=benchmark_instruction)
    actual_cache_time = time.time() - start_cache

    console.print(
        f"   Cache hit finished in: [bold green]{actual_cache_time:.4f}s[/bold green]"
    )

    # 5. Report
    vector_speedup = projected_naive_time / actual_loclean_time
    cache_total_speedup = projected_naive_time / max(actual_cache_time, 0.0001)

    table = Table(
        title="Performance Summary (Relative to Naive Baseline)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Strategy")
    table.add_column("Time", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Total Speedup", justify="left")

    table.add_row(
        "Naive Sequential Loop",
        f"{projected_naive_time:,.2f}s",
        f"{1 / actual_latency:.2f} rows/s",
        "Baseline",
    )
    table.add_row(
        "Loclean (Run 1: Vectorized)",
        f"{actual_loclean_time:.2f}s",
        f"{ROWS / actual_loclean_time:,.1f} rows/s",
        f"[bold green]{int(vector_speedup):,}x faster[/bold green]",
    )
    table.add_row(
        "Loclean (Run 2: Cached)",
        f"{actual_cache_time:.4f}s",
        f"{ROWS / actual_cache_time:,.1f} rows/s",
        f"[bold blue]{int(cache_total_speedup):,}x faster[/bold blue]",
    )

    console.print("\n", table)

    console.print(
        "\n[bold yellow]Sample of Cleaned Data "
        "(Normalizing complex text):[/bold yellow]"
    )
    console.print(result.select(["raw_info", "clean_value", "clean_unit"]).head(5))

    # 6. Verbose debug demo
    sample_to_debug = df.head(1)["raw_info"][0]
    console.print(
        "\n[bold red]BONUS: Debugging a sample using Verbose Mode "
        "(Forcing Cache Miss)[/bold red]"
    )
    console.print(f"Item: [dim]{sample_to_debug}[/dim]")

    debug_instruction = f"{benchmark_instruction} [FORCE_DEBUG_MISS_{int(time.time())}]"

    loclean.clean(
        pl.DataFrame({"raw_info": [sample_to_debug]}),
        "raw_info",
        instruction=debug_instruction,
        verbose=True,
    )


if __name__ == "__main__":
    run_benchmark()
