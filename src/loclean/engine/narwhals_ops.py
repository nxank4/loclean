import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import narwhals as nw
from narwhals.typing import IntoFrameT

if TYPE_CHECKING:
    from loclean.inference.base import InferenceEngine

from loclean.utils.logging import configure_module_logger
from loclean.utils.rich_output import (
    create_progress,
    log_batch_stats,
    log_processing_summary,
)

logger = configure_module_logger(__name__, level=logging.INFO)


class NarwhalsEngine:
    """
    Narwhals-based engine for efficient semantic data cleaning.
    """

    @staticmethod
    def _process_chunks_parallel(
        chunks: List[List[str]],
        inference_engine: "InferenceEngine",
        instruction: str,
        max_workers: int,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process chunks in parallel using ThreadPoolExecutor.

        Args:
            chunks: List of chunks (each chunk is a list of strings).
            inference_engine: Inference engine instance.
            instruction: Instruction to guide the LLM extraction.
            max_workers: Maximum number of worker threads.

        Returns:
            Dictionary mapping original_string -> result_dict or None.
        """
        mapping_results: Dict[str, Optional[Dict[str, Any]]] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(inference_engine.clean_batch, chunk, instruction): chunk
                for chunk in chunks
            }

            # Collect results as they complete - use Rich Progress if available
            from loclean.utils.rich_output import create_progress

            progress = create_progress(
                total=len(chunks), description="Processing batches in parallel"
            )
            if progress:
                with progress:
                    task_id = progress.add_task(
                        "[cyan]Processing batches[/cyan]", total=len(chunks)
                    )
                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        try:
                            batch_result: Dict[str, Optional[Dict[str, Any]]] = (
                                future.result()
                            )
                            mapping_results.update(batch_result)
                            completed += 1
                            progress.update(task_id, advance=1)
                        except Exception as e:
                            logger.error(
                                f"Error processing chunk {chunk[:3]}...: {e}",
                                exc_info=True,
                            )
                            # Mark failed items as None
                            for item in chunk:
                                mapping_results[item] = None
                            completed += 1
                            progress.update(task_id, advance=1)
            else:
                # Fallback: simple logging without progress bar
                logger.info(f"Processing {len(chunks)} batches in parallel...")
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        batch_result = future.result()
                        mapping_results.update(batch_result)
                        completed += 1
                        if completed % max(1, len(chunks) // 10) == 0:
                            logger.info(
                                f"Progress: {completed}/{len(chunks)} batches completed"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing chunk {chunk[:3]}...: {e}",
                            exc_info=True,
                        )
                        # Mark failed items as None
                        for item in chunk:
                            mapping_results[item] = None
                        completed += 1
                        if completed % max(1, len(chunks) // 10) == 0:
                            logger.info(
                                f"Progress: {completed}/{len(chunks)} batches completed"
                            )

        return mapping_results

    @staticmethod
    def process_column(
        df_native: IntoFrameT,
        col_name: str,
        inference_engine: "InferenceEngine",
        instruction: str,
        batch_size: int = 50,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> IntoFrameT:
        """
        Clean a specific column using the inference engine with batching
        and progress tracking.

        Args:
            df_native: Input DataFrame (pandas, Polars, PyArrow, etc.).
            col_name: Name of the column to clean.
            inference_engine: Inference engine instance for semantic extraction.
            instruction: Instruction to guide the LLM extraction.
            batch_size: Number of unique values to process per batch. Defaults to 50.
            parallel: Enable parallel processing using ThreadPoolExecutor.
                     Defaults to False for backward compatibility.
            max_workers: Maximum number of worker threads. If None, auto-detected
                        as min(cpu_count, len(chunks)). If 1, falls back to sequential.
                        Defaults to None.

        Returns:
            DataFrame with added 'clean_value', 'clean_unit', and 'clean_reasoning'
            columns. Return type matches input type
            (pandas -> pandas, Polars -> Polars, etc.)

        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        df = nw.from_native(df_native)  # type: ignore[type-var]

        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")

        # Cast to String before unique() to ensure consistent types for join later.
        unique_df_native = (
            df.select(nw.col(col_name).cast(nw.String)).unique().to_native()
        )

        if hasattr(unique_df_native, "to_pandas"):
            unique_df_native = unique_df_native.to_pandas()
            col_values = unique_df_native[col_name].tolist()
        elif hasattr(unique_df_native, "to_series"):
            col_values = unique_df_native[col_name].to_series().to_list()  # type: ignore[index]
        else:
            col_values = unique_df_native[col_name].to_list()  # type: ignore[index]

        uniques: List[str] = [
            str(x) for x in col_values if x is not None and str(x).strip() != ""
        ]

        if hasattr(inference_engine, "verbose") and inference_engine.verbose:
            logger.debug(
                f"[bold magenta]DEBUG:[/bold magenta] Processing column '{col_name}' "
                f"with {len(uniques)} unique values."
            )

        if not uniques:
            logger.warning(
                "No valid unique values found. Returning original DataFrame."
            )
            return df_native

        mapping_results: Dict[str, Optional[Dict[str, Any]]] = {}

        chunks: List[List[str]] = [
            uniques[i : i + batch_size] for i in range(0, len(uniques), batch_size)
        ]

        # Log batch statistics with Rich Panel
        log_batch_stats(
            total_patterns=len(uniques),
            num_batches=len(chunks),
            batch_size=batch_size,
            parallel=parallel,
            max_workers=max_workers if parallel else None,
            col_name=col_name,
        )

        if parallel and len(chunks) > 1:
            # Determine number of workers
            if max_workers is None:
                cpu_count = os.cpu_count() or 1
                max_workers = min(cpu_count, len(chunks))
            elif max_workers <= 0:
                logger.warning(
                    f"Invalid max_workers={max_workers}. Falling back to sequential."
                )
                max_workers = 1

            if max_workers == 1:
                # Fallback to sequential if only 1 worker
                parallel = False
                logger.info("Using sequential processing (max_workers=1)")

        start_time = time.time()

        if parallel and len(chunks) > 1:
            # Parallel processing
            # max_workers is guaranteed to be int here (checked above)
            assert isinstance(max_workers, int)
            mapping_results = NarwhalsEngine._process_chunks_parallel(
                chunks, inference_engine, instruction, max_workers
            )
        else:
            # Sequential processing (default) - use Rich Progress if available
            progress = create_progress(
                total=len(chunks), description="Processing batches"
            )
            if progress:
                with progress:
                    task_id = progress.add_task(
                        "[cyan]Processing batches[/cyan]", total=len(chunks)
                    )
                    for chunk in chunks:
                        batch_result: Dict[str, Optional[Dict[str, Any]]] = (
                            inference_engine.clean_batch(chunk, instruction=instruction)
                        )
                        mapping_results.update(batch_result)
                        progress.update(task_id, advance=1)
            else:
                # Fallback: simple logging without progress bar
                logger.info(f"Processing {len(chunks)} batches sequentially...")
                for idx, chunk in enumerate(chunks, 1):
                    batch_result = inference_engine.clean_batch(
                        chunk, instruction=instruction
                    )
                    mapping_results.update(batch_result)
                    if idx % max(1, len(chunks) // 10) == 0:
                        logger.info(f"Progress: {idx}/{len(chunks)} batches completed")

        elapsed_time = time.time() - start_time

        keys: List[str] = []
        clean_values: List[Optional[float]] = []
        clean_units: List[Optional[str]] = []
        clean_reasonings: List[Optional[str]] = []

        successful = 0
        failed = 0

        for original_val, clean_data in mapping_results.items():
            keys.append(original_val)
            if clean_data:
                clean_values.append(clean_data.get("value"))
                clean_units.append(clean_data.get("unit"))
                clean_reasonings.append(clean_data.get("reasoning"))
                successful += 1
            else:
                clean_values.append(None)
                clean_units.append(None)
                clean_reasonings.append(None)
                failed += 1

        if not keys:
            logger.warning(
                "No concepts were successfully extracted. Returning original DataFrame."
            )
            return df_native

        # Log processing summary with Rich Table
        log_processing_summary(
            total_processed=len(keys),
            successful=successful,
            failed=failed,
            time_taken=elapsed_time,
            context="Semantic Cleaning",
        )

        # 4. Create Mapping DataFrame using the same native backend as the input
        # Detect backend and create DataFrame with correct type
        native_df_cls = type(df_native)

        # Try to detect backend by module name
        module_name = native_df_cls.__module__

        if "polars" in module_name:
            import polars as pl

            map_df_native = pl.DataFrame(
                {
                    col_name: keys,
                    "clean_value": clean_values,
                    "clean_unit": clean_units,
                    "clean_reasoning": clean_reasonings,
                },
                schema={
                    col_name: pl.String,
                    "clean_value": pl.Float64,
                    "clean_unit": pl.String,
                    "clean_reasoning": pl.String,
                },
            )
        elif "pandas" in module_name:
            import pandas as pd

            map_df_native = pd.DataFrame(  # type: ignore[assignment]
                {
                    col_name: keys,
                    "clean_value": clean_values,
                    "clean_unit": clean_units,
                    "clean_reasoning": clean_reasonings,
                }
            )
        else:
            raise ValueError(
                f"Unsupported dataframe backend: {native_df_cls.__name__}. "
                "Loclean expects a Polars or Pandas DataFrame."
            )

        map_df = nw.from_native(map_df_native)

        try:
            # Create temporary join key by casting to String to handle type mismatches.
            # Original column might be Int/Float while mapping keys are String.
            result_df = (
                df.with_columns(
                    nw.col(col_name).cast(nw.String).alias(f"{col_name}_join_key")
                )
                .join(
                    map_df,  # type: ignore[arg-type]
                    left_on=f"{col_name}_join_key",
                    right_on=col_name,
                    how="left",
                )
                .drop(f"{col_name}_join_key")
                .to_native()
            )
            return result_df
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise
