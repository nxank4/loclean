"""Llama.cpp inference engine implementation.

This module provides the LlamaCppEngine class for local inference using
Llama.cpp with GBNF grammar constraints and prompt adapters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from semantix.inference.adapters import PromptAdapter, get_adapter
from semantix.inference.base import InferenceEngine

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Model registry mapping model names to their HuggingFace repo and filename
_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "phi-3-mini": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "qwen3-4b": {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "filename": "Qwen3-4B-Instruct-2507-GGUF.q4_k_m.gguf",
    },
    "gemma-3-4b": {
        "repo": "unsloth/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-GGUF.q4_k_m.gguf",
    },
    "deepseek-r1": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-GGUF.q4_k_m.gguf",
    },
}


class LlamaCppEngine(InferenceEngine):
    """
    Local inference engine using Llama.cpp with GBNF grammar constraints.

    Supports multiple GGUF models (Phi-3, Qwen, Gemma, DeepSeek) with automatic
    prompt adapter selection based on model name.
    """

    def __init__(
        self,
        model_name: str = "phi-3-mini",
        cache_dir: Optional[Path] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ):
        """
        Initialize the LlamaCppEngine.

        Args:
            model_name: Name of the model to use (e.g., "phi-3-mini", "qwen3-4b").
                       Must be in MODEL_REGISTRY. Defaults to "phi-3-mini".
            cache_dir: Optional custom directory for caching models.
                       Defaults to ~/.cache/semantix.
            n_ctx: Context window size. Defaults to 4096.
            n_gpu_layers: Number of GPU layers to use (0 = CPU only). Defaults to 0.
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "semantix"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Get model info from registry
        if model_name not in _MODEL_REGISTRY:
            logger.warning(f"Model '{model_name}' not in registry. Falling back to 'phi-3-mini'.")
            model_name = "phi-3-mini"
            self.model_name = model_name

        model_info = _MODEL_REGISTRY[model_name]
        self.model_repo = model_info["repo"]
        self.model_filename = model_info["filename"]

        # Get prompt adapter based on model name
        self.adapter: PromptAdapter = get_adapter(model_name)
        logger.info(f"Using adapter: {type(self.adapter).__name__} for model: {model_name}")

        # Get model path (download if needed)
        self.model_path = self._get_model_path()

        logger.info(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.grammar = self._get_json_grammar()

        from semantix.cache import SemantixCache

        self.cache = SemantixCache(cache_dir=self.cache_dir)

        logger.info(f"LlamaCppEngine initialized successfully with model: {model_name}")

    def _get_model_path(self) -> Path:
        """
        Ensure the model exists locally. Download if necessary.

        Returns:
            Path to the model file.
        """
        local_path = self.cache_dir / self.model_filename
        if local_path.exists():
            logger.info(f"Model found at {local_path}")
            return local_path

        logger.info(f"Model not found. Downloading {self.model_filename}...")
        path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_filename,
            local_dir=self.cache_dir,
        )
        return Path(path)

    def _get_json_grammar(self) -> LlamaGrammar:
        """
        Returns a GBNF grammar enforcing {"reasoning": <string>, "value": <number>, "unit": <string>}.

        Loads grammar from resources/grammars/json.gbnf using importlib.resources
        for zip-safe compatibility.

        Returns:
            LlamaGrammar instance for JSON extraction.
        """
        from semantix.utils.resources import load_grammar

        grammar_str = load_grammar("json.gbnf")
        return LlamaGrammar.from_string(grammar_str)

    def clean_batch(
        self,
        items: List[str],
        instruction: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process a batch of strings and extract structured data using GBNF.

        Result is cached to avoid re-computation.

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the task.

        Returns:
            Dictionary mapping original_string -> {"reasoning": str, "value": float, "unit": str} or None.
        """
        # 1. Check Cache
        cached_results = self.cache.get_batch(items, instruction)

        # 2. Identify Misses
        misses = [item for item in items if item not in cached_results]

        if not misses:
            return cached_results

        logger.info(f"Cache miss for {len(misses)} items. Running inference...")
        new_results: Dict[str, Optional[Dict[str, Any]]] = {}

        # Get stop tokens from adapter
        stop_tokens = self.adapter.get_stop_tokens()

        for item in misses:
            # Use adapter to format prompt
            prompt = self.adapter.format(instruction, item)

            output = None
            text = None
            try:
                output = self.llm.create_completion(
                    prompt=prompt,
                    grammar=self.grammar,
                    max_tokens=256,
                    stop=stop_tokens,
                    echo=False,
                )

                text = output["choices"][0]["text"].strip()

                data = json.loads(text)

                if "value" in data and "unit" in data and "reasoning" in data:
                    new_results[item] = data
                else:
                    logger.warning(f"Result for '{item}' missing keys. Raw: {text}")
                    new_results[item] = None

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to decode JSON for item '{item}': {e}. Raw text: '{text if 'text' in locals() else 'N/A'}'"
                )
                new_results[item] = None
            except Exception as e:
                logger.error(f"Inference error for item '{item}': {e}")
                new_results[item] = None

        # 4. Update Cache (only with valid results)
        valid_new_results = {k: v for k, v in new_results.items() if v is not None}
        if valid_new_results:
            self.cache.set_batch(list(valid_new_results.keys()), instruction, valid_new_results)

        # 5. Merge
        return {**cached_results, **new_results}

