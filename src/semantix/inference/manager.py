import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class LocalInferenceEngine:
    """
    A local inference engine using Llama.cpp with GBNF grammar constraints.
    """

    MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
    MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the inference engine. Auto-downloads the model if missing.

        Args:
            cache_dir: Optional custom directory for caching models.
                       Defaults to ~/.cache/semantix.
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "semantix"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self._get_model_path()

        logger.info(f"Loading model from {self.model_path}...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=4096,
            n_gpu_layers=0,  # CPU mode
            verbose=False,
        )
        self.grammar = self._get_json_grammar()
        
        from semantix.cache import SemantixCache
        self.cache = SemantixCache(cache_dir=self.cache_dir)
        
        logger.info("LocalInferenceEngine initialized successfully.")

    def _get_model_path(self) -> Path:
        """
        Ensure the model exists locally. Download if necessary.
        """
        local_path = self.cache_dir / self.MODEL_FILENAME
        if local_path.exists():
            logger.info(f"Model found at {local_path}")
            return local_path

        logger.info(f"Model not found. Downloading {self.MODEL_FILENAME}...")
        # We explicitly download to our cache dir effectively by just returning the path
        # customized via hf_hub_download if we wanted strict control,
        # but hf_hub_download manages its own cache typically.
        # User requested: "Check ~/.cache/semantix... If not, use hf_hub_download to fetch it."
        # hf_hub_download by default downloads to ~/.cache/huggingface.
        # To strictly follow "fetch TO ~/.cache/semantix", we use local_dir.
        
        path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.MODEL_FILENAME,
            local_dir=self.cache_dir,
        )
        return Path(path)

    def _get_json_grammar(self) -> LlamaGrammar:
        """
        Returns a GBNF grammar enforcing {"value": <number>, "unit": <string>}.
        """
        # GBNF string
        grammar_str = r"""
            root   ::= object
            object ::= "{" ws "\"value\"" ws ":" ws number "," ws "\"unit\"" ws ":" ws string ws "}"
            number ::= ("-"? ([0-9]+ ("." [0-9]+)?))
            string ::= "\"" ([^"]*) "\""
            ws     ::= [ \t\n]*
        """
        return LlamaGrammar.from_string(grammar_str)

    def clean_batch(
        self, 
        items: List[str], 
        instruction: str
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process a batch of strings and extract structured data using GBNF.
        Result is cached to avoid re-computation.

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the task.

        Returns:
            Dictionary mapping original_string -> {"value": <number>, "unit": <string>} or None.
        """
        # 1. Check Cache
        cached_results = self.cache.get_batch(items, instruction)
        
        # 2. Identify Misses
        misses = [item for item in items if item not in cached_results]
        
        if not misses:
            return cached_results

        logger.info(f"Cache miss for {len(misses)} items. Running inference...")
        new_results: Dict[str, Optional[Dict[str, Any]]] = {}

        for item in misses:
            # Use Phi-3 Instruct format with dynamic instruction
            prompt = f"""<|user|>
Task: {instruction}
Input Data: "{item}"

Return JSON with keys "value" (number) and "unit" (string).
<|end|>
<|assistant|>"""
            
            output = None
            text = None
            try:
                output = self.llm.create_completion(
                    prompt=prompt,
                    grammar=self.grammar,
                    max_tokens=128,
                    # Do NOT use '}' as stop, as it strips the closing brace needed for valid JSON.
                    # Grammar restricts output efficiently anyway.
                    stop=["<|end|>", "<|user|>"], 
                    echo=False
                )
                
                text = output['choices'][0]['text'].strip()
                # logger.debug(f"Raw LLM output for '{item}': {text}")

                data = json.loads(text)
                
                if "value" in data and "unit" in data:
<<<<<<< HEAD:src/semantix/inference/manager.py
                     new_results[item] = data
=======
                    results[item] = data
>>>>>>> origin/main:semantix/inference/manager.py
                else:
                    logger.warning(f"Result for '{item}' missing keys. Raw: {text}")
                    new_results[item] = None

            except json.JSONDecodeError as e:
<<<<<<< HEAD:src/semantix/inference/manager.py
                logger.warning(f"Failed to decode JSON for item '{item}': {e}. Raw text: '{text if 'text' in locals() else 'N/A'}'")
                new_results[item] = None
=======
                if output and text:
                    logger.warning(f"Failed to decode JSON for item '{item}': {e}. Raw text: '{text}'")
                else:
                    logger.warning(f"Failed to decode JSON for item '{item}': {e}")
                results[item] = None
>>>>>>> origin/main:semantix/inference/manager.py
            except Exception as e:
                logger.error(f"Inference error for item '{item}': {e}")
                new_results[item] = None
        
        # 4. Update Cache (only with valid results)
        valid_new_results = {k: v for k, v in new_results.items() if v is not None}
        if valid_new_results:
            self.cache.set_batch(list(valid_new_results.keys()), instruction, valid_new_results)

        # 5. Merge
        return {**cached_results, **new_results}
