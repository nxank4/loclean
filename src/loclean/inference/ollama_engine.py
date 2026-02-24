"""Ollama inference engine implementation.

This module provides the OllamaEngine class for local inference using
a running Ollama instance with structured JSON output via Pydantic schemas.

The engine is fully self-bootstrapping: it will automatically start the
Ollama daemon (if local) and pull the requested model when necessary.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from loclean.inference.base import InferenceEngine
from loclean.inference.daemon import ensure_daemon
from loclean.inference.model_manager import ensure_model
from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)

_DEFAULT_HOST = "http://localhost:11434"


class OllamaEngine(InferenceEngine):
    """Inference engine using a running Ollama instance.

    Connects to the Ollama HTTP API and leverages its native ``format``
    parameter to enforce structured JSON output matching Pydantic schemas.

    On construction the engine will:

    1. **Ensure the daemon is running** — if the host is local and
       unreachable, the ``ollama serve`` binary is launched automatically.
    2. **Ensure the model is pulled** — if the requested model is not in
       the local registry it is downloaded with a progress bar.
    """

    def __init__(
        self,
        model: str = "phi3",
        host: str = _DEFAULT_HOST,
        verbose: bool = False,
    ) -> None:
        """Initialize the OllamaEngine.

        Args:
            model: Ollama model tag (e.g. "phi3", "llama3", "gemma2").
            host: Ollama server URL. Defaults to http://localhost:11434.
            verbose: Enable detailed logging of prompts and outputs.

        Raises:
            FileNotFoundError: If the ``ollama`` binary is missing.
            ConnectionError: If the daemon cannot be reached after startup.
            RuntimeError: If the model pull fails.
        """
        self.model = model
        self.host = host
        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        ensure_daemon(self.host)

        import ollama as _ollama  # type: ignore[import-untyped]

        self._client = _ollama.Client(host=self.host)

        ensure_model(self._client, self.model)

        logger.info(
            f"[green]✓[/green] OllamaEngine ready — "
            f"[bold]{self.host}[/bold] · model: "
            f"[bold cyan]{self.model}[/bold cyan]"
        )

    # ------------------------------------------------------------------
    # Structured generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str:
        """Send a prompt to Ollama and return the raw text response.

        When *schema* is provided the Pydantic JSON Schema is passed via
        Ollama's ``format`` parameter so the model is constrained to
        produce valid JSON matching the schema.

        Args:
            prompt: The user prompt / instruction.
            schema: Optional Pydantic model for structured output.

        Returns:
            Raw text response from Ollama.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if schema is not None:
            kwargs["format"] = schema.model_json_schema()

        if self.verbose:
            logger.debug(f"[bold blue]PROMPT:[/bold blue]\n{prompt}")

        response = self._client.generate(**kwargs)
        text: str = response.get("response", "").strip()  # type: ignore[union-attr]

        if self.verbose:
            logger.debug(f"[bold green]RAW OUTPUT:[/bold green]\n{text}")

        return text

    # ------------------------------------------------------------------
    # InferenceEngine ABC implementation
    # ------------------------------------------------------------------

    def clean_batch(
        self,
        items: List[str],
        instruction: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Process a batch of strings and extract structured data.

        Uses Ollama with a JSON format constraint matching the default
        extraction schema (reasoning / value / unit).

        Args:
            items: List of raw strings to process.
            instruction: User-defined instruction for the task.

        Returns:
            Dictionary mapping original string → extracted dict or None.
        """
        from loclean.inference.schemas import ExtractionResult

        results: Dict[str, Optional[Dict[str, Any]]] = {}

        for item in items:
            prompt = f"{instruction}\n\nInput: {item}"
            try:
                raw = self.generate(prompt, schema=ExtractionResult)
                data = json.loads(raw)
                if "value" in data and "unit" in data and "reasoning" in data:
                    results[item] = data
                else:
                    logger.warning(
                        f"[yellow]⚠[/yellow] Missing required keys for: "
                        f"[dim]'{item[:50]}'[/dim]"
                    )
                    results[item] = None
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"[yellow]⚠[/yellow] JSON decode error for "
                    f"[dim]'{item[:50]}'[/dim]: {exc}"
                )
                results[item] = None
            except Exception as exc:
                logger.error(
                    f"[red]❌[/red] Inference error for [dim]'{item[:50]}'[/dim]: {exc}"
                )
                results[item] = None

        return results
