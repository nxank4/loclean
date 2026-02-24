"""Configuration system for inference engines.

This module provides hierarchical configuration management with the following
priority order (highest to lowest):
1. Runtime Parameters (passed directly to functions)
2. Environment Variables (prefixed with LOCLEAN_)
3. Project Config ([tool.loclean] in pyproject.toml)
4. Defaults (hardcoded fallbacks)
"""

import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration model for inference engines.

    Validates API keys and model parameters with hierarchical
    configuration support.
    """

    engine: Literal["ollama", "openai", "anthropic", "gemini"] = Field(
        default="ollama",
        description="Inference engine backend to use",
    )

    model: str = Field(
        default="phi3",
        description="Model identifier (Ollama tag or cloud model ID)",
    )

    host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud inference providers",
    )

    verbose: bool = Field(
        default=False,
        description=(
            "Enable detailed logging of prompts, outputs, and processing steps"
        ),
    )

    model_config = {
        "extra": "forbid",
    }


def _load_from_pyproject_toml() -> dict[str, Any]:
    """Load configuration from [tool.loclean] section in pyproject.toml.

    Returns:
        Dictionary with config values, or empty dict if not found.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # noqa: F401
        except ImportError:
            return {}

    from pathlib import Path

    current_dir = Path.cwd()
    for path in [current_dir] + list(current_dir.parents):
        pyproject_path = path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    if "tool" in data and "loclean" in data["tool"]:
                        result: dict[str, Any] = dict(data["tool"]["loclean"])
                        return result
            except Exception:
                continue

    return {}


def _load_from_env() -> dict[str, Any]:
    """Load configuration from environment variables (prefixed with LOCLEAN_).

    Returns:
        Dictionary with config values from environment.
    """
    config: dict[str, Any] = {}

    env_mapping = {
        "LOCLEAN_ENGINE": "engine",
        "LOCLEAN_MODEL": "model",
        "LOCLEAN_HOST": "host",
        "LOCLEAN_API_KEY": "api_key",
        "LOCLEAN_VERBOSE": "verbose",
    }

    for env_var, config_key in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            if config_key == "verbose":
                config[config_key] = value.lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
            else:
                config[config_key] = value

    return config


def load_config(
    engine: Optional[str] = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> EngineConfig:
    """Load configuration with hierarchical priority.

    Priority order (highest to lowest):
    1. Runtime Parameters (passed to this function)
    2. Environment Variables (LOCLEAN_*)
    3. Project Config ([tool.loclean] in pyproject.toml)
    4. Defaults (hardcoded in EngineConfig)

    Args:
        engine: Engine backend name.
        model: Model identifier.
        host: Ollama server URL.
        api_key: API key for cloud providers.
        verbose: Enable detailed logging.
        **kwargs: Additional configuration parameters.

    Returns:
        EngineConfig instance with merged configuration.
    """
    default_config = EngineConfig()
    file_config = _load_from_pyproject_toml()
    env_config = _load_from_env()

    runtime_config: dict[str, Any] = {}
    if engine is not None:
        runtime_config["engine"] = engine
    if model is not None:
        runtime_config["model"] = model
    if host is not None:
        runtime_config["host"] = host
    if api_key is not None:
        runtime_config["api_key"] = api_key
    if verbose is not None:
        runtime_config["verbose"] = verbose
    runtime_config.update(kwargs)

    merged_config = default_config.model_dump()
    merged_config.update(file_config)
    merged_config.update(env_config)
    merged_config.update(runtime_config)

    return EngineConfig(**merged_config)
