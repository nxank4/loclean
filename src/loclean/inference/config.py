"""Configuration system for inference engines.

This module provides hierarchical configuration management with the following
priority order (highest to lowest):
1. Runtime Parameters (passed directly to functions)
2. Environment Variables (prefixed with LOCLEAN_)
3. Project Config ([tool.loclean] in pyproject.toml)
4. Defaults (hardcoded fallbacks)
"""

import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EngineConfig(BaseModel):
    """
    Configuration model for inference engines.

    Validates paths, API keys, and model parameters with hierarchical
    configuration support.
    """

    engine: Literal["llama-cpp", "openai", "anthropic", "gemini"] = Field(
        default="llama-cpp",
        description="Inference engine backend to use",
    )

    model: str = Field(
        default="phi-3-mini-4k-instruct",
        description="Model identifier (GGUF path or cloud model ID)",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud inference providers",
    )

    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "loclean",
        description="Directory for caching models and inference results",
    )

    n_ctx: int = Field(
        default=4096,
        description="Context window size for Llama.cpp models",
        ge=512,
        le=32768,
    )

    n_gpu_layers: int = Field(
        default=0,
        description="Number of GPU layers to use (0 = CPU only)",
        ge=0,
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v: str | Path) -> Path:
        """Convert cache_dir to Path and ensure it exists."""
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "extra": "forbid",
    }


def _load_from_pyproject_toml() -> dict[str, Any]:
    """
    Load configuration from [tool.loclean] section in pyproject.toml.

    Returns:
        Dictionary with config values, or empty dict if not found.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # noqa: F401
        except ImportError:
            # tomli not available, return empty dict
            return {}

    # Try to find pyproject.toml in current directory or parent directories
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
                # If reading fails, continue searching
                continue

    return {}


def _load_from_env() -> dict[str, Any]:
    """
    Load configuration from environment variables (prefixed with LOCLEAN_).

    Returns:
        Dictionary with config values from environment.
    """
    config: dict[str, Any] = {}

    # Map environment variables to config keys
    env_mapping = {
        "LOCLEAN_ENGINE": "engine",
        "LOCLEAN_MODEL": "model",
        "LOCLEAN_API_KEY": "api_key",
        "LOCLEAN_CACHE_DIR": "cache_dir",
        "LOCLEAN_N_CTX": "n_ctx",
        "LOCLEAN_N_GPU_LAYERS": "n_gpu_layers",
    }

    for env_var, config_key in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            # Convert string values to appropriate types
            if config_key == "n_ctx" or config_key == "n_gpu_layers":
                try:
                    config[config_key] = int(value)
                except ValueError:
                    continue
            elif config_key == "cache_dir":
                config[config_key] = value
            else:
                config[config_key] = value

    return config


def load_config(
    engine: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    **kwargs: Any,
) -> EngineConfig:
    """
    Load configuration with hierarchical priority: Param > Env > Config File > Default.

    Priority order (highest to lowest):
    1. Runtime Parameters (passed to this function)
    2. Environment Variables (LOCLEAN_*)
    3. Project Config ([tool.loclean] in pyproject.toml)
    4. Defaults (hardcoded in EngineConfig)

    Args:
        engine: Engine backend name (overrides all other sources)
        model: Model identifier (overrides all other sources)
        api_key: API key (overrides all other sources)
        cache_dir: Cache directory (overrides all other sources)
        n_ctx: Context window size (overrides all other sources)
        n_gpu_layers: Number of GPU layers (overrides all other sources)
        **kwargs: Additional configuration parameters

    Returns:
        EngineConfig instance with merged configuration

    Example:
        >>> config = load_config(model="gpt-4o", api_key="sk-...")
        >>> config.model
        'gpt-4o'
    """
    default_config = EngineConfig()
    file_config = _load_from_pyproject_toml()
    env_config = _load_from_env()

    runtime_config: dict[str, Any] = {}
    if engine is not None:
        runtime_config["engine"] = engine
    if model is not None:
        runtime_config["model"] = model
    if api_key is not None:
        runtime_config["api_key"] = api_key
    if cache_dir is not None:
        runtime_config["cache_dir"] = cache_dir
    if n_ctx is not None:
        runtime_config["n_ctx"] = n_ctx
    if n_gpu_layers is not None:
        runtime_config["n_gpu_layers"] = n_gpu_layers
    runtime_config.update(kwargs)

    merged_config = default_config.model_dump()
    merged_config.update(file_config)
    merged_config.update(env_config)
    merged_config.update(runtime_config)

    return EngineConfig(**merged_config)
