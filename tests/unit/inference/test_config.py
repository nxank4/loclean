"""Test cases for configuration system."""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from loclean.inference.config import (
    EngineConfig,
    _load_from_env,
    _load_from_pyproject_toml,
    load_config,
)


class TestEngineConfig:
    """Test cases for EngineConfig Pydantic model."""

    def test_default_values(self) -> None:
        """Test that EngineConfig has correct default values."""
        config = EngineConfig()
        assert config.engine == "llama-cpp"
        assert config.model == "phi-3-mini-4k-instruct"
        assert config.api_key is None
        assert config.cache_dir == Path.home() / ".cache" / "loclean"
        assert config.n_ctx == 4096
        assert config.n_gpu_layers == 0

    def test_valid_creation_with_all_fields(self) -> None:
        """Test creating EngineConfig with all fields."""
        cache_dir = Path("/tmp/test_cache")
        config = EngineConfig(
            engine="openai",
            model="gpt-4o",
            api_key="sk-test123",
            cache_dir=cache_dir,
            n_ctx=8192,
            n_gpu_layers=10,
        )
        assert config.engine == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test123"
        assert config.cache_dir == cache_dir
        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 10

    def test_valid_engine_types(self) -> None:
        """Test that all valid engine types are accepted."""
        for engine in ["llama-cpp", "openai", "anthropic", "gemini"]:
            config = EngineConfig(engine=engine)  # type: ignore[arg-type]
            assert config.engine == engine

    def test_invalid_engine_type(self) -> None:
        """Test that invalid engine type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(engine="invalid-engine")  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("engine",) for error in errors)

    def test_cache_dir_from_string(self) -> None:
        """Test that cache_dir can be created from string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(cache_dir=tmpdir)  # type: ignore[arg-type]
            assert config.cache_dir == Path(tmpdir)
            assert config.cache_dir.exists()

    def test_cache_dir_auto_creation(self) -> None:
        """Test that cache_dir is automatically created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "new_cache"
            config = EngineConfig(cache_dir=cache_path)
            assert config.cache_dir.exists()
            assert config.cache_dir == cache_path

    def test_n_ctx_validation_min(self) -> None:
        """Test that n_ctx below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(n_ctx=100)

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("n_ctx",) for error in errors)

    def test_n_ctx_validation_max(self) -> None:
        """Test that n_ctx above maximum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(n_ctx=50000)

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("n_ctx",) for error in errors)

    def test_n_gpu_layers_validation_min(self) -> None:
        """Test that n_gpu_layers below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(n_gpu_layers=-1)

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("n_gpu_layers",) for error in errors)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(extra_field="should_fail")  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("extra" in str(error).lower() for error in errors)


class TestLoadFromEnv:
    """Test cases for _load_from_env function."""

    def test_load_all_env_variables(self) -> None:
        """Test loading all environment variables."""
        env_vars = {
            "LOCLEAN_ENGINE": "openai",
            "LOCLEAN_MODEL": "gpt-4o",
            "LOCLEAN_API_KEY": "sk-test123",
            "LOCLEAN_CACHE_DIR": "/tmp/test",
            "LOCLEAN_N_CTX": "8192",
            "LOCLEAN_N_GPU_LAYERS": "10",
        }

        with patch.dict(os.environ, env_vars):
            config = _load_from_env()
            assert config["engine"] == "openai"
            assert config["model"] == "gpt-4o"
            assert config["api_key"] == "sk-test123"
            assert config["cache_dir"] == "/tmp/test"
            assert config["n_ctx"] == 8192
            assert config["n_gpu_layers"] == 10

    def test_load_partial_env_variables(self) -> None:
        """Test loading only some environment variables."""
        env_vars = {
            "LOCLEAN_ENGINE": "anthropic",
            "LOCLEAN_MODEL": "claude-3",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = _load_from_env()
            assert config["engine"] == "anthropic"
            assert config["model"] == "claude-3"
            assert "api_key" not in config

    def test_load_no_env_variables(self) -> None:
        """Test loading when no environment variables are set."""
        # Remove all LOCLEAN_* env vars
        env_to_remove = [key for key in os.environ.keys() if key.startswith("LOCLEAN_")]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_to_remove:
                os.environ.pop(key, None)
            config = _load_from_env()
            assert config == {}

    def test_n_ctx_invalid_int(self) -> None:
        """Test that invalid integer for n_ctx is skipped."""
        env_vars = {
            "LOCLEAN_N_CTX": "not_an_int",
        }

        with patch.dict(os.environ, env_vars):
            config = _load_from_env()
            assert "n_ctx" not in config


class TestLoadFromPyprojectToml:
    """Test cases for _load_from_pyproject_toml function."""

    def test_load_from_existing_pyproject_toml(self, tmp_path: Any) -> None:
        """Test loading config from pyproject.toml."""
        pyproject_content = """
[tool.loclean]
engine = "gemini"
model = "gemini-pro"
api_key = "test-api-key"
n_ctx = 2048
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        with patch("loclean.inference.config.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config["engine"] == "gemini"
            assert config["model"] == "gemini-pro"
            assert config["api_key"] == "test-api-key"
            assert config["n_ctx"] == 2048

    def test_load_from_nonexistent_pyproject_toml(self, tmp_path: Any) -> None:
        """Test loading when pyproject.toml doesn't exist."""
        with patch("loclean.inference.config.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config == {}

    def test_load_from_pyproject_toml_without_loclean_section(
        self, tmp_path: Any
    ) -> None:
        """Test loading when pyproject.toml exists but has no
        [tool.loclean] section."""
        pyproject_content = """
[project]
name = "test"
version = "0.1.1"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        with patch("loclean.inference.config.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config == {}


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_with_defaults(self) -> None:
        """Test loading config with only defaults."""
        # Clear environment variables
        env_to_remove = [key for key in os.environ.keys() if key.startswith("LOCLEAN_")]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_to_remove:
                os.environ.pop(key, None)

            with patch(
                "loclean.inference.config._load_from_pyproject_toml", return_value={}
            ):
                config = load_config()
                assert config.engine == "llama-cpp"
                assert config.model == "phi-3-mini-4k-instruct"
                assert config.api_key is None
                assert config.n_ctx == 4096
                assert config.n_gpu_layers == 0

    def test_load_with_runtime_params(self) -> None:
        """Test that runtime parameters override everything."""
        env_vars = {
            "LOCLEAN_ENGINE": "openai",
            "LOCLEAN_MODEL": "gpt-3.5-turbo",
        }

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value={"engine": "anthropic", "model": "claude-3"},
            ):
                config = load_config(engine="gemini", model="gemini-pro")
                # Runtime params should override env and file
                assert config.engine == "gemini"
                assert config.model == "gemini-pro"

    def test_load_with_env_variables(self) -> None:
        """Test that environment variables override file config."""
        env_vars = {
            "LOCLEAN_ENGINE": "openai",
            "LOCLEAN_MODEL": "gpt-4o",
            "LOCLEAN_API_KEY": "sk-env-key",
        }

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value={"engine": "llama-cpp", "model": "phi-3"},
            ):
                config = load_config()
                # Env should override file
                assert config.engine == "openai"
                assert config.model == "gpt-4o"
                assert config.api_key == "sk-env-key"

    def test_load_with_file_config(self) -> None:
        """Test that file config is used when env vars are not set."""
        env_to_remove = [key for key in os.environ.keys() if key.startswith("LOCLEAN_")]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_to_remove:
                os.environ.pop(key, None)

            file_config = {
                "engine": "anthropic",
                "model": "claude-3-opus",
                "n_ctx": 2048,
            }

            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value=file_config,
            ):
                config = load_config()
                assert config.engine == "anthropic"
                assert config.model == "claude-3-opus"
                assert config.n_ctx == 2048

    def test_priority_order(self) -> None:
        """Test that priority order is correct: Param > Env > File > Default."""
        # File config
        file_config = {"engine": "llama-cpp", "model": "phi-3"}

        # Env config (should override file)
        env_vars = {"LOCLEAN_ENGINE": "openai", "LOCLEAN_MODEL": "gpt-3.5"}

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value=file_config,
            ):
                # Runtime param (should override env and file)
                config = load_config(engine="gemini", model="gemini-pro")
                assert config.engine == "gemini"  # Runtime param wins
                assert config.model == "gemini-pro"  # Runtime param wins

                # Without runtime params, env should win
                config2 = load_config()
                assert config2.engine == "openai"  # Env wins over file
                assert config2.model == "gpt-3.5"  # Env wins over file

    def test_load_with_cache_dir(self) -> None:
        """Test loading config with custom cache_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "custom_cache"
            config = load_config(cache_dir=cache_path)
            assert config.cache_dir == cache_path
            assert config.cache_dir.exists()

    def test_load_with_kwargs(self) -> None:
        """Test loading config with additional kwargs raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            load_config(engine="openai", custom_param="should_fail")

        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Should fail because extra fields are forbidden

    def test_load_merges_partial_configs(self) -> None:
        """Test that partial configs from different sources are merged correctly."""
        env_vars = {"LOCLEAN_ENGINE": "openai"}

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value={"model": "gpt-4o", "n_ctx": 8192},
            ):
                config = load_config(api_key="sk-runtime-key")
                # Should merge: engine from env, model from file, api_key from runtime
                assert config.engine == "openai"
                assert config.model == "gpt-4o"
                assert config.api_key == "sk-runtime-key"
                assert config.n_ctx == 8192
                # Defaults for fields not specified
                assert config.n_gpu_layers == 0
