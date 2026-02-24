"""Test cases for configuration system."""

import os
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
        assert config.engine == "ollama"
        assert config.model == "phi3"
        assert config.host == "http://localhost:11434"
        assert config.api_key is None
        assert config.verbose is False

    def test_valid_creation_with_all_fields(self) -> None:
        """Test creating EngineConfig with all fields."""
        config = EngineConfig(
            engine="openai",
            model="gpt-4o",
            api_key="sk-test123",
            host="http://custom:8080",
            verbose=True,
        )
        assert config.engine == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test123"
        assert config.host == "http://custom:8080"
        assert config.verbose is True

    def test_valid_engine_types(self) -> None:
        """Test that all valid engine types are accepted."""
        for engine in ["ollama", "openai", "anthropic", "gemini"]:
            config = EngineConfig(engine=engine)  # type: ignore[arg-type]
            assert config.engine == engine

    def test_invalid_engine_type(self) -> None:
        """Test that invalid engine type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EngineConfig(engine="invalid-engine")  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("engine",) for error in errors)

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
            "LOCLEAN_HOST": "http://custom:8080",
            "LOCLEAN_VERBOSE": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = _load_from_env()
            assert config["engine"] == "openai"
            assert config["model"] == "gpt-4o"
            assert config["api_key"] == "sk-test123"
            assert config["host"] == "http://custom:8080"
            assert config["verbose"] is True

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
        env_to_remove = [key for key in os.environ.keys() if key.startswith("LOCLEAN_")]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_to_remove:
                os.environ.pop(key, None)
            config = _load_from_env()
            assert config == {}

    def test_verbose_bool_parsing(self) -> None:
        """Test that LOCLEAN_VERBOSE parses various truthy values."""
        for val in ("true", "1", "yes", "on"):
            with patch.dict(os.environ, {"LOCLEAN_VERBOSE": val}):
                config = _load_from_env()
                assert config["verbose"] is True

        for val in ("false", "0", "no", "off"):
            with patch.dict(os.environ, {"LOCLEAN_VERBOSE": val}):
                config = _load_from_env()
                assert config["verbose"] is False


class TestLoadFromPyprojectToml:
    """Test cases for _load_from_pyproject_toml function."""

    def test_load_from_existing_pyproject_toml(self, tmp_path: Any) -> None:
        """Test loading config from pyproject.toml."""
        pyproject_content = """
[tool.loclean]
engine = "gemini"
model = "gemini-pro"
api_key = "test-api-key"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config["engine"] == "gemini"
            assert config["model"] == "gemini-pro"
            assert config["api_key"] == "test-api-key"

    def test_load_from_nonexistent_pyproject_toml(self, tmp_path: Any) -> None:
        """Test loading when pyproject.toml doesn't exist."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config == {}

    def test_load_from_pyproject_toml_without_loclean_section(
        self, tmp_path: Any
    ) -> None:
        """Test loading when pyproject.toml has no [tool.loclean] section."""
        pyproject_content = """
[project]
name = "test"
version = "0.1.1"
"""
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            config = _load_from_pyproject_toml()
            assert config == {}


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_with_defaults(self) -> None:
        """Test loading config with only defaults."""
        env_to_remove = [key for key in os.environ.keys() if key.startswith("LOCLEAN_")]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_to_remove:
                os.environ.pop(key, None)

            with patch(
                "loclean.inference.config._load_from_pyproject_toml", return_value={}
            ):
                config = load_config()
                assert config.engine == "ollama"
                assert config.model == "phi3"
                assert config.host == "http://localhost:11434"
                assert config.api_key is None

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
                return_value={"engine": "ollama", "model": "phi3"},
            ):
                config = load_config()
                assert config.engine == "openai"
                assert config.model == "gpt-4o"
                assert config.api_key == "sk-env-key"

    def test_priority_order(self) -> None:
        """Param > Env > File > Default."""
        file_config = {"engine": "ollama", "model": "phi3"}
        env_vars = {"LOCLEAN_ENGINE": "openai", "LOCLEAN_MODEL": "gpt-3.5"}

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value=file_config,
            ):
                config = load_config(engine="gemini", model="gemini-pro")
                assert config.engine == "gemini"
                assert config.model == "gemini-pro"

                config2 = load_config()
                assert config2.engine == "openai"
                assert config2.model == "gpt-3.5"

    def test_load_with_kwargs(self) -> None:
        """Extra kwargs raise ValidationError (extra fields forbidden)."""
        with pytest.raises(ValidationError):
            load_config(engine="openai", custom_param="should_fail")

    def test_load_merges_partial_configs(self) -> None:
        """Partial configs from different sources are merged correctly."""
        env_vars = {"LOCLEAN_ENGINE": "openai"}

        with patch.dict(os.environ, env_vars):
            with patch(
                "loclean.inference.config._load_from_pyproject_toml",
                return_value={"model": "gpt-4o"},
            ):
                config = load_config(api_key="sk-runtime-key")
                assert config.engine == "openai"
                assert config.model == "gpt-4o"
                assert config.api_key == "sk-runtime-key"
