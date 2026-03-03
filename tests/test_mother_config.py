"""
Phase 1: Tests for Mother configuration system.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from mother.config import (
    MotherConfig,
    load_config,
    save_config,
    detect_first_run,
    DEFAULT_MODELS,
    ENV_VARS,
    PROVIDERS,
)


class TestMotherConfig:
    """Test MotherConfig dataclass."""

    def test_defaults(self):
        config = MotherConfig()
        assert config.name == "Mother"
        assert config.personality == "composed"
        assert config.provider == "claude"
        assert config.setup_complete is False

    def test_get_model_returns_set_model(self):
        config = MotherConfig(model="gpt-4o")
        assert config.get_model() == "gpt-4o"

    def test_get_env_var(self):
        config = MotherConfig(provider="openai")
        assert config.get_env_var() == "OPENAI_API_KEY"


class TestLoadSaveConfig:
    """Test config persistence."""

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            name="Athena",
            personality="warm",
            provider="openai",
            setup_complete=True,
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.name == "Athena"
        assert loaded.personality == "warm"
        assert loaded.provider == "openai"
        assert loaded.setup_complete is True

    def test_load_missing_file_returns_defaults(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        config = load_config(path)
        assert config.name == "Mother"
        assert config.setup_complete is False

    def test_load_corrupt_json_returns_defaults(self, tmp_path):
        path = tmp_path / "mother.json"
        path.write_text("not json {{{")
        config = load_config(str(path))
        assert config.name == "Mother"

    def test_load_ignores_unknown_fields(self, tmp_path):
        path = tmp_path / "mother.json"
        path.write_text(json.dumps({"name": "Test", "unknown_field": True}))
        config = load_config(str(path))
        assert config.name == "Test"

    def test_save_creates_directory(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "mother.json")
        save_config(MotherConfig(), path)
        assert Path(path).exists()


class TestDetectFirstRun:
    """Test first-run detection."""

    def test_first_run_no_file(self, tmp_path):
        path = str(tmp_path / "mother.json")
        assert detect_first_run(path) is True

    def test_not_first_run_after_setup(self, tmp_path):
        path = str(tmp_path / "mother.json")
        save_config(MotherConfig(setup_complete=True), path)
        assert detect_first_run(path) is False
