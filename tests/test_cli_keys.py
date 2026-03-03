"""
Tests for CLI keys subcommand.

~6 tests covering create, list, revoke via CLI arg parsing + cmd_keys.
"""

import sys
import os
import pytest
from io import StringIO
from unittest.mock import patch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motherlabs_platform.auth import KeyStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_cli_keys.db")


@pytest.fixture
def store(db_path):
    return KeyStore(db_path=db_path)


class TestCLIKeysCreate:
    def test_create_key(self, store, db_path, capsys):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="create", name="test-key", rate_limit=100, budget=50.0, key_id=None)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            cmd_keys(args, cli, config)

        captured = capsys.readouterr()
        assert "API Key Created" in captured.out
        assert "test-key" in captured.out
        assert store.list_keys()[0].name == "test-key"

    def test_create_key_custom_limits(self, store, db_path, capsys):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="create", name="custom", rate_limit=500, budget=200.0, key_id=None)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            cmd_keys(args, cli, config)

        keys = store.list_keys()
        assert keys[0].rate_limit_per_hour == 500
        assert keys[0].budget_usd == 200.0


class TestCLIKeysList:
    def test_list_empty(self, db_path, capsys):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="list", name=None, rate_limit=100, budget=50.0, key_id=None)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            cmd_keys(args, cli, config)

        captured = capsys.readouterr()
        assert "No API keys" in captured.out

    def test_list_with_keys(self, store, db_path, capsys):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        store.create_key("alpha")
        store.create_key("beta")

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="list", name=None, rate_limit=100, budget=50.0, key_id=None)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            cmd_keys(args, cli, config)

        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out


class TestCLIKeysRevoke:
    def test_revoke_existing(self, store, db_path, capsys):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        key_id, _ = store.create_key("revoke-me")

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="revoke", key_id=key_id, name=None, rate_limit=100, budget=50.0)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            cmd_keys(args, cli, config)

        captured = capsys.readouterr()
        assert "revoked" in captured.out
        assert store.list_keys()[0].is_active is False

    def test_revoke_nonexistent(self, db_path):
        from cli.main import cmd_keys, CLI, ConfigManager
        import argparse

        cli = CLI()
        config = ConfigManager()
        args = argparse.Namespace(keys_command="revoke", key_id="fake-id", name=None, rate_limit=100, budget=50.0)

        with patch.dict(os.environ, {"MOTHERLABS_AUTH_DB": db_path}):
            with pytest.raises(SystemExit):
                cmd_keys(args, cli, config)
