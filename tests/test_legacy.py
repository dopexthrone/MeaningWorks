"""Tests for mother/legacy.py — legacy preservation and schema migration."""

import json
import time
from pathlib import Path

import pytest

from mother.legacy import (
    CURRENT_SCHEMA_VERSION,
    LegacyError,
    LegacyGuard,
    MigrationResult,
    _migrate_v0_to_v1,
)


# --- Fixtures ---

@pytest.fixture
def v0_config(tmp_path):
    """A version-0 config (no schema_version)."""
    config = {
        "name": "Mother",
        "provider": "claude",
        "model": "claude-sonnet-4-20250514",
    }
    path = tmp_path / "mother.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.fixture
def v1_config(tmp_path):
    """A current-version config."""
    config = {
        "schema_version": 1,
        "name": "Mother",
        "provider": "claude",
        "api_keys": {"claude": "sk-test"},
    }
    path = tmp_path / "mother.json"
    path.write_text(json.dumps(config))
    return path, config


# --- MigrationResult ---

class TestMigrationResult:
    def test_fields(self):
        r = MigrationResult(
            from_version=0,
            to_version=1,
            changes=["added version"],
            success=True,
            backup_path=Path("/tmp/backup"),
        )
        assert r.from_version == 0
        assert r.to_version == 1
        assert r.success is True
        assert r.backup_path == Path("/tmp/backup")


# --- read_config ---

class TestReadConfig:
    def test_missing_file(self, tmp_path):
        data, version = LegacyGuard.read_config(tmp_path / "nope.json")
        assert data == {}
        assert version == 0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        data, version = LegacyGuard.read_config(path)
        assert data == {}
        assert version == 0

    def test_no_version_field(self, v0_config):
        path, _ = v0_config
        data, version = LegacyGuard.read_config(path)
        assert version == 0
        assert data["name"] == "Mother"

    def test_with_version(self, v1_config):
        path, _ = v1_config
        data, version = LegacyGuard.read_config(path)
        assert version == 1
        assert data["name"] == "Mother"

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json!!!")
        data, version = LegacyGuard.read_config(path)
        assert data == {}
        assert version == 0

    def test_non_dict_json(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text('[1, 2, 3]')
        data, version = LegacyGuard.read_config(path)
        assert data == {}
        assert version == 0

    def test_non_int_version(self, tmp_path):
        path = tmp_path / "bad_version.json"
        path.write_text(json.dumps({"schema_version": "one", "name": "X"}))
        data, version = LegacyGuard.read_config(path)
        assert version == 0


# --- needs_migration ---

class TestNeedsMigration:
    def test_v0_needs_migration(self):
        assert LegacyGuard.needs_migration({}) is True
        assert LegacyGuard.needs_migration({"name": "Mother"}) is True

    def test_current_no_migration(self):
        assert LegacyGuard.needs_migration(
            {"schema_version": CURRENT_SCHEMA_VERSION}
        ) is False

    def test_non_int_version_needs_migration(self):
        assert LegacyGuard.needs_migration({"schema_version": "bad"}) is True


# --- migrate ---

class TestMigrate:
    def test_v0_to_v1(self):
        data = {"name": "Mother", "provider": "claude"}
        result = LegacyGuard.migrate(data, 0)
        assert result.success is True
        assert result.from_version == 0
        assert result.to_version == CURRENT_SCHEMA_VERSION
        assert len(result.changes) > 0

    def test_already_current(self):
        data = {"schema_version": CURRENT_SCHEMA_VERSION}
        result = LegacyGuard.migrate(data, CURRENT_SCHEMA_VERSION)
        assert result.success is True
        assert "Already at current version" in result.changes

    def test_future_version_raises(self):
        with pytest.raises(LegacyError, match="newer than supported"):
            LegacyGuard.migrate({"schema_version": 999}, 999)

    def test_does_not_mutate_input(self):
        data = {"name": "Mother"}
        original = data.copy()
        LegacyGuard.migrate(data, 0)
        assert data == original


# --- _migrate_v0_to_v1 ---

class TestMigrateV0ToV1:
    def test_adds_schema_version(self):
        data, changes = _migrate_v0_to_v1({"name": "Mother"})
        assert data["schema_version"] == 1
        assert any("schema_version" in c for c in changes)

    def test_adds_api_keys(self):
        data, changes = _migrate_v0_to_v1({"name": "Mother"})
        assert data["api_keys"] == {}
        assert any("api_keys" in c for c in changes)

    def test_preserves_existing_api_keys(self):
        data, changes = _migrate_v0_to_v1({
            "name": "Mother",
            "api_keys": {"claude": "sk-test"},
        })
        assert data["api_keys"] == {"claude": "sk-test"}

    def test_resets_invalid_api_keys(self):
        data, changes = _migrate_v0_to_v1({
            "name": "Mother",
            "api_keys": "not-a-dict",
        })
        assert data["api_keys"] == {}
        assert any("Reset" in c for c in changes)

    def test_converts_output_dir(self):
        data, changes = _migrate_v0_to_v1({
            "name": "Mother",
            "output_dir": 42,
        })
        assert data["output_dir"] == "42"

    def test_does_not_mutate_input(self):
        original = {"name": "Mother"}
        data_copy = original.copy()
        _migrate_v0_to_v1(data_copy)
        assert data_copy == original


# --- backup_before_migration ---

class TestBackup:
    def test_creates_backup(self, v0_config):
        path, _ = v0_config
        backup = LegacyGuard.backup_before_migration(path)
        assert backup.exists()
        assert ".bak." in backup.name
        assert backup.read_text() == path.read_text()

    def test_backup_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LegacyGuard.backup_before_migration(tmp_path / "nope.json")


# --- ensure_schema_version ---

class TestEnsureSchemaVersion:
    def test_stamps_missing(self):
        data = {"name": "Mother"}
        result = LegacyGuard.ensure_schema_version(data)
        assert result["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "schema_version" not in data  # original unchanged

    def test_preserves_existing(self):
        data = {"schema_version": 42, "name": "Mother"}
        result = LegacyGuard.ensure_schema_version(data)
        assert result["schema_version"] == 42

    def test_idempotent(self):
        data = {"name": "Mother"}
        r1 = LegacyGuard.ensure_schema_version(data)
        r2 = LegacyGuard.ensure_schema_version(r1)
        assert r1 == r2


# --- migrate_file ---

class TestMigrateFile:
    def test_migrate_v0_file(self, v0_config):
        path, _ = v0_config
        result = LegacyGuard.migrate_file(path)
        assert result.success is True
        assert result.backup_path is not None
        assert result.backup_path.exists()

        # Verify written file is now v1
        new_data = json.loads(path.read_text())
        assert new_data["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_no_migration_needed(self, v1_config):
        path, _ = v1_config
        result = LegacyGuard.migrate_file(path)
        assert result.success is True
        assert "No migration needed" in result.changes

    def test_dry_run(self, v0_config):
        path, original = v0_config
        result = LegacyGuard.migrate_file(path, dry_run=True)
        assert result.success is True
        assert result.backup_path is None
        # File unchanged
        current = json.loads(path.read_text())
        assert "schema_version" not in current
