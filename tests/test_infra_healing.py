"""Tests for mother/infra_healing.py — infrastructure self-healing."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from mother.infra_healing import (
    HealthCheck,
    HealingResult,
    InfraHealer,
)


# --- Fixtures ---

@pytest.fixture
def healer(tmp_path):
    """InfraHealer with temp config dir."""
    return InfraHealer(config_dir=tmp_path)


@pytest.fixture
def healthy_infra(tmp_path):
    """Set up a fully healthy infrastructure."""
    # Config dir
    tmp_path.mkdir(exist_ok=True)

    # Config file
    config = {"schema_version": 1, "name": "Mother"}
    (tmp_path / "mother.json").write_text(json.dumps(config))

    # Databases
    for db_name in ("history.db", "tools.db"):
        conn = sqlite3.connect(str(tmp_path / db_name))
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
        conn.close()

    return InfraHealer(config_dir=tmp_path)


# --- HealthCheck ---

class TestHealthCheck:
    def test_frozen(self):
        hc = HealthCheck(
            component="config",
            status="healthy",
            message="OK",
            auto_healable=True,
        )
        with pytest.raises(AttributeError):
            hc.status = "broken"  # type: ignore[misc]

    def test_fields(self):
        hc = HealthCheck(
            component="config",
            status="broken",
            message="corrupt",
            auto_healable=True,
        )
        assert hc.component == "config"
        assert hc.status == "broken"
        assert hc.auto_healable is True


# --- HealingResult ---

class TestHealingResult:
    def test_fields(self):
        hr = HealingResult(
            component="config",
            action="recreated",
            success=True,
            detail="done",
        )
        assert hr.component == "config"
        assert hr.action == "recreated"
        assert hr.success is True


# --- diagnose ---

class TestDiagnose:
    def test_returns_checks_for_all_components(self, healer):
        checks = healer.diagnose()
        components = {c.component for c in checks}
        assert "config_dir" in components
        assert "config" in components
        assert "history_db" in components
        assert "tools_db" in components
        assert "temp_files" in components
        assert "locks" in components

    def test_healthy_infra_all_healthy(self, healthy_infra):
        checks = healthy_infra.diagnose()
        for check in checks:
            assert check.status == "healthy", f"{check.component}: {check.message}"


# --- check_config_dir ---

class TestCheckConfigDir:
    def test_existing_dir(self, healer, tmp_path):
        check = healer.check_config_dir()
        assert check.status == "healthy"

    def test_missing_dir(self):
        healer = InfraHealer(config_dir=Path("/tmp/nonexistent_motherlabs_test"))
        check = healer.check_config_dir()
        assert check.status == "broken"
        assert check.auto_healable is True


# --- check_config ---

class TestCheckConfig:
    def test_missing_config(self, healer):
        check = healer.check_config()
        assert check.status == "degraded"
        assert check.auto_healable is True

    def test_valid_config(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text('{"name": "Mother"}')
        check = healer.check_config()
        assert check.status == "healthy"

    def test_empty_config(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text("")
        check = healer.check_config()
        assert check.status == "broken"

    def test_corrupt_json(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text("{invalid json!!!")
        check = healer.check_config()
        assert check.status == "broken"
        assert check.auto_healable is True

    def test_non_dict_json(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text("[1, 2, 3]")
        check = healer.check_config()
        assert check.status == "broken"


# --- check_history_db ---

class TestCheckHistoryDb:
    def test_missing_db(self, healer):
        check = healer.check_history_db()
        assert check.status == "degraded"

    def test_valid_db(self, healer, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "history.db"))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()
        check = healer.check_history_db()
        assert check.status == "healthy"

    def test_corrupt_db(self, healer, tmp_path):
        # Write a file with the SQLite magic header but corrupt page data.
        # This ensures sqlite3.connect() recognizes it as SQLite but fails
        # on any actual operation, reliably across all SQLite versions.
        header = b"SQLite format 3\x00"  # 16-byte magic
        corrupt_page = b"\xff" * 4080     # garbage page data
        (tmp_path / "history.db").write_bytes(header + corrupt_page)
        check = healer.check_history_db()
        assert check.status == "broken"
        assert check.auto_healable is True


# --- check_temp_files ---

class TestCheckTempFiles:
    def test_no_temp_dir(self, healer):
        check = healer.check_temp_files()
        assert check.status == "healthy"

    def test_no_stale_files(self, healer, tmp_path):
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        (temp_dir / "recent.txt").write_text("fresh")
        check = healer.check_temp_files()
        assert check.status == "healthy"

    def test_stale_files_detected(self, healer, tmp_path):
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        stale = temp_dir / "old.txt"
        stale.write_text("stale")
        # Set mtime to 48 hours ago
        old_time = time.time() - (48 * 3600)
        import os
        os.utime(stale, (old_time, old_time))
        check = healer.check_temp_files()
        assert check.status == "degraded"
        assert "1 stale" in check.message


# --- check_stale_locks ---

class TestCheckStaleLocks:
    def test_no_lock_dir(self, healer):
        check = healer.check_stale_locks()
        assert check.status == "healthy"

    def test_no_stale_locks(self, healer, tmp_path):
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        (lock_dir / "active.lock").write_text("pid:123")
        check = healer.check_stale_locks()
        assert check.status == "healthy"

    def test_stale_lock_detected(self, healer, tmp_path):
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        stale = lock_dir / "old.lock"
        stale.write_text("pid:999")
        old_time = time.time() - (60 * 60)  # 1 hour ago
        import os
        os.utime(stale, (old_time, old_time))
        check = healer.check_stale_locks()
        assert check.status == "degraded"
        assert "1 stale" in check.message


# --- heal ---

class TestHeal:
    def test_healthy_returns_empty(self, healthy_infra):
        results = healthy_infra.heal()
        assert results == []

    def test_heals_missing_config_dir(self):
        config_dir = Path("/tmp") / f"motherlabs_test_{int(time.time())}"
        try:
            healer = InfraHealer(config_dir=config_dir)
            results = healer.heal()
            # Should have healed config_dir at minimum
            config_dir_results = [r for r in results if r.component == "config_dir"]
            assert len(config_dir_results) == 1
            assert config_dir_results[0].success is True
            assert config_dir.is_dir()
        finally:
            import shutil
            shutil.rmtree(config_dir, ignore_errors=True)

    def test_heals_corrupt_config(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text("{corrupt!!!")
        results = healer.heal()
        config_results = [r for r in results if r.component == "config"]
        assert len(config_results) == 1
        assert config_results[0].success is True
        # Verify config was recreated
        new_config = json.loads((tmp_path / "mother.json").read_text())
        assert new_config["schema_version"] == 1

    def test_heals_corrupt_db(self, healer, tmp_path):
        # Write SQLite magic header + corrupt page data — reliably triggers DatabaseError.
        header = b"SQLite format 3\x00"
        (tmp_path / "history.db").write_bytes(header + b"\xff" * 4080)
        results = healer.heal()
        db_results = [r for r in results if r.component == "history_db"]
        assert len(db_results) == 1
        assert db_results[0].success is True

    def test_heals_stale_temp_files(self, healer, tmp_path):
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        stale = temp_dir / "old.txt"
        stale.write_text("stale")
        import os
        old_time = time.time() - (48 * 3600)
        os.utime(stale, (old_time, old_time))

        results = healer.heal()
        temp_results = [r for r in results if r.component == "temp_files"]
        assert len(temp_results) == 1
        assert temp_results[0].success is True
        assert not stale.exists()

    def test_heals_stale_locks(self, healer, tmp_path):
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        stale = lock_dir / "old.lock"
        stale.write_text("pid:999")
        import os
        old_time = time.time() - (60 * 60)
        os.utime(stale, (old_time, old_time))

        results = healer.heal()
        lock_results = [r for r in results if r.component == "locks"]
        assert len(lock_results) == 1
        assert lock_results[0].success is True
        assert not stale.exists()

    def test_heal_idempotent(self, healer, tmp_path):
        (tmp_path / "mother.json").write_text("{corrupt!!!")
        healer.heal()
        # Second heal should find everything healthy
        results = healer.heal()
        assert all(r.success for r in results) or results == []

    def test_heal_only_auto_healable(self, healer):
        """Heal skips non-auto-healable checks."""
        checks = [
            HealthCheck(
                component="custom",
                status="broken",
                message="manual fix needed",
                auto_healable=False,
            ),
        ]
        results = healer.heal(checks)
        assert len(results) == 1
        assert results[0].action == "skipped"
        assert results[0].success is False

    def test_heal_with_provided_checks(self, healer, tmp_path):
        checks = [
            HealthCheck(
                component="config",
                status="broken",
                message="corrupt",
                auto_healable=True,
            ),
        ]
        results = healer.heal(checks)
        assert len(results) == 1
        assert results[0].component == "config"
