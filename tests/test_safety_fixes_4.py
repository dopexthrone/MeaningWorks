"""
Tests for safety fixes 11-14 (Tier 3):
  Fix 11: Remove hardcoded credentials from config defaults
  Fix 12: Secure config file permissions + type validation
  Fix 13: Atomic PID lockfile with proper permissions
  Fix 14: Rate limit depth-chain enqueue in daemon
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.config import (
    MotherConfig,
    load_config,
    save_config,
    _validate_config_types,
)


# ---------------------------------------------------------------------------
# Fix 11: No hardcoded credentials in defaults
# ---------------------------------------------------------------------------

class TestNoHardcodedCredentials:
    """Verify API credentials are empty by default."""

    def test_twilio_account_sid_empty(self):
        config = MotherConfig()
        assert config.twilio_account_sid == ""

    def test_twilio_auth_token_empty(self):
        config = MotherConfig()
        assert config.twilio_auth_token == ""

    def test_ngrok_auth_token_empty(self):
        config = MotherConfig()
        assert config.ngrok_auth_token == ""

    def test_twilio_whatsapp_number_empty(self):
        config = MotherConfig()
        assert config.twilio_whatsapp_number == ""

    def test_user_whatsapp_number_empty(self):
        config = MotherConfig()
        assert config.user_whatsapp_number == ""

    def test_bluesky_app_password_empty(self):
        config = MotherConfig()
        assert config.bluesky_app_password == ""

    def test_no_credentials_in_source(self):
        """Source code should not contain real credential patterns."""
        import inspect
        source = inspect.getsource(MotherConfig)
        # Real Twilio SIDs start with AC + 32 hex chars
        assert "AC0" not in source
        # Auth tokens are 32 hex chars
        assert "741241" not in source
        # Ngrok tokens have a specific pattern
        assert "39hSJ" not in source

    def test_config_from_file_preserves_user_credentials(self, tmp_path):
        """User-set credentials in config file are preserved on load."""
        config_file = tmp_path / "mother.json"
        config_file.write_text(json.dumps({
            "twilio_account_sid": "ACuser123",
            "twilio_auth_token": "usertoken456",
        }))
        config = load_config(str(config_file))
        assert config.twilio_account_sid == "ACuser123"
        assert config.twilio_auth_token == "usertoken456"


# ---------------------------------------------------------------------------
# Fix 12: Secure config file permissions + type validation
# ---------------------------------------------------------------------------

class TestConfigFilePermissions:
    """Verify config file is saved with restricted permissions."""

    def test_saved_config_has_0600_permissions(self, tmp_path):
        config = MotherConfig()
        path = save_config(config, str(tmp_path / "test.json"))
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600", f"Config file has mode {mode}, expected 600"

    def test_saved_config_readable_by_owner(self, tmp_path):
        config = MotherConfig(name="TestMother")
        path = save_config(config, str(tmp_path / "test.json"))
        loaded = load_config(str(path))
        assert loaded.name == "TestMother"


class TestConfigTypeValidation:
    """Verify config type validation catches bad types."""

    def test_valid_types_pass_through(self):
        data = {"name": "Test", "cost_limit": 50.0, "setup_complete": True}
        cleaned = _validate_config_types(data)
        assert cleaned == data

    def test_string_in_int_field_rejected(self):
        data = {"daemon_health_check_interval": "malicious"}
        cleaned = _validate_config_types(data)
        assert "daemon_health_check_interval" not in cleaned

    def test_string_in_float_field_rejected(self):
        data = {"cost_limit": "not_a_number"}
        cleaned = _validate_config_types(data)
        assert "cost_limit" not in cleaned

    def test_string_in_bool_field_rejected(self):
        data = {"setup_complete": "true"}
        cleaned = _validate_config_types(data)
        assert "setup_complete" not in cleaned

    def test_int_in_str_field_rejected(self):
        data = {"name": 12345}
        cleaned = _validate_config_types(data)
        assert "name" not in cleaned

    def test_unknown_fields_dropped(self):
        data = {"nonexistent_field": "value", "name": "Test"}
        cleaned = _validate_config_types(data)
        assert "nonexistent_field" not in cleaned
        assert cleaned["name"] == "Test"

    def test_float_coerced_to_int(self):
        """Float with no decimal part is accepted for int fields."""
        data = {"daemon_health_check_interval": 300.0}
        cleaned = _validate_config_types(data)
        assert cleaned["daemon_health_check_interval"] == 300

    def test_int_accepted_for_float_field(self):
        """Int is valid for float fields."""
        data = {"cost_limit": 100}
        cleaned = _validate_config_types(data)
        assert cleaned["cost_limit"] == 100

    def test_malformed_json_returns_defaults(self, tmp_path):
        config_file = tmp_path / "mother.json"
        config_file.write_text("this is not json{{{")
        config = load_config(str(config_file))
        assert config.name == "Mother"  # default

    def test_mixed_valid_invalid_fields(self):
        """Valid fields pass, invalid types are dropped."""
        data = {
            "name": "Test",
            "cost_limit": "not_valid",  # should be float
            "setup_complete": True,
            "daemon_health_check_interval": "bad",  # should be int
        }
        cleaned = _validate_config_types(data)
        assert cleaned["name"] == "Test"
        assert cleaned["setup_complete"] is True
        assert "cost_limit" not in cleaned
        assert "daemon_health_check_interval" not in cleaned


# ---------------------------------------------------------------------------
# Fix 13: Atomic PID lockfile
# ---------------------------------------------------------------------------

from mother.app import acquire_lock, release_lock


class TestAtomicPidLockfile:
    """Verify PID lockfile atomicity and permissions."""

    def test_lock_creates_file(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        assert acquire_lock(pid_path)
        assert pid_path.exists()

    def test_lock_contains_current_pid(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        acquire_lock(pid_path)
        stored = int(pid_path.read_text().strip())
        assert stored == os.getpid()

    def test_lock_has_0600_permissions(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        acquire_lock(pid_path)
        mode = oct(pid_path.stat().st_mode)[-3:]
        assert mode == "600", f"Lock file has mode {mode}, expected 600"

    def test_lock_rejects_if_alive(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        # Write a PID that IS alive (our own PID)
        pid_path.write_text(str(os.getpid()))
        assert not acquire_lock(pid_path)

    def test_lock_overwrites_stale(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        # Write a PID that definitely doesn't exist
        pid_path.write_text("99999999")
        assert acquire_lock(pid_path)
        stored = int(pid_path.read_text().strip())
        assert stored == os.getpid()

    def test_release_removes_own_lock(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        acquire_lock(pid_path)
        release_lock(pid_path)
        assert not pid_path.exists()

    def test_release_preserves_other_pid(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        pid_path.write_text("99999999")
        release_lock(pid_path)
        # Should not delete because PID doesn't match
        assert pid_path.exists()

    def test_lock_handles_corrupt_file(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        pid_path.write_text("not_a_number")
        # Should overwrite corrupt file
        assert acquire_lock(pid_path)


# ---------------------------------------------------------------------------
# Fix 14: Rate limit depth-chain enqueue
# ---------------------------------------------------------------------------

from mother.daemon import DaemonMode, DaemonConfig


class TestDepthChainRateLimit:
    """Verify depth-chain enqueue is rate-limited."""

    def _make_daemon(self):
        daemon = DaemonMode(config=DaemonConfig(max_queue_size=20))
        daemon._running = True
        return daemon

    def test_daemon_has_rate_limit_state(self):
        daemon = self._make_daemon()
        assert hasattr(daemon, "_depth_chain_enqueue_times")
        assert hasattr(daemon, "_depth_chain_max_per_hour")
        assert daemon._depth_chain_max_per_hour == 10

    def _run(self, coro):
        """Run async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_rate_limit_blocks_after_max(self):
        """After max enqueues per hour, further chains are suppressed."""
        daemon = self._make_daemon()
        # Simulate 10 recent enqueues
        now = time.time()
        daemon._depth_chain_enqueue_times = [now - i for i in range(10)]

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore something", "chain_type": "frontier", "priority": 0.5}
        ]

        count = self._run(daemon._enqueue_depth_chains(result, "software", 0))
        assert count == 0

    def test_rate_limit_allows_after_expiry(self):
        """Old enqueues beyond 1 hour don't count against the limit."""
        daemon = self._make_daemon()
        # Simulate 10 enqueues from 2 hours ago
        two_hours_ago = time.time() - 7200
        daemon._depth_chain_enqueue_times = [two_hours_ago - i for i in range(10)]

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore fresh", "chain_type": "frontier", "priority": 0.5}
        ]

        count = self._run(daemon._enqueue_depth_chains(result, "software", 0))
        # Should be allowed (old entries expired)
        assert count >= 1

    def test_rate_limit_tracks_new_enqueues(self):
        """Each successful enqueue is recorded in the timestamp list."""
        daemon = self._make_daemon()
        initial_count = len(daemon._depth_chain_enqueue_times)

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore one", "chain_type": "frontier", "priority": 0.5},
            {"intent_text": "explore two", "chain_type": "low_conf", "priority": 0.3},
        ]

        count = self._run(daemon._enqueue_depth_chains(result, "software", 0))
        assert len(daemon._depth_chain_enqueue_times) == initial_count + count

    def test_depth_limit_still_enforced(self):
        """Depth limit (MAX_CHAIN_DEPTH) still blocks deep chains."""
        daemon = self._make_daemon()
        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "deep explore", "chain_type": "frontier", "priority": 0.5}
        ]

        count = self._run(daemon._enqueue_depth_chains(result, "software", daemon.MAX_CHAIN_DEPTH))
        assert count == 0

    def test_failed_result_not_chained(self):
        """Failed compilations don't enqueue depth chains."""
        daemon = self._make_daemon()
        result = MagicMock()
        result.success = False
        result.depth_chains = [
            {"intent_text": "should not enqueue", "chain_type": "frontier", "priority": 0.5}
        ]

        count = self._run(daemon._enqueue_depth_chains(result, "software", 0))
        assert count == 0
