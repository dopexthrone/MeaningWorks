"""Tests for mother/appendage_runtime.py — AppendageProcess + JSON Lines IPC."""

import json
import os
import sys
import textwrap
import time

import pytest

from mother.appendage_runtime import AppendageProcess, SpawnResult, InvokeResult


def _write_agent(tmp_path, script_content, filename="main.py"):
    """Write a Python script to tmp_path and return the directory."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    (agent_dir / filename).write_text(textwrap.dedent(script_content))
    return str(agent_dir)


class TestSpawnResult:
    def test_defaults(self):
        r = SpawnResult(success=True)
        assert r.pid == 0
        assert r.capabilities == ()
        assert r.error == ""

    def test_frozen(self):
        r = SpawnResult(success=True, pid=123)
        with pytest.raises(AttributeError):
            r.pid = 456


class TestInvokeResult:
    def test_defaults(self):
        r = InvokeResult(success=False, error="bad")
        assert r.output is None
        assert r.duration_seconds == 0.0


class TestAppendageProcess:
    def test_spawn_with_ready_signal(self, tmp_path):
        """Agent sends ready signal, spawn succeeds."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            sys.stdout.write(json.dumps({
                "id": "ready",
                "result": {"capabilities": ["count", "list"]}
            }) + "\\n")
            sys.stdout.flush()
            # Keep alive waiting for input
            for line in sys.stdin:
                req = json.loads(line.strip())
                sys.stdout.write(json.dumps({
                    "id": req["id"], "result": {"ok": True}, "error": None
                }) + "\\n")
                sys.stdout.flush()
        ''')

        proc = AppendageProcess(project_dir)
        result = proc.spawn(timeout=10.0)
        try:
            assert result.success is True
            assert result.pid > 0
            assert "count" in result.capabilities
            assert "list" in result.capabilities
        finally:
            proc.stop()

    def test_spawn_no_ready_signal_timeout(self, tmp_path):
        """Agent never sends ready signal, spawn times out."""
        project_dir = _write_agent(tmp_path, '''
            import time
            time.sleep(30)
        ''')

        proc = AppendageProcess(project_dir)
        result = proc.spawn(timeout=1.0)
        assert result.success is False
        assert "Timeout" in result.error

    def test_spawn_process_crashes(self, tmp_path):
        """Agent crashes on startup."""
        project_dir = _write_agent(tmp_path, '''
            raise RuntimeError("startup failure")
        ''')

        proc = AppendageProcess(project_dir)
        result = proc.spawn(timeout=3.0)
        assert result.success is False
        assert "exited during startup" in result.error or "Timeout" in result.error

    def test_spawn_entry_point_not_found(self, tmp_path):
        """Entry point doesn't exist."""
        project_dir = str(tmp_path / "nonexistent")
        os.makedirs(project_dir, exist_ok=True)

        proc = AppendageProcess(project_dir, entry_point="missing.py")
        result = proc.spawn()
        assert result.success is False
        assert "not found" in result.error

    def test_invoke_request_response(self, tmp_path):
        """Full request/response cycle."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            sys.stdout.write(json.dumps({
                "id": "ready", "result": {"capabilities": ["echo"]}
            }) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                req = json.loads(line.strip())
                params = req.get("params", {})
                sys.stdout.write(json.dumps({
                    "id": req["id"],
                    "result": {"echoed": params.get("message", "")},
                    "error": None
                }) + "\\n")
                sys.stdout.flush()
        ''')

        proc = AppendageProcess(project_dir)
        spawn = proc.spawn(timeout=10.0)
        assert spawn.success

        try:
            result = proc.invoke({"message": "hello"}, timeout=5.0)
            assert result.success is True
            assert result.output == {"echoed": "hello"}
            assert result.duration_seconds > 0
            assert result.error == ""
        finally:
            proc.stop()

    def test_invoke_with_error_response(self, tmp_path):
        """Agent returns an error in the response."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            sys.stdout.write(json.dumps({
                "id": "ready", "result": {"capabilities": []}
            }) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                req = json.loads(line.strip())
                sys.stdout.write(json.dumps({
                    "id": req["id"],
                    "result": None,
                    "error": "something went wrong"
                }) + "\\n")
                sys.stdout.flush()
        ''')

        proc = AppendageProcess(project_dir)
        proc.spawn(timeout=10.0)
        try:
            result = proc.invoke({"test": True}, timeout=5.0)
            assert result.success is False
            assert "something went wrong" in result.error
        finally:
            proc.stop()

    def test_invoke_process_not_running(self):
        """Invoke on a process that was never started."""
        proc = AppendageProcess("/tmp/nonexistent")
        result = proc.invoke({"test": True})
        assert result.success is False
        assert "not running" in result.error

    def test_is_alive(self, tmp_path):
        """is_alive reflects process state."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            sys.stdout.write(json.dumps({
                "id": "ready", "result": {"capabilities": []}
            }) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                pass
        ''')

        proc = AppendageProcess(project_dir)
        assert proc.is_alive() is False

        proc.spawn(timeout=10.0)
        assert proc.is_alive() is True

        proc.stop()
        assert proc.is_alive() is False

    def test_stop_idempotent(self, tmp_path):
        """Stopping a non-running process is safe."""
        proc = AppendageProcess(str(tmp_path))
        proc.stop()  # Should not raise
        proc.stop()  # Should not raise

    def test_spawn_already_running(self, tmp_path):
        """Cannot spawn twice."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            sys.stdout.write(json.dumps({
                "id": "ready", "result": {"capabilities": []}
            }) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                pass
        ''')

        proc = AppendageProcess(project_dir)
        proc.spawn(timeout=10.0)
        try:
            result = proc.spawn(timeout=1.0)
            assert result.success is False
            assert "already running" in result.error
        finally:
            proc.stop()

    def test_malformed_json_ignored(self, tmp_path):
        """Non-JSON output from agent is silently ignored."""
        project_dir = _write_agent(tmp_path, '''
            import json, sys
            # Some non-JSON output first
            print("Starting up...")
            print("Loading modules...")
            # Then the ready signal
            sys.stdout.write(json.dumps({
                "id": "ready", "result": {"capabilities": ["test"]}
            }) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                req = json.loads(line.strip())
                # Emit some garbage between responses
                print("debug: processing request")
                sys.stdout.write(json.dumps({
                    "id": req["id"], "result": {"ok": True}, "error": None
                }) + "\\n")
                sys.stdout.flush()
        ''')

        proc = AppendageProcess(project_dir)
        spawn = proc.spawn(timeout=10.0)
        assert spawn.success
        try:
            result = proc.invoke({"x": 1}, timeout=5.0)
            assert result.success
        finally:
            proc.stop()
