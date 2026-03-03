"""End-to-end integration tests for the First Instance.

Verifies the full loop: compile → write → start → message → self-extend.
Uses MockClient for compilation and real TCP for runtime communication.

All tests are marked @pytest.mark.slow — skipped by default.
Run with: pytest tests/test_first_instance.py -v --run-slow
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from core.domain_adapter import RuntimeCapabilities
from core.project_writer import write_project, ProjectConfig
from core.runtime_scaffold import (
    generate_runtime_py,
    generate_state_py,
    generate_tools_py,
    generate_llm_client_py,
    generate_config_py,
    generate_recompile_py,
)

slow = pytest.mark.slow


# =============================================================================
# FIXTURES
# =============================================================================

AGENT_BLUEPRINT = {
    "domain": "chat_agent",
    "core_need": "A simple chat agent for testing",
    "components": [
        {
            "name": "Chat Agent",
            "type": "agent",
            "description": "Handles chat messages",
        },
    ],
    "relationships": [],
    "constraints": [],
}

AGENT_CODE = {
    "Chat Agent": '''"""Chat Agent — handles incoming messages."""


class ChatAgent:
    """Simple chat agent for testing."""

    def __init__(self, state=None, llm=None, tools=None):
        self.state = state
        self.llm = llm
        self.tools = tools

    async def handle(self, message: dict) -> dict:
        """Handle an incoming message."""
        action = message.get("action", "")
        if action == "chat":
            text = message.get("message", "")
            return {"response": f"Echo: {text}"}
        return {"error": f"Unknown action: {action}"}
''',
}

RUNTIME_CAPABILITIES = RuntimeCapabilities(
    has_event_loop=True,
    has_llm_client=False,
    has_persistent_state=True,
    has_self_recompile=True,
    has_tool_execution=False,
    event_loop_type="asyncio",
    state_backend="sqlite",
    default_port=0,  # 0 = let OS pick a free port
)


def _write_test_project(tmpdir: str) -> str:
    """Write a minimal agent project for testing."""
    component_names = list(AGENT_CODE.keys())

    config = ProjectConfig(
        project_name="test_agent",
        runtime_capabilities=RUNTIME_CAPABILITIES,
    )
    manifest = write_project(
        AGENT_CODE, AGENT_BLUEPRINT, tmpdir,
        config=config,
    )
    return manifest.project_dir


def _patch_runtime_port(project_dir: str, port: int) -> None:
    """Patch config.py to use a specific port."""
    config_path = os.path.join(project_dir, "config.py")
    with open(config_path) as f:
        content = f.read()
    # Replace the default port with our test port
    content = content.replace('int(os.environ.get("PORT", "0"))',
                              f'int(os.environ.get("PORT", "{port}"))')
    with open(config_path, "w") as f:
        f.write(content)


async def _send_tcp_message(host: str, port: int, target: str, payload: dict) -> dict:
    """Send a single TCP JSON-line message and return the response."""
    reader, writer = await asyncio.open_connection(host, port)
    msg = json.dumps({"target": target, "payload": payload}) + "\n"
    writer.write(msg.encode())
    await writer.drain()
    data = await asyncio.wait_for(reader.readline(), timeout=10)
    writer.close()
    await writer.wait_closed()
    return json.loads(data.decode().strip())


# =============================================================================
# UNIT TESTS (no subprocess, fast)
# =============================================================================

class TestProjectWriteIntegration:
    """Verify project writing produces all expected files for agent system."""

    def test_writes_all_scaffold_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)
            expected = {"main.py", "runtime.py", "state.py", "config.py",
                        "recompile.py", "blueprint.json", "__init__.py",
                        "requirements.txt", "pyproject.toml", "README.md"}
            written = set(os.listdir(project_dir))
            for f in expected:
                assert f in written or any(f in w for w in written), f"Missing {f}"

    def test_runtime_py_has_system_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)
            with open(os.path.join(project_dir, "runtime.py")) as f:
                content = f.read()
            assert "_system" in content
            assert "_handle_system" in content

    def test_main_py_valid_syntax(self):
        import ast
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)
            with open(os.path.join(project_dir, "main.py")) as f:
                code = f.read()
            ast.parse(code)

    def test_recompile_py_has_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)
            with open(os.path.join(project_dir, "recompile.py")) as f:
                content = f.read()
            assert "restart_process" in content
            assert "os.execv" in content

    def test_no_variable_shadowing_in_main(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)
            with open(os.path.join(project_dir, "main.py")) as f:
                content = f.read()
            # "state = StateStore" should exist as infrastructure
            # "state = ChatAgent" should NOT exist (would shadow)
            lines = content.split("\n")
            state_assignments = [l.strip() for l in lines
                                 if l.strip().startswith("state") and "=" in l]
            # At most one state assignment (the infrastructure one)
            state_infra = [l for l in state_assignments if "StateStore" in l]
            state_component = [l for l in state_assignments if "ChatAgent" in l]
            assert len(state_infra) <= 1
            assert len(state_component) == 0


class TestMessageBridgeTranslation:
    """Verify messaging bridge translates commands correctly."""

    def test_learn_command_translation(self):
        from messaging.bridge import MessageBridge

        class TestBridge(MessageBridge):
            async def start(self): pass
            async def stop(self): pass

        bridge = TestBridge()
        msg = bridge.translate_to_runtime("/learn check the weather")
        assert msg == {
            "target": "_system",
            "payload": {"action": "learn", "skill": "check the weather"},
        }

    def test_status_command_translation(self):
        from messaging.bridge import MessageBridge

        class TestBridge(MessageBridge):
            async def start(self): pass
            async def stop(self): pass

        bridge = TestBridge()
        msg = bridge.translate_to_runtime("/status")
        assert msg == {
            "target": "_system",
            "payload": {"action": "status"},
        }


# =============================================================================
# SLOW TESTS (requires subprocess, real TCP)
# =============================================================================

@slow
class TestRuntimeIntegration:
    """Start a real runtime subprocess and communicate via TCP.

    These tests require: pytest --run-slow
    """

    def test_compile_start_message_respond(self):
        """Full loop: write project → start runtime → send message → get response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)

            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            test_port = sock.getsockname()[1]
            sock.close()

            _patch_runtime_port(project_dir, test_port)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            try:
                # Wait for READY signal
                deadline = time.time() + 15
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode()
                        pytest.fail(f"Runtime exited early: {stderr}")
                    line = proc.stdout.readline().decode().strip()
                    if "READY" in line:
                        ready = True
                        break

                assert ready, "Runtime did not start in time"

                async def _test():
                    resp = await _send_tcp_message(
                        "127.0.0.1", test_port,
                        "Chat Agent",
                        {"action": "chat", "message": "Hello!"},
                    )
                    assert "response" in resp
                    assert "Echo: Hello!" in resp["response"]

                asyncio.run(_test())

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

    def test_system_status_command(self):
        """Send _system/status and verify component list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)

            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            test_port = sock.getsockname()[1]
            sock.close()

            _patch_runtime_port(project_dir, test_port)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            try:
                deadline = time.time() + 15
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode()
                        pytest.fail(f"Runtime exited early: {stderr}")
                    line = proc.stdout.readline().decode().strip()
                    if "READY" in line:
                        ready = True
                        break

                assert ready, "Runtime did not start in time"

                async def _test():
                    resp = await _send_tcp_message(
                        "127.0.0.1", test_port,
                        "_system",
                        {"action": "status"},
                    )
                    assert "components" in resp
                    assert isinstance(resp["components"], list)

                asyncio.run(_test())

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

    def test_system_health_command(self):
        """Send _system/health and verify uptime/message count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)

            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            test_port = sock.getsockname()[1]
            sock.close()

            _patch_runtime_port(project_dir, test_port)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            try:
                deadline = time.time() + 15
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode()
                        pytest.fail(f"Runtime exited early: {stderr}")
                    line = proc.stdout.readline().decode().strip()
                    if "READY" in line:
                        ready = True
                        break

                assert ready, "Runtime did not start in time"

                async def _test():
                    resp = await _send_tcp_message(
                        "127.0.0.1", test_port,
                        "_system",
                        {"action": "health"},
                    )
                    assert "uptime" in resp
                    assert resp["uptime"] >= 0
                    assert "messages" in resp

                asyncio.run(_test())

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

    def test_system_learn_without_api(self):
        """Send _system/learn — should attempt recompile (will fail without API)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = _write_test_project(tmpdir)

            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            test_port = sock.getsockname()[1]
            sock.close()

            _patch_runtime_port(project_dir, test_port)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            try:
                deadline = time.time() + 15
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode()
                        pytest.fail(f"Runtime exited early: {stderr}")
                    line = proc.stdout.readline().decode().strip()
                    if "READY" in line:
                        ready = True
                        break

                assert ready, "Runtime did not start in time"

                async def _test():
                    resp = await _send_tcp_message(
                        "127.0.0.1", test_port,
                        "_system",
                        {"action": "learn", "skill": "check the weather"},
                    )
                    # Will either get recompile_requested or error (no httpx/API)
                    assert "status" in resp or "error" in resp

                asyncio.run(_test())

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
