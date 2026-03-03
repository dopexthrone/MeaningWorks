"""
Tests for Mother self-extension: auto-register, project memory,
tool awareness, and tool invocation.

Covers: bridge.register_build_as_tool, bridge.get_recent_projects,
        bridge.get_tool_details, bridge.run_tool,
        context.py additions, persona.py additions, chat.py wiring.
"""

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from mother.context import ContextData, synthesize_situation, synthesize_context
from mother.tool_runner import ToolRunResult, find_tool_project, run_tool


# =============================================================================
# Helpers
# =============================================================================

def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dataclass
class MockTrust:
    overall_score: float = 75.0
    verification_badge: str = "partial"
    structural: float = 80.0
    semantic: float = 70.0
    behavioral: float = 75.0
    interface: float = 65.0
    dependency: float = 80.0
    consistency: float = 85.0
    completeness: float = 70.0


@dataclass
class MockProjectManifest:
    project_dir: str = "/tmp/test-project"
    files_written: list = field(default_factory=lambda: ["main.py", "config.py"])
    total_lines: int = 100
    entry_point: str = "main.py"


@dataclass
class MockBuildResult:
    iterations: list = field(default_factory=lambda: [{}])
    components_fixed: list = field(default_factory=list)


@dataclass
class MockAgentResult:
    success: bool = True
    blueprint: Dict[str, Any] = field(default_factory=lambda: {
        "core_need": "hello-world-tool",
        "domain": "software",
        "components": [
            {"name": "main", "type": "module"},
            {"name": "config", "type": "module"},
        ],
        "relationships": [
            {"from": "main", "to": "config", "type": "uses"},
        ],
    })
    generated_code: Dict[str, str] = field(default_factory=lambda: {
        "main": "print('hello')",
        "config": "PORT = 8080",
    })
    trust: Optional[MockTrust] = field(default_factory=MockTrust)
    project_manifest: Optional[MockProjectManifest] = field(default_factory=MockProjectManifest)
    build_result: Optional[MockBuildResult] = field(default_factory=MockBuildResult)
    error: Optional[str] = None
    quality_score: float = 0.0


# =============================================================================
# Phase 1: Auto-register builds as tools
# =============================================================================

class TestRegisterBuildAsTool:
    def test_success(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult()

        with patch("core.determinism.compute_structural_fingerprint") as mock_fp, \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_store, \
             patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("core.tool_package.package_tool") as mock_pkg:

            mock_fp.return_value = MagicMock(hash_digest="abcd1234")
            inst = MagicMock()
            inst.instance_id = "test-instance-001"
            mock_store.return_value.get_or_create_self.return_value = inst

            registry = MagicMock()
            registry.find_by_fingerprint.return_value = None  # No duplicate
            mock_reg.return_value = registry

            pkg = MagicMock()
            pkg.package_id = "pkg001"
            pkg.name = "hello-world-tool"
            pkg.trust_score = 75.0
            pkg.verification_badge = "partial"
            mock_pkg.return_value = pkg

            info = _run_async(bridge.register_build_as_tool(result, "build a hello world tool"))

            assert info is not None
            assert info["package_id"] == "pkg001"
            assert info["name"] == "hello-world-tool"
            assert info["trust_score"] == 75.0
            registry.register_tool.assert_called_once_with(pkg, is_local=True)

    def test_missing_blueprint(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult(blueprint=None, generated_code=None)

        info = _run_async(bridge.register_build_as_tool(result, "test"))
        assert info is None

    def test_missing_generated_code(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult(generated_code=None)

        info = _run_async(bridge.register_build_as_tool(result, "test"))
        assert info is None

    def test_duplicate_fingerprint(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult()

        with patch("core.determinism.compute_structural_fingerprint") as mock_fp, \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_store, \
             patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("core.tool_package.package_tool") as mock_pkg:

            mock_fp.return_value = MagicMock(hash_digest="abcd1234")
            inst = MagicMock()
            inst.instance_id = "test-instance-001"
            mock_store.return_value.get_or_create_self.return_value = inst

            registry = MagicMock()
            # Already exists
            registry.find_by_fingerprint.return_value = MagicMock(package_id="existing")
            mock_reg.return_value = registry

            info = _run_async(bridge.register_build_as_tool(result, "test"))
            assert info is None
            registry.register_tool.assert_not_called()

    def test_no_trust(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult(trust=None)

        with patch("core.determinism.compute_structural_fingerprint") as mock_fp, \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_store, \
             patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("core.tool_package.package_tool") as mock_pkg:

            mock_fp.return_value = MagicMock(hash_digest="abcd5678")
            inst = MagicMock()
            inst.instance_id = "test-instance-001"
            mock_store.return_value.get_or_create_self.return_value = inst
            registry = MagicMock()
            registry.find_by_fingerprint.return_value = None
            mock_reg.return_value = registry
            pkg = MagicMock()
            pkg.package_id = "pkg002"
            pkg.name = "test-tool"
            pkg.trust_score = 0.0
            pkg.verification_badge = "unverified"
            mock_pkg.return_value = pkg

            info = _run_async(bridge.register_build_as_tool(result, "test"))
            assert info is not None
            assert info["trust_score"] == 0.0

    def test_empty_generated_code(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult(generated_code={})

        info = _run_async(bridge.register_build_as_tool(result, "test"))
        assert info is None

    def test_register_integrity_error(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")
        result = MockAgentResult()

        with patch("core.determinism.compute_structural_fingerprint") as mock_fp, \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_store, \
             patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("core.tool_package.package_tool") as mock_pkg:

            mock_fp.return_value = MagicMock(hash_digest="abcd1234")
            inst = MagicMock()
            inst.instance_id = "test-instance-001"
            mock_store.return_value.get_or_create_self.return_value = inst
            registry = MagicMock()
            registry.find_by_fingerprint.return_value = None
            registry.register_tool.side_effect = Exception("UNIQUE constraint")
            mock_reg.return_value = registry
            mock_pkg.return_value = MagicMock()

            info = _run_async(bridge.register_build_as_tool(result, "test"))
            assert info is None


# =============================================================================
# Phase 2: Project memory
# =============================================================================

class TestGetRecentProjects:
    def test_returns_projects(self, tmp_path):
        from mother.bridge import EngineBridge
        from mother.journal import BuildJournal, JournalEntry

        db_path = tmp_path / "test.db"
        journal = BuildJournal(db_path)
        journal.record(JournalEntry(
            event_type="build",
            description="A weather app",
            success=True,
            trust_score=80.0,
            project_path="/tmp/weather-app",
        ))
        journal.record(JournalEntry(
            event_type="build",
            description="A todo app",
            success=True,
            trust_score=70.0,
            project_path="/tmp/todo-app",
        ))

        bridge = EngineBridge(provider="grok", api_key="test")
        projects = _run_async(bridge.get_recent_projects(db_path))
        assert len(projects) == 2
        assert projects[0]["name"] == "todo-app"  # Most recent first
        assert projects[0]["description"] == "A todo app"
        assert projects[1]["name"] == "weather-app"

    def test_deduplicates(self, tmp_path):
        from mother.bridge import EngineBridge
        from mother.journal import BuildJournal, JournalEntry

        db_path = tmp_path / "test.db"
        journal = BuildJournal(db_path)
        # Same path twice
        journal.record(JournalEntry(
            event_type="build", description="v1", success=True, project_path="/tmp/app",
        ))
        journal.record(JournalEntry(
            event_type="build", description="v2", success=True, project_path="/tmp/app",
        ))

        bridge = EngineBridge(provider="grok", api_key="test")
        projects = _run_async(bridge.get_recent_projects(db_path))
        assert len(projects) == 1

    def test_filters_failed_builds(self, tmp_path):
        from mother.bridge import EngineBridge
        from mother.journal import BuildJournal, JournalEntry

        db_path = tmp_path / "test.db"
        journal = BuildJournal(db_path)
        journal.record(JournalEntry(
            event_type="build", description="failed", success=False, project_path="/tmp/fail",
        ))

        bridge = EngineBridge(provider="grok", api_key="test")
        projects = _run_async(bridge.get_recent_projects(db_path))
        assert len(projects) == 0

    def test_filters_no_path(self, tmp_path):
        from mother.bridge import EngineBridge
        from mother.journal import BuildJournal, JournalEntry

        db_path = tmp_path / "test.db"
        journal = BuildJournal(db_path)
        journal.record(JournalEntry(
            event_type="compile", description="just compiled", success=True, project_path="",
        ))

        bridge = EngineBridge(provider="grok", api_key="test")
        projects = _run_async(bridge.get_recent_projects(db_path))
        assert len(projects) == 0


# =============================================================================
# Phase 3: Tool awareness in context
# =============================================================================

class TestContextDataNewFields:
    def test_recent_projects_default(self):
        data = ContextData()
        assert data.recent_projects == []
        assert data.tool_names == []

    def test_recent_projects_renders(self):
        data = ContextData(
            tool_count=2,
            tool_verified_count=1,
            recent_projects=[
                {"name": "weather-app", "description": "A weather dashboard"},
                {"name": "todo-app", "description": "A simple todo list"},
            ],
        )
        situation = synthesize_situation(data)
        assert "Built projects:" in situation
        assert "weather-app" in situation
        assert "todo-app" in situation

    def test_tool_names_renders(self):
        data = ContextData(
            tool_count=3,
            tool_verified_count=1,
            tool_names=["weather-tool", "booking-system", "analytics"],
        )
        situation = synthesize_situation(data)
        assert "weather-tool" in situation
        assert "booking-system" in situation
        assert "analytics" in situation

    def test_tool_names_capped_at_10(self):
        names = [f"tool-{i}" for i in range(15)]
        data = ContextData(
            tool_count=15,
            tool_names=names,
        )
        situation = synthesize_situation(data)
        assert "tool-0" in situation
        assert "tool-9" in situation
        assert "tool-10" not in situation

    def test_recent_projects_capped_at_5(self):
        projects = [{"name": f"proj-{i}", "description": f"desc-{i}"} for i in range(8)]
        data = ContextData(
            recent_projects=projects,
        )
        situation = synthesize_situation(data)
        assert "proj-0" in situation
        assert "proj-4" in situation
        assert "proj-5" not in situation

    def test_no_projects_no_section(self):
        data = ContextData()
        situation = synthesize_situation(data)
        assert "Built projects:" not in situation

    def test_tool_names_empty_no_extra(self):
        data = ContextData(tool_count=2, tool_names=[])
        situation = synthesize_situation(data)
        assert "2 tools" in situation
        # Should not have a colon followed by names
        assert "2 tools." in situation

    def test_context_data_frozen(self):
        data = ContextData(tool_names=["a", "b"])
        with pytest.raises(AttributeError):
            data.tool_names = ["c"]


class TestGetToolDetails:
    def test_returns_details(self):
        from mother.bridge import EngineBridge

        mock_digest = MagicMock()
        mock_digest.name = "weather-tool"
        mock_digest.domain = "software"
        mock_digest.package_id = "pkg001"
        mock_digest.trust_score = 80.0

        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg:
            registry = MagicMock()
            registry.list_tools.return_value = [mock_digest]
            mock_reg.return_value = registry

            details = _run_async(bridge.get_tool_details())
            assert len(details) == 1
            assert details[0]["name"] == "weather-tool"
            assert details[0]["domain"] == "software"

    def test_caps_at_20(self):
        from mother.bridge import EngineBridge

        digests = []
        for i in range(25):
            d = MagicMock()
            d.name = f"tool-{i}"
            d.domain = "software"
            d.package_id = f"pkg-{i}"
            d.trust_score = 50.0
            digests.append(d)

        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg:
            registry = MagicMock()
            registry.list_tools.return_value = digests
            mock_reg.return_value = registry

            details = _run_async(bridge.get_tool_details())
            assert len(details) == 20

    def test_error_returns_empty(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg:
            mock_reg.side_effect = Exception("db error")
            details = _run_async(bridge.get_tool_details())
            assert details == []


# =============================================================================
# Phase 4: Tool invocation via bridge
# =============================================================================

class TestBridgeRunTool:
    def test_success(self, tmp_path):
        from mother.bridge import EngineBridge

        # Create a tool project
        project = tmp_path / "hello-world"
        project.mkdir()
        (project / "main.py").write_text("print('hello from tool')")

        mock_digest = MagicMock()
        mock_digest.name = "hello-world"
        mock_digest.package_id = "pkg001"

        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("mother.tool_runner.find_tool_project") as mock_find:

            registry = MagicMock()
            registry.search_tools.return_value = [mock_digest]
            mock_reg.return_value = registry
            mock_find.return_value = str(project)

            result = _run_async(bridge.run_tool("hello-world"))
            assert result["success"]
            assert "hello from tool" in result["output"]
            registry.record_usage.assert_called_once_with("pkg001", "success")
            registry.increment_usage_count.assert_called_once_with("pkg001")

    def test_not_found(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg:
            registry = MagicMock()
            registry.search_tools.return_value = []
            mock_reg.return_value = registry

            result = _run_async(bridge.run_tool("nonexistent"))
            assert not result["success"]
            assert "No tool found" in result["error"]

    def test_project_dir_not_found(self):
        from mother.bridge import EngineBridge

        mock_digest = MagicMock()
        mock_digest.name = "ghost-tool"
        mock_digest.package_id = "pkg002"

        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("mother.tool_runner.find_tool_project") as mock_find:

            registry = MagicMock()
            registry.search_tools.return_value = [mock_digest]
            mock_reg.return_value = registry
            mock_find.return_value = None

            result = _run_async(bridge.run_tool("ghost-tool"))
            assert not result["success"]
            assert "directory not found" in result["error"]

    def test_records_usage(self, tmp_path):
        from mother.bridge import EngineBridge

        project = tmp_path / "test-tool"
        project.mkdir()
        (project / "main.py").write_text("print('ok')")

        mock_digest = MagicMock()
        mock_digest.name = "test-tool"
        mock_digest.package_id = "pkg003"

        bridge = EngineBridge(provider="grok", api_key="test")

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("mother.tool_runner.find_tool_project") as mock_find:

            registry = MagicMock()
            registry.search_tools.return_value = [mock_digest]
            mock_reg.return_value = registry
            mock_find.return_value = str(project)

            _run_async(bridge.run_tool("test-tool"))
            registry.record_usage.assert_called_once()
            registry.increment_usage_count.assert_called_once()


# =============================================================================
# Persona additions
# =============================================================================

class TestPersonaIntentRouting:
    def test_use_tool_in_intent_routing(self):
        from mother.persona import INTENT_ROUTING
        assert "use_tool" in INTENT_ROUTING
        assert "weather-tool" in INTENT_ROUTING
        assert "booking-system" in INTENT_ROUTING

    def test_action_regex_matches_use_tool(self):
        import re
        _RE_ACTION = re.compile(r"\[ACTION:(\w+)\](.*?)\[/ACTION\]", re.DOTALL)
        text = "[ACTION:use_tool]weather-tool: run[/ACTION][VOICE]Running it now.[/VOICE]"
        match = _RE_ACTION.search(text)
        assert match
        assert match.group(1) == "use_tool"
        assert match.group(2) == "weather-tool: run"

    def test_use_tool_parse_colon_format(self):
        """The action arg for use_tool uses 'name: input' format."""
        arg = "booking-system: check availability"
        parts = arg.split(":", 1)
        assert parts[0].strip() == "booking-system"
        assert parts[1].strip() == "check availability"


# =============================================================================
# Integration: full context with all new fields
# =============================================================================

class TestFullContextIntegration:
    def test_context_with_tools_and_projects(self):
        data = ContextData(
            name="Mother",
            provider="grok",
            model="grok-3",
            tool_count=3,
            tool_verified_count=2,
            tool_names=["weather-tool", "booking-system", "analytics"],
            recent_projects=[
                {"name": "weather-app", "description": "A weather dashboard"},
                {"name": "todo-app", "description": "A simple todo list"},
            ],
        )
        ctx = synthesize_context(data)
        assert "weather-tool" in ctx
        assert "booking-system" in ctx
        assert "Built projects:" in ctx
        assert "weather-app" in ctx
        assert "You are Mother" in ctx

    def test_context_empty_new_fields(self):
        data = ContextData()
        ctx = synthesize_context(data)
        assert "Built projects:" not in ctx
        # Still valid context
        assert "You are Mother" in ctx
