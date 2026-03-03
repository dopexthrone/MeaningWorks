"""Tests for motherlabs_platform/tool_registry.py — SQLite-backed tool registry."""

import json
import os
import tempfile
import pytest

from core.tool_package import (
    ToolPackage,
    ToolDigest,
    package_tool,
    extract_digest,
)
from motherlabs_platform.tool_registry import ToolRegistry


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_tools.db")


@pytest.fixture
def registry(tmp_db):
    return ToolRegistry(db_path=tmp_db)


@pytest.fixture
def sample_blueprint():
    return {
        "core_need": "A task manager",
        "components": [
            {"name": "TaskStore", "type": "service", "description": "Manages tasks"},
            {"name": "UserAuth", "type": "service", "description": "Handles auth"},
        ],
        "relationships": [
            {"from": "TaskStore", "to": "UserAuth", "type": "depends_on"},
        ],
    }


@pytest.fixture
def sample_fidelity():
    return {
        "completeness": 80, "consistency": 75, "coherence": 85,
        "traceability": 70, "actionability": 60, "specificity": 65,
        "codegen_readiness": 70,
    }


@pytest.fixture
def sample_package(sample_blueprint, sample_fidelity):
    return package_tool(
        compilation_id="comp001",
        blueprint=sample_blueprint,
        generated_code={"TaskStore": "class TaskStore: pass", "UserAuth": "class UserAuth: pass"},
        trust_score=78.5,
        verification_badge="verified",
        fidelity_scores=sample_fidelity,
        fingerprint_hash="fp_abc123456789",
        instance_id="inst001",
        domain="software",
        provider="claude",
        input_hash="hash001",
        name="Task Manager",
    )


def _make_package(name, domain="software", fingerprint="fp_default", trust=80.0, badge="verified"):
    bp = {
        "core_need": name,
        "components": [{"name": f"{name}Core", "type": "service", "description": name}],
        "relationships": [],
    }
    return package_tool(
        compilation_id=f"comp_{name}",
        blueprint=bp,
        generated_code={f"{name}Core": f"class {name}Core: pass"},
        trust_score=trust,
        verification_badge=badge,
        fidelity_scores={"completeness": 80, "consistency": 75, "coherence": 85, "traceability": 70},
        fingerprint_hash=fingerprint,
        instance_id="inst001",
        domain=domain,
        name=name,
    )


# =============================================================================
# REGISTER + GET
# =============================================================================

class TestRegisterAndGet:
    def test_register_and_get(self, registry, sample_package):
        registry.register_tool(sample_package)
        retrieved = registry.get_tool(sample_package.package_id)
        assert retrieved is not None
        assert retrieved.package_id == sample_package.package_id
        assert retrieved.name == "Task Manager"
        assert retrieved.trust_score == 78.5
        assert retrieved.generated_code == sample_package.generated_code

    def test_get_nonexistent(self, registry):
        result = registry.get_tool("nonexistent_id")
        assert result is None

    def test_register_duplicate_raises(self, registry, sample_package):
        registry.register_tool(sample_package)
        with pytest.raises(Exception):
            registry.register_tool(sample_package)

    def test_register_local_vs_imported(self, registry, sample_package):
        registry.register_tool(sample_package, is_local=True)
        tools = registry.list_tools(local_only=True)
        assert len(tools) == 1

    def test_register_imported_has_imported_at(self, registry, tmp_db):
        pkg = _make_package("imported_tool", fingerprint="fp_imported")
        registry.register_tool(pkg, is_local=False)
        tools = registry.list_tools(local_only=False)
        assert len(tools) == 1
        # Imported tools should NOT appear in local_only list
        local_tools = registry.list_tools(local_only=True)
        assert len(local_tools) == 0


# =============================================================================
# LIST
# =============================================================================

class TestListTools:
    def test_list_empty(self, registry):
        tools = registry.list_tools()
        assert tools == []

    def test_list_all(self, registry):
        registry.register_tool(_make_package("Tool1", fingerprint="fp1"))
        registry.register_tool(_make_package("Tool2", fingerprint="fp2"))
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_filter_by_domain(self, registry):
        registry.register_tool(_make_package("SwTool", domain="software", fingerprint="fp1"))
        registry.register_tool(_make_package("ProcTool", domain="process", fingerprint="fp2"))
        sw_tools = registry.list_tools(domain="software")
        assert len(sw_tools) == 1
        assert sw_tools[0].name == "SwTool"

    def test_list_local_only(self, registry):
        registry.register_tool(_make_package("Local", fingerprint="fp1"), is_local=True)
        registry.register_tool(_make_package("Remote", fingerprint="fp2"), is_local=False)
        local = registry.list_tools(local_only=True)
        assert len(local) == 1
        assert local[0].name == "Local"

    def test_list_returns_digests(self, registry, sample_package):
        registry.register_tool(sample_package)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], ToolDigest)
        assert tools[0].package_id == sample_package.package_id


# =============================================================================
# SEARCH
# =============================================================================

class TestSearchTools:
    def test_search_by_name(self, registry):
        registry.register_tool(_make_package("TaskManager", fingerprint="fp1"))
        registry.register_tool(_make_package("UserAuth", fingerprint="fp2"))
        results = registry.search_tools("Task")
        assert len(results) == 1
        assert results[0].name == "TaskManager"

    def test_search_by_domain(self, registry):
        registry.register_tool(_make_package("Tool1", domain="software", fingerprint="fp1"))
        registry.register_tool(_make_package("Tool2", domain="process", fingerprint="fp2"))
        results = registry.search_tools("process")
        assert len(results) == 1

    def test_search_case_insensitive(self, registry):
        registry.register_tool(_make_package("MyTool", fingerprint="fp1"))
        results = registry.search_tools("mytool")
        assert len(results) == 1

    def test_search_no_results(self, registry):
        registry.register_tool(_make_package("Tool1", fingerprint="fp1"))
        results = registry.search_tools("nonexistent")
        assert len(results) == 0

    def test_search_ordered_by_trust(self, registry):
        registry.register_tool(_make_package("Low", trust=50.0, fingerprint="fp1"))
        registry.register_tool(_make_package("High", trust=90.0, fingerprint="fp2"))
        # Both match empty-ish search
        results = registry.search_tools("")
        assert results[0].name == "High"


# =============================================================================
# FINGERPRINT LOOKUP
# =============================================================================

class TestFindByFingerprint:
    def test_find_existing(self, registry):
        registry.register_tool(_make_package("Tool1", fingerprint="fp_unique"))
        result = registry.find_by_fingerprint("fp_unique")
        assert result is not None
        assert result.name == "Tool1"

    def test_find_nonexistent(self, registry):
        result = registry.find_by_fingerprint("fp_nonexistent")
        assert result is None

    def test_find_returns_digest(self, registry):
        registry.register_tool(_make_package("Tool1", fingerprint="fp_unique"))
        result = registry.find_by_fingerprint("fp_unique")
        assert isinstance(result, ToolDigest)


# =============================================================================
# USAGE TRACKING
# =============================================================================

class TestUsageTracking:
    def test_record_usage(self, registry):
        pkg = _make_package("Tool1", fingerprint="fp1")
        registry.register_tool(pkg)
        registry.record_usage(pkg.package_id, "export", "inst002")
        stats = registry.get_usage_stats(pkg.package_id)
        assert stats["action_counts"]["export"] == 1

    def test_increment_usage_count(self, registry):
        pkg = _make_package("Tool1", fingerprint="fp1")
        registry.register_tool(pkg)
        registry.increment_usage_count(pkg.package_id)
        registry.increment_usage_count(pkg.package_id)
        stats = registry.get_usage_stats(pkg.package_id)
        assert stats["usage_count"] == 2

    def test_multiple_actions(self, registry):
        pkg = _make_package("Tool1", fingerprint="fp1")
        registry.register_tool(pkg)
        registry.record_usage(pkg.package_id, "export")
        registry.record_usage(pkg.package_id, "query")
        registry.record_usage(pkg.package_id, "query")
        stats = registry.get_usage_stats(pkg.package_id)
        assert stats["action_counts"]["export"] == 1
        assert stats["action_counts"]["query"] == 2
        assert stats["total_events"] == 3

    def test_usage_stats_nonexistent(self, registry):
        stats = registry.get_usage_stats("nonexistent")
        assert stats["usage_count"] == 0
        assert stats["total_events"] == 0


# =============================================================================
# REMOVE
# =============================================================================

class TestRemoveTool:
    def test_remove_existing(self, registry):
        pkg = _make_package("Tool1", fingerprint="fp1")
        registry.register_tool(pkg)
        assert registry.remove_tool(pkg.package_id) is True
        assert registry.get_tool(pkg.package_id) is None

    def test_remove_nonexistent(self, registry):
        assert registry.remove_tool("nonexistent") is False

    def test_remove_cleans_usage_logs(self, registry):
        pkg = _make_package("Tool1", fingerprint="fp1")
        registry.register_tool(pkg)
        registry.record_usage(pkg.package_id, "export")
        registry.remove_tool(pkg.package_id)
        stats = registry.get_usage_stats(pkg.package_id)
        assert stats["total_events"] == 0
