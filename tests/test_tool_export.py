"""Tests for core/tool_export.py — Export/Import functions."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from core.tool_package import (
    ToolPackage,
    compute_package_id,
    package_tool,
    serialize_tool_package,
)
from core.tool_export import (
    export_tool,
    export_tool_to_file,
    load_tool_from_file,
    import_tool,
)
from core.governor_validation import ImportValidationResult
from motherlabs_platform.tool_registry import ToolRegistry


# =============================================================================
# FIXTURES
# =============================================================================

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
        "constraints": [],
    }


@pytest.fixture
def sample_code():
    return {
        "TaskStore": "class TaskStore:\n    pass\n",
        "UserAuth": "class UserAuth:\n    pass\n",
    }


@pytest.fixture
def sample_fidelity():
    return {
        "completeness": 80, "consistency": 75, "coherence": 85,
        "traceability": 70, "actionability": 60, "specificity": 65,
        "codegen_readiness": 70,
    }


@pytest.fixture
def sample_package(sample_blueprint, sample_code, sample_fidelity):
    return package_tool(
        compilation_id="comp001",
        blueprint=sample_blueprint,
        generated_code=sample_code,
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


@pytest.fixture
def tmp_registry(tmp_path):
    return ToolRegistry(db_path=str(tmp_path / "test_tools.db"))


@pytest.fixture
def mock_corpus(sample_blueprint, sample_code):
    """Mock corpus with a single compilation."""
    @dataclass
    class MockRecord:
        id: str = "comp001"
        input_text: str = "Build a task manager"
        domain: str = "software"
        provider: str = "claude"
        model: str = "sonnet"
        success: bool = True

    corpus = MagicMock()
    corpus.get.return_value = MockRecord()
    corpus.load_blueprint.return_value = sample_blueprint
    corpus.load_context_graph.return_value = {
        "generated_code": sample_code,
        "keywords": ["task", "manager"],
        "verification": {
            "completeness": {"score": 80},
            "consistency": {"score": 75},
            "coherence": {"score": 85},
            "traceability": {"score": 70},
            "actionability": {"score": 60},
            "specificity": {"score": 65},
            "codegen_readiness": {"score": 70},
        },
        "dimensional_metadata": {},
    }
    return corpus


# =============================================================================
# EXPORT
# =============================================================================

class TestExportTool:
    def test_export_from_corpus(self, mock_corpus):
        pkg = export_tool("comp001", mock_corpus, "inst001", name="My Tool")
        assert pkg.name == "My Tool"
        assert pkg.domain == "software"
        assert pkg.compilation_id == "comp001"
        assert pkg.source_instance_id == "inst001"
        assert len(pkg.provenance_chain) == 1
        assert len(pkg.package_id) == 16

    def test_export_not_found(self, mock_corpus):
        mock_corpus.get.return_value = None
        with pytest.raises(ValueError, match="not found"):
            export_tool("bad_id", mock_corpus, "inst001")

    def test_export_no_blueprint(self, mock_corpus):
        mock_corpus.load_blueprint.return_value = None
        with pytest.raises(ValueError, match="Blueprint not found"):
            export_tool("comp001", mock_corpus, "inst001")

    def test_export_default_name(self, mock_corpus):
        pkg = export_tool("comp001", mock_corpus, "inst001")
        assert pkg.name == "A task manager"

    def test_export_trust_computed(self, mock_corpus):
        pkg = export_tool("comp001", mock_corpus, "inst001")
        assert pkg.trust_score > 0
        assert pkg.verification_badge in ("verified", "partial", "unverified")


# =============================================================================
# FILE I/O
# =============================================================================

class TestExportToFile:
    def test_write_and_read(self, sample_package, tmp_path):
        output = str(tmp_path / "test.mtool")
        path = export_tool_to_file(sample_package, output)
        assert os.path.exists(path)

        loaded = load_tool_from_file(path)
        assert loaded.package_id == sample_package.package_id
        assert loaded.name == sample_package.name
        assert loaded.generated_code == sample_package.generated_code

    def test_default_filename(self, sample_package, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = export_tool_to_file(sample_package)
        assert path.endswith(".mtool")
        assert os.path.exists(path)

    def test_file_is_valid_json(self, sample_package, tmp_path):
        output = str(tmp_path / "test.mtool")
        export_tool_to_file(sample_package, output)

        with open(output) as f:
            data = json.load(f)
        assert data["format"] == "mtool"
        assert data["format_version"] == "1.0"

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_tool_from_file("/nonexistent/path.mtool")

    def test_load_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.mtool"
        bad_file.write_text("not json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_tool_from_file(str(bad_file))

    def test_load_wrong_format(self, tmp_path):
        wrong_file = tmp_path / "wrong.mtool"
        wrong_file.write_text(json.dumps({"format": "other"}))
        with pytest.raises(ValueError, match="Not a .mtool file"):
            load_tool_from_file(str(wrong_file))

    def test_load_malformed_mtool(self, tmp_path):
        bad_file = tmp_path / "bad.mtool"
        bad_file.write_text(json.dumps({"format": "mtool"}))
        with pytest.raises(ValueError, match="Malformed"):
            load_tool_from_file(str(bad_file))


# =============================================================================
# IMPORT
# =============================================================================

class TestImportTool:
    def test_successful_import(self, sample_package, tmp_registry):
        result = import_tool(sample_package, tmp_registry, "inst002")
        assert result.allowed is True
        # Verify it's in the registry
        retrieved = tmp_registry.get_tool(sample_package.package_id)
        assert retrieved is not None

    def test_duplicate_fingerprint_rejected(self, sample_package, tmp_registry):
        # First import succeeds
        result1 = import_tool(sample_package, tmp_registry, "inst002")
        assert result1.allowed is True

        # Second import with same fingerprint fails
        result2 = import_tool(sample_package, tmp_registry, "inst002")
        assert result2.allowed is False
        assert "duplicate" in result2.rejection_reason.lower()

    def test_low_trust_rejected(self, sample_fidelity, tmp_registry):
        bp = {"components": [{"name": "X", "type": "s"}], "relationships": []}
        pkg = package_tool(
            compilation_id="c1", blueprint=bp,
            generated_code={"X": "class X: pass"},
            trust_score=30.0,
            verification_badge="unverified",
            fidelity_scores={"completeness": 20, "consistency": 20, "coherence": 20, "traceability": 20},
            fingerprint_hash="fp_low", instance_id="i1",
        )
        result = import_tool(pkg, tmp_registry, "inst002")
        assert result.allowed is False
        assert result.trust_sufficient is False

    def test_require_verified_rejects_partial(self, sample_fidelity, tmp_registry):
        bp = {"components": [{"name": "X", "type": "s"}], "relationships": []}
        code = {"X": "class X: pass"}
        pid = compute_package_id(bp, code)
        pkg = package_tool(
            compilation_id="c1", blueprint=bp,
            generated_code=code,
            trust_score=80.0,
            verification_badge="partial",
            fidelity_scores={"completeness": 80, "consistency": 75, "coherence": 85, "traceability": 70},
            fingerprint_hash="fp_partial", instance_id="inst_partial",
        )
        result = import_tool(pkg, tmp_registry, "inst002", require_verified=True)
        assert result.allowed is False
        assert "verified" in result.rejection_reason.lower()

    def test_usage_recorded_on_import(self, sample_package, tmp_registry):
        import_tool(sample_package, tmp_registry, "inst002")
        stats = tmp_registry.get_usage_stats(sample_package.package_id)
        assert stats["action_counts"].get("import", 0) == 1

    def test_custom_min_trust(self, sample_package, tmp_registry):
        result = import_tool(sample_package, tmp_registry, "inst002", min_trust_score=90.0)
        assert result.allowed is False
