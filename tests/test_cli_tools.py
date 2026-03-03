"""Tests for CLI tools/instance commands and E11xxx error codes."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from core.error_catalog import CATALOG, get_entry
from core.tool_package import package_tool, serialize_tool_package
from core.tool_export import export_tool_to_file, load_tool_from_file


# =============================================================================
# ERROR CATALOG TESTS
# =============================================================================

class TestE11ErrorCodes:
    def test_e11001_exists(self):
        entry = get_entry("E11001")
        assert entry is not None
        assert "provenance" in entry.title.lower()

    def test_e11002_exists(self):
        entry = get_entry("E11002")
        assert entry is not None
        assert "trust" in entry.title.lower()

    def test_e11003_exists(self):
        entry = get_entry("E11003")
        assert entry is not None
        assert "code" in entry.title.lower() or "dangerous" in entry.title.lower()

    def test_e11004_exists(self):
        entry = get_entry("E11004")
        assert entry is not None
        assert "blueprint" in entry.title.lower()

    def test_e11005_exists(self):
        entry = get_entry("E11005")
        assert entry is not None
        assert "not found" in entry.title.lower()

    def test_e11006_exists(self):
        entry = get_entry("E11006")
        assert entry is not None
        assert "not verified" in entry.title.lower()

    def test_e11007_exists(self):
        entry = get_entry("E11007")
        assert entry is not None
        assert "deserialization" in entry.title.lower()

    def test_e11008_exists(self):
        entry = get_entry("E11008")
        assert entry is not None
        assert "duplicate" in entry.title.lower() or "fingerprint" in entry.title.lower()

    def test_all_e11_have_fix_examples(self):
        for code in ("E11001", "E11002", "E11003", "E11004",
                      "E11005", "E11006", "E11007", "E11008"):
            entry = get_entry(code)
            assert entry is not None, f"{code} missing"
            assert len(entry.fix_examples) > 0, f"{code} has no fix examples"


# =============================================================================
# CLI TOOLS COMMAND — INTEGRATION-STYLE TESTS
# =============================================================================

def _make_test_package(name="CliTool"):
    bp = {
        "core_need": name,
        "components": [{"name": f"{name}Core", "type": "service", "description": name}],
        "relationships": [],
    }
    return package_tool(
        compilation_id=f"comp_{name}",
        blueprint=bp,
        generated_code={f"{name}Core": f"class {name}Core: pass"},
        trust_score=80.0,
        verification_badge="verified",
        fidelity_scores={"completeness": 80, "consistency": 75, "coherence": 85, "traceability": 70},
        fingerprint_hash=f"fp_{name.lower()}",
        instance_id="inst_cli_test",
        domain="software",
        name=name,
    )


class TestToolFileRoundtrip:
    """Test the file-based export/import workflow that CLI uses."""

    def test_export_import_roundtrip(self, tmp_path):
        from motherlabs_platform.tool_registry import ToolRegistry
        from core.tool_export import import_tool

        pkg = _make_test_package("RoundtripTool")
        file_path = str(tmp_path / "roundtrip.mtool")

        # Export
        path = export_tool_to_file(pkg, file_path)
        assert os.path.exists(path)

        # Load
        loaded = load_tool_from_file(path)
        assert loaded.name == "RoundtripTool"
        assert loaded.trust_score == 80.0

        # Import
        registry = ToolRegistry(db_path=str(tmp_path / "tools.db"))
        result = import_tool(loaded, registry, "inst_target")
        assert result.allowed is True

        # Verify in registry
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "RoundtripTool"

    def test_mtool_is_human_readable(self, tmp_path):
        pkg = _make_test_package("ReadableTool")
        path = export_tool_to_file(pkg, str(tmp_path / "readable.mtool"))

        with open(path) as f:
            data = json.load(f)

        # Human-inspectable provenance
        assert "provenance_chain" in data
        assert len(data["provenance_chain"]) > 0
        chain_entry = data["provenance_chain"][0]
        assert "compilation_id" in chain_entry
        assert "instance_id" in chain_entry
        assert "timestamp" in chain_entry

    def test_governor_rejects_dangerous_import(self, tmp_path):
        from motherlabs_platform.tool_registry import ToolRegistry
        from core.tool_export import import_tool

        bp = {
            "core_need": "evil",
            "components": [{"name": "Evil", "type": "service", "description": "bad"}],
            "relationships": [],
        }
        pkg = package_tool(
            compilation_id="evil1",
            blueprint=bp,
            generated_code={"Evil": "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"},
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores={"completeness": 80, "consistency": 75, "coherence": 85, "traceability": 70},
            fingerprint_hash="fp_evil",
            instance_id="inst_evil",
        )

        registry = ToolRegistry(db_path=str(tmp_path / "tools.db"))
        result = import_tool(pkg, registry, "inst_target")
        assert result.allowed is False
        assert result.code_safe is False
