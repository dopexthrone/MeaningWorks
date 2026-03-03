"""Tests for core/tool_package.py — ToolPackage data structures and serialization."""

import json
import time
import pytest

from core.tool_package import (
    ProvenanceRecord,
    ToolPackage,
    ToolDigest,
    compute_package_id,
    package_tool,
    extract_digest,
    serialize_tool_package,
    deserialize_tool_package,
    serialize_digest,
    deserialize_digest,
    _serialize_provenance_record,
    _deserialize_provenance_record,
)


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
        "completeness": 80,
        "consistency": 75,
        "coherence": 85,
        "traceability": 70,
        "actionability": 60,
        "specificity": 65,
        "codegen_readiness": 70,
    }


@pytest.fixture
def sample_package(sample_blueprint, sample_code, sample_fidelity):
    return package_tool(
        compilation_id="abc123",
        blueprint=sample_blueprint,
        generated_code=sample_code,
        trust_score=78.5,
        verification_badge="verified",
        fidelity_scores=sample_fidelity,
        fingerprint_hash="deadbeef12345678",
        instance_id="inst001",
        domain="software",
        provider="claude",
        input_hash="a1b2c3d4e5f6g7h8",
    )


# =============================================================================
# PROVENANCE RECORD
# =============================================================================

class TestProvenanceRecord:
    def test_frozen(self):
        rec = ProvenanceRecord(
            compilation_id="c1", instance_id="i1",
            timestamp="2026-01-01T00:00:00Z", input_hash="h1",
            domain="software", provider="claude",
            trust_score=80.0, verification_badge="verified",
        )
        with pytest.raises(AttributeError):
            rec.trust_score = 90.0

    def test_serialize_deserialize(self):
        rec = ProvenanceRecord(
            compilation_id="c1", instance_id="i1",
            timestamp="2026-01-01T00:00:00Z", input_hash="h1",
            domain="software", provider="claude",
            trust_score=80.0, verification_badge="verified",
        )
        data = _serialize_provenance_record(rec)
        restored = _deserialize_provenance_record(data)
        assert restored.compilation_id == rec.compilation_id
        assert restored.trust_score == rec.trust_score


# =============================================================================
# PACKAGE ID
# =============================================================================

class TestComputePackageId:
    def test_deterministic(self, sample_blueprint, sample_code):
        id1 = compute_package_id(sample_blueprint, sample_code)
        id2 = compute_package_id(sample_blueprint, sample_code)
        assert id1 == id2

    def test_length(self, sample_blueprint, sample_code):
        pid = compute_package_id(sample_blueprint, sample_code)
        assert len(pid) == 16
        assert all(c in "0123456789abcdef" for c in pid)

    def test_different_code_different_id(self, sample_blueprint, sample_code):
        id1 = compute_package_id(sample_blueprint, sample_code)
        modified_code = {**sample_code, "TaskStore": "class TaskStore:\n    x = 1\n"}
        id2 = compute_package_id(sample_blueprint, modified_code)
        assert id1 != id2

    def test_different_blueprint_different_id(self, sample_code):
        bp1 = {"components": [{"name": "A", "type": "s"}], "relationships": []}
        bp2 = {"components": [{"name": "B", "type": "s"}], "relationships": []}
        assert compute_package_id(bp1, sample_code) != compute_package_id(bp2, sample_code)

    def test_empty_inputs(self):
        pid = compute_package_id({}, {})
        assert len(pid) == 16


# =============================================================================
# PACKAGE TOOL
# =============================================================================

class TestPackageTool:
    def test_basic(self, sample_package):
        assert sample_package.name == "A task manager"
        assert sample_package.domain == "software"
        assert sample_package.trust_score == 78.5
        assert sample_package.verification_badge == "verified"
        assert len(sample_package.provenance_chain) == 1
        assert sample_package.provenance_chain[0].compilation_id == "abc123"
        assert sample_package.provenance_chain[0].instance_id == "inst001"
        assert sample_package.usage_count == 0

    def test_auto_name_from_blueprint(self, sample_blueprint, sample_code, sample_fidelity):
        pkg = package_tool(
            compilation_id="x", blueprint=sample_blueprint,
            generated_code=sample_code, trust_score=50.0,
            verification_badge="partial", fidelity_scores=sample_fidelity,
            fingerprint_hash="abc", instance_id="i1",
        )
        assert pkg.name == "A task manager"

    def test_explicit_name(self, sample_blueprint, sample_code, sample_fidelity):
        pkg = package_tool(
            compilation_id="x", blueprint=sample_blueprint,
            generated_code=sample_code, trust_score=50.0,
            verification_badge="partial", fidelity_scores=sample_fidelity,
            fingerprint_hash="abc", instance_id="i1",
            name="My Custom Tool",
        )
        assert pkg.name == "My Custom Tool"

    def test_long_name_truncated(self, sample_code, sample_fidelity):
        long_name_bp = {"core_need": "x" * 100, "components": [], "relationships": []}
        pkg = package_tool(
            compilation_id="x", blueprint=long_name_bp,
            generated_code=sample_code, trust_score=50.0,
            verification_badge="partial", fidelity_scores=sample_fidelity,
            fingerprint_hash="abc", instance_id="i1",
        )
        assert len(pkg.name) == 80
        assert pkg.name.endswith("...")

    def test_package_id_computed(self, sample_package, sample_blueprint, sample_code):
        expected = compute_package_id(sample_blueprint, sample_code)
        assert sample_package.package_id == expected

    def test_frozen(self, sample_package):
        with pytest.raises(AttributeError):
            sample_package.trust_score = 100.0

    def test_version(self, sample_blueprint, sample_code, sample_fidelity):
        pkg = package_tool(
            compilation_id="x", blueprint=sample_blueprint,
            generated_code=sample_code, trust_score=50.0,
            verification_badge="partial", fidelity_scores=sample_fidelity,
            fingerprint_hash="abc", instance_id="i1",
            version="2.0.1",
        )
        assert pkg.version == "2.0.1"


# =============================================================================
# DIGEST
# =============================================================================

class TestExtractDigest:
    def test_basic(self, sample_package):
        digest = extract_digest(sample_package)
        assert digest.package_id == sample_package.package_id
        assert digest.name == sample_package.name
        assert digest.domain == sample_package.domain
        assert digest.trust_score == sample_package.trust_score
        assert digest.component_count == 2
        assert digest.relationship_count == 1
        assert digest.source_instance_id == "inst001"

    def test_digest_is_frozen(self, sample_package):
        digest = extract_digest(sample_package)
        with pytest.raises(AttributeError):
            digest.trust_score = 100.0


# =============================================================================
# SERIALIZATION — TOOL PACKAGE
# =============================================================================

class TestSerializeToolPackage:
    def test_roundtrip(self, sample_package):
        data = serialize_tool_package(sample_package)
        restored = deserialize_tool_package(data)
        assert restored.package_id == sample_package.package_id
        assert restored.name == sample_package.name
        assert restored.trust_score == sample_package.trust_score
        assert restored.verification_badge == sample_package.verification_badge
        assert restored.generated_code == sample_package.generated_code
        assert restored.blueprint == sample_package.blueprint
        assert len(restored.provenance_chain) == len(sample_package.provenance_chain)

    def test_json_safe(self, sample_package):
        data = serialize_tool_package(sample_package)
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_format_marker(self, sample_package):
        data = serialize_tool_package(sample_package)
        assert data["format"] == "mtool"
        assert data["format_version"] == "1.0"

    def test_provenance_chain_serialized(self, sample_package):
        data = serialize_tool_package(sample_package)
        chain = data["provenance_chain"]
        assert isinstance(chain, list)
        assert len(chain) == 1
        assert chain[0]["compilation_id"] == "abc123"

    def test_deserialize_with_defaults(self):
        minimal = {
            "package_id": "abc",
            "name": "test",
            "created_at": "2026-01-01T00:00:00Z",
            "blueprint": {"components": []},
            "generated_code": {},
        }
        pkg = deserialize_tool_package(minimal)
        assert pkg.domain == "software"
        assert pkg.file_extension == ".py"
        assert pkg.version == "1.0.0"
        assert pkg.usage_count == 0


# =============================================================================
# SERIALIZATION — DIGEST
# =============================================================================

class TestSerializeDigest:
    def test_roundtrip(self, sample_package):
        digest = extract_digest(sample_package)
        data = serialize_digest(digest)
        restored = deserialize_digest(data)
        assert restored.package_id == digest.package_id
        assert restored.name == digest.name
        assert restored.component_count == digest.component_count

    def test_json_safe(self, sample_package):
        digest = extract_digest(sample_package)
        data = serialize_digest(digest)
        json_str = json.dumps(data)
        assert len(json_str) > 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_empty_blueprint_name(self, sample_code, sample_fidelity):
        pkg = package_tool(
            compilation_id="x",
            blueprint={"components": [], "relationships": []},
            generated_code=sample_code,
            trust_score=0.0,
            verification_badge="unverified",
            fidelity_scores=sample_fidelity,
            fingerprint_hash="abc",
            instance_id="i1",
        )
        assert pkg.name == "unnamed-tool"

    def test_empty_code(self, sample_blueprint, sample_fidelity):
        pkg = package_tool(
            compilation_id="x",
            blueprint=sample_blueprint,
            generated_code={},
            trust_score=50.0,
            verification_badge="partial",
            fidelity_scores=sample_fidelity,
            fingerprint_hash="abc",
            instance_id="i1",
        )
        assert pkg.generated_code == {}
        assert len(pkg.package_id) == 16
