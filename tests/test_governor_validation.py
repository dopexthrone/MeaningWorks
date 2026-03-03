"""Tests for core/governor_validation.py — Governor import validation."""

import json
import pytest

from core.governor_validation import (
    ImportValidationResult,
    check_provenance_integrity,
    check_trust_thresholds,
    check_code_safety,
    check_blueprint_integrity,
    validate_import,
    _recompute_package_id,
)
from core.tool_package import compute_package_id


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_provenance():
    return [
        {
            "compilation_id": "abc123",
            "instance_id": "inst001",
            "timestamp": "2026-01-01T00:00:00Z",
            "input_hash": "a1b2c3d4",
            "domain": "software",
            "provider": "claude",
            "trust_score": 80.0,
            "verification_badge": "verified",
        }
    ]


@pytest.fixture
def valid_blueprint():
    return {
        "components": [
            {"name": "TaskStore", "type": "service", "description": "Manages tasks"},
            {"name": "UserAuth", "type": "service", "description": "Handles auth"},
        ],
        "relationships": [
            {"from": "TaskStore", "to": "UserAuth", "type": "depends_on"},
        ],
    }


@pytest.fixture
def valid_code():
    return {
        "TaskStore": "class TaskStore:\n    pass\n",
        "UserAuth": "class UserAuth:\n    pass\n",
    }


@pytest.fixture
def valid_fidelity():
    return {
        "completeness": 80,
        "consistency": 75,
        "coherence": 85,
        "traceability": 70,
    }


# =============================================================================
# PROVENANCE INTEGRITY
# =============================================================================

class TestCheckProvenanceIntegrity:
    def test_valid(self, valid_provenance):
        ok, reason = check_provenance_integrity(valid_provenance)
        assert ok is True
        assert reason == ""

    def test_empty_chain(self):
        ok, reason = check_provenance_integrity([])
        assert ok is False
        assert "empty" in reason.lower()

    def test_missing_compilation_id(self):
        chain = [{"instance_id": "i1", "timestamp": "2026-01-01T00:00:00Z", "domain": "s"}]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "compilation_id" in reason

    def test_missing_instance_id(self):
        chain = [{"compilation_id": "c1", "timestamp": "2026-01-01T00:00:00Z", "domain": "s"}]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "instance_id" in reason

    def test_missing_timestamp(self):
        chain = [{"compilation_id": "c1", "instance_id": "i1", "domain": "s"}]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "timestamp" in reason

    def test_invalid_timestamp_format(self):
        chain = [{
            "compilation_id": "c1", "instance_id": "i1",
            "timestamp": "not-a-timestamp", "domain": "s",
        }]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "timestamp" in reason.lower()

    def test_non_monotonic_timestamps(self):
        chain = [
            {"compilation_id": "c1", "instance_id": "inst001",
             "timestamp": "2026-01-02T00:00:00Z", "domain": "s"},
            {"compilation_id": "c2", "instance_id": "inst002",
             "timestamp": "2026-01-01T00:00:00Z", "domain": "s"},
        ]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "monotonic" in reason.lower()

    def test_monotonic_same_timestamp(self):
        chain = [
            {"compilation_id": "c1", "instance_id": "inst001",
             "timestamp": "2026-01-01T00:00:00Z", "domain": "s"},
            {"compilation_id": "c2", "instance_id": "inst002",
             "timestamp": "2026-01-01T00:00:00Z", "domain": "s"},
        ]
        ok, reason = check_provenance_integrity(chain)
        assert ok is True

    def test_short_instance_id(self):
        chain = [{
            "compilation_id": "c1", "instance_id": "ab",
            "timestamp": "2026-01-01T00:00:00Z", "domain": "s",
        }]
        ok, reason = check_provenance_integrity(chain)
        assert ok is False
        assert "instance_id" in reason

    def test_not_dict_record(self):
        ok, reason = check_provenance_integrity(["not a dict"])
        assert ok is False
        assert "not a dict" in reason.lower()

    def test_multi_record_chain(self):
        chain = [
            {"compilation_id": "c1", "instance_id": "inst1",
             "timestamp": "2026-01-01T00:00:00Z", "domain": "s"},
            {"compilation_id": "c2", "instance_id": "inst2",
             "timestamp": "2026-01-02T00:00:00Z", "domain": "s"},
        ]
        ok, reason = check_provenance_integrity(chain)
        assert ok is True


# =============================================================================
# TRUST THRESHOLDS
# =============================================================================

class TestCheckTrustThresholds:
    def test_valid(self, valid_fidelity):
        ok, reason = check_trust_thresholds(80.0, "verified", valid_fidelity)
        assert ok is True
        assert reason == ""

    def test_score_too_low(self, valid_fidelity):
        ok, reason = check_trust_thresholds(50.0, "verified", valid_fidelity)
        assert ok is False
        assert "below minimum" in reason.lower()

    def test_unverified_badge(self, valid_fidelity):
        ok, reason = check_trust_thresholds(80.0, "unverified", valid_fidelity)
        assert ok is False
        assert "unverified" in reason.lower()

    def test_partial_badge_ok(self, valid_fidelity):
        ok, reason = check_trust_thresholds(80.0, "partial", valid_fidelity)
        assert ok is True

    def test_core_dimension_too_low(self):
        fidelity = {
            "completeness": 30,  # Below 40 threshold
            "consistency": 75,
            "coherence": 85,
            "traceability": 70,
        }
        ok, reason = check_trust_thresholds(80.0, "verified", fidelity)
        assert ok is False
        assert "completeness" in reason

    def test_custom_min_score(self, valid_fidelity):
        ok, _ = check_trust_thresholds(80.0, "verified", valid_fidelity, min_trust_score=90.0)
        assert ok is False

    def test_custom_min_dimension(self):
        fidelity = {"completeness": 50, "consistency": 50, "coherence": 50, "traceability": 50}
        ok, _ = check_trust_thresholds(80.0, "verified", fidelity, min_core_dimension=60)
        assert ok is False

    def test_missing_dimension_defaults_zero(self):
        ok, reason = check_trust_thresholds(80.0, "verified", {})
        assert ok is False
        assert "completeness" in reason  # First dimension checked

    def test_exact_threshold(self, valid_fidelity):
        ok, _ = check_trust_thresholds(60.0, "verified", valid_fidelity, min_trust_score=60.0)
        assert ok is True


# =============================================================================
# CODE SAFETY
# =============================================================================

class TestCheckCodeSafety:
    def test_safe_code(self):
        code = {"main": "class Foo:\n    def bar(self):\n        return 42\n"}
        safe, warnings = check_code_safety(code)
        assert safe is True

    def test_exec_detected(self):
        code = {"main": "exec('print(1)')"}
        safe, warnings = check_code_safety(code)
        assert safe is False
        assert any("exec" in w for w in warnings)

    def test_eval_detected(self):
        code = {"main": "result = eval('2+2')"}
        safe, warnings = check_code_safety(code)
        assert safe is False
        assert any("eval" in w for w in warnings)

    def test_import_detected(self):
        code = {"main": "__import__('os')"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_os_system_detected(self):
        code = {"main": "import os\nos.system('rm -rf /')"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_subprocess_detected(self):
        code = {"main": "import subprocess\nsubprocess.run(['ls'])"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_network_patterns_allowed(self):
        """Network patterns (requests, httpx, urllib, socket) are allowed — not exfiltration."""
        for snippet in [
            "import requests\nrequests.get('https://api.example.com')",
            "import httpx\nhttpx.get('https://api.example.com')",
            "import urllib.request\nurllib.request.urlopen('https://example.com')",
            "import socket\ns = socket.socket()",
        ]:
            safe, warnings = check_code_safety({"main": snippet})
            assert safe is True, f"Network code should be allowed: {snippet}"

    def test_size_limit(self):
        code = {"main": "x = 1\n" * 100_000}
        safe, warnings = check_code_safety(code, max_size_bytes=100)
        assert safe is False
        assert any("size" in w.lower() for w in warnings)

    def test_empty_code(self):
        safe, warnings = check_code_safety({})
        assert safe is True
        assert warnings == []

    def test_non_python_returns_safe_with_warning(self):
        code = {"config": "key: value\nother: data"}
        safe, warnings = check_code_safety(code, file_extension=".yaml")
        assert safe is True
        assert any("limited" in w.lower() for w in warnings)

    def test_syntax_error_warning(self):
        code = {"main": "def foo(\n  # missing closing paren"}
        safe, warnings = check_code_safety(code)
        # Syntax errors are warnings, not rejections
        assert safe is True
        assert any("syntax" in w.lower() for w in warnings)

    def test_multiple_dangerous_patterns(self):
        code = {
            "evil1": "exec('bad')",
            "evil2": "eval('worse')",
        }
        safe, warnings = check_code_safety(code)
        assert safe is False
        assert len(warnings) >= 2

    def test_setattr_detected(self):
        code = {"main": "setattr(obj, 'x', 42)"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_shutil_rmtree_detected(self):
        code = {"main": "import shutil\nshutil.rmtree('/tmp/foo')"}
        safe, warnings = check_code_safety(code)
        assert safe is False


# =============================================================================
# BLUEPRINT INTEGRITY
# =============================================================================

class TestCheckBlueprintIntegrity:
    def test_valid(self, valid_blueprint):
        ok, reason = check_blueprint_integrity(valid_blueprint)
        assert ok is True
        assert reason == ""

    def test_empty_blueprint(self):
        ok, reason = check_blueprint_integrity({})
        assert ok is False
        assert "empty" in reason.lower()

    def test_none_blueprint(self):
        ok, reason = check_blueprint_integrity(None)
        assert ok is False

    def test_no_components(self):
        ok, reason = check_blueprint_integrity({"components": []})
        assert ok is False
        assert "no components" in reason.lower()

    def test_component_missing_name(self):
        bp = {"components": [{"type": "service"}]}
        ok, reason = check_blueprint_integrity(bp)
        assert ok is False
        assert "no name" in reason.lower()

    def test_component_missing_type(self):
        bp = {"components": [{"name": "Foo"}]}
        ok, reason = check_blueprint_integrity(bp)
        assert ok is False
        assert "no type" in reason.lower()

    def test_duplicate_component_names(self):
        bp = {"components": [
            {"name": "Foo", "type": "service"},
            {"name": "Foo", "type": "entity"},
        ]}
        ok, reason = check_blueprint_integrity(bp)
        assert ok is False
        assert "duplicate" in reason.lower()

    def test_relationship_unknown_from(self):
        bp = {
            "components": [{"name": "A", "type": "s"}],
            "relationships": [{"from": "Unknown", "to": "A", "type": "x"}],
        }
        ok, reason = check_blueprint_integrity(bp)
        assert ok is False
        assert "Unknown" in reason

    def test_relationship_unknown_to(self):
        bp = {
            "components": [{"name": "A", "type": "s"}],
            "relationships": [{"from": "A", "to": "Missing", "type": "x"}],
        }
        ok, reason = check_blueprint_integrity(bp)
        assert ok is False
        assert "Missing" in reason

    def test_components_not_list(self):
        ok, reason = check_blueprint_integrity({"components": "not a list"})
        assert ok is False

    def test_component_not_dict(self):
        ok, reason = check_blueprint_integrity({"components": ["not a dict"]})
        assert ok is False


# =============================================================================
# PACKAGE ID VERIFICATION
# =============================================================================

class TestRecomputePackageId:
    def test_matches_tool_package(self, valid_blueprint, valid_code):
        from_governor = _recompute_package_id(valid_blueprint, valid_code)
        from_tool_package = compute_package_id(valid_blueprint, valid_code)
        assert from_governor == from_tool_package


# =============================================================================
# FULL VALIDATION
# =============================================================================

class TestValidateImport:
    def test_all_checks_pass(self, valid_blueprint, valid_code, valid_provenance, valid_fidelity):
        package_id = compute_package_id(valid_blueprint, valid_code)
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc123",
            package_id=package_id,
        )
        assert result.allowed is True
        assert result.rejection_reason == ""
        assert result.provenance_valid is True
        assert result.trust_sufficient is True
        assert result.code_safe is True
        assert len(result.checks_performed) == 5

    def test_provenance_fails(self, valid_blueprint, valid_code, valid_fidelity):
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=[],
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id="abc",
        )
        assert result.allowed is False
        assert result.provenance_valid is False
        assert "provenance_integrity" in result.checks_performed

    def test_trust_fails(self, valid_blueprint, valid_code, valid_provenance, valid_fidelity):
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=30.0,
            verification_badge="unverified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id="abc",
        )
        assert result.allowed is False
        assert result.trust_sufficient is False

    def test_code_safety_fails(self, valid_blueprint, valid_provenance, valid_fidelity):
        dangerous_code = {"main": "exec('import os')"}
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=dangerous_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id="abc",
        )
        assert result.allowed is False
        assert result.code_safe is False

    def test_blueprint_fails(self, valid_code, valid_provenance, valid_fidelity):
        result = validate_import(
            blueprint={},
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id="abc",
        )
        assert result.allowed is False
        assert "empty" in result.rejection_reason.lower()

    def test_package_id_mismatch(self, valid_blueprint, valid_code, valid_provenance, valid_fidelity):
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id="wrong_id_here_x",
        )
        assert result.allowed is False
        assert "mismatch" in result.rejection_reason.lower()

    def test_result_is_frozen(self, valid_blueprint, valid_code, valid_provenance, valid_fidelity):
        package_id = compute_package_id(valid_blueprint, valid_code)
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id=package_id,
        )
        with pytest.raises(AttributeError):
            result.allowed = False

    def test_custom_thresholds(self, valid_blueprint, valid_code, valid_provenance, valid_fidelity):
        package_id = compute_package_id(valid_blueprint, valid_code)
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=valid_provenance,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id=package_id,
            min_trust_score=90.0,
        )
        assert result.allowed is False
        assert result.trust_sufficient is False

    def test_provenance_depth_reported(self, valid_blueprint, valid_code, valid_fidelity):
        chain = [
            {"compilation_id": f"c{i}", "instance_id": f"inst{i:03d}",
             "timestamp": f"2026-01-0{i+1}T00:00:00Z", "domain": "s"}
            for i in range(3)
        ]
        package_id = compute_package_id(valid_blueprint, valid_code)
        result = validate_import(
            blueprint=valid_blueprint,
            generated_code=valid_code,
            provenance_chain=chain,
            trust_score=80.0,
            verification_badge="verified",
            fidelity_scores=valid_fidelity,
            fingerprint="abc",
            package_id=package_id,
        )
        assert result.provenance_depth == 3
