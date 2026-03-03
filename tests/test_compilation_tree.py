"""
Tests for Phase 25: Compilation Trees.

Tests cover:
- 25.1: Frozen dataclasses + pure functions (decomposition, L2, integration)
- 25.2: Engine compile_tree() integration
- 25.3: Enhanced L2 pattern synthesis + formatting
"""

import pytest
from datetime import datetime

from core.compilation_tree import (
    SubsystemSpec,
    TreeDecomposition,
    CrossCuttingComponent,
    InterfaceGap,
    L2Synthesis,
    IntegrationReport,
    ChildResult,
    TreeResult,
    normalize_component_name,
    decompose_root,
    build_subsystem_description,
    extract_shared_vocabulary,
    find_cross_cutting_components,
    detect_interface_gaps,
    extract_relationship_patterns,
    synthesize_l2_patterns,
    verify_integration,
    compute_tree_health,
    format_l2_patterns_section,
    serialize_tree_result,
    deserialize_tree_result,
)
from core.determinism import compute_structural_fingerprint


# =============================================================================
# FIXTURES
# =============================================================================

def _make_blueprint(components, relationships=None, constraints=None):
    """Helper to build a minimal blueprint dict."""
    return {
        "version": "3.0",
        "components": [
            {"name": n, "type": t, "description": d, "derived_from": "test"}
            for n, t, d in components
        ],
        "relationships": relationships or [],
        "constraints": constraints or [],
        "unresolved": [],
    }


def _make_child_result(name, success=True, blueprint=None, verification_score=0.8):
    """Helper to build a ChildResult."""
    bp = blueprint or _make_blueprint([("A", "entity", "test")])
    fp = compute_structural_fingerprint(bp)
    return ChildResult(
        subsystem_name=name,
        success=success,
        blueprint=bp,
        fingerprint_hash=fp.hash_digest,
        component_count=len(bp.get("components", [])),
        relationship_count=len(bp.get("relationships", [])),
        verification_score=verification_score,
    )


# =============================================================================
# 25.1: DATACLASS CONSTRUCTION + IMMUTABILITY
# =============================================================================

class TestDataclasses:

    def test_subsystem_spec_frozen(self):
        spec = SubsystemSpec(
            name="Auth",
            description="Authentication subsystem",
            canonical_components=("UserAuth", "TokenManager"),
            parent_components=("Gateway",),
            derived_from="architect_artifact",
        )
        assert spec.name == "Auth"
        assert spec.canonical_components == ("UserAuth", "TokenManager")
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_tree_decomposition_frozen(self):
        decomp = TreeDecomposition(
            subsystem_specs=(),
            root_component_count=5,
            decomposition_source="none",
            decomposition_confidence=0.0,
            unassigned_components=("A", "B"),
        )
        assert decomp.root_component_count == 5
        with pytest.raises(AttributeError):
            decomp.decomposition_source = "changed"

    def test_l2_synthesis_frozen(self):
        l2 = L2Synthesis(
            shared_vocabulary=(("auth", 3),),
            cross_cutting_components=(),
            relationship_patterns=(),
            interface_gaps=(),
            integration_constraints=(),
            pattern_count=1,
            synthesis_confidence=0.5,
        )
        assert l2.pattern_count == 1
        with pytest.raises(AttributeError):
            l2.pattern_count = 99

    def test_child_result_frozen(self):
        cr = _make_child_result("Auth")
        assert cr.subsystem_name == "Auth"
        assert cr.success is True
        with pytest.raises(AttributeError):
            cr.success = False

    def test_tree_result_frozen(self):
        result = TreeResult(
            root_blueprint={},
            decomposition=TreeDecomposition((), 0, "none", 0.0, ()),
            child_results=(),
            l2_synthesis=L2Synthesis((), (), (), (), (), 0, 0.0),
            integration_report=IntegrationReport(0, 0, 0, (), 1.0),
            tree_health=0.5,
            total_components=10,
            timestamp="2026-02-09T00:00:00",
        )
        assert result.tree_health == 0.5
        with pytest.raises(AttributeError):
            result.tree_health = 1.0


# =============================================================================
# 25.1: normalize_component_name
# =============================================================================

class TestNormalize:

    def test_lowercase(self):
        assert normalize_component_name("UserAuth") == "userauth"

    def test_strip_spaces(self):
        assert normalize_component_name("User Auth") == "userauth"

    def test_strip_underscores(self):
        assert normalize_component_name("user_auth") == "userauth"

    def test_strip_hyphens(self):
        assert normalize_component_name("user-auth") == "userauth"

    def test_mixed(self):
        assert normalize_component_name("User_Auth-Service") == "userauthservice"

    def test_empty(self):
        assert normalize_component_name("") == ""


# =============================================================================
# 25.1: decompose_root
# =============================================================================

class TestDecomposeRoot:

    def test_from_architect_artifact(self):
        bp = _make_blueprint([
            ("UserAuth", "entity", "auth"),
            ("TokenManager", "process", "tokens"),
            ("Gateway", "interface", "gateway"),
            ("Logger", "process", "logging"),
        ])
        artifact = {
            "subsystems": [
                {"name": "Auth", "contains": ["UserAuth", "TokenManager"], "description": "Auth system"},
                {"name": "Infra", "contains": ["Logger"], "description": "Infrastructure"},
            ]
        }
        decomp = decompose_root(bp, architect_artifact=artifact)
        assert decomp.decomposition_source == "architect_artifact"
        assert len(decomp.subsystem_specs) == 2
        assert decomp.subsystem_specs[0].name == "Auth"
        assert "UserAuth" in decomp.subsystem_specs[0].canonical_components
        assert "Gateway" in decomp.unassigned_components

    def test_from_subsystem_hints(self):
        bp = _make_blueprint([
            ("A", "entity", "a"),
            ("B", "process", "b"),
            ("C", "interface", "c"),
        ])
        hints = {"Core": ["A", "B"]}
        decomp = decompose_root(bp, subsystem_hints=hints)
        assert decomp.decomposition_source == "subsystem_hint"
        assert len(decomp.subsystem_specs) == 1
        assert decomp.subsystem_specs[0].name == "Core"
        assert "C" in decomp.unassigned_components

    def test_from_blueprint_subsystem_type(self):
        bp = {
            "components": [
                {"name": "AuthSub", "type": "subsystem", "description": "Auth subsystem"},
                {"name": "UserAuth", "type": "entity", "description": "auth"},
                {"name": "TokenMgr", "type": "process", "description": "tokens"},
            ],
            "relationships": [
                {"from": "AuthSub", "to": "UserAuth", "type": "contains"},
                {"from": "AuthSub", "to": "TokenMgr", "type": "contains"},
            ],
            "constraints": [],
            "unresolved": [],
        }
        decomp = decompose_root(bp)
        assert decomp.decomposition_source == "blueprint_subsystem_type"
        assert len(decomp.subsystem_specs) == 1
        assert "UserAuth" in decomp.subsystem_specs[0].canonical_components
        assert "TokenMgr" in decomp.subsystem_specs[0].canonical_components

    def test_no_subsystems(self):
        bp = _make_blueprint([("A", "entity", "a"), ("B", "process", "b")])
        decomp = decompose_root(bp)
        assert decomp.decomposition_source == "none"
        assert len(decomp.subsystem_specs) == 0
        assert decomp.decomposition_confidence == 0.0
        assert "A" in decomp.unassigned_components

    def test_architect_takes_priority_over_hints(self):
        bp = _make_blueprint([("A", "entity", "a")])
        artifact = {"subsystems": [{"name": "Sub1", "contains": ["A"], "description": ""}]}
        hints = {"Sub2": ["A"]}
        decomp = decompose_root(bp, architect_artifact=artifact, subsystem_hints=hints)
        assert decomp.decomposition_source == "architect_artifact"
        assert decomp.subsystem_specs[0].name == "Sub1"

    def test_empty_blueprint(self):
        decomp = decompose_root({})
        assert decomp.decomposition_source == "none"
        assert decomp.root_component_count == 0


# =============================================================================
# 25.1: build_subsystem_description
# =============================================================================

class TestBuildSubsystemDescription:

    def test_basic(self):
        spec = SubsystemSpec(
            name="Auth",
            description="Authentication subsystem",
            canonical_components=("UserAuth", "TokenManager"),
            parent_components=(),
            derived_from="architect_artifact",
        )
        bp = _make_blueprint([
            ("UserAuth", "entity", "auth"),
            ("TokenManager", "process", "tokens"),
        ])
        desc = build_subsystem_description("Build a web app", spec, bp)
        assert "Auth" in desc
        assert "Authentication subsystem" in desc
        assert "UserAuth" in desc
        assert "web app" in desc

    def test_with_relationships(self):
        spec = SubsystemSpec(
            name="Auth",
            description="",
            canonical_components=("UserAuth",),
            parent_components=(),
            derived_from="architect_artifact",
        )
        bp = {
            "components": [{"name": "UserAuth", "type": "entity", "description": ""}],
            "relationships": [
                {"from": "UserAuth", "to": "Database", "type": "accesses", "description": "stores users"},
            ],
            "constraints": [],
            "unresolved": [],
        }
        desc = build_subsystem_description("Build a system", spec, bp)
        assert "UserAuth" in desc
        assert "accesses" in desc


# =============================================================================
# 25.1: extract_shared_vocabulary
# =============================================================================

class TestExtractSharedVocabulary:

    def test_shared_terms(self):
        bp1 = _make_blueprint([
            ("UserAuth", "entity", "handles authentication and validation"),
        ])
        bp2 = _make_blueprint([
            ("TokenAuth", "entity", "token-based authentication and security"),
        ])
        vocab = extract_shared_vocabulary([bp1, bp2], ["Auth", "Token"])
        # "authentication" should appear in both
        terms = dict(vocab)
        assert "authentication" in terms
        assert terms["authentication"] == 2

    def test_single_child_returns_empty(self):
        bp1 = _make_blueprint([("A", "entity", "test")])
        vocab = extract_shared_vocabulary([bp1], ["A"])
        assert vocab == ()

    def test_empty(self):
        vocab = extract_shared_vocabulary([], [])
        assert vocab == ()


# =============================================================================
# 25.1: find_cross_cutting_components
# =============================================================================

class TestFindCrossCuttingComponents:

    def test_all_siblings(self):
        bp1 = _make_blueprint([("Database", "entity", ""), ("Auth", "process", "")])
        bp2 = _make_blueprint([("Database", "entity", ""), ("Logger", "process", "")])
        bp3 = _make_blueprint([("Database", "entity", ""), ("Cache", "process", "")])
        result = find_cross_cutting_components([bp1, bp2, bp3], ["A", "B", "C"])
        assert len(result) == 1
        assert result[0].normalized_name == "database"
        assert result[0].frequency == 1.0

    def test_two_of_three(self):
        bp1 = _make_blueprint([("Auth", "process", ""), ("Database", "entity", "")])
        bp2 = _make_blueprint([("Auth", "process", ""), ("Logger", "process", "")])
        bp3 = _make_blueprint([("Cache", "process", ""), ("Logger", "process", "")])
        result = find_cross_cutting_components([bp1, bp2, bp3], ["A", "B", "C"])
        names = {r.normalized_name for r in result}
        assert "auth" in names
        assert "logger" in names

    def test_no_overlap(self):
        bp1 = _make_blueprint([("A", "entity", "")])
        bp2 = _make_blueprint([("B", "entity", "")])
        result = find_cross_cutting_components([bp1, bp2], ["C1", "C2"])
        assert len(result) == 0

    def test_normalized_matching(self):
        bp1 = _make_blueprint([("User Auth", "entity", "")])
        bp2 = _make_blueprint([("user_auth", "entity", "")])
        result = find_cross_cutting_components([bp1, bp2], ["A", "B"])
        assert len(result) == 1
        assert result[0].normalized_name == "userauth"
        assert len(result[0].variants) == 2


# =============================================================================
# 25.1: detect_interface_gaps
# =============================================================================

class TestDetectInterfaceGaps:

    def test_missing_contract(self):
        bp1 = _make_blueprint(
            [("Auth", "entity", "")],
            relationships=[{"from": "Auth", "to": "Database", "type": "accesses"}],
        )
        bp2 = _make_blueprint([("Database", "entity", "")])
        gaps = detect_interface_gaps([bp1, bp2], ["AuthSub", "DataSub"])
        gap_types = [g.gap_type for g in gaps]
        assert "missing_contract" in gap_types

    def test_type_mismatch(self):
        bp1 = _make_blueprint([("Logger", "entity", "")])
        bp2 = _make_blueprint([("Logger", "process", "")])
        gaps = detect_interface_gaps([bp1, bp2], ["A", "B"])
        gap_types = [g.gap_type for g in gaps]
        assert "type_mismatch" in gap_types

    def test_dangling_reference(self):
        bp1 = _make_blueprint(
            [("Auth", "entity", "")],
            relationships=[{"from": "Auth", "to": "Nonexistent", "type": "accesses"}],
        )
        bp2 = _make_blueprint([("Other", "entity", "")])
        gaps = detect_interface_gaps([bp1, bp2], ["A", "B"])
        gap_types = [g.gap_type for g in gaps]
        assert "dangling_reference" in gap_types

    def test_no_gaps_clean(self):
        bp1 = _make_blueprint([("Auth", "entity", "")])
        bp2 = _make_blueprint([("Logger", "entity", "")])
        gaps = detect_interface_gaps([bp1, bp2], ["A", "B"])
        assert len(gaps) == 0


# =============================================================================
# 25.1: synthesize_l2_patterns
# =============================================================================

class TestSynthesizeL2Patterns:

    def test_basic_synthesis(self):
        bp1 = _make_blueprint([
            ("Database", "entity", "stores authentication data"),
            ("Auth", "process", "handles login"),
        ])
        bp2 = _make_blueprint([
            ("Database", "entity", "stores user data"),
            ("Logger", "process", "logs events"),
        ])
        l2 = synthesize_l2_patterns([bp1, bp2], ["AuthSub", "DataSub"])
        assert l2.pattern_count > 0
        assert len(l2.cross_cutting_components) >= 1
        cc_names = [cc.normalized_name for cc in l2.cross_cutting_components]
        assert "database" in cc_names

    def test_empty_returns_zero(self):
        l2 = synthesize_l2_patterns([], [])
        assert l2.pattern_count == 0
        assert l2.synthesis_confidence == 0.0

    def test_l2_confidence_drops_with_gaps(self):
        # Two blueprints where Logger has type mismatch → gaps
        bp1 = _make_blueprint([("Logger", "entity", "log stuff")])
        bp2 = _make_blueprint([("Logger", "process", "log stuff")])
        l2 = synthesize_l2_patterns([bp1, bp2], ["A", "B"])
        # Should have gaps, which reduce confidence
        assert len(l2.interface_gaps) > 0
        # Compare with clean version
        bp3 = _make_blueprint([("Logger", "entity", "log stuff")])
        bp4 = _make_blueprint([("Logger", "entity", "log stuff")])
        l2_clean = synthesize_l2_patterns([bp3, bp4], ["A", "B"])
        assert l2_clean.synthesis_confidence >= l2.synthesis_confidence

    def test_l2_pattern_count_correct(self):
        bp1 = _make_blueprint([
            ("Auth", "entity", "authentication module"),
            ("Database", "entity", "storage layer"),
        ], relationships=[{"from": "Auth", "to": "Database", "type": "accesses"}])
        bp2 = _make_blueprint([
            ("Auth", "entity", "auth system"),
            ("Database", "entity", "data store"),
        ], relationships=[{"from": "Auth", "to": "Database", "type": "accesses"}])
        l2 = synthesize_l2_patterns([bp1, bp2], ["Sub1", "Sub2"])
        expected_count = (
            len(l2.shared_vocabulary)
            + len(l2.cross_cutting_components)
            + len(l2.relationship_patterns)
        )
        assert l2.pattern_count == expected_count


# =============================================================================
# 25.1: verify_integration
# =============================================================================

class TestVerifyIntegration:

    def test_clean_integration(self):
        bp1 = _make_blueprint([("Auth", "entity", "")])
        bp2 = _make_blueprint([("Logger", "entity", "")])
        report = verify_integration([bp1, bp2], ["A", "B"])
        assert report.gap_count == 0
        assert report.overall_score == 1.0

    def test_with_gaps(self):
        bp1 = _make_blueprint([("Logger", "entity", "")])
        bp2 = _make_blueprint([("Logger", "process", "")])
        report = verify_integration([bp1, bp2], ["A", "B"])
        assert report.gap_count > 0
        assert report.overall_score < 1.0

    def test_single_child(self):
        bp1 = _make_blueprint([("A", "entity", "")])
        report = verify_integration([bp1], ["Sub1"])
        assert report.overall_score == 1.0
        assert report.total_interfaces_checked == 0


# =============================================================================
# 25.1: compute_tree_health
# =============================================================================

class TestComputeTreeHealth:

    def test_all_succeed(self):
        cr1 = _make_child_result("A", success=True, verification_score=0.9)
        cr2 = _make_child_result("B", success=True, verification_score=0.8)
        l2 = L2Synthesis((), (), (), (), (), 5, 0.7)
        ir = IntegrationReport(5, 5, 0, (), 1.0)
        health = compute_tree_health((cr1, cr2), l2, ir)
        assert 0.0 <= health <= 1.0
        assert health > 0.5  # All succeed, good scores

    def test_partial_failure(self):
        cr1 = _make_child_result("A", success=True, verification_score=0.9)
        cr2 = _make_child_result("B", success=False, verification_score=0.1)
        l2 = L2Synthesis((), (), (), (), (), 0, 0.0)
        ir = IntegrationReport(0, 0, 0, (), 1.0)
        health = compute_tree_health((cr1, cr2), l2, ir)
        assert health < 0.8  # Partial failure drags it down

    def test_empty_children(self):
        l2 = L2Synthesis((), (), (), (), (), 0, 0.0)
        ir = IntegrationReport(0, 0, 0, (), 1.0)
        health = compute_tree_health((), l2, ir)
        assert health == 0.0


# =============================================================================
# 25.1: SERIALIZATION ROUND-TRIP
# =============================================================================

class TestSerialization:

    def test_round_trip(self):
        bp = _make_blueprint([("Auth", "entity", "auth"), ("DB", "entity", "db")])
        cr = _make_child_result("AuthSub", blueprint=bp)
        gap = InterfaceGap("Auth", "DB", "AuthSub", "DataSub", "missing_contract", "test gap")
        l2 = L2Synthesis(
            shared_vocabulary=(("auth", 2),),
            cross_cutting_components=(
                CrossCuttingComponent("auth", ("Auth",), 1.0, ("AuthSub",), "entity"),
            ),
            relationship_patterns=(("auth", "db", "accesses", 2),),
            interface_gaps=(gap,),
            integration_constraints=("Auth must be consistent",),
            pattern_count=3,
            synthesis_confidence=0.7,
        )
        ir = IntegrationReport(1, 1, 0, (), 0.9)
        decomp = TreeDecomposition(
            subsystem_specs=(SubsystemSpec("AuthSub", "auth", ("Auth",), (), "test"),),
            root_component_count=5,
            decomposition_source="architect_artifact",
            decomposition_confidence=0.8,
            unassigned_components=("Extra",),
        )
        result = TreeResult(
            root_blueprint=bp,
            decomposition=decomp,
            child_results=(cr,),
            l2_synthesis=l2,
            integration_report=ir,
            tree_health=0.85,
            total_components=10,
            timestamp="2026-02-09T00:00:00",
        )

        data = serialize_tree_result(result)
        restored = deserialize_tree_result(data)

        assert restored.tree_health == result.tree_health
        assert restored.total_components == result.total_components
        assert restored.timestamp == result.timestamp
        assert len(restored.child_results) == 1
        assert restored.child_results[0].subsystem_name == "AuthSub"
        assert restored.decomposition.decomposition_source == "architect_artifact"
        assert len(restored.l2_synthesis.cross_cutting_components) == 1
        assert restored.l2_synthesis.synthesis_confidence == 0.7
        assert len(restored.l2_synthesis.interface_gaps) == 1
        assert restored.l2_synthesis.interface_gaps[0].gap_type == "missing_contract"


# =============================================================================
# 25.3: extract_relationship_patterns
# =============================================================================

class TestExtractRelationshipPatterns:

    def test_shared_patterns(self):
        bp1 = _make_blueprint(
            [("Auth", "entity", ""), ("Database", "entity", "")],
            relationships=[{"from": "Auth", "to": "Database", "type": "accesses"}],
        )
        bp2 = _make_blueprint(
            [("Auth", "entity", ""), ("Database", "entity", "")],
            relationships=[{"from": "Auth", "to": "Database", "type": "accesses"}],
        )
        patterns = extract_relationship_patterns([bp1, bp2], ["A", "B"])
        assert len(patterns) >= 1
        # (from_norm, to_norm, type, freq)
        assert patterns[0][0] == "auth"
        assert patterns[0][1] == "database"
        assert patterns[0][2] == "accesses"
        assert patterns[0][3] == 2

    def test_no_overlap(self):
        bp1 = _make_blueprint(
            [("A", "entity", ""), ("B", "entity", "")],
            relationships=[{"from": "A", "to": "B", "type": "triggers"}],
        )
        bp2 = _make_blueprint(
            [("C", "entity", ""), ("D", "entity", "")],
            relationships=[{"from": "C", "to": "D", "type": "accesses"}],
        )
        patterns = extract_relationship_patterns([bp1, bp2], ["S1", "S2"])
        assert len(patterns) == 0


# =============================================================================
# 25.3: format_l2_patterns_section
# =============================================================================

class TestFormatL2Patterns:

    def test_with_data(self):
        l2 = L2Synthesis(
            shared_vocabulary=(("auth", 3), ("user", 2)),
            cross_cutting_components=(
                CrossCuttingComponent("database", ("Database",), 1.0, ("A", "B"), "entity"),
            ),
            relationship_patterns=(("auth", "database", "accesses", 2),),
            interface_gaps=(
                InterfaceGap("Auth", "DB", "A", "B", "missing_contract", "test gap"),
            ),
            integration_constraints=("Database must be consistent",),
            pattern_count=5,
            synthesis_confidence=0.7,
        )
        text = format_l2_patterns_section(l2)
        assert text is not None
        assert "Cross-Subsystem Patterns" in text
        assert "database" in text
        assert "Recurring Relationships" in text
        assert "Integration Constraints" in text

    def test_empty_returns_none(self):
        l2 = L2Synthesis((), (), (), (), (), 0, 0.0)
        text = format_l2_patterns_section(l2)
        assert text is None

    def test_synthesize_includes_relationship_patterns(self):
        bp1 = _make_blueprint(
            [("Auth", "entity", "auth module"), ("DB", "entity", "database")],
            relationships=[{"from": "Auth", "to": "DB", "type": "accesses"}],
        )
        bp2 = _make_blueprint(
            [("Auth", "entity", "auth module"), ("DB", "entity", "database")],
            relationships=[{"from": "Auth", "to": "DB", "type": "accesses"}],
        )
        l2 = synthesize_l2_patterns([bp1, bp2], ["Sub1", "Sub2"])
        assert len(l2.relationship_patterns) >= 1


# =============================================================================
# 25.2: ENGINE INTEGRATION — compile_tree()
# =============================================================================

class TestCompileTree:
    """Tests for MotherlabsEngine.compile_tree() using MockClient."""

    @pytest.fixture
    def engine(self):
        from core.llm import MockClient
        from core.engine import MotherlabsEngine
        return MotherlabsEngine(llm_client=MockClient(), auto_store=False)

    def test_compile_tree_returns_tree_result(self, engine):
        result = engine.compile_tree("Build an e-commerce platform with authentication and payments")
        assert isinstance(result, TreeResult)
        assert result.root_blueprint is not None
        assert result.timestamp

    def test_compile_tree_no_subsystems_returns_root_only(self, engine):
        # Short description unlikely to produce subsystems
        result = engine.compile_tree("Build a simple calculator")
        assert isinstance(result, TreeResult)
        # Even if no children, should be a valid TreeResult
        assert result.decomposition is not None
        assert result.root_blueprint is not None

    def test_compile_tree_empty_description_raises(self, engine):
        with pytest.raises(Exception):
            engine.compile_tree("")

    def test_compile_tree_health_in_range(self, engine):
        result = engine.compile_tree("Build a web application with user management")
        assert 0.0 <= result.tree_health <= 1.0

    def test_compile_tree_integration_report_populated(self, engine):
        result = engine.compile_tree("Build a task management system")
        assert result.integration_report is not None
        assert 0.0 <= result.integration_report.overall_score <= 1.0

    def test_compile_tree_l2_synthesis_populated(self, engine):
        result = engine.compile_tree("Build a project management tool")
        assert result.l2_synthesis is not None
        assert result.l2_synthesis.synthesis_confidence >= 0.0

    def test_compile_tree_max_children_cap(self, engine):
        result = engine.compile_tree(
            "Build a massive enterprise system",
            max_children=2,
        )
        assert isinstance(result, TreeResult)
        assert len(result.child_results) <= 2

    def test_compile_tree_child_results_have_fingerprints(self, engine):
        result = engine.compile_tree("Build a content management system")
        for cr in result.child_results:
            assert cr.fingerprint_hash  # Non-empty string
            assert len(cr.fingerprint_hash) == 16  # SHA256[:16]
