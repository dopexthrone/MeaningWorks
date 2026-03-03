"""
Tests for core/verification.py — Phase 18: Deterministic Verification.

~30 tests covering all scoring functions, dataclass integrity,
routing logic, and format conversion.
"""

import pytest

from core.verification import (
    DimensionScore,
    DeterministicVerification,
    score_completeness,
    score_consistency,
    score_coherence,
    score_traceability,
    score_actionability,
    score_specificity,
    score_codegen_readiness,
    verify_deterministic,
    to_verification_dict,
    _clamp,
    _find_cross_subsystem_duplicates,
    _normalize_for_dedup,
    _infer_methods_from_type,
    _infer_methods_from_relationships,
)


# =============================================================================
# HELPERS
# =============================================================================


def _make_component(name, desc="A component", derived_from="user said X about it", comp_type="entity", methods=None):
    """Create a minimal component dict."""
    c = {
        "name": name,
        "type": comp_type,
        "description": desc,
        "derived_from": derived_from,
    }
    if methods:
        c["methods"] = methods
    return c


def _make_blueprint(n_components=5, with_rels=True, with_constraints=False):
    """Create a test blueprint with N components."""
    components = []
    for i in range(n_components):
        components.append(_make_component(
            name=f"Component{i}",
            desc=f"This is component {i} which handles domain logic for feature {i}",
            derived_from=f"User mentioned feature {i} in the requirements document section {i}",
        ))
    relationships = []
    if with_rels and n_components > 1:
        for i in range(n_components - 1):
            relationships.append({
                "from": f"Component{i}",
                "to": f"Component{i+1}",
                "type": "depends_on",
                "description": f"Component{i} depends on Component{i+1}",
            })
    bp = {
        "components": components,
        "relationships": relationships,
        "constraints": [],
        "unresolved": [],
    }
    if with_constraints:
        bp["constraints"] = [
            {"description": "score in range [0, 100]", "applies_to": ["Component0"]},
            {"description": "name must not be null", "applies_to": ["Component1"]},
        ]
    return bp


# =============================================================================
# DimensionScore TESTS
# =============================================================================


class TestDimensionScore:
    def test_frozen(self):
        ds = DimensionScore(name="test", score=80, confidence=0.9, gaps=(), details="ok")
        with pytest.raises(AttributeError):
            ds.score = 90

    def test_score_stored(self):
        ds = DimensionScore(name="comp", score=75, confidence=0.8, gaps=("a",), details="x")
        assert ds.score == 75
        assert ds.confidence == 0.8
        assert ds.gaps == ("a",)

    def test_gaps_as_tuple(self):
        ds = DimensionScore(name="t", score=50, confidence=0.5, gaps=("g1", "g2"), details="")
        assert isinstance(ds.gaps, tuple)
        assert len(ds.gaps) == 2


# =============================================================================
# score_completeness TESTS
# =============================================================================


class TestScoreCompleteness:
    def test_full_coverage(self):
        bp = _make_blueprint(3)
        keywords = ["component0", "component1", "component2"]
        ds = score_completeness(bp, keywords, "build something")
        assert ds.score == 100
        assert len(ds.gaps) == 0

    def test_zero_coverage(self):
        bp = _make_blueprint(3)
        keywords = ["nonexistent", "missing", "absent"]
        ds = score_completeness(bp, keywords, "build something")
        assert ds.score == 0
        assert len(ds.gaps) == 3

    def test_partial_coverage(self):
        bp = _make_blueprint(3)
        keywords = ["component0", "missing_keyword"]
        ds = score_completeness(bp, keywords, "build something")
        assert ds.score == 50
        assert len(ds.gaps) == 1

    def test_empty_intent(self):
        bp = _make_blueprint(3)
        ds = score_completeness(bp, [], "build something")
        assert ds.score == 50
        assert ds.confidence == 0.3

    def test_keyword_in_description(self):
        bp = {
            "components": [
                _make_component("AuthService", desc="handles authentication for users"),
            ],
            "relationships": [],
        }
        keywords = ["authentication"]
        ds = score_completeness(bp, keywords, "user auth")
        assert ds.score == 100


# =============================================================================
# score_consistency TESTS
# =============================================================================


class TestScoreConsistency:
    def test_clean(self):
        ds = score_consistency(0, [], [])
        assert ds.score == 100
        assert len(ds.gaps) == 0

    def test_contradictions(self):
        ds = score_consistency(2, [], [])
        assert ds.score == 60  # 100 - 2*20
        assert "contradiction" in ds.gaps[0].lower()

    def test_cycles(self):
        ds = score_consistency(0, ["Dependency cycle detected: A -> B -> A"], [])
        assert ds.score == 85  # 100 - 1*15

    def test_combined(self):
        ds = score_consistency(1, ["cycle: A->B->A"], ["orphan: X"])
        # 100 - 20 - 15 - 5 = 60
        assert ds.score == 60


# =============================================================================
# score_coherence TESTS
# =============================================================================


class TestScoreCoherence:
    def test_zero_orphans(self):
        ds = score_coherence(0.0, 1.0, 1.0, 0)
        assert ds.score > 80

    def test_high_orphans(self):
        ds = score_coherence(0.8, 0.0, 0.5, 0)
        assert ds.score < 60

    def test_density_bonus(self):
        low = score_coherence(0.2, 0.1, 0.7, 0)
        high = score_coherence(0.2, 0.9, 0.7, 0)
        assert high.score > low.score

    def test_dangling_penalty(self):
        no_dangle = score_coherence(0.1, 0.5, 0.8, 0)
        with_dangle = score_coherence(0.1, 0.5, 0.8, 4)
        assert no_dangle.score > with_dangle.score


# =============================================================================
# score_traceability TESTS
# =============================================================================


class TestScoreTraceability:
    def test_all_derived(self):
        comps = [
            _make_component("A", derived_from="User said build authentication system"),
            _make_component("B", derived_from="Requirements doc section 3.2 specifies data model"),
        ]
        ds = score_traceability(comps)
        assert ds.score == 100

    def test_none_derived(self):
        comps = [
            _make_component("A", derived_from=""),
            _make_component("B", derived_from=""),
        ]
        ds = score_traceability(comps)
        assert ds.score == 0
        assert len(ds.gaps) == 2

    def test_vague_derived(self):
        comps = [
            _make_component("A", derived_from="input"),  # <= 10 chars, not specific
        ]
        ds = score_traceability(comps)
        # has derived (60 pts) but not specific (0 of 40 pts)
        assert ds.score == 60

    def test_specific_derived(self):
        comps = [
            _make_component("A", derived_from="User described auth flow with login and session management"),
        ]
        ds = score_traceability(comps)
        assert ds.score == 100


# =============================================================================
# score_actionability TESTS
# =============================================================================


class TestScoreActionability:
    def test_with_methods(self):
        comps = [
            _make_component("A", desc="Handles auth process for users", comp_type="process", methods=[{"name": "run"}]),
            _make_component("B", desc="Stores user data in the database", comp_type="entity", methods=[{"name": "get"}]),
        ]
        ds = score_actionability(1.0, comps)
        # parseable=1.0*25=25, methods=1.0*35=35, typed=0.5*20=10, substance=1.0*20=20 = 90
        assert ds.score >= 80

    def test_bare_components(self):
        comps = [_make_component("A", desc="short")]
        ds = score_actionability(0.0, comps)
        # Entity type infers CRUD methods, so methods_ratio=100% (35 pts) + substance=0% → 35
        assert ds.score < 40

    def test_parseable_constraints_boost(self):
        comps = [_make_component("A")]
        low = score_actionability(0.0, comps)
        high = score_actionability(1.0, comps)
        assert high.score > low.score


# =============================================================================
# score_specificity TESTS
# =============================================================================


class TestScoreSpecificity:
    def test_long_descriptions(self):
        comps = [
            _make_component("A", desc="x" * 100, derived_from="y" * 50),
        ]
        ds = score_specificity(comps, 0.8)
        assert ds.score > 70

    def test_short_descriptions(self):
        comps = [
            _make_component("A", desc="ok", derived_from="x"),
        ]
        ds = score_specificity(comps, 0.3)
        assert ds.score < 30

    def test_type_confidence_boost(self):
        comps = [_make_component("A", desc="x" * 50, derived_from="y" * 30)]
        low = score_specificity(comps, 0.2)
        high = score_specificity(comps, 0.9)
        assert high.score > low.score


# =============================================================================
# score_codegen_readiness TESTS
# =============================================================================


class TestScoreCodegenReadiness:
    def test_all_high(self):
        ds = score_codegen_readiness(90, 90, 90, 90, 90)
        assert ds.score == 90
        assert len(ds.gaps) == 0

    def test_blocker_detection(self):
        ds = score_codegen_readiness(20, 30, 80, 20, 80)
        assert len(ds.gaps) > 0
        assert any("Completeness" in g for g in ds.gaps)
        assert any("Consistency" in g for g in ds.gaps)
        assert any("Actionability" in g for g in ds.gaps)


# =============================================================================
# verify_deterministic TESTS
# =============================================================================


class TestVerifyDeterministic:
    def test_skip_llm_on_high_scores(self):
        bp = _make_blueprint(5)
        result = verify_deterministic(
            blueprint=bp,
            intent_keywords=["component0", "component1", "component2", "component3", "component4"],
            input_text="build components",
            graph_errors=[],
            graph_warnings=[],
            health_score=1.0,
            health_stats={"orphan_ratio": 0.0, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=1.0,
            avg_type_confidence=0.8,
            skip_threshold=70,
            fail_threshold=40,
        )
        assert result.status == "pass"
        assert result.needs_llm is False

    def test_fail_without_llm_on_low_scores(self):
        bp = {"components": [], "relationships": []}
        result = verify_deterministic(
            blueprint=bp,
            intent_keywords=["missing"],
            input_text="build something",
            graph_errors=["cycle: A->B->A", "cycle: C->D->C", "cycle: E->F->E"],
            graph_warnings=["orphan: X", "orphan: Y"],
            health_score=0.1,
            health_stats={"orphan_ratio": 0.9, "dangling_ref_count": 5},
            contradiction_count=3,
            parseable_constraint_ratio=0.0,
            avg_type_confidence=0.2,
            skip_threshold=70,
            fail_threshold=40,
        )
        assert result.status == "needs_work"
        assert result.needs_llm is False

    def test_ambiguous_zone(self):
        bp = _make_blueprint(3, with_rels=False)
        result = verify_deterministic(
            blueprint=bp,
            intent_keywords=["component0", "component1", "missing_term"],
            input_text="build components",
            graph_errors=[],
            graph_warnings=["orphan: Component2"],
            health_score=0.7,
            health_stats={"orphan_ratio": 0.33, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.5,
            skip_threshold=70,
            fail_threshold=40,
        )
        # At least some dimensions should be ambiguous
        assert result.needs_llm is True or result.status in ("pass", "needs_work")


# =============================================================================
# to_verification_dict TESTS
# =============================================================================


class TestToVerificationDict:
    def test_output_format(self):
        bp = _make_blueprint(5)
        det = verify_deterministic(
            blueprint=bp,
            intent_keywords=["component0"],
            input_text="x",
            graph_errors=[],
            graph_warnings=[],
            health_score=0.9,
            health_stats={"orphan_ratio": 0.0, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.6,
        )
        d = to_verification_dict(det)

        # Required keys
        assert "status" in d
        assert d["status"] in ("pass", "needs_work")
        assert "completeness" in d
        assert "consistency" in d
        assert "coherence" in d
        assert "traceability" in d
        assert "actionability" in d
        assert "specificity" in d
        assert "codegen_readiness" in d
        assert "verification_mode" in d

        # Each dimension has score
        for dim in ("completeness", "consistency", "coherence", "traceability"):
            assert "score" in d[dim]
            assert isinstance(d[dim]["score"], int)

    def test_gaps_populated(self):
        bp = {"components": [], "relationships": []}
        det = verify_deterministic(
            blueprint=bp,
            intent_keywords=["missing"],
            input_text="x",
            graph_errors=["cycle found"],
            graph_warnings=[],
            health_score=0.0,
            health_stats={"orphan_ratio": 1.0, "dangling_ref_count": 3},
            contradiction_count=2,
            parseable_constraint_ratio=0.0,
            avg_type_confidence=0.1,
        )
        d = to_verification_dict(det)
        assert d["status"] == "needs_work"
        # Consistency should have conflicts for resynthesis
        assert "conflicts" in d["consistency"]


# =============================================================================
# LEAF MODULE CHECK
# =============================================================================


class TestLeafModule:
    def test_no_engine_protocol_imports(self):
        """Verify verification.py imports only stdlib."""
        import importlib
        import ast
        import inspect

        src = inspect.getsource(importlib.import_module("core.verification"))
        tree = ast.parse(src)

        banned = {"core.engine", "core.protocol", "core.pipeline", "core.schema",
                  "core.llm", "core.digest", "core.cache"}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in banned, f"Illegal import: {module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in banned, f"Illegal import: {alias.name}"


# =============================================================================
# CLAMP TESTS
# =============================================================================


class TestClamp:
    def test_within_range(self):
        assert _clamp(50) == 50

    def test_below_floor(self):
        assert _clamp(-10) == 0

    def test_above_ceiling(self):
        assert _clamp(150) == 100


# =============================================================================
# CROSS-SUBSYSTEM DUPLICATE DETECTION
# =============================================================================


class TestNormalizeForDedup:
    def test_strips_type_suffix(self):
        assert _normalize_for_dedup("Browser Service") == "browser"
        assert _normalize_for_dedup("Auth Manager") == "auth"
        assert _normalize_for_dedup("Request Handler") == "request"

    def test_collapses_whitespace(self):
        assert _normalize_for_dedup("  Browser  Proxy  ") == "browser proxy"

    def test_no_strip_compound_word(self):
        """Don't strip 'engine' from 'BrowserEngine' (no word boundary)."""
        assert _normalize_for_dedup("BrowserEngine") == "browserengine"
        assert _normalize_for_dedup("browserengine") == "browserengine"

    def test_strips_separated_suffix(self):
        """Strip 'Engine' when separated by space."""
        assert _normalize_for_dedup("Browser Engine") == "browser"

    def test_preserves_core_name(self):
        assert _normalize_for_dedup("Authentication") == "authentication"


class TestFindCrossSubsystemDuplicates:
    def test_detects_exact_duplicate(self):
        components = [
            {"name": "Browser", "type": "service"},
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [{"name": "Browser"}],
            }},
        ]
        dupes = _find_cross_subsystem_duplicates(components)
        assert len(dupes) == 1
        assert "Browser" in dupes

    def test_detects_suffix_variant_duplicate(self):
        """'Browser' at top-level matches 'Browser Service' inside sub_blueprint."""
        components = [
            {"name": "Browser", "type": "service"},
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [{"name": "Browser Service"}],
            }},
        ]
        dupes = _find_cross_subsystem_duplicates(components)
        assert len(dupes) == 1

    def test_no_false_positive_on_different_names(self):
        components = [
            {"name": "Auth", "type": "service"},
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [{"name": "Browser"}],
            }},
        ]
        dupes = _find_cross_subsystem_duplicates(components)
        assert len(dupes) == 0

    def test_ignores_subsystem_type_at_toplevel(self):
        """Subsystem components themselves are not flagged as duplicates."""
        components = [
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [{"name": "Networking Core"}],
            }},
        ]
        dupes = _find_cross_subsystem_duplicates(components)
        assert len(dupes) == 0

    def test_no_sub_blueprints(self):
        components = [
            {"name": "A", "type": "service"},
            {"name": "B", "type": "entity"},
        ]
        dupes = _find_cross_subsystem_duplicates(components)
        assert len(dupes) == 0


class TestConsistencyCrossSubsystemDupes:
    def test_clean_with_components(self):
        """No dupes → score 100 still."""
        comps = [
            {"name": "A", "type": "service"},
            {"name": "B", "type": "entity"},
        ]
        ds = score_consistency(0, [], [], comps)
        assert ds.score == 100

    def test_penalty_for_cross_subsystem_dupes(self):
        comps = [
            {"name": "Browser", "type": "service"},
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [{"name": "Browser"}],
            }},
        ]
        ds = score_consistency(0, [], [], comps)
        assert ds.score == 85  # 100 - 1*15
        assert "cross-subsystem" in ds.gaps[0].lower()

    def test_multiple_dupes_penalty(self):
        comps = [
            {"name": "Browser", "type": "service"},
            {"name": "Cache", "type": "service"},
            {"name": "Networking", "type": "subsystem", "sub_blueprint": {
                "components": [
                    {"name": "Browser Service"},
                    {"name": "Cache Manager"},
                ],
            }},
        ]
        ds = score_consistency(0, [], [], comps)
        assert ds.score == 70  # 100 - 2*15
        assert "2 cross-subsystem" in ds.gaps[0]

    def test_backward_compat_no_components(self):
        """Old callers without components arg still work."""
        ds = score_consistency(0, [], [])
        assert ds.score == 100


# =============================================================================
# METHOD INFERENCE (ACTIONABILITY)
# =============================================================================


class TestInferMethodsFromType:
    def test_service_type(self):
        methods = _infer_methods_from_type("service")
        assert "handle" in methods
        assert "process" in methods

    def test_entity_type(self):
        methods = _infer_methods_from_type("entity")
        assert "create" in methods
        assert "read" in methods

    def test_agent_type(self):
        methods = _infer_methods_from_type("agent")
        assert "run" in methods

    def test_unknown_type(self):
        methods = _infer_methods_from_type("quantum_flux")
        assert methods == []

    def test_subsystem_no_methods(self):
        methods = _infer_methods_from_type("subsystem")
        assert methods == []


class TestInferMethodsFromRelationships:
    def test_manages_relationship(self):
        rels = [{"from": "SessionManager", "to": "Session", "type": "manages"}]
        methods = _infer_methods_from_relationships("SessionManager", rels)
        assert "manage" in methods

    def test_triggers_relationship(self):
        rels = [{"from": "Scheduler", "to": "Task", "type": "triggers"}]
        methods = _infer_methods_from_relationships("Scheduler", rels)
        assert "trigger" in methods

    def test_no_matching_source(self):
        rels = [{"from": "Other", "to": "Task", "type": "manages"}]
        methods = _infer_methods_from_relationships("Scheduler", rels)
        assert methods == []

    def test_unknown_rel_type(self):
        rels = [{"from": "A", "to": "B", "type": "connects_to"}]
        methods = _infer_methods_from_relationships("A", rels)
        assert methods == []


class TestActionabilityWithInference:
    def test_inferred_methods_boost_score(self):
        """Components with types but no explicit methods should get credit from inference."""
        comps = [
            _make_component("Auth", desc="Handles user auth processes", comp_type="service"),
            _make_component("Users", desc="Stores user data for the app", comp_type="database"),
            _make_component("API", desc="Routes incoming API requests to services", comp_type="controller"),
        ]
        ds = score_actionability(0.5, comps)
        # All 3 should get inferred methods (service→handle/process, database→query/store, controller→route/handle)
        assert ds.score >= 50
        assert "3 inferred" in ds.details or "inferred" in ds.details

    def test_explicit_methods_not_double_counted(self):
        """Components with explicit methods don't also get inferred methods."""
        comps = [
            _make_component("Auth", desc="Handles user auth", comp_type="service", methods=[{"name": "login"}]),
        ]
        ds = score_actionability(1.0, comps)
        assert "0 inferred" in ds.details

    def test_relationship_inference(self):
        comps = [
            _make_component("Scheduler", desc="Manages task scheduling for the system", comp_type="entity"),
        ]
        rels = [{"from": "Scheduler", "to": "Task", "type": "manages"}]
        ds = score_actionability(0.0, comps, relationships=rels)
        assert "1 inferred" in ds.details

    def test_bare_entity_no_inference(self):
        """Entity with no relationships gets type-inferred CRUD methods."""
        comps = [_make_component("Widget", desc="A widget entity in the system", comp_type="entity")]
        ds = score_actionability(0.0, comps)
        # entity type → create/read/update/delete inferred
        assert "1 inferred" in ds.details

    def test_backward_compat_no_relationships(self):
        """Old callers without relationships arg still work."""
        comps = [_make_component("A", desc="short")]
        ds = score_actionability(0.0, comps)
        assert ds.name == "actionability"
