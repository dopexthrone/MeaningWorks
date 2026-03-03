"""
Phase 14: Staged Pipeline Tests.

Tests for core/pipeline.py — dataclasses, artifact parsers, gate predicates,
StageExecutor, StagedPipeline orchestrator, and engine integration.
"""

import json
import time
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from core.pipeline import (
    StageResult,
    StageRecord,
    PipelineState,
    # Parsers
    parse_expand_artifact,
    parse_decompose_artifact,
    parse_ground_artifact,
    parse_constrain_artifact,
    parse_architect_artifact,
    _is_likely_state_or_attribute,
    _assign_depths,
    _compute_max_depth,
    # Gates
    gate_expand,
    gate_decompose,
    gate_ground,
    gate_constrain,
    gate_architect,
    # Orchestrator
    StageExecutor,
    StagedPipeline,
    # Helpers
    format_precomputed_structure,
    _build_ground_prime,
    STAGE_CONFIGS,
    STAGE_PROMPTS,
    PIPELINE_GATES,
    ARTIFACT_PARSERS,
    PRIME_BUILDERS,
)
from core.protocol import SharedState, Message, MessageType


# =============================================================================
# HELPERS
# =============================================================================


def _make_state_with_messages(messages):
    """Create a SharedState with given messages."""
    state = SharedState()
    for sender, content in messages:
        msg = Message(
            sender=sender,
            content=content,
            message_type=MessageType.PROPOSITION,
        )
        state.add_message(msg)
    return state


def _make_pipeline_state(input_text="Build a user auth system", nouns=None, verbs=None):
    """Create a basic PipelineState for testing."""
    intent = {
        "core_need": "Build a user auth system",
        "domain": "auth",
        "actors": ["User", "Admin"],
        "constraints": ["Must be secure"],
        "explicit_components": ["User", "Session", "AuthService"],
        "explicit_relationships": [],
        "insight": "Core need is auth",
    }
    return PipelineState(
        original_input=input_text,
        intent=intent,
        personas=[
            {"name": "Architect", "perspective": "Security focus"},
            {"name": "UX", "perspective": "User experience"},
        ],
    )


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_success_result(self):
        r = StageResult(success=True)
        assert r.success
        assert r.errors == []
        assert r.warnings == []

    def test_failure_result(self):
        r = StageResult(success=False, errors=["Too few nouns"])
        assert not r.success
        assert "Too few nouns" in r.errors

    def test_warnings(self):
        r = StageResult(success=True, warnings=["High orphan ratio"])
        assert r.success
        assert len(r.warnings) == 1


class TestStageRecord:
    """Tests for StageRecord dataclass."""

    def test_creation(self):
        state = SharedState()
        r = StageRecord(
            name="expand",
            state=state,
            artifact={"nodes": [{"name": "User", "source": "input", "depth": 0}]},
            gate_result=StageResult(success=True),
            turn_count=4,
            duration_seconds=5.0,
        )
        assert r.name == "expand"
        assert r.turn_count == 4
        assert r.duration_seconds == 5.0
        assert r.artifact["nodes"][0]["name"] == "User"


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_creation(self):
        ps = _make_pipeline_state()
        assert ps.original_input == "Build a user auth system"
        assert len(ps.stages) == 0
        assert len(ps.all_insights) == 0

    def test_add_stage_accumulates_insights(self):
        ps = _make_pipeline_state()
        state = SharedState()
        state.insights = ["User = email + password"]
        record = StageRecord(
            name="expand",
            state=state,
            artifact={},
            gate_result=StageResult(success=True),
            turn_count=2,
            duration_seconds=1.0,
        )
        ps.add_stage(record)
        assert len(ps.stages) == 1
        assert "User = email + password" in ps.all_insights

    def test_add_stage_accumulates_unknowns_dedup(self):
        ps = _make_pipeline_state()
        state1 = SharedState()
        state1.unknown = ["What is session TTL?"]
        state2 = SharedState()
        state2.unknown = ["What is session TTL?", "How to hash?"]

        for name, state in [("expand", state1), ("decompose", state2)]:
            ps.add_stage(StageRecord(
                name=name, state=state, artifact={},
                gate_result=StageResult(success=True),
                turn_count=2, duration_seconds=1.0,
            ))

        assert ps.all_unknowns.count("What is session TTL?") == 1
        assert "How to hash?" in ps.all_unknowns

    def test_add_stage_accumulates_conflicts(self):
        ps = _make_pipeline_state()
        state = SharedState()
        state.conflicts = [{"topic": "User type", "resolved": False}]
        ps.add_stage(StageRecord(
            name="expand", state=state, artifact={},
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        assert len(ps.all_conflicts) == 1

    def test_get_artifact(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={"nodes": [{"name": "User", "source": "", "depth": 0},
                                {"name": "Session", "source": "", "depth": 0}],
                      "nouns": ["User", "Session"]},
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        assert [n["name"] for n in ps.get_artifact("expand")["nodes"]] == ["User", "Session"]
        assert ps.get_artifact("decompose") is None


# =============================================================================
# ARTIFACT PARSER TESTS
# =============================================================================


class TestParseExpandArtifact:
    """Tests for parse_expand_artifact — recursion map parser."""

    def test_parses_nodes_with_source(self):
        state = _make_state_with_messages([
            ("Entity", 'Some analysis.\nNODE: User (source: "user account management")\nINSIGHT: User has credentials'),
        ])
        art = parse_expand_artifact(state)
        names = [n["name"] for n in art["nodes"]]
        assert "User" in names
        assert art["noun_sources"]["User"] == "user account management"

    def test_parses_containment(self):
        state = _make_state_with_messages([
            ("Entity", 'NODE: System (source: "the system")\nNODE: User (source: "user")\n'
                        'CONTAINS: System > User (source: "system has users")\nINSIGHT: containment mapped'),
        ])
        art = parse_expand_artifact(state)
        assert len(art["containment"]) == 1
        assert art["containment"][0]["parent"] == "System"
        assert art["containment"][0]["child"] == "User"
        assert art["containment"][0]["source"] == "system has users"

    def test_parses_simple_nodes(self):
        state = _make_state_with_messages([
            ("Entity", "NODE: Session\nNODE: Token\nINSIGHT: sessions exist"),
        ])
        art = parse_expand_artifact(state)
        names = [n["name"] for n in art["nodes"]]
        assert "Session" in names
        assert "Token" in names

    def test_parses_self_references(self):
        state = _make_state_with_messages([
            ("Process", "SELF_REF: Pipeline | path: Pipeline > Stage > Pipeline | depth: 2\nINSIGHT: recursive"),
        ])
        art = parse_expand_artifact(state)
        assert len(art["self_references"]) == 1
        assert art["self_references"][0]["node"] == "Pipeline"
        assert art["self_references"][0]["path"] == ["Pipeline", "Stage", "Pipeline"]
        assert art["self_references"][0]["depth"] == 2

    def test_deduplicates_nodes(self):
        state = _make_state_with_messages([
            ("Entity", "NODE: User\nINSIGHT: x"),
            ("Entity", "NODE: User\nINSIGHT: y"),
        ])
        art = parse_expand_artifact(state)
        names = [n["name"] for n in art["nodes"]]
        assert names.count("User") == 1

    def test_ignores_non_agent_messages(self):
        state = _make_state_with_messages([
            ("System", "NODE: Fake\nINSIGHT: should not appear"),
        ])
        art = parse_expand_artifact(state)
        assert len(art["nodes"]) == 0

    def test_parses_max_depth(self):
        state = _make_state_with_messages([
            ("Process", "MAX_DEPTH: 3 (reason: \"finite domain\")\nINSIGHT: bounded"),
        ])
        art = parse_expand_artifact(state)
        assert art["max_depth"] == 3

    def test_auto_registers_nodes_from_containment(self):
        state = _make_state_with_messages([
            ("Entity", 'CONTAINS: System > User\nINSIGHT: auto-register'),
        ])
        art = parse_expand_artifact(state)
        names = [n["name"] for n in art["nodes"]]
        assert "System" in names
        assert "User" in names

    def test_backward_compat_nouns_key(self):
        state = _make_state_with_messages([
            ("Entity", 'NODE: User (source: "input")\nNODE: Session (source: "input")\nINSIGHT: x'),
        ])
        art = parse_expand_artifact(state)
        assert "User" in art["nouns"]
        assert "Session" in art["nouns"]

    def test_backward_compat_verbs_empty(self):
        state = _make_state_with_messages([
            ("Entity", 'NODE: User (source: "input")\nINSIGHT: x'),
        ])
        art = parse_expand_artifact(state)
        assert art["verbs"] == []

    def test_assigns_depths_from_containment(self):
        state = _make_state_with_messages([
            ("Entity", 'NODE: System (source: "sys")\nNODE: User (source: "usr")\nNODE: Profile (source: "prof")\n'
                        'CONTAINS: System > User\nCONTAINS: User > Profile\nINSIGHT: x'),
        ])
        art = parse_expand_artifact(state)
        depth_map = {n["name"]: n["depth"] for n in art["nodes"]}
        assert depth_map["System"] == 0
        assert depth_map["User"] == 1
        assert depth_map["Profile"] == 2

    def test_compute_max_depth_from_tree(self):
        nodes = [
            {"name": "A", "source": "", "depth": None},
            {"name": "B", "source": "", "depth": None},
            {"name": "C", "source": "", "depth": None},
        ]
        containment = [
            {"parent": "A", "child": "B", "source": ""},
            {"parent": "B", "child": "C", "source": ""},
        ]
        assert _compute_max_depth(containment) == 2


class TestParseDecomposeArtifact:
    """Tests for parse_decompose_artifact."""

    def test_parses_components(self):
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user account"\nINSIGHT: typed'),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["components"]) == 1
        assert art["components"][0]["name"] == "User"
        assert art["components"][0]["type"] == "entity"

    def test_parses_methods(self):
        state = _make_state_with_messages([
            ("Process", "METHOD: User.login(email: str, password: str) -> bool\nINSIGHT: login method"),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["method_assignments"]) == 1
        assert art["method_assignments"][0]["component"] == "User"
        assert art["method_assignments"][0]["name"] == "login"

    def test_deduplicates_components(self):
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity\nINSIGHT: x'),
            ("Process", 'COMPONENT: User | type=entity\nINSIGHT: y'),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["components"]) == 1


class TestParseGroundArtifact:
    """Tests for parse_ground_artifact."""

    def test_parses_relationships(self):
        state = _make_state_with_messages([
            ("Entity", 'RELATIONSHIP: User -> Session | type=contains | description="User has sessions"\nINSIGHT: x'),
        ])
        art = parse_ground_artifact(state)
        assert len(art["relationships"]) == 1
        assert art["relationships"][0]["from"] == "User"
        assert art["relationships"][0]["to"] == "Session"
        assert art["relationships"][0]["type"] == "contains"

    def test_deduplicates_relationships(self):
        state = _make_state_with_messages([
            ("Entity", 'RELATIONSHIP: A -> B | type=triggers\nINSIGHT: x'),
            ("Process", 'RELATIONSHIP: A -> B | type=triggers\nINSIGHT: y'),
        ])
        art = parse_ground_artifact(state)
        assert len(art["relationships"]) == 1

    def test_identifies_data_flows(self):
        state = _make_state_with_messages([
            ("Process", 'RELATIONSHIP: Auth -> DB | type=reads | description="reads user records"\nINSIGHT: x'),
        ])
        art = parse_ground_artifact(state)
        assert len(art["data_flows"]) == 1


class TestParseConstrainArtifact:
    """Tests for parse_constrain_artifact."""

    def test_parses_constraints(self):
        state = _make_state_with_messages([
            ("Entity", 'CONSTRAINT: User | description="email must be unique" | derived_from="input"\nINSIGHT: x'),
        ])
        art = parse_constrain_artifact(state)
        assert len(art["constraints"]) == 1
        assert art["constraints"][0]["description"] == "email must be unique"
        assert art["constraints"][0]["applies_to"] == ["User"]

    def test_parses_methods_via_digest(self):
        state = _make_state_with_messages([
            ("Process", "METHOD: Auth.validate(token: str) -> bool\nINSIGHT: validation method"),
        ])
        art = parse_constrain_artifact(state)
        assert len(art["methods"]) == 1
        assert art["methods"][0]["component"] == "Auth"
        assert art["methods"][0]["name"] == "validate"


class TestParseArchitectArtifact:
    """Tests for parse_architect_artifact."""

    def test_parses_subsystems(self):
        state = _make_state_with_messages([
            ("Entity", 'SUBSYSTEM: AuthModule | contains=[User, Session, Token] | description="Auth boundary"\nINSIGHT: x'),
        ])
        art = parse_architect_artifact(state)
        assert len(art["subsystems"]) == 1
        assert art["subsystems"][0]["name"] == "AuthModule"
        assert "User" in art["subsystems"][0]["contains"]
        assert "Session" in art["subsystems"][0]["contains"]

    def test_empty_architect(self):
        state = _make_state_with_messages([
            ("Entity", "No subsystems needed.\nINSIGHT: flat architecture"),
        ])
        art = parse_architect_artifact(state)
        assert len(art["subsystems"]) == 0


# =============================================================================
# GATE PREDICATE TESTS
# =============================================================================


class TestGateExpand:
    """Tests for gate_expand — structural containment validation."""

    def _make_expand_artifact(self, nodes=None, containment=None, self_references=None, max_depth=1):
        """Helper to build a valid expand artifact."""
        nodes = nodes or []
        containment = containment or []
        self_references = self_references or []
        return {
            "nodes": nodes,
            "containment": containment,
            "self_references": self_references,
            "max_depth": max_depth,
            "nouns": [n["name"] for n in nodes],
        }

    def test_pass_basic(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "User", "source": "src", "depth": 0},
                {"name": "Session", "source": "src", "depth": 1},
                {"name": "Token", "source": "src", "depth": 1},
            ],
            containment=[
                {"parent": "User", "child": "Session", "source": "src"},
                {"parent": "User", "child": "Token", "source": "src"},
            ],
            max_depth=1,
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert result.success

    def test_fail_too_few_nodes(self):
        artifact = self._make_expand_artifact(
            nodes=[{"name": "User", "source": "", "depth": 0}],
            containment=[
                {"parent": "User", "child": "Session", "source": ""},
                {"parent": "User", "child": "Token", "source": ""},
            ],
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert not result.success
        assert any("Too few nodes" in e for e in result.errors)

    def test_fail_too_few_containment(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 0},
                {"name": "C", "source": "", "depth": 0},
            ],
            containment=[{"parent": "A", "child": "B", "source": ""}],
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert not result.success
        assert any("Too few containment" in e for e in result.errors)

    def test_warn_missing_explicit_component(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "User", "source": "", "depth": 0},
                {"name": "Session", "source": "", "depth": 1},
                {"name": "Token", "source": "", "depth": 1},
            ],
            containment=[
                {"parent": "User", "child": "Session", "source": ""},
                {"parent": "User", "child": "Token", "source": ""},
            ],
        )
        intent = {"explicit_components": ["PaymentGateway"]}
        result = gate_expand(artifact, intent)
        assert result.success  # Still passes, just warns
        assert len(result.warnings) > 0

    def test_warn_orphan_nodes(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 0},
                {"name": "C", "source": "", "depth": 0},
                {"name": "D", "source": "", "depth": 0},
                {"name": "E", "source": "", "depth": 0},
            ],
            containment=[
                {"parent": "A", "child": "B", "source": ""},
                {"parent": "A", "child": "C", "source": ""},
            ],
            max_depth=1,
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert result.success
        assert any("orphan" in w.lower() for w in result.warnings)

    def test_explicit_component_stem_match(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "Users", "source": "", "depth": 0},
                {"name": "Sessions", "source": "", "depth": 1},
                {"name": "Tokens", "source": "", "depth": 1},
            ],
            containment=[
                {"parent": "Users", "child": "Sessions", "source": ""},
                {"parent": "Users", "child": "Tokens", "source": ""},
            ],
        )
        intent = {"explicit_components": ["User"]}
        result = gate_expand(artifact, intent)
        assert result.success
        # "Users" matches "User" via stem

    def test_gate_self_ref_valid_path(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 1},
                {"name": "C", "source": "", "depth": 2},
            ],
            containment=[
                {"parent": "A", "child": "B", "source": ""},
                {"parent": "B", "child": "C", "source": ""},
            ],
            self_references=[
                {"node": "A", "path": ["A", "B", "C", "A"], "depth": 3},
            ],
            max_depth=2,
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert result.success
        # No warnings about self-ref since path loops correctly

    def test_gate_self_ref_degenerate_warning(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 1},
                {"name": "C", "source": "", "depth": 2},
            ],
            containment=[
                {"parent": "A", "child": "B", "source": ""},
                {"parent": "B", "child": "C", "source": ""},
            ],
            self_references=[
                {"node": "A", "path": ["A", "B", "C"], "depth": 3},  # doesn't loop
            ],
            max_depth=2,
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert any("does not loop" in w for w in result.warnings)

    def test_gate_max_depth_zero_warning(self):
        artifact = self._make_expand_artifact(
            nodes=[
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 1},
                {"name": "C", "source": "", "depth": 1},
            ],
            containment=[
                {"parent": "A", "child": "B", "source": ""},
                {"parent": "A", "child": "C", "source": ""},
            ],
            max_depth=0,
        )
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert any("max_depth" in w for w in result.warnings)

    def test_gate_undefined_node_in_containment(self):
        artifact = {
            "nodes": [
                {"name": "A", "source": "", "depth": 0},
                {"name": "B", "source": "", "depth": 1},
                {"name": "C", "source": "", "depth": 1},
            ],
            "containment": [
                {"parent": "A", "child": "B", "source": ""},
                {"parent": "X", "child": "C", "source": ""},  # X not in nodes
            ],
            "self_references": [],
            "max_depth": 1,
            "nouns": ["A", "B", "C"],
        }
        intent = {"explicit_components": []}
        result = gate_expand(artifact, intent)
        assert any("not in declared nodes" in w for w in result.warnings)


class TestGateDecompose:
    """Tests for gate_decompose."""

    def test_pass_basic(self):
        artifact = {
            "components": [
                {"name": "User", "type": "entity", "derived_from": "input"},
                {"name": "Session", "type": "entity", "derived_from": "input"},
                {"name": "Auth", "type": "process", "derived_from": "input"},
            ],
            "type_assignments": {"User": "entity", "Session": "entity", "Auth": "process"},
            "folded": [],
            "interfaces": [],
        }
        expand = {"nouns": ["User", "Session", "Auth"]}
        result = gate_decompose(artifact, expand)
        assert result.success

    def test_fail_too_few_components(self):
        artifact = {
            "components": [{"name": "User", "type": "entity"}],
            "type_assignments": {},
            "folded": [],
            "interfaces": [],
        }
        expand = {"nouns": ["User"]}
        result = gate_decompose(artifact, expand)
        assert not result.success
        assert any("Too few" in e for e in result.errors)

    def test_warn_missing_derivation(self):
        artifact = {
            "components": [
                {"name": "A", "type": "entity", "derived_from": ""},
                {"name": "B", "type": "entity", "derived_from": "x"},
                {"name": "C", "type": "process", "derived_from": "y"},
            ],
            "type_assignments": {},
            "folded": [],
            "interfaces": [],
        }
        expand = {"nouns": []}
        result = gate_decompose(artifact, expand)
        assert result.success
        assert any("derived_from" in w for w in result.warnings)


class TestGateGround:
    """Tests for gate_ground."""

    def test_pass_basic(self):
        artifact = {
            "relationships": [
                {"from": "User", "to": "Session", "type": "contains"},
                {"from": "Auth", "to": "User", "type": "accesses"},
            ],
        }
        decompose = {
            "components": [
                {"name": "User"}, {"name": "Session"}, {"name": "Auth"}
            ]
        }
        result = gate_ground(artifact, decompose)
        assert result.success

    def test_fail_too_few_relationships(self):
        artifact = {"relationships": []}
        decompose = {"components": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        result = gate_ground(artifact, decompose)
        assert not result.success
        assert any("Too few" in e for e in result.errors)

    def test_warn_unknown_endpoints(self):
        artifact = {
            "relationships": [
                {"from": "User", "to": "Unknown", "type": "accesses"},
                {"from": "User", "to": "Session", "type": "contains"},
            ],
        }
        decompose = {"components": [{"name": "User"}, {"name": "Session"}]}
        result = gate_ground(artifact, decompose)
        # May still pass (not enough rels for 2 components though)
        assert any("Unknown" in str(w) for w in result.warnings)

    def test_warn_high_orphan_ratio(self):
        artifact = {
            "relationships": [
                {"from": "A", "to": "B", "type": "triggers"},
            ],
        }
        decompose = {
            "components": [
                {"name": "A"}, {"name": "B"}, {"name": "C"},
                {"name": "D"}, {"name": "E"},
            ]
        }
        result = gate_ground(artifact, decompose)
        assert any("orphan" in w.lower() for w in result.warnings)


class TestGateConstrain:
    """Tests for gate_constrain."""

    def test_always_passes(self):
        """Constrain gate always returns success=True (warnings only)."""
        artifact = {"constraints": [], "methods": [], "state_machines": []}
        decompose = {"components": [{"name": "User"}]}
        intent = {"constraints": []}
        result = gate_constrain(artifact, decompose, intent)
        assert result.success

    def test_warn_unknown_constraint_target(self):
        artifact = {
            "constraints": [{"description": "email unique", "applies_to": ["Unknown"]}],
        }
        decompose = {"components": [{"name": "User"}]}
        intent = {"constraints": []}
        result = gate_constrain(artifact, decompose, intent)
        assert result.success
        assert any("Unknown" in str(w) for w in result.warnings)

    def test_warn_low_constraint_coverage(self):
        artifact = {"constraints": []}
        decompose = {"components": [{"name": "User"}]}
        intent = {"constraints": ["Must be secure", "Must be fast", "Must scale"]}
        result = gate_constrain(artifact, decompose, intent)
        assert result.success
        assert any("coverage" in w.lower() for w in result.warnings)


class TestGateArchitect:
    """Tests for gate_architect."""

    def test_always_passes(self):
        artifact = {"subsystems": []}
        decompose = {"components": []}
        result = gate_architect(artifact, decompose)
        assert result.success

    def test_warn_unknown_subsystem_component(self):
        artifact = {
            "subsystems": [{"name": "AuthModule", "contains": ["Unknown"]}],
        }
        decompose = {"components": [{"name": "User"}]}
        result = gate_architect(artifact, decompose)
        assert result.success
        assert any("Unknown" in str(w) for w in result.warnings)


# =============================================================================
# FORMAT PRECOMPUTED STRUCTURE TESTS
# =============================================================================


class TestFormatPrecomputedStructure:
    """Tests for format_precomputed_structure."""

    def test_empty_pipeline_promotes_canonical(self):
        """Even with no stages, canonical components from intent are promoted."""
        ps = _make_pipeline_state()
        result = format_precomputed_structure(ps)
        # Canonical components from intent are promoted
        assert "User" in result
        assert "Session" in result
        assert "AuthService" in result

    def test_truly_empty_pipeline(self):
        """Pipeline with no intent explicit_components and no stages = empty."""
        ps = PipelineState(
            original_input="test",
            intent={"core_need": "test", "domain": "test", "actors": []},
            personas=[],
        )
        result = format_precomputed_structure(ps)
        assert result == ""

    def test_with_components(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "User", "type": "entity", "derived_from": "input"},
                    {"name": "Auth", "type": "process", "derived_from": "input"},
                ],
            },
            gate_result=StageResult(success=True),
            turn_count=4, duration_seconds=2.0,
        ))
        result = format_precomputed_structure(ps)
        assert "User" in result
        assert "COMPONENTS" in result

    def test_with_relationships(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={"components": [{"name": "A", "type": "entity"}]},
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        ps.add_stage(StageRecord(
            name="ground", state=SharedState(),
            artifact={
                "relationships": [
                    {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B"},
                ],
            },
            gate_result=StageResult(success=True),
            turn_count=4, duration_seconds=2.0,
        ))
        result = format_precomputed_structure(ps)
        assert "RELATIONSHIPS" in result
        assert "triggers" in result


# =============================================================================
# PRIME BUILDER TESTS
# =============================================================================


class TestPrimeBuilders:
    """Tests for prime content builder functions."""

    def test_expand_prime(self):
        ps = _make_pipeline_state()
        prime = PRIME_BUILDERS["expand"](ps)
        assert "Build a user auth system" in prime
        assert "containment" in prime.lower() or "CONTAINMENT" in prime

    def test_decompose_prime_includes_nodes(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={
                "nodes": [
                    {"name": "User", "source": "input", "depth": 0},
                    {"name": "Session", "source": "input", "depth": 1},
                ],
                "containment": [
                    {"parent": "User", "child": "Session", "source": "input"},
                ],
                "self_references": [],
                "max_depth": 1,
                "nouns": ["User", "Session"],
                "verbs": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = PRIME_BUILDERS["decompose"](ps)
        assert "User" in prime
        assert "Session" in prime
        # Should include containment structure
        assert "User > Session" in prime or "containment" in prime.lower()

    def test_ground_prime_includes_components(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "User", "type": "entity"},
                    {"name": "Auth", "type": "process"},
                ],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = PRIME_BUILDERS["ground"](ps)
        assert "User" in prime
        assert "Auth" in prime


# =============================================================================
# STAGE CONFIGS TESTS
# =============================================================================


class TestStageConfigs:
    """Tests for stage configuration constants."""

    def test_five_stages(self):
        assert len(STAGE_CONFIGS) == 5

    def test_stage_names(self):
        names = [s[0] for s in STAGE_CONFIGS]
        assert names == ["expand", "decompose", "ground", "constrain", "architect"]

    def test_all_stages_have_prompts(self):
        for name, _, _, _ in STAGE_CONFIGS:
            assert name in STAGE_PROMPTS
            entity_prompt, process_prompt = STAGE_PROMPTS[name]
            assert "Entity Agent" in entity_prompt
            assert "Process Agent" in process_prompt

    def test_all_stages_have_gates(self):
        for name, _, _, _ in STAGE_CONFIGS:
            assert name in PIPELINE_GATES

    def test_all_stages_have_parsers(self):
        for name, _, _, _ in STAGE_CONFIGS:
            assert name in ARTIFACT_PARSERS

    def test_all_stages_have_prime_builders(self):
        for name, _, _, _ in STAGE_CONFIGS:
            assert name in PRIME_BUILDERS

    def test_total_turns_within_budget(self):
        total = sum(s[1] for s in STAGE_CONFIGS)
        assert total == 32  # Matches current pipeline max_turns sum

    def test_total_timeout_within_budget(self):
        total = sum(s[3] for s in STAGE_CONFIGS)
        assert total == 2100


# =============================================================================
# STAGE EXECUTOR TESTS
# =============================================================================


class TestStageExecutor:
    """Tests for StageExecutor."""

    def _make_mock_llm(self, responses):
        """Create a mock LLM that returns preset responses."""
        mock = Mock()
        mock.complete_with_system = Mock(side_effect=responses)
        return mock

    def test_executor_runs_dialogue(self):
        """Executor runs a sub-dialogue and returns a StageRecord."""
        responses = [
            "Entity analysis.\nNODE: User (source: \"users\")\nNODE: Session (source: \"sessions\")\nNODE: Token (source: \"tokens\")\n"
            "CONTAINS: User > Session (source: \"user has sessions\")\nCONTAINS: User > Token (source: \"user has tokens\")\nINSIGHT: User has sessions",
            "Process analysis.\nMAX_DEPTH: 1 (reason: \"flat domain\")\nINSIGHT: login/logout flow",
            "I agree with the analysis. Sufficient coverage.\nINSIGHT: coverage good",
            "Agreed, sufficient.\nINSIGHT: done",
        ]
        llm = self._make_mock_llm(responses)
        ps = _make_pipeline_state()

        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="Entity prompt",
            process_prompt="Process prompt",
            llm_client=llm,
            max_turns=4,
            min_turns=2,
            gate_fn=gate_expand,
        )

        record = executor.run(ps, "Analyze this system.")
        assert record.name == "expand"
        assert record.turn_count > 0
        assert record.duration_seconds > 0
        assert isinstance(record.artifact, dict)

    def test_executor_detects_agreement_early_termination(self):
        """Executor stops on 2 consecutive agreements after min_turns."""
        responses = [
            "NODE: A (source: \"a\")\nNODE: B (source: \"b\")\nNODE: C (source: \"c\")\n"
            "CONTAINS: A > B (source: \"ab\")\nCONTAINS: A > C (source: \"ac\")\nINSIGHT: x",
            "MAX_DEPTH: 1\nI agree. Sufficient.\nINSIGHT: y",
            "I agree. Agreed. Sufficient.\nINSIGHT: z",
        ]
        llm = self._make_mock_llm(responses)
        ps = _make_pipeline_state()

        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="p1",
            process_prompt="p2",
            llm_client=llm,
            max_turns=6,
            min_turns=2,
            gate_fn=gate_expand,
        )

        record = executor.run(ps, "prime")
        # Should stop at 3 turns due to 2 consecutive agreements (turns 2 and 3)
        assert record.turn_count <= 4


# =============================================================================
# STAGED PIPELINE ORCHESTRATOR TESTS
# =============================================================================


class TestStagedPipeline:
    """Tests for StagedPipeline orchestrator."""

    def _make_mock_llm_for_full_pipeline(self):
        """Create a mock LLM that provides reasonable responses for all 5 stages."""
        expand_responses = [
            "NODE: User (source: \"user account\")\nNODE: Session (source: \"session management\")\nNODE: AuthService (source: \"authentication\")\n"
            "CONTAINS: AuthService > User (source: \"auth manages users\")\nCONTAINS: User > Session (source: \"user has sessions\")\nINSIGHT: three core entities",
            "MAX_DEPTH: 2 (reason: \"AuthService > User > Session\")\nINSIGHT: three core actions",
            "I agree. Sufficient.\nINSIGHT: coverage good",
            "Agreed, sufficient.\nINSIGHT: done",
        ]
        decompose_responses = [
            'COMPONENT: User | type=entity | boundary=[Session] | derived_from="user account"\n'
            'COMPONENT: Session | type=entity | derived_from="session management"\n'
            'COMPONENT: AuthService | type=process | derived_from="authentication"\n'
            'FOLD: Credentials INTO User | reason="credentials is an attribute of user"\n'
            'INSIGHT: three components typed',
            'METHOD: User.login(email: str) -> bool\n'
            'INTERFACE: AuthProtocol | pattern=recursive | connects=[AuthService, User] | derived_from="auth flow"\n'
            'INSIGHT: login method on User',
            "I agree. Sufficient.\nINSIGHT: coverage good",
            "Agreed, sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: converge",
            "Agreed.\nINSIGHT: converge",
        ]
        ground_responses = [
            'RELATIONSHIP: User -> Session | type=contains | description="user has sessions"\n'
            'RELATIONSHIP: AuthService -> User | type=accesses | description="auth reads users"\n'
            'INSIGHT: two relationships found',
            'RELATIONSHIP: AuthService -> Session | type=generates | description="auth creates sessions"\n'
            'INSIGHT: auth creates sessions',
            "I agree. Sufficient.\nINSIGHT: grounding done",
            "Agreed, sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: converge",
            "Agreed.\nINSIGHT: converge",
        ]
        constrain_responses = [
            'CONSTRAINT: User | description="email must be unique" | derived_from="security"\n'
            'METHOD: AuthService.validate(token: str) -> bool\nINSIGHT: validation method',
            "I agree with constraints. Sufficient.\nINSIGHT: constraints complete",
            "Agreed, sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: converge",
        ]
        architect_responses = [
            'SUBSYSTEM: AuthModule | contains=[User, Session, AuthService] | description="auth boundary"\n'
            'INSIGHT: single auth subsystem',
            "I agree. Sufficient.\nINSIGHT: architecture done",
            "Agreed, sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: converge",
        ]

        all_responses = (
            expand_responses
            + decompose_responses
            + ground_responses
            + constrain_responses
            + architect_responses
        )

        mock = Mock()
        mock.complete_with_system = Mock(side_effect=all_responses)
        return mock

    def test_full_pipeline_runs(self):
        """Full pipeline runs all 5 stages and returns PipelineState."""
        llm = self._make_mock_llm_for_full_pipeline()
        ps = _make_pipeline_state()

        pipeline = StagedPipeline(llm_client=llm)
        result = pipeline.run(ps)

        assert len(result.stages) == 5
        assert result.stages[0].name == "expand"
        assert result.stages[1].name == "decompose"
        assert result.stages[2].name == "ground"
        assert result.stages[3].name == "constrain"
        assert result.stages[4].name == "architect"

    def test_insights_accumulated_across_stages(self):
        """Insights from all stages accumulate in PipelineState."""
        llm = self._make_mock_llm_for_full_pipeline()
        ps = _make_pipeline_state()

        pipeline = StagedPipeline(llm_client=llm)
        result = pipeline.run(ps)

        assert len(result.all_insights) >= 2  # Insights accumulated across stages (provenance gate filters most)

    def test_artifacts_available_after_pipeline(self):
        """All stage artifacts are accessible after pipeline completion."""
        llm = self._make_mock_llm_for_full_pipeline()
        ps = _make_pipeline_state()

        pipeline = StagedPipeline(llm_client=llm)
        result = pipeline.run(ps)

        expand = result.get_artifact("expand")
        assert expand is not None
        decompose = result.get_artifact("decompose")
        assert decompose is not None
        ground = result.get_artifact("ground")
        assert ground is not None


# =============================================================================
# ENGINE INTEGRATION TESTS
# =============================================================================


class TestEngineIntegration:
    """Tests for engine.py integration with staged pipeline."""

    def test_engine_accepts_pipeline_mode_param(self):
        """Engine constructor accepts pipeline_mode parameter."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine(llm_client=MockClient(), pipeline_mode="staged")
        assert engine.pipeline_mode == "staged"

    def test_engine_default_pipeline_mode_is_legacy(self):
        """Default pipeline_mode is 'legacy' for backwards compatibility."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine(llm_client=MockClient())
        assert engine.pipeline_mode == "legacy"

    def test_compile_accepts_pipeline_mode_override(self):
        """compile() accepts pipeline_mode parameter to override instance default."""
        from core.llm import MockClient
        # Just verify the parameter signature accepts it
        from core.engine import MotherlabsEngine
        import inspect
        sig = inspect.signature(MotherlabsEngine.compile)
        assert "pipeline_mode" in sig.parameters


# =============================================================================
# API MODEL TESTS
# =============================================================================


class TestAPIModels:
    """Tests for pipeline_mode in API models."""

    def test_compile_request_accepts_pipeline_mode(self):
        from api.models import CompileRequest
        req = CompileRequest(
            description="Build a user auth system",
            pipeline_mode="staged",
        )
        assert req.pipeline_mode == "staged"

    def test_compile_request_default_pipeline_mode_is_none(self):
        from api.models import CompileRequest
        req = CompileRequest(description="Build something good for testing quality gate")
        assert req.pipeline_mode is None


# =============================================================================
# DETERMINISTIC FILTER TESTS
# =============================================================================


class TestStateAttributeFilter:
    """Tests for _is_likely_state_or_attribute deterministic filter."""

    def test_all_caps_with_underscore_is_state(self):
        assert _is_likely_state_or_attribute("AWAITING_INPUT") is True
        assert _is_likely_state_or_attribute("ENTITY_TURN") is True
        assert _is_likely_state_or_attribute("GOVERNOR_CHECK") is True

    def test_known_state_patterns(self):
        assert _is_likely_state_or_attribute("DIALOGUE") is True
        assert _is_likely_state_or_attribute("SYNTHESIS") is True
        assert _is_likely_state_or_attribute("VERIFICATION") is True
        assert _is_likely_state_or_attribute("OUTPUT") is True
        assert _is_likely_state_or_attribute("PROCESS_TURN") is True

    def test_known_attribute_patterns(self):
        assert _is_likely_state_or_attribute("Known") is True
        assert _is_likely_state_or_attribute("Unknown") is True
        assert _is_likely_state_or_attribute("Ontology") is True
        assert _is_likely_state_or_attribute("Personas") is True
        assert _is_likely_state_or_attribute("History") is True
        assert _is_likely_state_or_attribute("Confidence") is True

    def test_real_components_pass_through(self):
        assert _is_likely_state_or_attribute("SharedState") is False
        assert _is_likely_state_or_attribute("Entity Agent") is False
        assert _is_likely_state_or_attribute("Governor Agent") is False
        assert _is_likely_state_or_attribute("ConfidenceVector") is False
        assert _is_likely_state_or_attribute("DialogueProtocol") is False
        assert _is_likely_state_or_attribute("Corpus") is False
        assert _is_likely_state_or_attribute("ConflictOracle") is False
        assert _is_likely_state_or_attribute("Message") is False
        assert _is_likely_state_or_attribute("AuditTrail") is False
        assert _is_likely_state_or_attribute("PatternLearner") is False

    def test_decompose_parser_filters_states(self):
        """DECOMPOSE parser should filter out state/attribute names."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: SharedState | type=entity | derived_from="SharedState"\n'
                        'COMPONENT: Known | type=entity | derived_from="Known"\n'
                        'COMPONENT: AWAITING_INPUT | type=process | derived_from="state"'),
            ("Process", 'COMPONENT: Governor Agent | type=agent | derived_from="agent"\n'
                        'COMPONENT: DIALOGUE | type=process | derived_from="state"'),
        ])
        artifact = parse_decompose_artifact(state)
        names = [c["name"] for c in artifact["components"]]
        assert "SharedState" in names
        assert "Governor Agent" in names
        assert "Known" not in names
        assert "AWAITING_INPUT" not in names
        assert "DIALOGUE" not in names
        assert len(artifact.get("filtered_out", [])) == 3

    def test_precomputed_structure_promotes_undeclared_endpoints(self):
        """format_precomputed_structure promotes relationship endpoints not in DECOMPOSE."""
        ps = _make_pipeline_state()
        # DECOMPOSE has SharedState
        from core.pipeline import StageRecord, StageResult
        dec_state = SharedState()
        dec_record = StageRecord(
            name="decompose",
            state=dec_state,
            artifact={
                "components": [{"name": "SharedState", "type": "entity", "derived_from": "input"}],
                "type_assignments": {"SharedState": "entity"},
                "method_assignments": [],
                "orphan_verbs": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2,
            duration_seconds=1.0,
        )
        # GROUND references ConfidenceVector which isn't in DECOMPOSE
        gnd_state = SharedState()
        gnd_record = StageRecord(
            name="ground",
            state=gnd_state,
            artifact={
                "relationships": [
                    {"from": "SharedState", "to": "ConfidenceVector", "type": "contains", "description": ""},
                    {"from": "SharedState", "to": "DIALOGUE", "type": "contains", "description": ""},
                ],
                "data_flows": [],
                "orphan_components": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2,
            duration_seconds=1.0,
        )
        ps.stages = [dec_record, gnd_record]
        result = format_precomputed_structure(ps)
        # ConfidenceVector should be promoted (real component)
        assert "ConfidenceVector" in result
        # DIALOGUE should NOT be promoted as a component (it's a state name)
        # It may appear in RELATIONSHIPS section, but not in COMPONENTS section
        comp_section = result.split("RELATIONSHIPS")[0] if "RELATIONSHIPS" in result else result
        comp_lines = [l for l in comp_section.split("\n") if "DIALOGUE" in l]
        assert len(comp_lines) == 0


# =============================================================================
# RETRY/RECOVERY TESTS
# =============================================================================


class TestTurnLevelRetry:
    """Tests for transient error retry in StageExecutor."""

    def test_turn_retry_on_provider_error(self):
        """Executor retries a turn when ProviderError is raised."""
        from core.exceptions import ProviderError
        from core.pipeline import _TURN_MAX_RETRIES

        good_response = (
            "Entity analysis.\nNODE: User (source: \"users\")\nNODE: Session (source: \"sessions\")\nNODE: Token (source: \"tokens\")\n"
            "CONTAINS: User > Session (source: \"has\")\nCONTAINS: User > Token (source: \"has\")\nINSIGHT: found entities"
        )
        llm = Mock()
        # First call fails, second succeeds, rest succeed
        llm.complete_with_system = Mock(side_effect=[
            ProviderError("500 Internal Server Error", provider="test"),
            good_response,
            "MAX_DEPTH: 1\nINSIGHT: depth found",
            "I agree. Sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: done",
        ])

        ps = _make_pipeline_state()
        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="Entity prompt",
            process_prompt="Process prompt",
            llm_client=llm,
            max_turns=4,
            min_turns=2,
            gate_fn=gate_expand,
        )

        with patch("core.pipeline.time.sleep"):  # Skip actual delay
            record = executor.run(ps, "Analyze this system.")

        assert record.turn_count > 0
        # LLM was called at least 3 times (1 fail + 1 success for turn 1, then more turns)
        assert llm.complete_with_system.call_count >= 3

    def test_turn_retry_exhausted_raises(self):
        """Executor raises after exhausting turn retries."""
        from core.exceptions import ProviderError
        from core.pipeline import _TURN_MAX_RETRIES

        llm = Mock()
        # All calls fail
        llm.complete_with_system = Mock(
            side_effect=ProviderError("always fails", provider="test")
        )

        ps = _make_pipeline_state()
        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="Entity prompt",
            process_prompt="Process prompt",
            llm_client=llm,
            max_turns=4,
            min_turns=2,
            gate_fn=gate_expand,
        )

        with patch("core.pipeline.time.sleep"):
            with pytest.raises(ProviderError):
                executor.run(ps, "Analyze this system.")

        # Should have tried _TURN_MAX_RETRIES + 1 times
        assert llm.complete_with_system.call_count == _TURN_MAX_RETRIES + 1

    def test_turn_retry_on_timeout_error(self):
        """Executor retries a turn when TimeoutError is raised."""
        from core.exceptions import TimeoutError as MotherlabsTimeout

        llm = Mock()
        llm.complete_with_system = Mock(side_effect=[
            MotherlabsTimeout("timed out"),
            "NODE: A (source: \"a\")\nNODE: B (source: \"b\")\nNODE: C (source: \"c\")\n"
            "CONTAINS: A > B (source: \"ab\")\nCONTAINS: A > C (source: \"ac\")\nINSIGHT: found nodes",
            "MAX_DEPTH: 1\nINSIGHT: depth",
            "I agree. Sufficient.\nINSIGHT: done",
            "Agreed.\nINSIGHT: done",
        ])

        ps = _make_pipeline_state()
        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="p1",
            process_prompt="p2",
            llm_client=llm,
            max_turns=4,
            min_turns=2,
            gate_fn=gate_expand,
        )

        with patch("core.pipeline.time.sleep"):
            record = executor.run(ps, "prime")

        assert record.turn_count > 0

    def test_non_retryable_error_raises_immediately(self):
        """Non-transient errors (ValueError, etc.) propagate immediately."""
        llm = Mock()
        llm.complete_with_system = Mock(side_effect=ValueError("bad input"))

        ps = _make_pipeline_state()
        executor = StageExecutor(
            stage_name="expand",
            entity_prompt="p1",
            process_prompt="p2",
            llm_client=llm,
            max_turns=4,
            min_turns=2,
            gate_fn=gate_expand,
        )

        with pytest.raises(ValueError, match="bad input"):
            executor.run(ps, "prime")

        # Called exactly once — no retry for non-transient errors
        assert llm.complete_with_system.call_count == 1


class TestStageLevelRetry:
    """Tests for stage-level retry in StagedPipeline."""

    def test_stage_retry_on_provider_error(self):
        """StagedPipeline retries entire stage when provider error propagates from executor."""
        from core.exceptions import ProviderError

        # We'll patch StageExecutor.run to fail once then succeed
        expand_record = StageRecord(
            name="expand",
            state=SharedState(),
            artifact={
                "nodes": [
                    {"name": "User", "source": "", "depth": 0},
                    {"name": "Session", "source": "", "depth": 1},
                    {"name": "Token", "source": "", "depth": 1},
                ],
                "containment": [
                    {"parent": "User", "child": "Session", "source": ""},
                    {"parent": "User", "child": "Token", "source": ""},
                ],
                "self_references": [],
                "max_depth": 1,
                "nouns": ["User", "Session", "Token"],
                "verbs": [],
            },
            gate_result=StageResult(success=True),
            turn_count=3,
            duration_seconds=1.0,
        )

        call_count = {"n": 0}

        def mock_executor_run(pipeline, prime_content, retry_context=""):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ProviderError("500 error", provider="test")
            return expand_record

        llm = Mock()
        pipeline_obj = StagedPipeline(llm_client=llm)

        with patch.object(StageExecutor, 'run', side_effect=mock_executor_run), \
             patch("core.pipeline.time.sleep"):
            # Only run expand stage by limiting STAGE_CONFIGS
            with patch("core.pipeline.STAGE_CONFIGS", [("expand", 4, 2, 30)]):
                ps = _make_pipeline_state()
                result = pipeline_obj.run(ps)

        assert len(result.stages) == 1
        assert result.stages[0].name == "expand"
        # Should have been called twice: 1 failure + 1 success
        assert call_count["n"] == 2

    def test_stage_retry_exhausted_raises(self):
        """StagedPipeline raises when both stage attempts fail."""
        from core.exceptions import ProviderError

        llm = Mock()
        pipeline_obj = StagedPipeline(llm_client=llm)

        with patch.object(
            StageExecutor, 'run',
            side_effect=ProviderError("always fails", provider="test")
        ), patch("core.pipeline.time.sleep"):
            with patch("core.pipeline.STAGE_CONFIGS", [("expand", 4, 2, 30)]):
                ps = _make_pipeline_state()
                with pytest.raises(ProviderError, match="always fails"):
                    pipeline_obj.run(ps)

    def test_stage_retry_does_not_catch_non_transient(self):
        """StagedPipeline does not retry non-transient errors like CompilationError."""
        from core.exceptions import CompilationError

        llm = Mock()
        pipeline_obj = StagedPipeline(llm_client=llm)

        with patch.object(
            StageExecutor, 'run',
            side_effect=CompilationError("hollow artifact", stage="expand", error_code="E3003")
        ):
            with patch("core.pipeline.STAGE_CONFIGS", [("expand", 4, 2, 30)]):
                ps = _make_pipeline_state()
                with pytest.raises(CompilationError, match="hollow artifact"):
                    pipeline_obj.run(ps)


# =============================================================================
# NEW DECOMPOSE PARSER TESTS — FOLD, PROMOTE, INTERFACE, BREAK, BOUNDARY
# =============================================================================


class TestDecomposeParserNewLineTypes:
    """Tests for new DECOMPOSE line types: FOLD, PROMOTE, INTERFACE, BREAK, boundary."""

    def test_parses_component_with_boundary(self):
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: AuthService | type=entity | boundary=[User, Session] | derived_from="auth"\nINSIGHT: x'),
        ])
        art = parse_decompose_artifact(state)
        comp = next(c for c in art["components"] if c["name"] == "AuthService")
        assert comp["boundary"] == ["User", "Session"]
        assert comp["type"] in ("entity", "agent")  # Classifier may override type
        assert comp["derived_from"] == "auth"

    def test_parses_fold_lines(self):
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user"\n'
                        'COMPONENT: Session | type=entity | derived_from="session"\n'
                        'COMPONENT: Token | type=entity | derived_from="token"\n'
                        'FOLD: Credentials INTO User | reason="credentials is a field of user"\n'
                        'INSIGHT: folded'),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["folded"]) == 1
        assert art["folded"][0]["child"] == "Credentials"
        assert art["folded"][0]["into"] == "User"
        assert art["folded"][0]["reason"] == "credentials is a field of user"

    def test_parses_promote_lines(self):
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user"\n'
                        'COMPONENT: Session | type=entity | derived_from="session"\n'
                        'PROMOTE: Token | type=entity | derived_from="token management"\n'
                        'INSIGHT: promoted'),
        ])
        art = parse_decompose_artifact(state)
        names = [c["name"] for c in art["components"]]
        assert "Token" in names
        token = next(c for c in art["components"] if c["name"] == "Token")
        assert token["type"] == "entity"
        assert token["derived_from"] == "token management"

    def test_parses_interface_lines(self):
        state = _make_state_with_messages([
            ("Process", 'INTERFACE: AuthProtocol | pattern=recursive | connects=[AuthService, User, Session] | derived_from="auth flow"\n'
                         'COMPONENT: AuthService | type=process | derived_from="auth"\n'
                         'COMPONENT: User | type=entity | derived_from="user"\n'
                         'COMPONENT: Session | type=entity | derived_from="session"\n'
                         'INSIGHT: interface found'),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["interfaces"]) == 1
        iface = art["interfaces"][0]
        assert iface["name"] == "AuthProtocol"
        assert iface["pattern"] == "recursive"
        assert iface["connects"] == ["AuthService", "User", "Session"]
        assert iface["derived_from"] == "auth flow"

    def test_parses_break_lines_logged(self):
        state = _make_state_with_messages([
            ("Process", 'BREAK: Config | reason="config self-reference is incidental"\n'
                         'COMPONENT: User | type=entity | derived_from="user"\n'
                         'COMPONENT: Session | type=entity | derived_from="session"\n'
                         'COMPONENT: Token | type=entity | derived_from="token"\n'
                         'INSIGHT: break logged'),
        ])
        art = parse_decompose_artifact(state)
        # BREAK lines are logged only — not in components, folded, or interfaces
        names = [c["name"] for c in art["components"]]
        assert "Config" not in names
        assert all(f["child"] != "Config" for f in art.get("folded", []))
        assert all(i["name"] != "Config" for i in art.get("interfaces", []))

    def test_fold_not_in_components(self):
        """FOLD lines should NOT appear as components."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user"\n'
                        'COMPONENT: Session | type=entity | derived_from="session"\n'
                        'COMPONENT: Token | type=entity | derived_from="token"\n'
                        'FOLD: Password INTO User | reason="field"\n'
                        'INSIGHT: x'),
        ])
        art = parse_decompose_artifact(state)
        names = [c["name"] for c in art["components"]]
        assert "Password" not in names
        assert art["folded"][0]["child"] == "Password"

    def test_promote_appears_as_component(self):
        """PROMOTE lines become regular components."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: UserService | type=entity | derived_from="user service handles users"\n'
                        'COMPONENT: SessionManager | type=entity | derived_from="session management"\n'
                        'PROMOTE: TokenValidator | type=process | derived_from="token validation process"\n'
                        'INSIGHT: x'),
        ])
        state.known["input"] = "Build UserService for user management, SessionManager for sessions, TokenValidator for tokens"
        art = parse_decompose_artifact(state)
        names = [c["name"] for c in art["components"]]
        assert "TokenValidator" in names
        c = next(comp for comp in art["components"] if comp["name"] == "TokenValidator")
        assert c["type"] in ("process", "agent")  # Classifier may override

    def test_backward_compat_old_component_format(self):
        """Old COMPONENT: format without boundary still parses."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user"\n'
                        'COMPONENT: Session | type=entity | derived_from="session"\n'
                        'COMPONENT: Auth | type=process | derived_from="auth"\n'
                        'INSIGHT: x'),
        ])
        art = parse_decompose_artifact(state)
        assert len(art["components"]) == 3
        assert all("boundary" not in c for c in art["components"])
        # New keys present with empty defaults
        assert art["folded"] == []
        assert art["interfaces"] == []

    def test_dedup_promote_with_component(self):
        """If COMPONENT and PROMOTE emit same name, dedup keeps first."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: User | type=entity | derived_from="user"\n'
                        'COMPONENT: Session | type=entity | derived_from="session"\n'
                        'COMPONENT: Token | type=entity | derived_from="token"\n'
                        'INSIGHT: x'),
            ("Process", 'PROMOTE: User | type=agent | derived_from="user agent"\n'
                         'INSIGHT: y'),
        ])
        art = parse_decompose_artifact(state)
        user_comps = [c for c in art["components"] if c["name"] == "User"]
        assert len(user_comps) == 1
        assert user_comps[0]["type"] == "entity"  # First one wins

    def test_artifact_has_new_keys(self):
        """Artifact always has folded, interfaces, and updated _parse_health."""
        state = _make_state_with_messages([
            ("Entity", 'COMPONENT: A | type=entity | derived_from="a"\n'
                        'COMPONENT: B | type=entity | derived_from="b"\n'
                        'COMPONENT: C | type=entity | derived_from="c"\n'
                        'FOLD: D INTO A | reason="field"\n'
                        'INTERFACE: Proto | pattern=recursive | connects=[A, B]\n'
                        'INSIGHT: x'),
        ])
        art = parse_decompose_artifact(state)
        assert "folded" in art
        assert "interfaces" in art
        assert art["_parse_health"]["folded_count"] == 1
        assert art["_parse_health"]["interfaces_found"] == 1


# =============================================================================
# NEW GATE TESTS — PARENT/SELF-REF ACCOUNTING, FOLD COVERAGE
# =============================================================================


class TestGateDecomposeNewChecks:
    """Tests for new gate_decompose checks: parent accounting, self-ref, fold coverage."""

    def test_gate_warns_missing_parent_nodes(self):
        """Gate warns when parent nodes from EXPAND are not accounted for."""
        artifact = {
            "components": [
                {"name": "User", "type": "entity", "derived_from": "x"},
                {"name": "Session", "type": "entity", "derived_from": "x"},
                {"name": "Token", "type": "entity", "derived_from": "x"},
            ],
            "type_assignments": {"User": "entity", "Session": "entity", "Token": "entity"},
            "folded": [],
            "interfaces": [],
        }
        expand = {
            "nouns": ["User", "Session", "Token", "AuthService"],
            "containment": [
                {"parent": "AuthService", "child": "User", "source": ""},
                {"parent": "User", "child": "Session", "source": ""},
            ],
        }
        result = gate_decompose(artifact, expand)
        assert result.success  # Warns, not errors
        assert any("Unaccounted parent" in w for w in result.warnings)

    def test_gate_warns_unaddressed_self_references(self):
        """Gate warns when self-references from EXPAND are not addressed."""
        artifact = {
            "components": [
                {"name": "A", "type": "entity", "derived_from": "x"},
                {"name": "B", "type": "entity", "derived_from": "x"},
                {"name": "C", "type": "entity", "derived_from": "x"},
            ],
            "type_assignments": {"A": "entity", "B": "entity", "C": "entity"},
            "folded": [],
            "interfaces": [],  # No interfaces — self-refs unaddressed
        }
        expand = {
            "nouns": ["A", "B", "C"],
            "self_references": [
                {"node": "Pipeline", "path": ["Pipeline", "Stage", "Pipeline"], "depth": 2},
            ],
        }
        result = gate_decompose(artifact, expand)
        assert result.success
        assert any("Unaddressed self-reference" in w for w in result.warnings)

    def test_gate_node_coverage_includes_folded(self):
        """Folded nodes count toward node coverage."""
        artifact = {
            "components": [
                {"name": "User", "type": "entity", "derived_from": "x"},
                {"name": "Session", "type": "entity", "derived_from": "x"},
                {"name": "Token", "type": "entity", "derived_from": "x"},
            ],
            "type_assignments": {"User": "entity", "Session": "entity", "Token": "entity"},
            "folded": [
                {"child": "Password", "into": "User", "reason": "field"},
                {"child": "Email", "into": "User", "reason": "field"},
            ],
            "interfaces": [],
        }
        expand = {
            "nouns": ["User", "Session", "Token", "Password", "Email"],
        }
        result = gate_decompose(artifact, expand)
        assert result.success
        # All 5 nouns accounted for (3 components + 2 folded = 100%)
        assert not any("node coverage" in w.lower() for w in result.warnings)

    def test_gate_passes_with_fold_accounting(self):
        """Gate passes when parent nodes are accounted for via FOLD."""
        artifact = {
            "components": [
                {"name": "User", "type": "entity", "derived_from": "x"},
                {"name": "Session", "type": "entity", "derived_from": "x"},
                {"name": "Auth", "type": "process", "derived_from": "x"},
            ],
            "type_assignments": {"User": "entity", "Session": "entity", "Auth": "process"},
            "folded": [
                {"child": "OldParent", "into": "Auth", "reason": "merged"},
            ],
            "interfaces": [],
        }
        expand = {
            "nouns": ["User", "Session", "Auth"],
            "containment": [
                {"parent": "Auth", "child": "User", "source": ""},
            ],
        }
        result = gate_decompose(artifact, expand)
        assert result.success
        # Auth is in components, so no parent warning
        assert not any("Unaccounted parent" in w for w in result.warnings)

    def test_gate_backward_compat_no_containment(self):
        """Gate works when EXPAND has no containment (old format)."""
        artifact = {
            "components": [
                {"name": "A", "type": "entity", "derived_from": "x"},
                {"name": "B", "type": "entity", "derived_from": "x"},
                {"name": "C", "type": "entity", "derived_from": "x"},
            ],
            "type_assignments": {"A": "entity", "B": "entity", "C": "entity"},
        }
        expand = {"nouns": ["A", "B", "C"]}
        result = gate_decompose(artifact, expand)
        assert result.success


# =============================================================================
# NEW PRIME/FORMAT TESTS
# =============================================================================


class TestDecomposePrimeNewFeatures:
    """Tests for enhanced _build_decompose_prime()."""

    def test_decompose_prime_includes_self_references(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={
                "nodes": [
                    {"name": "Pipeline", "source": "pipeline desc", "depth": 0},
                    {"name": "Stage", "source": "stage desc", "depth": 1},
                ],
                "containment": [
                    {"parent": "Pipeline", "child": "Stage", "source": ""},
                ],
                "self_references": [
                    {"node": "Pipeline", "path": ["Pipeline", "Stage", "Pipeline"], "depth": 2},
                ],
                "max_depth": 2,
                "nouns": ["Pipeline", "Stage"],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = PRIME_BUILDERS["decompose"](ps)
        assert "SELF-REFERENCES" in prime
        assert "Pipeline" in prime
        assert "1 loops" in prime

    def test_decompose_prime_includes_node_sources(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={
                "nodes": [
                    {"name": "User", "source": "user account management", "depth": 0},
                    {"name": "Session", "source": "session handling", "depth": 1},
                    {"name": "Token", "source": "auth tokens", "depth": 1},
                ],
                "containment": [
                    {"parent": "User", "child": "Session", "source": ""},
                    {"parent": "User", "child": "Token", "source": ""},
                ],
                "self_references": [],
                "max_depth": 1,
                "nouns": ["User", "Session", "Token"],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = PRIME_BUILDERS["decompose"](ps)
        assert "user account management" in prime
        assert "session handling" in prime
        assert "CRYSTALLIZE BOUNDARIES" in prime


class TestFormatPrecomputedNewFeatures:
    """Tests for folded/interfaces in format_precomputed_structure."""

    def test_format_includes_folded_section(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "User", "type": "entity", "derived_from": "input"},
                    {"name": "Auth", "type": "process", "derived_from": "input"},
                ],
                "folded": [
                    {"child": "Password", "into": "User", "reason": "field of user"},
                ],
                "interfaces": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        result = format_precomputed_structure(ps)
        assert "FOLDED ATTRIBUTES" in result
        assert "Password" in result
        assert "User" in result

    def test_format_includes_interfaces_section(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "A", "type": "entity", "derived_from": "input"},
                    {"name": "B", "type": "process", "derived_from": "input"},
                ],
                "folded": [],
                "interfaces": [
                    {"name": "AuthProto", "pattern": "recursive", "connects": ["A", "B"], "derived_from": "x"},
                ],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        result = format_precomputed_structure(ps)
        assert "INTERFACES" in result
        assert "AuthProto" in result
        assert "recursive" in result

    def test_format_includes_boundary_on_components(self):
        ps = _make_pipeline_state()
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "Auth", "type": "entity", "derived_from": "input",
                     "boundary": ["User", "Session"]},
                ],
                "folded": [],
                "interfaces": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        result = format_precomputed_structure(ps)
        assert "boundary=[User, Session]" in result


# =============================================================================
# Case-insensitive dedup in parse_expand_artifact
# =============================================================================

class TestExpandCaseDedup:
    """Verify that parse_expand_artifact collapses case-variant nodes."""

    def test_dedup_nodes_by_case(self):
        """'tasks' and 'Tasks' should collapse into one node."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                'NODE: tasks (source: "projects contain tasks") [depth: 1]\n'
                'NODE: Tasks (source: "Tasks subsystem") [depth: 1]\n'
                'NODE: projects (source: "projects") [depth: 0]\n'
                'CONTAINS: projects > tasks\n'
                'CONTAINS: projects > Tasks\n'
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        names = [n["name"] for n in artifact["nodes"]]
        # Should have 2 unique nodes, not 3
        assert len(names) == 2
        assert "tasks" in names
        assert "projects" in names
        assert "Tasks" not in names

    def test_dedup_containment_edges(self):
        """Duplicate containment edges after case-dedup should collapse."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                'NODE: projects (source: "projects") [depth: 0]\n'
                'NODE: tasks (source: "tasks") [depth: 1]\n'
                'NODE: Tasks (source: "Tasks") [depth: 1]\n'
                'CONTAINS: projects > tasks\n'
                'CONTAINS: projects > Tasks\n'
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        assert len(artifact["containment"]) == 1
        assert artifact["containment"][0]["parent"] == "projects"
        assert artifact["containment"][0]["child"] == "tasks"

    def test_dedup_self_references(self):
        """Duplicate self-references after case-dedup should collapse."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                'NODE: tasks (source: "tasks") [depth: 1]\n'
                'NODE: Tasks (source: "Tasks") [depth: 1]\n'
                'NODE: subtasks (source: "subtasks") [depth: 2]\n'
                'CONTAINS: tasks > subtasks\n'
                'SELF_REF: tasks | path: tasks > subtasks > tasks | depth: 2\n'
                'SELF_REF: Tasks | path: Tasks > subtasks > Tasks | depth: 2\n'
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        assert len(artifact["self_references"]) == 1
        assert artifact["self_references"][0]["node"] == "tasks"

    def test_dedup_preserves_source(self):
        """When deduping, keep source from the first-seen node."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                'NODE: Comments (source: "each task has comments") [depth: 2]\n'
                'NODE: comments (source: "comments section") [depth: 2]\n'
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        assert len(artifact["nodes"]) == 1
        assert artifact["nodes"][0]["name"] == "Comments"
        assert artifact["nodes"][0]["source"] == "each task has comments"

    def test_nouns_list_reflects_dedup(self):
        """The backward-compat 'nouns' list should also be deduped."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                'NODE: teams (source: "teams") [depth: 0]\n'
                'NODE: Teams (source: "Teams") [depth: 0]\n'
                'NODE: users (source: "users") [depth: 1]\n'
                'CONTAINS: teams > users\n'
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        assert len(artifact["nouns"]) == 2


# =============================================================================
# Case-insensitive self-ref accounting in gate_decompose
# =============================================================================

class TestGateDecomposeSelfRefCaseInsensitive:
    """Verify self-reference accounting uses case-insensitive matching."""

    def test_interface_connects_covers_self_ref(self):
        """INTERFACE connects=['Tasks', 'Subtasks'] should cover self-ref 'tasks'."""
        artifact = {
            "components": [
                {"name": "Tasks", "type": "subsystem", "derived_from": "tasks"},
                {"name": "Subtasks", "type": "entity", "derived_from": "subtasks"},
                {"name": "Projects", "type": "subsystem", "derived_from": "projects"},
            ],
            "type_assignments": {"Tasks": "subsystem", "Subtasks": "entity", "Projects": "subsystem"},
            "folded": [],
            "interfaces": [
                {"name": "SubtaskRecursion", "pattern": "recursive",
                 "connects": ["Tasks", "Subtasks"], "derived_from": "tasks recurse"},
            ],
            "classification_scores": [],
            "classification_rejected": [],
        }
        expand_artifact = {
            "nouns": ["tasks", "subtasks", "projects"],
            "containment": [],
            "self_references": [
                {"node": "tasks", "path": ["tasks", "subtasks", "tasks"], "depth": 2},
            ],
        }
        result = gate_decompose(artifact, expand_artifact)
        assert result.success
        # Should NOT warn about unaddressed self-refs
        sr_warnings = [w for w in result.warnings if "Unaddressed self-ref" in w]
        assert len(sr_warnings) == 0

    def test_dedup_self_refs_by_case(self):
        """Self-refs 'tasks' and 'Tasks' should only count once."""
        artifact = {
            "components": [
                {"name": "Tasks", "type": "subsystem", "derived_from": "tasks"},
                {"name": "Subtasks", "type": "entity", "derived_from": "subtasks"},
                {"name": "Projects", "type": "subsystem", "derived_from": "projects"},
            ],
            "type_assignments": {},
            "folded": [],
            "interfaces": [],
            "classification_scores": [],
            "classification_rejected": [],
        }
        expand_artifact = {
            "nouns": ["tasks", "subtasks", "projects"],
            "containment": [],
            "self_references": [
                {"node": "tasks", "path": ["tasks", "subtasks", "tasks"], "depth": 2},
                {"node": "Tasks", "path": ["Tasks", "Subtasks", "Tasks"], "depth": 2},
            ],
        }
        result = gate_decompose(artifact, expand_artifact)
        # Both should be reported as one unaddressed (no interface)
        sr_warnings = [w for w in result.warnings if "Unaddressed self-ref" in w]
        if sr_warnings:
            # Should mention only one node, not two
            assert "tasks" in sr_warnings[0].lower()


# =============================================================================
# GROUND prime includes folded names
# =============================================================================

class TestGroundPrimeFoldedNames:
    """Verify _build_ground_prime includes folded name warnings."""

    def test_includes_folded_names(self):
        ps = PipelineState(
            original_input="Build a project manager with tasks and subtasks",
            intent={"explicit_components": []},
            personas=[],
        )
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={"nodes": [], "containment": [], "self_references": []},
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "Tasks", "type": "subsystem", "derived_from": "tasks"},
                    {"name": "Subtasks", "type": "entity", "derived_from": "subtasks"},
                ],
                "type_assignments": {"Tasks": "subsystem", "Subtasks": "entity"},
                "folded": [
                    {"child": "tasks", "into": "Tasks", "reason": "case variant"},
                    {"child": "sub-tasks", "into": "Subtasks", "reason": "case variant"},
                ],
                "interfaces": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = _build_ground_prime(ps)
        assert "tasks" in prime and "Tasks" in prime  # folded name mentioned
        assert "IMPORTANT" in prime  # fold warning present
        assert "Do NOT reference" in prime

    def test_no_fold_hint_when_empty(self):
        ps = PipelineState(
            original_input="Build something simple",
            intent={"explicit_components": []},
            personas=[],
        )
        ps.add_stage(StageRecord(
            name="expand", state=SharedState(),
            artifact={"nodes": [], "containment": [], "self_references": []},
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        ps.add_stage(StageRecord(
            name="decompose", state=SharedState(),
            artifact={
                "components": [
                    {"name": "Widget", "type": "entity", "derived_from": "widget"},
                ],
                "type_assignments": {},
                "folded": [],
                "interfaces": [],
            },
            gate_result=StageResult(success=True),
            turn_count=2, duration_seconds=1.0,
        ))
        prime = _build_ground_prime(ps)
        assert "IMPORTANT" not in prime  # no fold hint when no folds
