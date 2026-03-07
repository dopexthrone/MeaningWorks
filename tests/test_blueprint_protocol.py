import pytest
from pydantic import ValidationError

from core.blueprint_protocol import (
    BLUEPRINTS_SSOT_VERSION,
    BlueprintMetadata,
    BlueprintNode,
    CompiledBlueprint,
    CompiledEntity,
    CompiledFunction,
    CompiledRelationship,
    CompiledRule,
    CompiledStateMachine,
    CompiledTest,
    ContextBudget,
    DepthReport,
    FailedPaths,
    FreshnessPolicy,
    GovernanceReport,
    IntentContract,
    LayerCoverage,
    build_blueprint_semantic_gates,
    build_blueprint_semantic_nodes,
    build_semantic_gate_escalations,
    make_node_ref,
    NodeProvenance,
    normalize_node_name,
    project_legacy_blueprint_nodes,
    RuntimeAnchor,
    parse_node_ref,
    parse_postcode,
)


def _intent_contract() -> IntentContract:
    return IntentContract(
        seed_text="Build a semantic compiler workbench",
        goals=["Compile intent into deterministic blueprints"],
        constraints=["Preserve provenance"],
        layers_in_scope=["INT", "SEM", "EXC", "OBS"],
        domains_in_scope=["SFT", "COG"],
        known_unknowns=["Final renderer bridge"],
        budget_limit=50.0,
        anti_goals=[
            {
                "description": "Do not hide unresolved ambiguity",
                "derived_from": "Compile intent into deterministic blueprints",
                "severity": "critical",
                "detection": "Gap count grows while UI reports completion",
            }
        ],
        runtime_anchor={
            "enabled": True,
            "invariants": ["Blueprint is sole source of truth"],
            "check_mode": "continuous",
        },
        context_budget={
            "total": 200000,
            "reserved": 20000,
            "available": 180000,
            "per_agent": 25714,
            "compression_trigger": 0.85,
        },
        seed_hash="seed-hash",
        contract_hash="contract-hash",
    )


def _node_provenance() -> NodeProvenance:
    return NodeProvenance(
        source_ref=["user:intake"],
        agent_id="intent",
        run_id="run-1",
        timestamp="2026-03-07T22:00:00Z",
        human_input=True,
    )


class TestPostcode:
    def test_parse_postcode(self):
        postcode = parse_postcode("EXC.FNC.CMP.HOW.SFT")
        assert postcode.layer == "EXC"
        assert postcode.concern == "FNC"
        assert postcode.scope == "CMP"
        assert postcode.dimension == "HOW"
        assert postcode.domain == "SFT"
        assert postcode.key == "EXC.FNC.CMP.HOW.SFT"
        assert postcode.depth == 4

    def test_rejects_unknown_codes(self):
        with pytest.raises(ValueError):
            parse_postcode("ZZZ.FNC.CMP.HOW.SFT")

        with pytest.raises(ValueError):
            parse_postcode("EXC.FNC.CMP.HOW.BAD")

    def test_security_layer_is_supported(self):
        postcode = parse_postcode("SEC.PLY.APP.HOW.SFT")
        assert postcode.layer == "SEC"

    def test_parse_node_ref(self):
        node_ref = parse_node_ref("EXC.FNC.CMP.HOW.SFT/divide")
        assert node_ref.postcode == "EXC.FNC.CMP.HOW.SFT"
        assert node_ref.name == "divide"
        assert node_ref.key == "EXC.FNC.CMP.HOW.SFT/divide"

    def test_make_node_ref_normalizes_name(self):
        assert normalize_node_name("Division Guard") == "division_guard"
        assert make_node_ref("CTR.PLY.APP.HOW.SFT", "Division Guard") == "CTR.PLY.APP.HOW.SFT/division_guard"


class TestBlueprintNode:
    def test_node_derives_axes_from_postcode(self):
        node = BlueprintNode(
            id="node-1",
            postcode="EXC.FNC.CMP.HOW.SFT",
            primitive="compile_stage",
            description="Executes a constrained compilation step.",
            notes=["Derived from execution layer"],
            fill_state="F",
            confidence=0.92,
            status="promoted",
            version=1,
            created_at="2026-03-07T22:00:00Z",
            updated_at="2026-03-07T22:05:00Z",
            last_verified="2026-03-07T22:05:00Z",
            freshness=FreshnessPolicy(),
            provenance=_node_provenance(),
            token_cost=1800,
            references={
                "read_before": ["INT.SEM.APP.WHY.SFT/purpose"],
                "read_after": ["OBS.TRC.CMP.WHAT.SFT/trace_surface"],
                "see_also": [],
                "deep_dive": ["doc:workbench"],
                "warns": ["CTR.LMT.CMP.IF.SFT/numeric_bounds"],
            },
        )

        assert node.layer == "EXC"
        assert node.concern == "FNC"
        assert node.scope == "CMP"
        assert node.dimension == "HOW"
        assert node.domain == "SFT"
        assert node.depth == 4

    def test_node_rejects_axis_conflict(self):
        with pytest.raises(ValidationError):
            BlueprintNode(
                id="node-1",
                postcode="EXC.FNC.CMP.HOW.SFT",
                primitive="compile_stage",
                description="Executes a constrained compilation step.",
                notes=[],
                fill_state="F",
                confidence=0.92,
                status="promoted",
                version=1,
                created_at="2026-03-07T22:00:00Z",
                updated_at="2026-03-07T22:05:00Z",
                last_verified="2026-03-07T22:05:00Z",
                freshness=FreshnessPolicy(),
                provenance=_node_provenance(),
                token_cost=1800,
                connections=["STR.ENT.APP.WHAT.SFT/expression"],
                layer="INT",
            )

    def test_effective_confidence_respects_floor(self):
        node = BlueprintNode(
            id="node-2",
            postcode="OBS.TRC.CMP.WHAT.SFT",
            primitive="trace_surface",
            description="Captures provenance output.",
            notes=[],
            fill_state="P",
            confidence=0.8,
            status="candidate",
            version=1,
            created_at="2026-03-07T22:00:00Z",
            updated_at="2026-03-07T22:05:00Z",
            last_verified="2026-03-07T22:05:00Z",
            freshness=FreshnessPolicy(decay_rate=0.1, floor=0.6, stale_after=90),
            provenance=_node_provenance(),
            token_cost=900,
        )

        assert node.effective_confidence(days_since_verified=1) == pytest.approx(0.7)
        assert node.effective_confidence(days_since_verified=4) == pytest.approx(0.6)


class TestCompiledBlueprint:
    def test_compiled_blueprint_validates(self):
        blueprint = CompiledBlueprint(
            metadata=BlueprintMetadata(
                id="bp-1",
                seed="Build Motherlabs workbench",
                seed_hash="seed-hash",
                blueprint_hash="blueprint-hash",
                created_at="2026-03-07T23:00:00Z",
                version=BLUEPRINTS_SSOT_VERSION,
                compilation_depth=DepthReport(label="standard", average_scope_depth=5.4),
            ),
            intent_contract=_intent_contract(),
            layers=[
                LayerCoverage(
                    layer="EXC",
                    nodeCount=2,
                    coverage=["EXC.FNC.CMP.HOW.SFT", "EXC.FLW.FET.HOW.SFT"],
                )
            ],
            entities=[
                CompiledEntity(
                    name="BlueprintNode",
                    postcode="STR.ENT.CMP.WHAT.SFT",
                    description="Addressable semantic node.",
                    attributes=["postcode", "fill_state", "references"],
                    confidence=0.96,
                )
            ],
            functions=[
                CompiledFunction(
                    name="parse_postcode",
                    postcode="EXC.FNC.FNC.HOW.SFT",
                    description="Parses postcode coordinates.",
                    inputs=["raw"],
                    outputs=["Postcode"],
                    rules=["5-axis strict parsing"],
                    confidence=0.94,
                )
            ],
            rules=[
                CompiledRule(
                    name="Blueprint Supremacy",
                    postcode="CTR.PLY.APP.IF.SFT",
                    description="Renderer obeys the blueprint.",
                    type="policy",
                    confidence=0.95,
                )
            ],
            relationships=[
                CompiledRelationship(
                    **{
                        "from": "STR.ENT.CMP.WHAT.SFT",
                        "to": "EXC.FNC.FNC.HOW.SFT",
                        "relation": "constrained_by",
                        "postcode": "SEM.REL.FNC.IF.SFT",
                        "confidence": 0.9,
                    }
                )
            ],
            state_machines=[
                CompiledStateMachine(
                    name="CompilationLifecycle",
                    postcode="STA.STA.APP.WHEN.SFT",
                    states=["idle", "intent_phase", "complete"],
                    transitions=[{"from": "idle", "to": "intent_phase", "trigger": "start"}],
                )
            ],
            silences=[{"layer": "NET", "reason": "No external APIs in scope", "type": "out_of_scope", "decided_by": "intent_contract"}],
            gaps=[{"layer": "RES", "concern": "MET", "scope": "APP", "reason": "Cost benchmark not compiled yet", "priority": "medium"}],
            failed_paths=FailedPaths(
                rejected=[{"postcode": "EMG.CND.FET.WHAT.COG", "reason": "Not approved"}],
                deferred=[{"postcode": "NET.FLW.DOM.HOW.SFT", "reason": "Deferred to later release"}],
                conflicts=[{"nodes": ["STR.ENT.CMP.WHAT.SFT", "DAT.SCH.CMP.WHAT.SFT"], "resolution": "Keep schema local to component"}],
            ),
            governance_report=GovernanceReport(
                total_nodes=24,
                promoted=18,
                quarantined=[{"postcode": "EMG.CND.FET.WHAT.COG", "reason": "Speculation without approval"}],
                escalated=[{"postcode": "RES.MET.APP.HOW_MUCH.ECN", "question": "Increase budget?", "answer": "yes"}],
                axiom_violations=[],
                human_decisions=[{"postcode": "RES.MET.APP.HOW_MUCH.ECN", "question": "Increase budget?", "answer": "yes", "timestamp": "2026-03-07T23:05:00Z"}],
                coverage=91.0,
                anti_goals_checked=3,
            ),
            tests=[
                CompiledTest(
                    source_postcode="CTR.PLY.APP.IF.SFT",
                    test_type="rule",
                    description="Renderer refuses out-of-blueprint writes.",
                    assertion="All emitted files cite a source postcode.",
                )
            ],
            )

        assert blueprint.metadata.version == "1.0.0"
        assert blueprint.governance_report.coverage == pytest.approx(91.0)
        assert blueprint.relationships[0].from_postcode == "STR.ENT.CMP.WHAT.SFT"

    def test_intent_contract_rejects_unknown_domain(self):
        with pytest.raises(ValidationError):
            IntentContract(
                seed_text="x",
                goals=[],
                constraints=[],
                layers_in_scope=["INT"],
                domains_in_scope=["BAD"],
                known_unknowns=[],
                budget_limit=1.0,
                anti_goals=[],
                runtime_anchor=RuntimeAnchor(enabled=False, invariants=[], check_mode="startup"),
                context_budget=ContextBudget(
                    total=1,
                    reserved=0,
                    available=1,
                    per_agent=1,
                    compression_trigger=0.5,
                ),
                seed_hash="s",
                contract_hash="c",
            )


class TestLegacyProjection:
    def test_projects_legacy_blueprint_to_semantic_nodes(self):
        nodes = project_legacy_blueprint_nodes(
            {
                "components": [
                    {
                        "name": "AuthService",
                        "type": "entity",
                        "description": "Handles authentication",
                        "derived_from": "Build secure auth",
                        "attributes": {"provider": "email"},
                        "methods": [],
                        "validation_rules": ["Must validate token"],
                    },
                    {
                        "name": "AuthenticateUser",
                        "type": "process",
                        "description": "Checks credentials and creates a session",
                        "derived_from": "Build secure auth",
                        "attributes": {},
                        "methods": [{"name": "run", "parameters": [], "return_type": "Session", "description": "", "derived_from": "Build secure auth"}],
                        "validation_rules": [],
                    },
                ],
                "relationships": [
                    {
                        "from_component": "AuthService",
                        "to_component": "AuthenticateUser",
                        "type": "depends_on",
                        "description": "AuthenticateUser uses AuthService",
                        "derived_from": "Build secure auth",
                    }
                ],
                "constraints": [
                    {
                        "description": "Must support secure token verification",
                        "applies_to": ["AuthService"],
                        "derived_from": "Build secure auth",
                    }
                ],
                "core_need": "Build secure auth",
                "unresolved": ["AuthService needs provider fallback"],
            },
            seed_text="Build secure auth",
            trust={"overall_score": 87.0, "gap_report": ["AuthService needs provider fallback"]},
            verification={"completeness": {"score": 80}},
            run_id="run-42",
        )

        refs = {make_node_ref(node.postcode, node.primitive): node for node in nodes}

        assert "INT.SEM.APP.WHY.SFT/purpose" in refs
        assert "STR.ENT.APP.WHAT.SFT/authservice" in refs
        assert "EXC.FNC.APP.HOW.SFT/authenticateuser" in refs

        auth = refs["STR.ENT.APP.WHAT.SFT/authservice"]
        flow = refs["EXC.FNC.APP.HOW.SFT/authenticateuser"]

        assert auth.fill_state == "B"
        assert auth.references.read_before == ["INT.SEM.APP.WHY.SFT/purpose"]
        assert flow.references.read_before == ["STR.ENT.APP.WHAT.SFT/authservice"]
        assert auth.constraints[0].description == "Must validate token"
        assert any("secure token verification" in constraint.description for constraint in auth.constraints)
        assert flow.references.read_after == []

    def test_prefers_native_blueprint_semantic_nodes_when_present(self):
        nodes = build_blueprint_semantic_nodes(
            {
                "components": [
                    {
                        "name": "AuthService",
                        "type": "entity",
                        "description": "Handles authentication",
                        "derived_from": "Build secure auth",
                        "attributes": {},
                        "methods": [],
                        "validation_rules": [],
                    }
                ],
                "relationships": [],
                "constraints": [],
                "semantic_nodes": [
                    {
                        "postcode": "EXC.FNC.APP.HOW.SFT",
                        "primitive": "authenticate",
                        "description": "Validate credentials and issue session access.",
                        "fill_state": "F",
                        "confidence": 0.93,
                        "connections": ["STR.ENT.APP.WHAT.SFT/authservice"],
                        "source_ref": ["Build secure auth"],
                    }
                ],
            },
            seed_text="Build secure auth",
            run_id="native-nodes",
        )

        refs = {make_node_ref(node.postcode, node.primitive): node for node in nodes}
        assert "EXC.FNC.APP.HOW.SFT/authenticate" in refs
        assert "INT.SEM.APP.WHY.SFT/purpose" in refs

    def test_builds_conflict_escalations_from_context_graph(self):
        blueprint = {
            "components": [
                {
                    "name": "AuthService",
                    "type": "entity",
                    "description": "Handles authentication",
                    "derived_from": "Build secure auth",
                    "attributes": {},
                    "methods": [],
                    "validation_rules": [],
                }
            ],
            "relationships": [],
            "constraints": [],
            "core_need": "Build secure auth",
            "unresolved": [],
        }
        nodes = project_legacy_blueprint_nodes(
            blueprint,
            seed_text="Build secure auth",
            trust={"overall_score": 84.0, "gap_report": []},
            verification={"completeness": {"score": 80}},
            run_id="run-43",
        )

        blueprint["semantic_gates"] = build_blueprint_semantic_gates(
            blueprint,
            trust={"gap_report": []},
            context_graph={
                "conflict_summary": {
                    "unresolved": [
                        {
                            "topic": "AuthService: storage strategy",
                            "category": "MISSING_INFO",
                            "positions": {
                                "Entity": "Persist sessions in PostgreSQL",
                                "Process": "Keep sessions stateless with JWT",
                            },
                        }
                    ]
                }
            },
        )

        escalations = build_semantic_gate_escalations(
            nodes,
            blueprint=blueprint,
            trust={"gap_report": []},
            context_graph={},
        )

        assert escalations[0]["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert "storage strategy" in escalations[0]["question"].lower()
        assert escalations[0]["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert escalations[0]["kind"] == "semantic_conflict"
        assert escalations[0]["options"] == [
            "Persist sessions in PostgreSQL",
            "Keep sessions stateless with JWT",
        ]

    def test_builds_native_blueprint_semantic_gates(self):
        gates = build_blueprint_semantic_gates(
            {
                "components": [
                    {
                        "name": "AuthService",
                        "type": "entity",
                        "description": "Handles authentication",
                        "derived_from": "Build secure auth",
                        "attributes": {},
                        "methods": [],
                        "validation_rules": [],
                    }
                ],
                "relationships": [],
                "constraints": [],
                "unresolved": ["AuthService needs provider fallback"],
            },
            trust={"gap_report": ["AuthService needs provider fallback"]},
            context_graph={},
        )

        assert gates[0]["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert gates[0]["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert gates[0]["kind"] == "gap"

    def test_normalizes_model_emitted_semantic_gates(self):
        gates = build_blueprint_semantic_gates(
            {
                "components": [
                    {
                        "name": "AuthService",
                        "type": "entity",
                        "description": "Handles authentication",
                        "derived_from": "Build secure auth",
                        "attributes": {},
                        "methods": [],
                        "validation_rules": [],
                    }
                ],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
                "semantic_gates": [
                    {
                        "owner_component": "AuthService",
                        "question": "Clarify provider fallback strategy",
                        "kind": "gap",
                        "options": ["Anthropic", "OpenAI"],
                        "stage": "verification",
                    }
                ],
            },
            trust={"gap_report": []},
            context_graph={},
        )

        assert gates[0]["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert gates[0]["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert gates[0]["options"] == ["Anthropic", "OpenAI"]

    def test_merges_verification_semantic_gates_into_native_blueprint_gates(self):
        gates = build_blueprint_semantic_gates(
            {
                "components": [
                    {
                        "name": "AuthService",
                        "type": "entity",
                        "description": "Handles authentication",
                        "derived_from": "Build secure auth",
                        "attributes": {},
                        "methods": [],
                        "validation_rules": [],
                    }
                ],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            },
            verification={
                "semantic_gates": [
                    {
                        "owner_component": "AuthService",
                        "question": "Which provider fallback should AuthService use?",
                        "kind": "semantic_conflict",
                        "options": ["Anthropic", "OpenAI"],
                        "stage": "verification",
                    }
                ]
            },
            trust={"gap_report": []},
            context_graph={},
        )

        assert gates[0]["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert gates[0]["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert gates[0]["kind"] == "semantic_conflict"
