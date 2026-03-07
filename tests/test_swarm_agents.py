"""Tests for swarm agents — base + CompileAgent with mocked engine."""

from unittest.mock import patch, MagicMock
from swarm.agents.base import SwarmAgent
from swarm.agents.compile import CompileAgent
from swarm.state import SwarmState


class TestSwarmAgentBase:
    """SwarmAgent ABC."""

    def test_cannot_instantiate(self):
        try:
            agent = SwarmAgent()
            agent.execute(SwarmState(intent="test"), {})
            assert False, "Should not be directly usable"
        except TypeError:
            pass

    def test_default_keys(self):
        """Concrete subclass inherits default empty keys."""
        class DummyAgent(SwarmAgent):
            name = "dummy"
            def execute(self, state, config):
                return state

        agent = DummyAgent()
        assert agent.input_keys == []
        assert agent.output_keys == []
        assert agent.criticality == "medium"


class TestCompileAgent:
    """CompileAgent with mocked engine."""

    def test_metadata(self):
        agent = CompileAgent()
        assert agent.name == "compile"
        assert agent.criticality == "critical"
        assert "intent" in agent.input_keys
        assert "blueprint" in agent.output_keys
        assert "trust" in agent.output_keys

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_execute_basic(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """CompileAgent calls engine.compile() and populates state."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "Auth"}]}
        mock_result.verification = {"passed": True}
        mock_result.context_graph = {"keywords": ["auth"]}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {"overall_score": 85.0}

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        assert new_state.blueprint == {"components": [{"name": "Auth"}]}
        assert new_state.trust == {"overall_score": 85.0}
        assert new_state.verification == {"passed": True}
        assert new_state.compile_result is not None
        assert new_state.compile_result["success"] is True

        # Original state unchanged
        assert state.blueprint is None

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_enriched_description_with_research(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """Research context is prepended to intent."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {}
        mock_result.verification = {}
        mock_result.context_graph = {}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {}

        state = SwarmState(
            intent="Build a payments API",
            research_context={"findings": "Stripe uses webhooks for async events"},
        )

        agent = CompileAgent()
        agent.execute(state, {})

        call_args = mock_engine.compile.call_args
        description = call_args[1].get("description", call_args[0][0] if call_args[0] else "")
        assert "Stripe uses webhooks" in description
        assert "Build a payments API" in description

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_no_context_uses_raw_intent(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """Without pre-compilation context, uses raw intent."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {}
        mock_result.verification = {}
        mock_result.context_graph = {}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {}

        state = SwarmState(intent="Build a todo app")
        agent = CompileAgent()
        agent.execute(state, {})

        call_args = mock_engine.compile.call_args
        description = call_args[1].get("description", "")
        assert description == "Build a todo app"

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_preserves_glass_box_fields(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """CompileAgent keeps structured insights and difficulty in compile_result."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "Auth"}]}
        mock_result.verification = {"passed": True}
        mock_result.context_graph = {"keywords": ["auth"]}
        mock_result.dimensional_metadata = {"coverage": {"entity": 0.9}}
        mock_result.interface_map = {"Auth": {"methods": ["login"]}}
        mock_result.stage_results = [
            MagicMock(stage="intent", success=True, errors=[], warnings=[], retries=0),
            MagicMock(stage="verification", success=False, errors=["missing edge case"], warnings=["low coverage"], retries=1),
        ]
        mock_result.stage_timings = {"intent": 0.4, "verification": 1.2}
        mock_result.retry_counts = {"verification": 1}
        mock_result.structured_insights = [
            {"text": "Identified: Auth, Session", "category": "discovery", "stage": "entity_extraction"},
        ]
        mock_result.difficulty = {"unknown_count": 2, "irritation_depth": 0.4}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {"triggered": False}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {"overall_score": 85.0}

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        assert new_state.compile_result["structured_insights"] == mock_result.structured_insights
        assert new_state.compile_result["difficulty"] == mock_result.difficulty
        assert new_state.compile_result["dimensional_metadata"] == mock_result.dimensional_metadata
        assert new_state.compile_result["interface_map"] == mock_result.interface_map
        assert new_state.compile_result["stage_results"][1]["stage"] == "verification"
        assert new_state.compile_result["stage_results"][1]["retries"] == 1
        assert new_state.compile_result["stage_timings"]["verification"] == 1.2
        assert new_state.compile_result["retry_counts"]["verification"] == 1
        assert new_state.compile_result["interrogation"]["triggered"] is False

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_preserves_pause_fields(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """CompileAgent preserves pause/error fields for approval loops."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.blueprint = {}
        mock_result.verification = {}
        mock_result.context_graph = {}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.stage_results = []
        mock_result.stage_timings = {}
        mock_result.retry_counts = {}
        mock_result.structured_insights = []
        mock_result.difficulty = {}
        mock_result.error = "Clarification required before compilation can continue"
        mock_result.fracture = {
            "stage": "interrogation",
            "competing_configs": ["A", "B"],
            "collapsing_constraint": "Which direction should I take?",
            "agent": "Interrogation",
        }
        mock_result.interrogation = {"triggered": True}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {"overall_score": 0.0}

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        assert new_state.compile_result["success"] is False
        assert new_state.compile_result["error"] == mock_result.error
        assert new_state.compile_result["fracture"]["stage"] == "interrogation"
        assert new_state.compile_result["interrogation"]["triggered"] is True

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_builds_blocking_semantic_gate_escalations(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """Blocked semantic nodes should be surfaced as blocking escalations."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {
            "components": [
                {
                    "name": "AuthService",
                    "type": "entity",
                    "description": "Handles auth",
                    "derived_from": "Build auth",
                    "attributes": {},
                    "methods": [],
                    "validation_rules": [],
                }
            ],
            "relationships": [],
            "constraints": [],
            "unresolved": ["AuthService needs provider fallback"],
        }
        mock_result.verification = {"passed": True}
        mock_result.context_graph = {"keywords": ["auth"]}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.stage_results = []
        mock_result.stage_timings = {}
        mock_result.retry_counts = {}
        mock_result.structured_insights = []
        mock_result.difficulty = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {"triggered": False}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {
            "overall_score": 78.0,
            "gap_report": ["AuthService needs provider fallback"],
        }

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        assert new_state.compile_result["success"] is True
        assert len(new_state.compile_result["semantic_nodes"]) >= 1
        assert new_state.compile_result["blocking_escalations"][0]["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert new_state.compile_result["blocking_escalations"][0]["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert "provider fallback" in new_state.compile_result["blocking_escalations"][0]["question"].lower()

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_builds_conflict_gate_escalations(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """Unresolved semantic conflicts should become blocking escalations with options."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {
            "components": [
                {
                    "name": "AuthService",
                    "type": "entity",
                    "description": "Handles auth",
                    "derived_from": "Build auth",
                    "attributes": {},
                    "methods": [],
                    "validation_rules": [],
                }
            ],
            "relationships": [],
            "constraints": [],
            "unresolved": [],
        }
        mock_result.verification = {"passed": True}
        mock_result.context_graph = {
            "keywords": ["auth"],
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
            },
        }
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.stage_results = []
        mock_result.stage_timings = {}
        mock_result.retry_counts = {}
        mock_result.structured_insights = []
        mock_result.difficulty = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {"triggered": False}

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {
            "overall_score": 82.0,
            "gap_report": [],
        }

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        escalation = new_state.compile_result["blocking_escalations"][0]
        assert escalation["postcode"] == "STR.ENT.APP.WHAT.SFT"
        assert escalation["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert "storage strategy" in escalation["question"].lower()
        assert escalation["options"] == [
            "Persist sessions in PostgreSQL",
            "Keep sessions stateless with JWT",
        ]

    @patch("core.trust.serialize_trust_indicators")
    @patch("core.trust.compute_trust_indicators")
    @patch("core.engine.MotherlabsEngine")
    @patch("core.adapter_registry.get_adapter")
    def test_preserves_engine_semantic_pause_payload(self, mock_adapter, mock_engine_cls, mock_trust, mock_serialize):
        """CompileAgent should preserve engine-emitted nodes and blocking escalations."""
        mock_adapter.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.blueprint = {"components": []}
        mock_result.verification = {}
        mock_result.context_graph = {}
        mock_result.dimensional_metadata = {}
        mock_result.interface_map = {}
        mock_result.stage_results = []
        mock_result.stage_timings = {}
        mock_result.retry_counts = {}
        mock_result.structured_insights = []
        mock_result.difficulty = {}
        mock_result.error = None
        mock_result.fracture = None
        mock_result.interrogation = {}
        mock_result.semantic_nodes = [
            {
                "id": "node-1-purpose",
                "postcode": "INT.SEM.APP.WHY.SFT",
                "primitive": "purpose",
                "description": "Build auth system.",
                "notes": [],
                "fill_state": "F",
                "confidence": 0.98,
                "status": "promoted",
                "version": 1,
                "created_at": "2026-03-07T22:00:00Z",
                "updated_at": "2026-03-07T22:00:00Z",
                "last_verified": "2026-03-07T22:00:00Z",
                "freshness": {"decay_rate": 0.001, "floor": 0.6, "stale_after": 90},
                "parent": None,
                "children": [],
                "connections": [],
                "references": {"read_before": [], "read_after": [], "see_also": [], "deep_dive": [], "warns": []},
                "provenance": {
                    "source_ref": ["Build auth system"],
                    "agent_id": "Intent",
                    "run_id": "engine:test",
                    "timestamp": "2026-03-07T22:00:00Z",
                    "human_input": True,
                },
                "token_cost": 0,
                "constraints": [],
                "constraint_source": [],
            }
        ]
        mock_result.blocking_escalations = [
            {
                "postcode": "STR.ENT.APP.WHAT.SFT",
                "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
                "options": ["Persist sessions in PostgreSQL", "Keep sessions stateless with JWT"],
            }
        ]

        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_trust.return_value = MagicMock()
        mock_serialize.return_value = {"overall_score": 82.0}

        state = SwarmState(intent="Build auth system", domain="software")
        agent = CompileAgent()
        new_state = agent.execute(state, {})

        assert new_state.compile_result["semantic_nodes"] == mock_result.semantic_nodes
        assert new_state.compile_result["blocking_escalations"] == mock_result.blocking_escalations

    def test_enrich_description_all_contexts(self):
        """All three context types are included when present."""
        agent = CompileAgent()
        state = SwarmState(
            intent="Build X",
            research_context={"findings": "Research data"},
            retrieval_context={"relevant_documents": "Doc content"},
            memory_context={"relevant_patterns": "Past patterns"},
        )
        desc = agent._enrich_description(state)
        assert "[Research Context]" in desc
        assert "[Retrieved Context]" in desc
        assert "[Memory Context]" in desc
        assert "[User Intent]" in desc
        assert "Build X" in desc
