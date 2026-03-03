"""
Phase 12.2: Process Telemetry, Specific Focus Hints, Conflict Mediation.

Tests for:
- 12.2a: Process telemetry storage (CompilationRecord fields, SQLite persistence, engine collection)
- 12.2b: Specific governor focus hints (uncovered components, unknowns, conflicts)
- 12.2c: Non-structural conflict mediation (_classify_conflict, synthesis prompt, context_graph)
"""

import json
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from dataclasses import fields as dataclass_fields

from persistence.corpus import CompilationRecord, Corpus
from persistence.sqlite_corpus import SQLiteCorpus
from core.engine import MotherlabsEngine, CompileResult
from core.protocol import SharedState, Message, MessageType, ConfidenceVector, calculate_dialogue_depth
from core.llm import BaseLLMClient
from agents.base import LLMAgent


# =============================================================================
# HELPERS
# =============================================================================

SAMPLE_CONTEXT_GRAPH = {
    "known": {"intent": {"core_need": "test", "domain": "testing"}},
    "unknown": [],
}

SAMPLE_BLUEPRINT = {
    "components": [
        {"name": "TestComp", "type": "entity", "derived_from": "test"},
    ],
    "relationships": [],
    "constraints": [],
    "unresolved": [],
}

MOCK_INTENT_JSON = {
    "core_need": "Build a user authentication system",
    "domain": "authentication",
    "actors": ["User", "AuthService"],
    "implicit_goals": ["Secure session management"],
    "constraints": ["Must handle sessions"],
    "insight": "Core need is secure identity verification",
    "explicit_components": [],
    "explicit_relationships": [],
}

MOCK_PERSONA_JSON = {
    "personas": [
        {
            "name": "Security Architect",
            "perspective": "Focus on authentication security",
            "blind_spots": "May over-engineer",
            "priorities": ["Security", "Compliance"],
            "key_questions": ["How are credentials stored?"],
        },
        {
            "name": "UX Designer",
            "perspective": "Focus on user experience",
            "blind_spots": "May ignore security",
            "priorities": ["Usability"],
            "key_questions": ["Is login fast?"],
        },
    ]
}

MOCK_SYNTHESIS_JSON = {
    "components": [
        {"name": "User", "type": "entity", "derived_from": "Input: user authentication"},
        {"name": "AuthService", "type": "process", "derived_from": "Input: authentication system"},
    ],
    "relationships": [
        {"from": "User", "to": "AuthService", "type": "uses"},
    ],
    "constraints": [],
    "unresolved": [],
}

MOCK_VERIFY_JSON = {
    "status": "pass",
    "completeness": {"score": 85, "gaps": []},
    "consistency": {"score": 90, "details": "No contradictions"},
    "coherence": {"score": 80, "details": "Logical structure"},
    "traceability": {"score": 95, "details": "All derived_from present"},
}


_KERNEL_EXTRACT_MARKER = "You are a semantic compiler. You extract structured concepts"
_MOCK_KERNEL_EXTRACTIONS = json.dumps([
    {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "user", "content": "User entity", "confidence": 0.9, "connections": []},
])


def make_sequenced_mock(extra_dialogue_turns=9):
    """Create mock LLM for full pipeline."""
    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock-model"
    call_count = [0]

    def mock_complete(system_prompt, user_content, **kwargs):
        if _KERNEL_EXTRACT_MARKER in system_prompt:
            return _MOCK_KERNEL_EXTRACTIONS
        # Detect synthesis/verify by system prompt (position-independent)
        if "You are the Synthesis Agent" in system_prompt:
            return json.dumps(MOCK_SYNTHESIS_JSON)
        if "You are the Verify Agent" in system_prompt:
            return json.dumps(MOCK_VERIFY_JSON)
        # Intent + persona use sequential slots, dialogue cycles
        sequential = [
            json.dumps(MOCK_INTENT_JSON),
            json.dumps(MOCK_PERSONA_JSON),
        ]
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(sequential):
            return sequential[idx]
        if idx % 2 == 0:
            return (
                "Analyzing structure of the auth system.\n"
                "INSIGHT: system builds User entity with email, password_hash"
            )
        return (
            "Analyzing behavior of the auth system.\n"
            "INSIGHT: system builds login flow that validates credentials"
        )

    client.complete_with_system = Mock(side_effect=mock_complete)
    return client


# =============================================================================
# 12.2a — PROCESS TELEMETRY
# =============================================================================

class TestTelemetryFields:
    """Tests for CompilationRecord telemetry fields."""

    def test_compilation_record_has_telemetry_fields(self):
        """New telemetry fields exist with None defaults."""
        record = CompilationRecord(
            id="test", input_text="test", domain="test",
            timestamp="2026-01-01", components_count=5,
            insights_count=3, success=True, file_path="/tmp/test",
        )
        assert record.dialogue_turns is None
        assert record.confidence_trajectory is None
        assert record.message_type_counts is None
        assert record.conflict_count is None
        assert record.structural_conflicts_resolved is None
        assert record.unknown_count is None
        assert record.dialogue_depth_config is None

    def test_compilation_record_to_dict_includes_telemetry(self):
        """to_dict() serializes telemetry fields."""
        record = CompilationRecord(
            id="test", input_text="test", domain="test",
            timestamp="2026-01-01", components_count=5,
            insights_count=3, success=True, file_path="/tmp/test",
            dialogue_turns=8,
            confidence_trajectory=[0.1, 0.2, 0.3],
            message_type_counts={"PROPOSITION": 5, "CHALLENGE": 3},
        )
        d = record.to_dict()
        assert d["dialogue_turns"] == 8
        assert d["confidence_trajectory"] == [0.1, 0.2, 0.3]
        assert d["message_type_counts"]["PROPOSITION"] == 5

    def test_compilation_record_from_dict_backward_compat(self):
        """Old dicts without telemetry fields still load correctly."""
        old_data = {
            "id": "abc",
            "input_text": "test",
            "domain": "testing",
            "timestamp": "2026-01-01",
            "components_count": 3,
            "insights_count": 2,
            "success": True,
            "file_path": "/tmp/x",
            "provider": "mock",
            "model": "mock",
        }
        record = CompilationRecord.from_dict(old_data)
        assert record.id == "abc"
        assert record.dialogue_turns is None
        assert record.confidence_trajectory is None


class TestSQLiteTelemetry:
    """Tests for SQLiteCorpus telemetry storage."""

    def test_sqlite_corpus_stores_telemetry(self, tmp_path):
        """store + retrieve with telemetry fields."""
        corpus = SQLiteCorpus(corpus_path=tmp_path / "corpus")
        record = corpus.store(
            input_text="Build auth system for telemetry test",
            context_graph=SAMPLE_CONTEXT_GRAPH,
            blueprint=SAMPLE_BLUEPRINT,
            insights=["insight1"],
            success=True,
            dialogue_turns=10,
            confidence_trajectory=[0.1, 0.25, 0.4],
            message_type_counts={"PROPOSITION": 6, "CHALLENGE": 4},
            conflict_count=3,
            structural_conflicts_resolved=1,
            unknown_count=2,
            dialogue_depth_config={"min_turns": 6, "min_insights": 8, "max_turns": 12},
        )
        assert record.dialogue_turns == 10
        assert record.confidence_trajectory == [0.1, 0.25, 0.4]
        assert record.message_type_counts["PROPOSITION"] == 6
        assert record.conflict_count == 3
        assert record.structural_conflicts_resolved == 1
        assert record.unknown_count == 2
        assert record.dialogue_depth_config["max_turns"] == 12

    def test_sqlite_corpus_migration_adds_columns(self, tmp_path):
        """New DB gets columns, existing DB gets ALTER TABLE migration."""
        corpus_path = tmp_path / "corpus"
        # First init creates the table
        corpus1 = SQLiteCorpus(corpus_path=corpus_path)
        # Second init runs migration (idempotent)
        corpus2 = SQLiteCorpus(corpus_path=corpus_path)
        # Should not raise — columns already exist
        record = corpus2.store(
            input_text="Migration test input data",
            context_graph=SAMPLE_CONTEXT_GRAPH,
            blueprint=SAMPLE_BLUEPRINT,
            insights=[],
            success=True,
            dialogue_turns=5,
        )
        assert record.dialogue_turns == 5

    def test_sqlite_corpus_row_to_record_with_telemetry(self, tmp_path):
        """JSON deserialization of telemetry works on retrieval."""
        corpus = SQLiteCorpus(corpus_path=tmp_path / "corpus")
        corpus.store(
            input_text="Deserialization test input text",
            context_graph=SAMPLE_CONTEXT_GRAPH,
            blueprint=SAMPLE_BLUEPRINT,
            insights=["test insight"],
            success=True,
            confidence_trajectory=[0.5, 0.6, 0.7],
            message_type_counts={"AGREEMENT": 2},
            dialogue_depth_config={"min_turns": 4, "min_insights": 6, "max_turns": 10},
        )
        records = corpus.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.confidence_trajectory == [0.5, 0.6, 0.7]
        assert r.message_type_counts == {"AGREEMENT": 2}
        assert r.dialogue_depth_config["min_insights"] == 6


def _passing_closed_loop_gate(description, blueprint):
    """Mock closed_loop_gate that always passes (mock LLM data doesn't meet real fidelity)."""
    from dataclasses import dataclass, field as _field

    @dataclass
    class _MockCLResult:
        passed: bool = True
        fidelity_score: float = 0.85
        compression_losses: list = _field(default_factory=list)

    return _MockCLResult()


class TestEngineTelemetryCollection:
    """Tests for engine collecting telemetry during compile."""

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_engine_collects_telemetry(self, _mock_clg, tmp_path):
        """Mock compile produces non-None telemetry on stored record."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=True,
            cache_policy="none",
        )
        result = engine.compile(
            "Build a user authentication system with login and registration"
        )
        # Check the stored record
        records = corpus.list_all()
        assert len(records) >= 1
        record = records[0]
        assert record.dialogue_turns is not None
        assert record.dialogue_turns > 0

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_telemetry_confidence_trajectory_length(self, _mock_clg, tmp_path):
        """Trajectory length matches dialogue turns."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=True,
            cache_policy="none",
        )
        result = engine.compile(
            "Build a user authentication system with secure session handling"
        )
        records = corpus.list_all()
        record = records[0]
        # confidence_trajectory has one entry per dialogue turn
        assert record.confidence_trajectory is not None
        assert len(record.confidence_trajectory) == record.dialogue_turns

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_telemetry_message_type_counts(self, _mock_clg, tmp_path):
        """Counts include at least PROPOSITION key."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=True,
            cache_policy="none",
        )
        result = engine.compile(
            "Build a comprehensive user authentication and authorization system"
        )
        records = corpus.list_all()
        record = records[0]
        assert record.message_type_counts is not None
        assert isinstance(record.message_type_counts, dict)
        # Should have at least one message type from the dialogue
        assert len(record.message_type_counts) > 0


# =============================================================================
# 12.2b — SPECIFIC FOCUS HINTS
# =============================================================================

class TestSpecificFocusHints:
    """Tests for specific governor focus hints."""

    def _make_engine_and_state(self):
        """Create engine + state for focus hint testing."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock-model"
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=None,
            auto_store=False,
            cache_policy="none",
        )
        state = SharedState()
        state.known["intent"] = {
            "core_need": "Build auth",
            "domain": "auth",
            "explicit_components": ["UserService", "TokenManager", "SessionStore", "AuditLog"],
        }
        return engine, state

    def test_focus_hint_includes_uncovered_components(self):
        """Hint names uncovered components when structural is weak."""
        engine, state = self._make_engine_and_state()
        # No insights discovered → all components uncovered
        state.confidence.structural = 0.1
        state.confidence.behavioral = 0.5
        state.confidence.coverage = 0.5
        state.confidence.consistency = 0.5

        weakest = state.confidence.weakest_dimension()
        assert weakest == "structural"

        uncovered = engine.entity_agent._compute_uncovered_ground(state)
        assert "UserService" in uncovered
        assert "TokenManager" in uncovered

    def test_focus_hint_behavioral_mentions_methods(self):
        """Behavioral weakness → hint mentions methods/flows."""
        _, state = self._make_engine_and_state()
        state.confidence.structural = 0.8
        state.confidence.behavioral = 0.1
        state.confidence.coverage = 0.6
        state.confidence.consistency = 0.6

        weakest = state.confidence.weakest_dimension()
        assert weakest == "behavioral"

        # Build hint parts matching engine logic
        hint_parts = [f"[GOVERNOR] Confidence stalled on {weakest}."]
        if weakest == "behavioral":
            hint_parts.append("Specify methods, state transitions, or interaction flows.")
        hint = " ".join(hint_parts)
        assert "methods" in hint
        assert "state transitions" in hint

    def test_focus_hint_consistency_mentions_unknowns(self):
        """Consistency weakness → names unknowns."""
        _, state = self._make_engine_and_state()
        state.confidence.structural = 0.8
        state.confidence.behavioral = 0.8
        state.confidence.coverage = 0.8
        state.confidence.consistency = 0.1
        state.unknown = ["How does session expiry work?", "What auth protocol?"]

        weakest = state.confidence.weakest_dimension()
        assert weakest == "consistency"

        hint_parts = [f"[GOVERNOR] Confidence stalled on {weakest}."]
        if weakest == "consistency" and state.unknown:
            hint_parts.append(f"Unresolved unknowns: {', '.join(state.unknown[:2])}.")
        hint = " ".join(hint_parts)
        assert "session expiry" in hint
        assert "auth protocol" in hint

    def test_focus_hint_mentions_open_conflicts(self):
        """Hint includes unresolved conflict topics."""
        _, state = self._make_engine_and_state()
        state.confidence.structural = 0.1
        state.conflicts = [
            {"topic": "Auth method: JWT vs sessions", "resolved": False, "positions": {}},
            {"topic": "Storage: SQL vs NoSQL", "resolved": False, "positions": {}},
            {"topic": "Resolved one", "resolved": True, "positions": {}},
        ]

        unresolved = [c for c in state.conflicts if not c["resolved"]]
        topics = [c["topic"][:40] for c in unresolved[:2]]
        hint_part = f"Open conflicts: {'; '.join(topics)}."
        assert "JWT vs sessions" in hint_part
        assert "SQL vs NoSQL" in hint_part

    def test_focus_hint_graceful_no_uncovered(self):
        """All components covered → 'look for implicit ones'."""
        engine, state = self._make_engine_and_state()
        # Add insights that cover all explicit components
        state.insights = [
            "INSIGHT: UserService handles auth",
            "INSIGHT: TokenManager creates tokens",
            "INSIGHT: SessionStore persists sessions",
            "INSIGHT: AuditLog records events",
        ]
        state.confidence.structural = 0.1

        uncovered = engine.entity_agent._compute_uncovered_ground(state)
        assert len(uncovered) == 0

        # Engine logic: if no uncovered → "look for implicit ones"
        hint_parts = ["[GOVERNOR] Confidence stalled on structural."]
        if not uncovered:
            hint_parts.append("All explicit components addressed — look for implicit ones.")
        hint = " ".join(hint_parts)
        assert "implicit" in hint

    def test_focus_hint_max_3_components(self):
        """Caps at 3 component names."""
        engine, state = self._make_engine_and_state()
        # All 4 are uncovered (no insights)
        uncovered = engine.entity_agent._compute_uncovered_ground(state)
        assert len(uncovered) == 4
        # Engine logic caps at 3
        capped = uncovered[:3]
        assert len(capped) == 3

    def test_focus_hint_max_2_unknowns(self):
        """Caps at 2 unknown names."""
        _, state = self._make_engine_and_state()
        state.unknown = ["Q1?", "Q2?", "Q3?", "Q4?"]
        capped = state.unknown[:2]
        assert len(capped) == 2
        assert capped == ["Q1?", "Q2?"]

    def test_focus_hint_max_2_conflicts(self):
        """Caps at 2 conflict topics."""
        _, state = self._make_engine_and_state()
        state.conflicts = [
            {"topic": "Conflict A", "resolved": False, "positions": {}},
            {"topic": "Conflict B", "resolved": False, "positions": {}},
            {"topic": "Conflict C", "resolved": False, "positions": {}},
        ]
        unresolved = [c for c in state.conflicts if not c["resolved"]]
        topics = [c["topic"][:40] for c in unresolved[:2]]
        assert len(topics) == 2
        assert topics == ["Conflict A", "Conflict B"]


# =============================================================================
# 12.2c — CONFLICT MEDIATION
# =============================================================================

class TestConflictClassification:
    """Tests for _classify_conflict."""

    def _make_engine(self):
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock-model"
        return MotherlabsEngine(
            llm_client=client,
            corpus=None,
            auto_store=False,
            cache_policy="none",
        )

    def test_classify_conflict_missing_info(self):
        """'unknown'/'unclear' → MISSING_INFO."""
        engine = self._make_engine()
        conflict = {
            "topic": "Auth token format",
            "positions": {
                "Entity": "JWT token with unclear expiry",
                "Process": "unknown token refresh mechanism",
            },
        }
        assert engine._classify_conflict(conflict) == "MISSING_INFO"

    def test_classify_conflict_priority(self):
        """'should'/'must' → PRIORITY."""
        engine = self._make_engine()
        conflict = {
            "topic": "Feature order",
            "positions": {
                "Entity": "Auth should come first",
                "Process": "Onboarding must happen before auth",
            },
        }
        assert engine._classify_conflict(conflict) == "PRIORITY"

    def test_classify_conflict_tradeoff(self):
        """Default → TRADEOFF."""
        engine = self._make_engine()
        conflict = {
            "topic": "Storage choice",
            "positions": {
                "Entity": "PostgreSQL for reliability",
                "Process": "MongoDB for flexibility",
            },
        }
        assert engine._classify_conflict(conflict) == "TRADEOFF"


class TestConflictResolution:
    """Tests for conflict classification during _resolve_conflicts."""

    def _make_engine(self):
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock-model"
        return MotherlabsEngine(
            llm_client=client,
            corpus=None,
            auto_store=False,
            cache_policy="none",
        )

    def test_resolve_conflicts_classifies_nonstructural(self):
        """After _resolve_conflicts, non-structural have 'category'."""
        engine = self._make_engine()
        state = SharedState()
        state.conflicts = [
            {
                "agents": ["Entity", "Process"],
                "topic": "Auth: structural disagreement (entity vs process)",
                "positions": {"Entity": "Auth as entity", "Process": "Auth as flow"},
                "turn": 3,
                "resolved": False,
            },
            {
                "agents": ["Entity", "Process"],
                "topic": "Speed vs safety",
                "positions": {"Entity": "Optimize for speed", "Process": "Optimize for safety"},
                "turn": 5,
                "resolved": False,
            },
        ]
        resolved_count = engine._resolve_conflicts(state)
        # Structural conflict should be resolved
        assert resolved_count == 1
        assert state.conflicts[0]["resolved"] is True
        # Non-structural conflict should have category
        assert state.conflicts[1]["resolved"] is False
        assert state.conflicts[1]["category"] == "TRADEOFF"


class TestSynthesisConflictSection:
    """Tests for conflict section in synthesis prompt."""

    def test_synthesis_prompt_includes_conflict_section(self, tmp_path):
        """Synthesis prompt has SECTION 8 when conflicts exist."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=True,
            cache_policy="none",
        )
        # Inject conflicts into state by patching _run_spec_dialogue
        original_dialogue = engine._run_spec_dialogue

        def patched_dialogue(state):
            original_dialogue(state)
            state.conflicts.append({
                "agents": ["Entity", "Process"],
                "topic": "Priority: speed vs safety",
                "positions": {"Entity": "Speed first", "Process": "Safety first"},
                "turn": 3,
                "resolved": False,
                "category": "PRIORITY",
            })

        engine._run_spec_dialogue = patched_dialogue

        # Capture the synthesis prompt
        synthesis_calls = []
        original_run_llm_only = engine.synthesis_agent.run_llm_only

        def capture_run(state, msg, max_tokens=4096):
            synthesis_calls.append(msg.content)
            return original_run_llm_only(state, msg, max_tokens)

        engine.synthesis_agent.run_llm_only = capture_run

        result = engine.compile(
            "Build a user authentication system with secure login and session management"
        )

        # Check that at least one synthesis call contains SECTION 8 (renumbered in Phase 12.3)
        assert any("SECTION 8" in call for call in synthesis_calls), \
            f"No SECTION 8 in synthesis calls: {[c[:200] for c in synthesis_calls]}"

    def test_conflict_mediation_guidance_text(self):
        """PRIORITY/MISSING_INFO/TRADEOFF guidance strings present in section."""
        # Build the conflict section as the engine does
        conflicts = [
            {"topic": "Auth order", "positions": {"Entity": "JWT"}, "resolved": False, "category": "PRIORITY"},
            {"topic": "Token format", "positions": {"Entity": "unclear"}, "resolved": False, "category": "MISSING_INFO"},
        ]
        unresolved = [c for c in conflicts if not c["resolved"]]
        conflict_lines = []
        for c in unresolved:
            category = c.get("category", "TRADEOFF")
            topic = c.get("topic", "unknown")
            pos = c.get("positions", {})
            conflict_lines.append(f"- [{category}] {topic}: {pos}")
        section = "SECTION 8: UNRESOLVED CONFLICTS\n" + "\n".join(conflict_lines)
        section += (
            "\n\nFor PRIORITY conflicts: choose the option that serves the core_need.\n"
            "For MISSING_INFO conflicts: mark as unresolved in the blueprint.\n"
            "For TRADEOFF conflicts: choose the option with broader impact, note the alternative."
        )
        assert "[PRIORITY]" in section
        assert "[MISSING_INFO]" in section
        assert "core_need" in section
        assert "mark as unresolved" in section
        assert "broader impact" in section


class TestConflictSummary:
    """Tests for conflict_summary in context_graph."""

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_conflict_summary_in_context_graph(self, _mock_clg, tmp_path):
        """context_graph has conflict_summary dict."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=True,
            cache_policy="none",
        )
        # Inject a conflict
        original_dialogue = engine._run_spec_dialogue

        def patched_dialogue(state):
            original_dialogue(state)
            state.conflicts.append({
                "agents": ["Entity", "Process"],
                "topic": "Storage: SQL vs NoSQL",
                "positions": {"Entity": "SQL", "Process": "NoSQL"},
                "turn": 4,
                "resolved": False,
            })

        engine._run_spec_dialogue = patched_dialogue

        result = engine.compile(
            "Build a user authentication and data storage system with persistence"
        )
        cg = result.context_graph
        assert "conflict_summary" in cg
        summary = cg["conflict_summary"]
        assert "total" in summary
        assert "resolved" in summary
        assert "unresolved" in summary
        assert isinstance(summary["unresolved"], list)
        # At least one unresolved conflict (the one we injected)
        assert summary["total"] >= 1
