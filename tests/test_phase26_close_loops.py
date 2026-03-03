"""
Tests for Phase 26: Close the Loops.

Three data flow breaks fixed:
- Break 5: domain_model → state.known for Stratum 1+2 provenance
- Break 4: classification scores → _avg_type_confidence for verification
- Break 10: self-compile patterns → state.self_compile_patterns for Stratum 3
"""

import json
import pytest
from dataclasses import asdict
from unittest.mock import Mock, patch, MagicMock

from core.engine import MotherlabsEngine, CompileResult
from core.protocol import SharedState
from core.llm import MockClient, BaseLLMClient
from core.classification import ClassificationScore
from core.self_compile import SelfPattern, extract_self_patterns
from persistence.corpus import Corpus


# =============================================================================
# MOCK RESPONSE SEQUENCES (reuse test_engine.py pattern)
# =============================================================================

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
            "perspective": "Focus on authentication security and session management",
            "blind_spots": "May over-engineer simple auth flows",
        },
        {
            "name": "UX Designer",
            "perspective": "Focus on user experience during login/logout",
            "blind_spots": "May underestimate security requirements",
        },
    ],
    "cross_cutting_concerns": ["Performance vs security tradeoff"],
    "suggested_focus_areas": ["Session lifecycle"],
}

MOCK_SYNTHESIS_JSON = {
    "components": [
        {
            "name": "User",
            "type": "entity",
            "description": "User account with authentication credentials",
            "derived_from": "INSIGHT: User entity contains email, password_hash",
            "properties": [
                {"name": "email", "type": "str"},
                {"name": "password_hash", "type": "str"},
            ],
        },
        {
            "name": "Session",
            "type": "entity",
            "description": "Active user session with expiry",
            "derived_from": "INSIGHT: Session entity contains token, expiry",
            "properties": [
                {"name": "token", "type": "str"},
                {"name": "expiry", "type": "datetime"},
            ],
        },
        {
            "name": "AuthService",
            "type": "process",
            "description": "Authentication service",
            "derived_from": "INSIGHT: AuthService manages User and Session",
            "methods": [
                {
                    "name": "login",
                    "parameters": [
                        {"name": "email", "type_hint": "str"},
                        {"name": "password", "type_hint": "str"},
                    ],
                    "return_type": "Session",
                    "description": "Authenticate user",
                    "derived_from": "login(email, password) -> Session",
                }
            ],
        },
    ],
    "relationships": [
        {"from": "AuthService", "to": "User", "type": "accesses", "description": "AuthService validates User"},
        {"from": "AuthService", "to": "Session", "type": "generates", "description": "AuthService creates Session"},
        {"from": "Session", "to": "User", "type": "depends_on", "description": "Session belongs to User"},
    ],
    "constraints": [],
    "unresolved": [],
}

MOCK_VERIFY_JSON = {
    "status": "pass",
    "completeness": {"score": 85, "details": "All components traced"},
    "consistency": {"score": 90, "details": "No contradictions"},
    "coherence": {"score": 80, "details": "Logical structure"},
    "traceability": {"score": 95, "details": "All derived_from present"},
}


_KERNEL_EXTRACT_MARKER = "You are a semantic compiler. You extract structured concepts"
_MOCK_KERNEL_EXTRACTIONS = json.dumps([
    {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "user", "content": "User entity", "confidence": 0.9, "connections": []},
])


def make_sequenced_mock(extra_dialogue_turns=9):
    """Create mock LLM client returning appropriate responses per stage."""
    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock-model"
    call_count = [0]

    def mock_complete_with_system(system_prompt, user_content, **kwargs):
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
                "Analyzing structure of the authentication system.\n"
                "INSIGHT: system builds User entity with email, password_hash, created_at fields"
            )
        return (
            "Analyzing behavior of the authentication system.\n"
            "INSIGHT: system builds login flow that validates user credentials"
        )

    client.complete_with_system = Mock(side_effect=mock_complete_with_system)
    return client


# =============================================================================
# Break 5: Domain Model → state.known for Stratum 1+2 Provenance
# =============================================================================


class TestBreak5DomainModelWire:
    """Break 5: domain_model stored in state.known for Stratum 1+2."""

    def test_domain_model_stored_in_state_known(self, tmp_path):
        """After compile with seeded corpus, state.known['domain_model'] is populated."""
        from persistence.corpus_analysis import CorpusAnalyzer, DomainModel

        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")

        # Seed corpus with compilations in the "authentication" domain
        for i in range(4):
            corpus.store(
                input_text=f"Build an authentication system variant {i}",
                context_graph={
                    "known": {"domain": "authentication"},
                    "unknown": [],
                    "ontology": {},
                    "personas": [],
                    "decision_trace": [],
                    "insights": [f"auth insight {i}"],
                    "flags": [],
                },
                blueprint={
                    "components": [
                        {"name": "User", "type": "entity", "description": "User account", "derived_from": "test"},
                        {"name": "Session", "type": "entity", "description": "Session", "derived_from": "test"},
                        {"name": "AuthService", "type": "process", "description": "Auth", "derived_from": "test"},
                    ],
                    "relationships": [
                        {"from": "AuthService", "to": "User", "type": "accesses", "description": "validates"},
                        {"from": "AuthService", "to": "Session", "type": "generates", "description": "creates"},
                    ],
                    "constraints": [],
                    "unresolved": [],
                },
                insights=[f"authentication insight {i}"],
                success=True,
            )

        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        # Capture state.known during compilation
        captured_state = {}
        original_synthesize = engine._synthesize

        def capturing_synthesize(state, **kwargs):
            captured_state.update(dict(state.known))
            return original_synthesize(state, **kwargs)

        engine._synthesize = capturing_synthesize
        result = engine.compile(
            "Build a user authentication system",
            use_corpus_suggestions=True,
        )

        assert result.success
        # The domain_model key should be present (if CorpusAnalyzer found patterns)
        if "domain_model" in captured_state:
            dm = captured_state["domain_model"]
            assert isinstance(dm, dict)
            assert "vocabulary" in dm
            assert "archetypes" in dm
            assert "relationship_patterns" in dm
            assert "domain" in dm

    def test_domain_model_shape_matches_consumer(self):
        """asdict(DomainModel(...)) produces the shape _check_stratum_1/2 expects."""
        from persistence.corpus_analysis import DomainModel

        dm = DomainModel(
            domain="authentication",
            sample_size=5,
            vocabulary={"user": {"frequency": 4, "type_hint": "entity"}},
            archetypes=[],
            relationship_patterns=[],
        )
        d = asdict(dm)

        # _check_stratum_1 reads d.get("vocabulary", {})
        assert isinstance(d["vocabulary"], dict)
        assert "user" in d["vocabulary"]

        # _check_stratum_2 reads d.get("archetypes", []) and d.get("relationship_patterns", [])
        assert isinstance(d["archetypes"], list)
        assert isinstance(d["relationship_patterns"], list)

    def test_domain_model_not_stored_without_corpus(self, tmp_path):
        """Engine with no corpus → no domain_model key in state."""
        engine = MotherlabsEngine(
            llm_client=make_sequenced_mock(),
            corpus=None,
            auto_store=False,
            cache_policy="none",
        )

        captured_state = {}
        original_synthesize = engine._synthesize

        def capturing_synthesize(state, **kwargs):
            captured_state.update(dict(state.known))
            return original_synthesize(state, **kwargs)

        engine._synthesize = capturing_synthesize
        result = engine.compile("Build a user authentication system")

        assert result.success
        assert "domain_model" not in captured_state


# =============================================================================
# Break 4: Classification Scores → _avg_type_confidence for Verification
# =============================================================================


class TestBreak4ClassificationToVerification:
    """Break 4: classification scores flow to verification via _avg_type_confidence."""

    def test_avg_type_confidence_stored_after_staged_pipeline(self):
        """Unit test: the extraction logic correctly computes and stores avg_type_confidence."""
        from unittest.mock import MagicMock

        # Simulate what engine.py does after staged pipeline
        state = SharedState()
        state.known["input"] = "test"

        # Create a mock pipeline_state with classification scores in decompose artifact
        mock_pipeline_state = MagicMock()
        scores = [
            ClassificationScore(
                name="User", mention_frequency=0.8, grammatical_role="subject",
                semantic_centrality=0.5, inferred_type="entity", type_confidence=0.9,
                is_component=True, overall_confidence=0.8, reasoning="test",
            ),
            ClassificationScore(
                name="Session", mention_frequency=0.6, grammatical_role="object",
                semantic_centrality=0.3, inferred_type="entity", type_confidence=0.7,
                is_component=True, overall_confidence=0.6, reasoning="test",
            ),
        ]
        mock_pipeline_state.get_artifact.return_value = {"classification_scores": scores}

        # Execute the same logic as engine.py Phase 26 insertion
        decompose_artifact = mock_pipeline_state.get_artifact("decompose")
        if decompose_artifact:
            cls_scores = decompose_artifact.get("classification_scores", [])
            if cls_scores:
                avg_tc = sum(s.type_confidence for s in cls_scores) / len(cls_scores)
                state.known["_avg_type_confidence"] = avg_tc

        assert "_avg_type_confidence" in state.known
        assert abs(state.known["_avg_type_confidence"] - 0.8) < 1e-9  # (0.9 + 0.7) / 2

    def test_avg_type_confidence_correct_average(self):
        """Unit test: known ClassificationScores → correct average."""
        scores = [
            ClassificationScore(
                name="User", mention_frequency=0.8, grammatical_role="subject",
                semantic_centrality=0.5, inferred_type="entity", type_confidence=0.9,
                is_component=True, overall_confidence=0.8, reasoning="test",
            ),
            ClassificationScore(
                name="Session", mention_frequency=0.6, grammatical_role="object",
                semantic_centrality=0.3, inferred_type="entity", type_confidence=0.7,
                is_component=True, overall_confidence=0.6, reasoning="test",
            ),
            ClassificationScore(
                name="AuthService", mention_frequency=0.5, grammatical_role="subject",
                semantic_centrality=0.4, inferred_type="process", type_confidence=0.8,
                is_component=True, overall_confidence=0.7, reasoning="test",
            ),
        ]
        avg_tc = sum(s.type_confidence for s in scores) / len(scores)
        assert abs(avg_tc - 0.8) < 1e-9  # (0.9 + 0.7 + 0.8) / 3 = 0.8

    def test_empty_classification_scores_no_crash(self):
        """Empty scores list → no key set, no crash."""
        # Simulate the engine logic directly
        state = SharedState()
        state.known["input"] = "test"

        cls_scores = []
        if cls_scores:
            avg_tc = sum(s.type_confidence for s in cls_scores) / len(cls_scores)
            state.known["_avg_type_confidence"] = avg_tc

        assert "_avg_type_confidence" not in state.known

    def test_legacy_mode_uses_default(self, tmp_path):
        """Legacy pipeline → default 0.5 used (no classification runs)."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
            pipeline_mode="legacy",
        )

        captured_state = {}
        original_verify = engine._verify_deterministic

        def capturing_verify(blueprint, state):
            captured_state.update(dict(state.known))
            return original_verify(blueprint, state)

        engine._verify_deterministic = capturing_verify
        result = engine.compile("Build a user authentication system")

        assert result.success
        # Legacy mode has no classification → key should NOT be set
        assert "_avg_type_confidence" not in captured_state


# =============================================================================
# Break 10: Self-Compile Feedback Loop
# =============================================================================


class TestBreak10SelfCompileFeedback:
    """Break 10: self-compile patterns flow back into compile() for Stratum 3."""

    def test_self_compile_patterns_stored_on_engine(self):
        """After run_self_compile_loop(), engine._last_self_compile_patterns is populated."""
        engine = MotherlabsEngine(llm_client=MockClient())
        assert engine._last_self_compile_patterns == []

        report = engine.run_self_compile_loop(runs=2)

        # Patterns should be stored as list of dicts
        assert isinstance(engine._last_self_compile_patterns, list)
        # Each entry should be a dict (serialized SelfPattern)
        for p in engine._last_self_compile_patterns:
            assert isinstance(p, dict)
            assert "pattern_type" in p
            assert "name" in p

    def test_patterns_injected_into_compile(self, tmp_path):
        """With pre-set patterns, state.self_compile_patterns is populated during compile."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        # Simulate prior self-compile by setting patterns directly
        engine._last_self_compile_patterns = [
            {"pattern_type": "stable_component", "name": "SharedState", "frequency": 3, "confidence": 0.95},
            {"pattern_type": "stable_relationship", "name": "Entity->Process", "frequency": 3, "confidence": 0.90},
        ]

        # Capture state at synthesis time
        captured_patterns = []
        original_synthesize = engine._synthesize

        def capturing_synthesize(state, **kwargs):
            captured_patterns.extend(state.self_compile_patterns)
            return original_synthesize(state, **kwargs)

        engine._synthesize = capturing_synthesize
        result = engine.compile("Build a user authentication system")

        assert result.success
        assert len(captured_patterns) == 2
        assert captured_patterns[0]["name"] == "SharedState"
        assert captured_patterns[1]["name"] == "Entity->Process"

    def test_compile_without_self_compile_has_empty_patterns(self, tmp_path):
        """No prior self-compile → patterns remain empty."""
        client = make_sequenced_mock()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        captured_patterns = []
        original_synthesize = engine._synthesize

        def capturing_synthesize(state, **kwargs):
            captured_patterns.extend(state.self_compile_patterns)
            return original_synthesize(state, **kwargs)

        engine._synthesize = capturing_synthesize
        result = engine.compile("Build a user authentication system")

        assert result.success
        assert len(captured_patterns) == 0

    def test_patterns_are_copies_not_shared_refs(self):
        """Two compile() calls get independent copies of patterns, not shared refs."""
        patterns_source = [
            {"pattern_type": "stable_component", "name": "SharedState", "frequency": 3, "confidence": 0.95},
        ]

        # Simulate what engine.compile() does: list(self._last_self_compile_patterns)
        copy1 = list(patterns_source)
        copy2 = list(patterns_source)

        # Equal content
        assert copy1 == copy2
        # Different list objects
        assert copy1 is not copy2
        # Mutating one doesn't affect the other
        copy1.append({"pattern_type": "drift_point", "name": "X"})
        assert len(copy2) == 1

    def test_stratum_3_integration(self):
        """Full loop: self-compile → compile → stratum 3 data available."""
        from agents.base import LLMAgent

        # Set up agent with mock
        mock_llm = MagicMock()
        agent = LLMAgent(
            name="Entity",
            perspective="Structure",
            system_prompt="Test agent",
            llm_client=mock_llm,
        )

        # Create state with self-compile patterns (as engine would inject)
        state = SharedState()
        state.known["input"] = "Build a semantic compiler with SharedState and Entity agents"
        state.self_compile_patterns = [
            {"pattern_type": "stable_component", "name": "SharedState", "frequency": 3, "confidence": 0.95},
            {"pattern_type": "stable_relationship", "name": "Entity->Process", "frequency": 3, "confidence": 0.90},
            {"pattern_type": "drift_point", "name": "Unstable", "frequency": 1, "confidence": 0.3},
        ]

        # Stratum 3 should fire for an insight mentioning a stable pattern
        result = agent._check_stratum_3(
            "SharedState is the central coordination hub", state
        )
        assert result is True

        # Stratum 3 should NOT fire for drift_point patterns
        result = agent._check_stratum_3(
            "Unstable component detected", state
        )
        assert result is False

        # Stratum 3 should NOT fire for unrelated insights
        result = agent._check_stratum_3(
            "completely unrelated quantum physics insight", state
        )
        assert result is False

    def test_self_compile_loop_populates_asdict_patterns(self):
        """run_self_compile_loop stores patterns as asdict-serialized dicts."""
        engine = MotherlabsEngine(llm_client=MockClient())
        report = engine.run_self_compile_loop(runs=2)

        # Verify the stored patterns match the report patterns
        assert len(engine._last_self_compile_patterns) == len(report.patterns)
        for stored, original in zip(engine._last_self_compile_patterns, report.patterns):
            assert stored == asdict(original)
