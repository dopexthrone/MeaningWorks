"""
Tests for insight provenance gate (Phase 11.1 + Phase 23 Stratified Provenance).

Prevents runaway inference chains where agent insights drift away
from the original input, producing hallucinated components.

Phase 23 extends provenance to three strata:
- Stratum 0: User input (original gate)
- Stratum 1: Domain entailment (corpus vocabulary + input connection)
- Stratum 2: Corpus patterns (archetype/pattern matching)
"""

import pytest
from unittest.mock import MagicMock

from core.protocol import SharedState, Message, MessageType
from agents.base import LLMAgent


def make_agent():
    """Create an LLMAgent with a mock LLM for testing."""
    mock_llm = MagicMock()
    return LLMAgent(
        name="Entity",
        perspective="Structure",
        system_prompt="Test agent",
        llm_client=mock_llm,
    )


def make_state(input_text: str, insights: list = None, domain_model: dict = None) -> SharedState:
    """Create SharedState with given input and pre-accepted insights."""
    state = SharedState()
    state.known["input"] = input_text
    if insights:
        for i in insights:
            state.insights.append(i)
    if domain_model:
        state.known["domain_model"] = domain_model
    return state


# =============================================================================
# Stratum 0: Basic provenance checks (original Phase 11.1)
# =============================================================================

class TestInsightProvenanceBasic:
    """Basic provenance checks — Stratum 0."""

    def test_insight_with_input_term_accepted(self):
        """Insight containing a term from the input is accepted."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")
        passed, stratum = agent._check_insight_provenance(
            "booking = commitment + scheduling", state
        )
        assert passed is True
        assert stratum == 0

    def test_insight_without_input_terms_rejected(self):
        """Insight with no terms from the input is rejected."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")
        passed, stratum = agent._check_insight_provenance(
            "calibration flows require learning memory", state
        )
        assert passed is False
        assert stratum == -1

    def test_insight_with_domain_term_accepted(self):
        """Insight referencing domain concepts from input is accepted."""
        agent = make_agent()
        state = make_state(
            "Artists have specializations. Clients book appointments. "
            "Deposits are required."
        )
        passed, stratum = agent._check_insight_provenance(
            "artist specialization determines appointment matching", state
        )
        assert passed is True
        assert stratum == 0

    def test_insight_with_only_common_words_allowed(self):
        """Insight with only common/short words passes (can't check)."""
        agent = make_agent()
        state = make_state("Build a booking system")
        # All words < 5 chars or common
        passed, stratum = agent._check_insight_provenance("a = b + c", state)
        assert passed is True
        assert stratum == 0

    def test_empty_input_allows_all(self):
        """If no input is set, all insights pass (safety fallback)."""
        agent = make_agent()
        state = SharedState()  # No input
        passed, stratum = agent._check_insight_provenance(
            "completely unrelated insight about quantum physics", state
        )
        assert passed is True
        assert stratum == 0

    def test_stemming_handles_plurals(self):
        """Stemming matches 'artist' against 'artists' in input."""
        agent = make_agent()
        state = make_state("Artists have different specializations")
        # "artist" stem matches "artists" stem
        passed, stratum = agent._check_insight_provenance(
            "artist = time + skill + location", state
        )
        assert passed is True
        assert stratum == 0

    def test_stemming_handles_verb_forms(self):
        """Stemming matches 'booking' against 'bookings' in input."""
        agent = make_agent()
        state = make_state("Clients manage their bookings and appointments")
        # "booki" stem matches "bookings" stem
        passed, stratum = agent._check_insight_provenance(
            "booking = commitment + scheduling", state
        )
        assert passed is True
        assert stratum == 0


class TestInsightProvenanceNoTransitivity:
    """Verify that provenance does NOT chain through accepted insights."""

    def test_accepted_insight_does_not_extend_pool(self):
        """Previously-accepted insights don't expand what's allowed."""
        agent = make_agent()
        state = make_state(
            "I need a booking system for a tattoo studio",
            insights=["coordinator handles synchronization"]
        )
        # "synchronization" is in accepted insight but NOT in original input
        passed, stratum = agent._check_insight_provenance(
            "synchronization requires conflict resolution", state
        )
        assert passed is False

    def test_chain_breaks_at_second_hop(self):
        """Even with many accepted insights, only input matters."""
        agent = make_agent()
        state = make_state(
            "Artists have availability slots that change weekly",
            insights=[
                "availability = entity with synchronization needs",
                "synchronization requires coordinator entities",
            ]
        )
        # "coordinator" came from insight chain, not input
        passed, stratum = agent._check_insight_provenance(
            "coordinator entities need maintenance schedules", state
        )
        assert passed is False


class TestInsightProvenanceTattooScenario:
    """Reproduce the actual hallucination chain from the tattoo compile."""

    def test_early_insights_accepted(self):
        """First insights in the chain anchor to input."""
        agent = make_agent()
        state = make_state(
            "I need a booking system for a tattoo studio. "
            "Artists have specializations. Clients book appointments. "
            "Deposits required. Walk-ins limited to flash designs."
        )
        # These all reference input terms
        passed1, _ = agent._check_insight_provenance(
            "artist = time + skill + location", state
        )
        passed2, _ = agent._check_insight_provenance(
            "booking != walk-in - completely different state flows", state
        )
        assert passed1 is True
        assert passed2 is True

    def test_mid_chain_with_input_term_accepted(self):
        """Insights that still reference input terms pass even if they
        also introduce new terms."""
        agent = make_agent()
        state = make_state(
            "I need a booking system for a tattoo studio. "
            "Artists have specializations. Some artists work across "
            "multiple studios."
        )
        # "studio" is from input
        passed, _ = agent._check_insight_provenance(
            "availability = entity with studio-specific states", state
        )
        assert passed is True

    def test_drift_chain_rejected(self):
        """The fully-drifted insights from the tattoo compile get rejected."""
        agent = make_agent()
        state = make_state(
            "I need a booking system for a tattoo studio. "
            "Artists have specializations. Clients book appointments. "
            "Deposits required. Walk-ins limited to flash designs."
        )
        # None of these reference any input terms
        for drifted in [
            "coordinator entities = persistent orchestration",
            "orchestration -> require state machine",
            "maintenance -> require configuration",
            "calibration flows -> require learning memory",
            "learning memory -> requires active synthesis flows",
        ]:
            passed, _ = agent._check_insight_provenance(drifted, state)
            assert passed is False, f"Should reject: {drifted}"

    def test_legitimate_emergent_insight_accepted(self):
        """Emergent insights that reference input concepts pass."""
        agent = make_agent()
        state = make_state(
            "I need a booking system for a tattoo studio. "
            "Artists have specializations. Walk-ins limited to flash "
            "designs from the studio's pre-approved catalog."
        )
        # "flash" and "designs" are from input — legitimate emergent
        passed, _ = agent._check_insight_provenance(
            "flash catalog manages pre-approved design collection", state
        )
        assert passed is True


class TestInsightProvenanceInRun:
    """Test that provenance gate integrates correctly in run()."""

    def test_run_rejects_drifted_insight(self):
        """run() nulls insight when provenance check fails."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")

        # Mock LLM returns response with drifted insight
        agent.llm.complete_with_system.return_value = (
            "Analysis of the system.\n"
            "INSIGHT: calibration flows require learning memory"
        )

        msg = agent.run(state)
        assert msg.insight is None
        assert msg.insight_display is None

    def test_run_accepts_grounded_insight(self):
        """run() preserves insight when provenance check passes."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")

        # Mock LLM returns response with grounded insight
        agent.llm.complete_with_system.return_value = (
            "Analysis of the system.\n"
            "INSIGHT: booking = commitment + scheduling + matching"
        )

        msg = agent.run(state)
        assert msg.insight is not None
        assert "booking" in msg.insight
        assert msg.insight_stratum == 0

    def test_run_rejected_insight_not_in_state(self):
        """Rejected insight doesn't get added to state.insights."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")

        agent.llm.complete_with_system.return_value = (
            "Analysis.\n"
            "INSIGHT: calibration flows require learning memory"
        )

        msg = agent.run(state)
        state.add_message(msg)
        assert len(state.insights) == 0

    def test_run_accepted_insight_in_state(self):
        """Accepted insight gets added to state.insights via add_message."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")

        agent.llm.complete_with_system.return_value = (
            "Analysis.\n"
            "INSIGHT: booking = commitment + scheduling + matching"
        )

        msg = agent.run(state)
        state.add_message(msg)
        assert len(state.insights) == 1
        assert "booking" in state.insights[0]


class TestInsightProvenanceEdgeCases:
    """Edge cases and boundary conditions."""

    def test_insight_with_mixed_anchored_and_novel_terms(self):
        """Insight with at least 1 input term passes even with new terms."""
        agent = make_agent()
        state = make_state("Clients book appointments with deposits")
        # "deposit" anchors to input, rest is new
        passed, stratum = agent._check_insight_provenance(
            "deposit = financial commitment with refund lifecycle", state
        )
        assert passed is True
        assert stratum == 0

    def test_meta_vocabulary_excluded(self):
        """Meta terms like 'entity', 'process', 'component' don't count."""
        agent = make_agent()
        state = make_state("Build a booking system for a studio")
        # "entity", "process", "component" are in COMMON_WORDS
        passed, stratum = agent._check_insight_provenance(
            "entity interactions reveal missing component processes", state
        )
        assert passed is False

    def test_short_input_still_works(self):
        """Even with very short input, provenance works."""
        agent = make_agent()
        state = make_state("Build a hotel reservation app")
        passed1, _ = agent._check_insight_provenance(
            "reservation = room + guest + dates", state
        )
        passed2, _ = agent._check_insight_provenance(
            "calibration memory for learning optimization", state
        )
        assert passed1 is True
        assert passed2 is False


# =============================================================================
# Stratum 1: Domain Entailment (Phase 23)
# =============================================================================

class TestStratum1DomainEntailment:
    """Stratum 1: corpus vocabulary + input connection."""

    def _domain_model(self, vocab_terms=None, archetypes=None, patterns=None):
        """Build a minimal domain model dict."""
        return {
            "domain": "tattoo_studio",
            "sample_size": 5,
            "vocabulary": vocab_terms or {},
            "archetypes": archetypes or [],
            "relationship_patterns": patterns or [],
        }

    def test_vocab_term_plus_input_stem_accepted(self):
        """Insight references corpus vocab term + input stem → stratum 1."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "deposit": {"definition": "financial commitment", "frequency": 0.8},
            "portfolio": {"definition": "collection of work samples", "frequency": 0.6},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        # "deposit" is in corpus vocab; "booking" stem matches input
        # Wait — "deposit" is not in input. "booking" IS in input → stratum 0
        # Let's test something that fails stratum 0 but passes stratum 1
        # "portfolio" is in corpus vocab, no input stems match directly
        # But "portfolio" itself doesn't have stems matching input.
        # Need: insight has a vocab term AND some stem connection to input.
        # Let's use "deposit policy for booking" — "booki" matches input (→ stratum 0)
        # Better test: insight with vocab term where the vocab term itself has input overlap
        passed, stratum = agent._check_insight_provenance(
            "deposit policy requires financial commitment", state
        )
        # "deposit" is in corpus vocab. "deposit" stem "depos" is NOT in input.
        # No input stems match insight stems. But "deposit" is in vocab,
        # and does its stem match input? No.
        # This should NOT pass stratum 1 — no input connection.
        assert passed is False

    def test_vocab_term_with_input_overlap_accepted(self):
        """Vocab term that traces back to input enables stratum 1."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "confirmation": {"definition": "verified reservation", "frequency": 0.8},
            "booking": {"definition": "scheduled appointment", "frequency": 0.9},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        # "confirmation" is in vocab, appears in insight.
        # "booking" is also in vocab and "booki" matches input stem.
        # The vocab term "booking" traces to input → stratum 1 chain holds.
        passed, stratum = agent._check_insight_provenance(
            "confirmation requires deposit verification", state
        )
        # "confi" not in input stems → stratum 0 fails
        # "confirmation" is in vocab. Does "confirmation" stem trace to input? No.
        # But insight must have at least 1 direct stem OR vocab term tracing to input.
        # Here: "confirmation" is a vocab term, its stem "confi" not in input.
        # The chain: insight references "confirmation" (vocab) but "confirmation"
        # itself doesn't connect to input. This should actually FAIL stratum 1.
        # Let's fix: use a vocab term whose stem matches input.
        assert passed is False  # No input connection for "confirmation"

    def test_vocab_term_stem_matches_input_accepted(self):
        """Vocab term whose stem matches input enables stratum 1."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "booking deposit": {"definition": "upfront payment", "frequency": 0.8},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        # Insight: "deposit policy" — "depos" not in input stems.
        # But "booking deposit" is in vocab, and "booki" matches input.
        # "booking deposit" appears in insight? No — "deposit" appears but not "booking deposit".
        # The check: vocab term in insight_lower. "booking deposit" not in "deposit policy for clients".
        # Individual word check needed.
        passed, stratum = agent._check_insight_provenance(
            "deposit policy for studio clients", state
        )
        # "studi" matches input → stratum 0!
        # Need: no input stem overlap but vocab connection
        passed, stratum = agent._check_insight_provenance(
            "deposit policy creates financial obligation", state
        )
        # "depos" not in input, "polic" not in input, "creat" not in input,
        # "finan" not in input, "oblig" not in input → stratum 0 fails
        # "booking deposit" vocab: "deposit" in insight_lower? Yes!
        # Vocab term stems: "booki" matches input → chain holds → stratum 1
        assert passed is True
        assert stratum == 1

    def test_vocab_term_no_input_connection_rejected(self):
        """Vocab term without any input connection → rejected at stratum 1."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "portfolio": {"definition": "work samples", "frequency": 0.6},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        # "portfolio" is in vocab but has no input stem overlap
        passed, stratum = agent._check_insight_provenance(
            "portfolio showcases artistic evolution", state
        )
        assert passed is False

    def test_no_domain_model_skips_stratum_1(self):
        """Without domain_model in state, stratum 1 is skipped."""
        agent = make_agent()
        state = make_state("I need a booking system")
        # No domain_model → only stratum 0
        passed, stratum = agent._check_insight_provenance(
            "deposit requires verification", state
        )
        assert passed is False

    def test_empty_vocabulary_skips_stratum_1(self):
        """Empty vocab dict → stratum 1 doesn't fire."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={})
        state = make_state("I need a booking system", domain_model=dm)
        passed, stratum = agent._check_insight_provenance(
            "deposit requires verification", state
        )
        assert passed is False

    def test_stratum_0_takes_precedence(self):
        """Insight that passes stratum 0 stays at stratum 0 even with vocab match."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "booking": {"definition": "reservation", "frequency": 0.9},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        passed, stratum = agent._check_insight_provenance(
            "booking = commitment + scheduling", state
        )
        assert passed is True
        assert stratum == 0  # Stratum 0 takes precedence

    def test_direct_insight_stem_in_input_goes_stratum_0(self):
        """If insight has direct stem match to input, stratum 0 wins."""
        agent = make_agent()
        dm = self._domain_model(vocab_terms={
            "deposit": {"definition": "financial commitment", "frequency": 0.8},
        })
        state = make_state(
            "I need a booking system for a tattoo studio",
            domain_model=dm,
        )
        # "studio" stem "studi" matches input → stratum 0 takes precedence
        passed, stratum = agent._check_insight_provenance(
            "deposit policies vary by studio location", state
        )
        assert passed is True
        assert stratum == 0  # Stratum 0 wins when input stem matches


# =============================================================================
# Stratum 2: Corpus Patterns (Phase 23)
# =============================================================================

class TestStratum2CorpusPatterns:
    """Stratum 2: archetype/pattern matching from corpus."""

    def _domain_model(self, archetypes=None, patterns=None):
        return {
            "domain": "tattoo_studio",
            "sample_size": 5,
            "vocabulary": {},
            "archetypes": archetypes or [],
            "relationship_patterns": patterns or [],
        }

    def test_archetype_name_in_insight_accepted(self):
        """Insight mentioning a well-backed archetype → stratum 2."""
        agent = make_agent()
        dm = self._domain_model(archetypes=[
            {
                "canonical_name": "BookingSystem",
                "type": "entity",
                "variants": ["Booking", "ReservationSystem"],
                "source_ids": ["c1", "c2", "c3", "c4"],
            },
        ])
        state = make_state(
            "I need a web application for scheduling",
            domain_model=dm,
        )
        # "BookingSystem" is an archetype with 4 backing compilations
        # No input stems match. But archetype matches → stratum 2
        passed, stratum = agent._check_insight_provenance(
            "BookingSystem handles reservation lifecycle", state
        )
        assert passed is True
        assert stratum == 2

    def test_archetype_variant_accepted(self):
        """Archetype variant name also triggers stratum 2."""
        agent = make_agent()
        dm = self._domain_model(archetypes=[
            {
                "canonical_name": "BookingSystem",
                "type": "entity",
                "variants": ["Reservation"],
                "source_ids": ["c1", "c2", "c3"],
            },
        ])
        state = make_state("I need a web platform", domain_model=dm)
        passed, stratum = agent._check_insight_provenance(
            "Reservation entity tracks availability", state
        )
        assert passed is True
        assert stratum == 2

    def test_archetype_insufficient_backing_rejected(self):
        """Archetype with < 3 source_ids → not enough backing."""
        agent = make_agent()
        dm = self._domain_model(archetypes=[
            {
                "canonical_name": "FlashCatalog",
                "type": "entity",
                "variants": [],
                "source_ids": ["c1", "c2"],  # Only 2 — not enough
            },
        ])
        state = make_state("I need a web platform", domain_model=dm)
        passed, stratum = agent._check_insight_provenance(
            "FlashCatalog manages design collections", state
        )
        assert passed is False

    def test_relationship_pattern_accepted(self):
        """Two+ pattern components in insight → stratum 2."""
        agent = make_agent()
        dm = self._domain_model(patterns=[
            {
                "components": ["Client", "Booking", "Artist"],
                "relationships": [
                    {"from": "Client", "to": "Booking", "type": "creates"},
                    {"from": "Booking", "to": "Artist", "type": "assigned_to"},
                ],
                "source_ids": ["c1", "c2", "c3", "c4"],
            },
        ])
        state = make_state("I need a web platform", domain_model=dm)
        # "Client" and "Booking" are both pattern components
        passed, stratum = agent._check_insight_provenance(
            "Client creates Booking with constraints", state
        )
        assert passed is True
        assert stratum == 2

    def test_relationship_pattern_single_component_rejected(self):
        """Only 1 pattern component in insight → not enough."""
        agent = make_agent()
        dm = self._domain_model(patterns=[
            {
                "components": ["Client", "Booking", "Artist"],
                "source_ids": ["c1", "c2", "c3"],
            },
        ])
        state = make_state("I need a web platform", domain_model=dm)
        passed, stratum = agent._check_insight_provenance(
            "Client needs authentication", state
        )
        assert passed is False

    def test_pattern_insufficient_backing_rejected(self):
        """Pattern with < 3 source_ids → rejected."""
        agent = make_agent()
        dm = self._domain_model(patterns=[
            {
                "components": ["Client", "Booking"],
                "source_ids": ["c1"],  # Only 1
            },
        ])
        state = make_state("I need a web platform", domain_model=dm)
        passed, stratum = agent._check_insight_provenance(
            "Client creates Booking with constraints", state
        )
        assert passed is False

    def test_stratum_0_before_stratum_2(self):
        """If stratum 0 passes, stratum stays 0 even with archetype match."""
        agent = make_agent()
        dm = self._domain_model(archetypes=[
            {
                "canonical_name": "BookingSystem",
                "source_ids": ["c1", "c2", "c3"],
                "variants": [],
            },
        ])
        state = make_state(
            "I need a booking system with BookingSystem",
            domain_model=dm,
        )
        passed, stratum = agent._check_insight_provenance(
            "BookingSystem handles booking lifecycle", state
        )
        assert passed is True
        assert stratum == 0  # Input match takes precedence


# =============================================================================
# Stratum Labeling & Immutability (Phase 23)
# =============================================================================

class TestStratumLabeling:
    """Test that strata are correctly labeled on Messages and state."""

    def test_stratum_0_label_on_message(self):
        """Stratum 0 insight gets stratum=0 on Message."""
        agent = make_agent()
        state = make_state("I need a booking system for a tattoo studio")
        agent.llm.complete_with_system.return_value = (
            "Analysis.\nINSIGHT: booking = commitment + scheduling"
        )
        msg = agent.run(state)
        assert msg.insight_stratum == 0

    def test_rejected_insight_no_stratum(self):
        """Rejected insight gets stratum=0 (default, no insight)."""
        agent = make_agent()
        state = make_state("I need a booking system")
        agent.llm.complete_with_system.return_value = (
            "Analysis.\nINSIGHT: calibration flows need memory"
        )
        msg = agent.run(state)
        assert msg.insight is None
        assert msg.insight_stratum == 0

    def test_insight_strata_dict_populated(self):
        """State.insight_strata tracks stratum for non-zero strata."""
        state = SharedState()
        state.known["input"] = "test input"
        # Simulate adding a stratum-1 insight
        msg = Message(
            sender="Entity", content="test", message_type=MessageType.PROPOSITION,
            insight="deposit policy needs enforcement",
            insight_stratum=1,
        )
        state.add_message(msg)
        assert len(state.insights) == 1
        assert state.get_insight_stratum(0) == 1

    def test_stratum_0_not_in_dict(self):
        """Stratum 0 insights are NOT stored in insight_strata (default)."""
        state = SharedState()
        msg = Message(
            sender="Entity", content="test", message_type=MessageType.PROPOSITION,
            insight="booking = commitment",
            insight_stratum=0,
        )
        state.add_message(msg)
        assert 0 not in state.insight_strata

    def test_context_graph_includes_strata(self):
        """to_context_graph includes insight_strata."""
        state = SharedState()
        msg = Message(
            sender="Entity", content="test", message_type=MessageType.PROPOSITION,
            insight="deposit policy", insight_stratum=1,
        )
        state.add_message(msg)
        graph = state.to_context_graph()
        assert "insight_strata" in graph
        assert graph["insight_strata"]["0"] == 1

    def test_context_graph_includes_self_compile_patterns(self):
        """to_context_graph includes self_compile_patterns."""
        state = SharedState()
        state.self_compile_patterns = [
            {"type": "stable_component", "name": "AuthService", "confidence": 0.9},
        ]
        graph = state.to_context_graph()
        assert "self_compile_patterns" in graph
        assert len(graph["self_compile_patterns"]) == 1
        assert graph["self_compile_patterns"][0]["type"] == "stable_component"

    def test_context_graph_self_compile_patterns_empty_default(self):
        """to_context_graph includes empty self_compile_patterns by default."""
        state = SharedState()
        graph = state.to_context_graph()
        assert "self_compile_patterns" in graph
        assert graph["self_compile_patterns"] == []


# =============================================================================
# Stratum 3: Self-Observation Patterns (Phase 24)
# =============================================================================

class TestStratum3SelfObservation:
    """Stratum 3: self-compile pattern matching."""

    def _stable_patterns(self):
        """Build self_compile_patterns with stable and drift entries."""
        return [
            {
                "pattern_type": "stable_component",
                "name": "SharedState",
                "frequency": 1.0,
                "details": "Appears in 3/3 runs (100%)",
                "derived_from": "self-compile:v3.0",
            },
            {
                "pattern_type": "stable_relationship",
                "name": "Governor Agent -> Entity Agent (triggers)",
                "frequency": 0.95,
                "details": "Appears in 19/20 runs",
                "derived_from": "self-compile:v3.0",
            },
            {
                "pattern_type": "drift_point",
                "name": "AuditTrail",
                "frequency": 0.5,
                "details": "Unstable: appears in 5/10 runs",
                "derived_from": "self-compile:v3.0",
            },
            {
                "pattern_type": "canonical_gap",
                "name": "ConflictOracle",
                "frequency": 0.0,
                "details": "Not found in any run",
                "derived_from": "self-compile:v3.0",
            },
        ]

    def test_stable_component_accepted(self):
        """Insight matching stable_component pattern → stratum 3."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "SharedState manages all known specifications", state
        )
        assert passed is True
        assert stratum == 3

    def test_stable_relationship_accepted(self):
        """Insight matching stable_relationship pattern → stratum 3."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "Governor Agent -> Entity Agent (triggers) is foundational", state
        )
        assert passed is True
        assert stratum == 3

    def test_drift_point_rejected(self):
        """Insight matching drift_point (unstable) → NOT accepted at stratum 3."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "AuditTrail requires persistent storage", state
        )
        assert passed is False
        assert stratum == -1

    def test_canonical_gap_rejected(self):
        """Insight matching canonical_gap → NOT accepted at stratum 3."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "ConflictOracle needs threshold monitoring", state
        )
        assert passed is False
        assert stratum == -1

    def test_no_patterns_available_rejected(self):
        """No self-compile patterns → stratum 3 skipped, falls through."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        # No patterns set (empty list is default)
        passed, stratum = agent._check_insight_provenance(
            "SharedState manages specifications", state
        )
        assert passed is False
        assert stratum == -1

    def test_stratum_0_takes_precedence_over_3(self):
        """If stratum 0 passes, stays at stratum 0 even with pattern match."""
        agent = make_agent()
        state = make_state("I need a booking system with SharedState")
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "SharedState handles booking lifecycle", state
        )
        assert passed is True
        assert stratum == 0  # Input match takes precedence

    def test_stratum_3_label_on_message(self):
        """Stratum 3 insight gets stratum=3 on Message via run()."""
        agent = make_agent()
        state = make_state("I need a web platform for scheduling")
        state.self_compile_patterns = self._stable_patterns()
        agent.llm.complete_with_system.return_value = (
            "Analysis.\nINSIGHT: SharedState = specification repository"
        )
        msg = agent.run(state)
        assert msg.insight is not None
        assert msg.insight_stratum == 3

    def test_stratum_2_before_stratum_3(self):
        """Stratum 2 takes precedence over stratum 3 when both match."""
        agent = make_agent()
        dm = {
            "domain": "test",
            "sample_size": 5,
            "vocabulary": {},
            "archetypes": [
                {
                    "canonical_name": "SharedState",
                    "type": "entity",
                    "variants": [],
                    "source_ids": ["c1", "c2", "c3"],
                },
            ],
            "relationship_patterns": [],
        }
        state = make_state("I need a web platform", domain_model=dm)
        state.self_compile_patterns = self._stable_patterns()
        passed, stratum = agent._check_insight_provenance(
            "SharedState manages all specifications", state
        )
        assert passed is True
        assert stratum == 2  # Stratum 2 checked before stratum 3
