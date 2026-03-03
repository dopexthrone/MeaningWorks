"""
Phase 8.1: Tests for core/digest.py - Dialogue digest for synthesis.

Tests build_dialogue_digest() with various SharedState configurations.
"""

import pytest
from core.digest import (
    build_dialogue_digest,
    _find_insight_source,
    _estimate_tokens,
    _truncate_to_budget,
    extract_dialogue_algorithms,
    MAX_DIGEST_CHARS,
    MAX_EXCHANGE_CHARS,
    MAX_EXCHANGES,
)
from core.protocol import (
    SharedState,
    Message,
    MessageType,
    ConfidenceVector,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def empty_state():
    """SharedState with no data."""
    return SharedState()


@pytest.fixture
def state_with_insights():
    """SharedState with insights from dialogue."""
    state = SharedState()
    # Add messages that produce insights
    m1 = Message(
        sender="Entity",
        content="User entity has email, password_hash fields",
        message_type=MessageType.PROPOSITION,
        insight="User entity contains email, password_hash",
    )
    m2 = Message(
        sender="Process",
        content="Login flow validates credentials then creates session",
        message_type=MessageType.PROPOSITION,
        insight="login(email, password) -> Session validates credentials",
    )
    state.add_message(m1)
    state.add_message(m2)
    return state


@pytest.fixture
def state_with_conflicts():
    """SharedState with conflicts."""
    state = SharedState()
    state.add_conflict(
        "Entity", "Process",
        "session storage",
        {"Entity": "Session stored in DB", "Process": "Session stored in Redis"}
    )
    state.add_conflict(
        "Entity", "Process",
        "auth method",
        {"Entity": "JWT tokens", "Process": "Session cookies"}
    )
    state.resolve_conflict(1, "Use JWT tokens with cookie transport")
    return state


@pytest.fixture
def state_with_challenges():
    """SharedState with CHALLENGE and ACCOMMODATION messages."""
    state = SharedState()
    state.add_message(Message(
        sender="Entity",
        content="I see User as a simple entity with email and name fields.",
        message_type=MessageType.PROPOSITION,
    ))
    state.add_message(Message(
        sender="Process",
        content="CHALLENGE: User needs state tracking for login attempts. A simple entity misses the behavioral aspect of account lockout after N failures.",
        message_type=MessageType.CHALLENGE,
    ))
    state.add_message(Message(
        sender="Entity",
        content="ACCOMMODATION: Adding login_attempts and locked_until fields to User entity to support lockout behavior.",
        message_type=MessageType.ACCOMMODATION,
    ))
    return state


@pytest.fixture
def rich_state():
    """Full SharedState with all sections populated."""
    state = SharedState()
    state.known["input"] = "Build an auth system"
    state.known["intent"] = {"core_need": "authentication", "domain": "security"}

    # Personas
    state.personas = [
        {
            "name": "Security Architect",
            "perspective": "Focus on authentication security",
            "priorities": ["Prevent credential stuffing", "Secure session management"],
        },
        {
            "name": "UX Designer",
            "perspective": "Focus on user experience during login",
            "priorities": ["Minimal friction login", "Clear error messages"],
        },
    ]

    # Confidence
    state.confidence = ConfidenceVector(
        structural=0.7, behavioral=0.6, coverage=0.5, consistency=0.8
    )

    # History with insights
    for i in range(6):
        sender = "Entity" if i % 2 == 0 else "Process"
        mtype = MessageType.PROPOSITION if i < 4 else MessageType.AGREEMENT
        m = Message(
            sender=sender,
            content=f"Turn {i+1} analysis of auth system component {i}",
            message_type=mtype,
            insight=f"Insight {i+1}: component {i} detail",
        )
        state.add_message(m)

    # Challenge and accommodation
    state.add_message(Message(
        sender="Process",
        content="CHALLENGE: Missing rate limiting on login endpoint",
        message_type=MessageType.CHALLENGE,
    ))
    state.add_message(Message(
        sender="Entity",
        content="ACCOMMODATION: Adding RateLimiter entity with window and max_attempts",
        message_type=MessageType.ACCOMMODATION,
    ))

    # Conflicts
    state.add_conflict("Entity", "Process", "token format",
                       {"Entity": "Opaque tokens", "Process": "JWT"})

    # Unknowns
    state.unknown = ["Password hashing algorithm", "Token expiry duration"]

    return state


# =============================================================================
# BASIC TESTS
# =============================================================================


class TestDigestFromEmptyState:
    """Digest from empty state should be safe and minimal."""

    def test_empty_state_returns_empty_string(self, empty_state):
        digest = build_dialogue_digest(empty_state)
        assert digest == ""

    def test_empty_state_no_error(self, empty_state):
        # Should not raise
        build_dialogue_digest(empty_state)


class TestDigestInsights:
    """Test insight section of digest."""

    def test_insights_appear_in_digest(self, state_with_insights):
        digest = build_dialogue_digest(state_with_insights)
        assert "INSIGHTS:" in digest
        assert "User entity contains email, password_hash" in digest
        assert "login(email, password) -> Session" in digest

    def test_insight_source_attribution(self, state_with_insights):
        digest = build_dialogue_digest(state_with_insights)
        assert "Entity" in digest
        assert "Process" in digest

    def test_insight_count_matches(self, state_with_insights):
        digest = build_dialogue_digest(state_with_insights)
        # 2 insights, each on its own line (Phase 12.1c: tiered format)
        insight_section = digest.split("INSIGHTS:\n")[1]
        lines = [l for l in insight_section.strip().split("\n") if "[" in l and "]" in l and "/" in l]
        assert len(lines) == 2


class TestDigestConfidence:
    """Test confidence section of digest."""

    def test_confidence_appears_when_nonzero(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert "CONFIDENCE:" in digest
        assert "structural: 70%" in digest
        assert "behavioral: 60%" in digest

    def test_confidence_absent_when_zero(self, empty_state):
        digest = build_dialogue_digest(empty_state)
        assert "CONFIDENCE:" not in digest


class TestDigestConflicts:
    """Test conflict section of digest."""

    def test_conflicts_appear(self, state_with_conflicts):
        digest = build_dialogue_digest(state_with_conflicts)
        assert "CONFLICTS:" in digest
        assert "session storage" in digest

    def test_resolved_conflict_marked(self, state_with_conflicts):
        digest = build_dialogue_digest(state_with_conflicts)
        assert "RESOLVED" in digest
        assert "UNRESOLVED" in digest

    def test_resolution_shown(self, state_with_conflicts):
        digest = build_dialogue_digest(state_with_conflicts)
        assert "JWT tokens with cookie transport" in digest


class TestDigestUnknowns:
    """Test unknowns section of digest."""

    def test_unknowns_appear(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert "UNKNOWNS:" in digest
        assert "Password hashing algorithm" in digest
        assert "Token expiry duration" in digest

    def test_no_unknowns_section_when_empty(self, empty_state):
        digest = build_dialogue_digest(empty_state)
        assert "UNKNOWNS:" not in digest


class TestDigestKeyExchanges:
    """Test key exchanges section (CHALLENGE/ACCOMMODATION messages)."""

    def test_challenges_appear(self, state_with_challenges):
        digest = build_dialogue_digest(state_with_challenges)
        assert "KEY EXCHANGES:" in digest
        assert "CHALLENGE" in digest

    def test_accommodations_appear(self, state_with_challenges):
        digest = build_dialogue_digest(state_with_challenges)
        assert "ACCOMMODATION" in digest

    def test_propositions_excluded(self, state_with_challenges):
        """PROPOSITION messages should NOT appear in key exchanges."""
        digest = build_dialogue_digest(state_with_challenges)
        exchanges_section = digest.split("KEY EXCHANGES:\n")[1] if "KEY EXCHANGES:" in digest else ""
        assert "PROPOSITION" not in exchanges_section

    def test_exchange_content_truncated(self):
        """Long messages should be truncated to MAX_EXCHANGE_CHARS."""
        state = SharedState()
        long_content = "X" * 1000
        state.add_message(Message(
            sender="Process",
            content=long_content,
            message_type=MessageType.CHALLENGE,
        ))
        digest = build_dialogue_digest(state)
        assert "KEY EXCHANGES:" in digest
        assert "..." in digest

    def test_max_exchanges_limit(self):
        """Only MAX_EXCHANGES messages should appear."""
        state = SharedState()
        for i in range(MAX_EXCHANGES + 5):
            state.add_message(Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content=f"Challenge {i}",
                message_type=MessageType.CHALLENGE,
            ))
        digest = build_dialogue_digest(state)
        # Count exchange lines
        exchanges_section = digest.split("KEY EXCHANGES:\n")[1]
        exchange_lines = [l for l in exchanges_section.strip().split("\n") if l.strip().startswith("[")]
        assert len(exchange_lines) == MAX_EXCHANGES


class TestDigestPersonas:
    """Test persona priorities section."""

    def test_persona_priorities_appear(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert "PERSONA PRIORITIES:" in digest
        assert "Security Architect" in digest
        assert "UX Designer" in digest

    def test_top_2_priorities_shown(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert "Prevent credential stuffing" in digest
        assert "Secure session management" in digest

    def test_fallback_to_perspective(self):
        """When no priorities, use perspective."""
        state = SharedState()
        state.personas = [
            {"name": "TestPersona", "perspective": "Focus on testing things"}
        ]
        digest = build_dialogue_digest(state)
        assert "TestPersona" in digest
        assert "Focus on testing things" in digest


class TestDigestTokenBudget:
    """Test token budget enforcement."""

    def test_digest_within_budget(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert len(digest) <= MAX_DIGEST_CHARS

    def test_large_state_truncated(self):
        """Very large state should be truncated to budget."""
        state = SharedState()
        # Add many long insights
        for i in range(200):
            m = Message(
                sender="Entity",
                content="A" * 500,
                message_type=MessageType.PROPOSITION,
                insight=f"Very long insight number {i}: " + "detail " * 50,
            )
            state.add_message(m)
        digest = build_dialogue_digest(state)
        assert len(digest) <= MAX_DIGEST_CHARS + 50  # Small margin for truncation message


class TestDigestDeterminism:
    """Test deterministic output."""

    def test_same_input_same_output(self, rich_state):
        digest1 = build_dialogue_digest(rich_state)
        digest2 = build_dialogue_digest(rich_state)
        assert digest1 == digest2

    def test_deterministic_across_calls(self, state_with_insights):
        results = [build_dialogue_digest(state_with_insights) for _ in range(5)]
        assert all(r == results[0] for r in results)


class TestDigestAllSections:
    """Test full digest with all sections."""

    def test_all_sections_present(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        assert "INSIGHTS:" in digest
        assert "CONFIDENCE:" in digest
        assert "CONFLICTS:" in digest
        assert "UNKNOWNS:" in digest
        assert "KEY EXCHANGES:" in digest
        assert "PERSONA PRIORITIES:" in digest

    def test_sections_ordered(self, rich_state):
        digest = build_dialogue_digest(rich_state)
        # Sections should appear in order
        sections = ["INSIGHTS:", "CONFIDENCE:", "CONFLICTS:", "UNKNOWNS:", "KEY EXCHANGES:", "PERSONA PRIORITIES:"]
        positions = [digest.index(s) for s in sections]
        assert positions == sorted(positions)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 0

    def test_short(self):
        assert _estimate_tokens("hello world") == 2  # 11 / 4 = 2


class TestTruncateToBudget:
    def test_no_truncation_needed(self):
        text = "short text"
        assert _truncate_to_budget(text, 100) == text

    def test_truncation_at_newline(self):
        text = "line1\nline2\nline3\nline4\nline5"
        result = _truncate_to_budget(text, 20)
        assert "[...truncated]" in result
        assert len(result) < len(text) + 20


class TestFindInsightSource:
    def test_direct_match(self, state_with_insights):
        source = _find_insight_source(
            state_with_insights,
            "User entity contains email, password_hash",
            0
        )
        assert "Entity" in source

    def test_fallback_attribution(self):
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content="analysis",
            message_type=MessageType.PROPOSITION,
        ))
        state.insights.append("manual insight")
        source = _find_insight_source(state, "manual insight", 0)
        assert "Entity" in source


# =============================================================================
# ALGORITHM EXTRACTION TESTS
# =============================================================================


class TestExtractDialogueAlgorithms:
    """Tests for extract_dialogue_algorithms()."""

    def test_extract_algorithms_basic(self):
        """Single ALGORITHM: block → dict with component, method_name, steps."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                "ALGORITHM: PayloadAggregator.aggregate\n"
                "  1. Collect all pending payloads\n"
                "  2. If payloads empty then return None\n"
                "  3. Return merged payload\n"
            ),
            message_type=MessageType.PROPOSITION,
        ))
        algos = extract_dialogue_algorithms(state)
        assert len(algos) == 1
        assert algos[0]["component"] == "PayloadAggregator"
        assert algos[0]["method_name"] == "aggregate"
        assert len(algos[0]["steps"]) == 3
        assert "Collect all pending" in algos[0]["steps"][0]
        assert algos[0]["source"] == "dialogue"

    def test_extract_algorithms_multiple(self):
        """Two algorithms on different components."""
        state = SharedState()
        state.add_message(Message(
            sender="Process",
            content=(
                "ALGORITHM: Router.dispatch\n"
                "  1. Parse incoming request\n"
                "  2. Match route\n"
                "\n"
                "ALGORITHM: Cache.evict\n"
                "  1. Find least recently used entry\n"
                "  2. Remove from cache\n"
            ),
            message_type=MessageType.PROPOSITION,
        ))
        algos = extract_dialogue_algorithms(state)
        assert len(algos) == 2
        assert algos[0]["component"] == "Router"
        assert algos[1]["component"] == "Cache"

    def test_extract_algorithms_with_pre_post(self):
        """PRE: and POST: lines parsed."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content=(
                "ALGORITHM: Validator.validate\n"
                "  PRE: input must not be empty\n"
                "  1. Check schema\n"
                "  2. Return result\n"
                "  POST: validation flag is set\n"
            ),
            message_type=MessageType.PROPOSITION,
        ))
        algos = extract_dialogue_algorithms(state)
        assert len(algos) == 1
        assert algos[0]["preconditions"] == ["input must not be empty"]
        assert algos[0]["postconditions"] == ["validation flag is set"]
        assert len(algos[0]["steps"]) == 2

    def test_extract_algorithms_no_match(self):
        """No ALGORITHM: blocks → empty list."""
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content="METHOD: Foo.bar() -> None\nJust some text",
            message_type=MessageType.PROPOSITION,
        ))
        algos = extract_dialogue_algorithms(state)
        assert algos == []

    def test_extract_algorithms_non_agent_ignored(self):
        """System/Governor messages skipped."""
        state = SharedState()
        state.add_message(Message(
            sender="System",
            content="ALGORITHM: Foo.bar\n  1. Do something\n",
            message_type=MessageType.PROPOSITION,
        ))
        algos = extract_dialogue_algorithms(state)
        assert algos == []

    def test_parse_constrain_includes_algorithms(self):
        """parse_constrain_artifact returns algorithms key."""
        from core.pipeline import parse_constrain_artifact
        state = SharedState()
        state.known["input"] = "test input"
        state.add_message(Message(
            sender="Entity",
            content=(
                'CONSTRAINT: Router | description="must handle errors" | derived_from="input"\n'
                "ALGORITHM: Router.dispatch\n"
                "  1. Parse request\n"
                "  2. Route to handler\n"
            ),
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_constrain_artifact(state)
        assert "algorithms" in artifact
        assert len(artifact["algorithms"]) == 1
        assert artifact["algorithms"][0]["component"] == "Router"

    def test_handoff_includes_algorithms(self):
        """_extract_handoff for constrain includes algorithm strings."""
        from core.pipeline import _extract_handoff
        artifact = {
            "constraints": [{"description": "must handle errors"}],
            "algorithms": [
                {"component": "Router", "method_name": "dispatch"},
            ],
        }
        handoff = _extract_handoff("constrain", artifact)
        assert "Router.dispatch" in handoff.algorithms
        assert handoff.stage_source == "constrain"

    def test_format_precomputed_includes_algorithms(self):
        """format_precomputed_structure renders ALGORITHM lines."""
        from core.pipeline import format_precomputed_structure, PipelineState, StageRecord, StageResult
        pipeline = PipelineState(
            original_input="test",
            intent={"core_need": "test", "domain": "test"},
            personas=[],
        )
        # Add decompose stage (required for components)
        decompose_state = SharedState()
        decompose_artifact = {
            "components": [{"name": "Router", "type": "process", "derived_from": "test"}],
            "type_assignments": {"Router": "process"},
        }
        pipeline.add_stage(StageRecord(
            name="decompose", state=decompose_state, artifact=decompose_artifact,
            gate_result=StageResult(success=True), turn_count=2, duration_seconds=1.0,
        ))
        # Add constrain stage with algorithms
        constrain_state = SharedState()
        constrain_artifact = {
            "constraints": [],
            "methods": [],
            "state_machines": [],
            "algorithms": [
                {"component": "Router", "method_name": "dispatch",
                 "steps": ["1. Parse request", "2. Route to handler"]},
            ],
        }
        pipeline.add_stage(StageRecord(
            name="constrain", state=constrain_state, artifact=constrain_artifact,
            gate_result=StageResult(success=True), turn_count=2, duration_seconds=1.0,
        ))
        result = format_precomputed_structure(pipeline)
        assert "ALGORITHM: Router.dispatch" in result
        assert "Parse request" in result
