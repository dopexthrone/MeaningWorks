"""
Phase 12.3: Method & State Machine Extraction Pipeline.

Tests for:
- 12.3a: Extract METHOD: and STATES: lines from dialogue (16 tests)
- 12.3a: Format extracted data for synthesis prompt (3 tests)
- 12.3b: Engine integration — extraction, dedup, synthesis prompt (5 tests)
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from core.digest import (
    extract_dialogue_methods,
    _parse_method_params,
    extract_dialogue_state_machines,
    extract_pattern_method_stubs,
    format_method_section,
    _extract_pattern_matches,
)
from core.protocol import SharedState, Message, MessageType, ConfidenceVector
from core.engine import MotherlabsEngine, CompileResult
from core.llm import BaseLLMClient
from agents.base import AgentCallResult


def _synthesis_result(content: str) -> AgentCallResult:
    """Wrap a JSON content string in an AgentCallResult for synthesis mocks."""
    msg = Message(sender="Synthesis", content=content, message_type=MessageType.PROPOSITION)
    return AgentCallResult(
        agent_name="Synthesis", response_text=content, message=msg,
        conflicts=(), unknowns=(), fractures=(),
        confidence_boost=0.0, agent_dimension="", has_insight=False,
        token_usage={},
    )


# =============================================================================
# HELPERS
# =============================================================================


def _make_state_with_messages(*messages):
    """Build a SharedState with the given messages."""
    state = SharedState()
    for sender, content in messages:
        msg = Message(
            sender=sender,
            content=content,
            message_type=MessageType.PROPOSITION,
        )
        state.add_message(msg)
    return state


# =============================================================================
# 12.3a — METHOD EXTRACTION (tests 1-10)
# =============================================================================


class TestExtractDialogueMethods:
    """Test extract_dialogue_methods() parsing of METHOD: lines."""

    def test_extract_method_basic(self):
        """Single METHOD: line parsed correctly."""
        state = _make_state_with_messages(
            ("Entity", "Analyzing structure.\nMETHOD: AuditTrail.append_link(source, target) -> DerivationLink")
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        m = methods[0]
        assert m["component"] == "AuditTrail"
        assert m["name"] == "append_link"
        assert m["return_type"] == "DerivationLink"
        assert m["source"] == "dialogue"
        assert len(m["parameters"]) == 2
        assert m["parameters"][0]["name"] == "source"

    def test_extract_method_with_return_type(self):
        """-> ReturnType captured correctly."""
        state = _make_state_with_messages(
            ("Process", "METHOD: Pipeline.execute(input: str) -> Blueprint")
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert methods[0]["return_type"] == "Blueprint"

    def test_extract_method_no_return_type(self):
        """No return type defaults to 'None'."""
        state = _make_state_with_messages(
            ("Entity", "METHOD: Cache.invalidate(key: str)")
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert methods[0]["return_type"] == "None"

    def test_extract_method_multiple_params(self):
        """3+ parameters parsed correctly."""
        state = _make_state_with_messages(
            ("Entity", "METHOD: AuditTrail.append_link(source: Any, target: Any, contribution_type: Any, phase: Any, turn: Any) -> DerivationLink")
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert len(methods[0]["parameters"]) == 5
        assert methods[0]["parameters"][2]["name"] == "contribution_type"
        assert methods[0]["parameters"][2]["type_hint"] == "Any"

    def test_extract_method_no_params(self):
        """method() with no params -> empty params list."""
        state = _make_state_with_messages(
            ("Process", "METHOD: Engine.reset() -> None")
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert methods[0]["parameters"] == []

    def test_extract_multiple_methods_one_message(self):
        """Message with 2+ METHOD: lines."""
        state = _make_state_with_messages(
            ("Entity", (
                "Analyzing AuditTrail structure.\n"
                "METHOD: AuditTrail.append_link(source, target) -> DerivationLink\n"
                "METHOD: AuditTrail.trace_back(element: Any) -> List"
            ))
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 2
        assert methods[0]["name"] == "append_link"
        assert methods[1]["name"] == "trace_back"

    def test_extract_methods_across_messages(self):
        """Methods from Entity and Process messages both captured."""
        state = _make_state_with_messages(
            ("Entity", "METHOD: User.validate(email: str) -> bool"),
            ("Process", "METHOD: AuthFlow.login(email: str, password: str) -> Session"),
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 2
        assert {m["component"] for m in methods} == {"User", "AuthFlow"}

    def test_no_methods_returns_empty(self):
        """Dialogue without METHOD: lines returns []."""
        state = _make_state_with_messages(
            ("Entity", "User entity has email, password_hash fields"),
            ("Process", "Login flow validates credentials"),
        )
        methods = extract_dialogue_methods(state)
        assert methods == []

    def test_ignores_non_agent_messages(self):
        """System/Governor METHOD: lines skipped."""
        state = _make_state_with_messages(
            ("System", "METHOD: Fake.method() -> None"),
            ("Governor", "METHOD: Also.fake() -> None"),
            ("Entity", "METHOD: Real.method() -> str"),
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert methods[0]["component"] == "Real"

    def test_malformed_method_skipped(self):
        """Bad format doesn't crash, just skipped."""
        state = _make_state_with_messages(
            ("Entity", "METHOD: not a valid method line\nMETHOD: Good.method(x: int) -> bool"),
        )
        methods = extract_dialogue_methods(state)
        assert len(methods) == 1
        assert methods[0]["name"] == "method"


# =============================================================================
# 12.3a — STATE MACHINE EXTRACTION (tests 11-14)
# =============================================================================


class TestExtractDialogueStateMachines:
    """Test extract_dialogue_state_machines() parsing of STATES: blocks."""

    def test_extract_state_machine_basic(self):
        """STATES: block parsed into states list."""
        state = _make_state_with_messages(
            ("Process", (
                "STATES: CompilationPipeline\n"
                "  states: [AWAITING_INPUT, INTENT_EXTRACTION, DIALOGUE, SYNTHESIS]\n"
                "  transitions:\n"
                '    - AWAITING_INPUT -> INTENT_EXTRACTION on "user submits"\n'
            ))
        )
        machines = extract_dialogue_state_machines(state)
        assert len(machines) == 1
        sm = machines[0]
        assert sm["component"] == "CompilationPipeline"
        assert "AWAITING_INPUT" in sm["states"]
        assert "SYNTHESIS" in sm["states"]
        assert len(sm["states"]) == 4
        assert sm["source"] == "dialogue"

    def test_extract_state_machine_transitions(self):
        """A -> B on "trigger" transitions parsed."""
        state = _make_state_with_messages(
            ("Process", (
                "STATES: DialogueLoop\n"
                "  states: [ENTITY_TURN, PROCESS_TURN, GOVERNOR_CHECK]\n"
                "  transitions:\n"
                '    - ENTITY_TURN -> PROCESS_TURN on "entity responds"\n'
                '    - PROCESS_TURN -> GOVERNOR_CHECK on "process responds"\n'
                '    - GOVERNOR_CHECK -> ENTITY_TURN on "continue"\n'
            ))
        )
        machines = extract_dialogue_state_machines(state)
        assert len(machines) == 1
        sm = machines[0]
        assert len(sm["transitions"]) == 3
        assert sm["transitions"][0]["from"] == "ENTITY_TURN"
        assert sm["transitions"][0]["to"] == "PROCESS_TURN"
        assert sm["transitions"][0]["trigger"] == "entity responds"
        assert sm["transitions"][2]["trigger"] == "continue"

    def test_extract_multiple_state_machines(self):
        """Multiple STATES: blocks from one dialogue."""
        state = _make_state_with_messages(
            ("Process", (
                "STATES: Pipeline\n"
                "  states: [A, B]\n"
                "  transitions:\n"
                '    - A -> B on "go"\n'
            )),
            ("Process", (
                "STATES: Dialogue\n"
                "  states: [X, Y]\n"
                "  transitions:\n"
                '    - X -> Y on "next"\n'
            )),
        )
        machines = extract_dialogue_state_machines(state)
        assert len(machines) == 2
        assert {m["component"] for m in machines} == {"Pipeline", "Dialogue"}

    def test_no_states_returns_empty(self):
        """No STATES: blocks returns []."""
        state = _make_state_with_messages(
            ("Process", "The pipeline processes input through several stages."),
        )
        machines = extract_dialogue_state_machines(state)
        assert machines == []


# =============================================================================
# 12.3a — PATTERN METHOD STUBS (tests 15-16)
# =============================================================================


class TestExtractPatternMethodStubs:
    """Test extract_pattern_method_stubs() from pattern transfer hints."""

    def test_pattern_hints_to_stubs(self):
        """'X works like Y — a, b, c' produces 3 method stubs."""
        insights = ["flash catalog works like Pinterest — browse + preview + select"]
        stubs = extract_pattern_method_stubs(insights)
        assert len(stubs) == 3
        assert stubs[0]["name"] == "browse"
        assert stubs[0]["source"] == "pattern_transfer"
        assert stubs[0]["return_type"] == "None"
        assert stubs[0]["parameters"] == []
        assert "Pinterest" in stubs[0]["derived_from"]

    def test_pattern_no_hints_no_stubs(self):
        """No hints in pattern → no stubs."""
        # An insight with "like" but no hints after
        insights = ["simple concept is like Basic storage"]
        stubs = extract_pattern_method_stubs(insights)
        assert stubs == []


# =============================================================================
# 12.3a — FORMATTING (tests 17-19)
# =============================================================================


class TestFormatMethodSection:
    """Test format_method_section() output formatting."""

    def test_format_groups_by_component(self):
        """Methods for same component grouped together."""
        methods = [
            {
                "component": "AuditTrail",
                "name": "append_link",
                "parameters": [{"name": "source", "type_hint": "Any"}],
                "return_type": "DerivationLink",
                "derived_from": "METHOD: AuditTrail.append_link(source) -> DerivationLink",
                "source": "dialogue",
            },
            {
                "component": "AuditTrail",
                "name": "trace_back",
                "parameters": [{"name": "element", "type_hint": "Any"}],
                "return_type": "List",
                "derived_from": "METHOD: AuditTrail.trace_back(element) -> List",
                "source": "dialogue",
            },
        ]
        result = format_method_section(methods, [])
        assert "AuditTrail:" in result
        assert "append_link(source: Any) -> DerivationLink" in result
        assert "trace_back(element: Any) -> List" in result
        # Should be grouped — AuditTrail appears only once as header
        assert result.count("AuditTrail:") == 1

    def test_format_includes_state_machines(self):
        """State machines included in output."""
        machines = [
            {
                "component": "Pipeline",
                "states": ["A", "B", "C"],
                "transitions": [{"from": "A", "to": "B", "trigger": "start"}],
                "derived_from": "STATES: Pipeline",
                "source": "dialogue",
            },
        ]
        result = format_method_section([], machines)
        assert "Pipeline:" in result
        assert "states: [A, B, C]" in result
        assert 'A -> B on "start"' in result

    def test_format_empty_returns_empty(self):
        """No methods/states → empty string."""
        result = format_method_section([], [])
        assert result == ""


# =============================================================================
# 12.3b — ENGINE INTEGRATION (tests 20-24)
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


class TestEngineExtractMethods:
    """Test _extract_methods_for_synthesis() on the engine."""

    def test_engine_extracts_methods(self):
        """_extract_methods_for_synthesis returns methods from mock dialogue."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content="METHOD: AuditTrail.append_link(source: Any, target: Any) -> DerivationLink",
            message_type=MessageType.PROPOSITION,
        ))
        state.add_message(Message(
            sender="Process",
            content="METHOD: Pipeline.run(input: str) -> Blueprint",
            message_type=MessageType.PROPOSITION,
        ))
        state.insights = []

        methods, machines = engine._extract_methods_for_synthesis(state)
        assert len(methods) == 2
        assert {m["component"] for m in methods} == {"AuditTrail", "Pipeline"}
        assert machines == []

    def test_engine_dedup_prefers_dialogue(self):
        """Same method from dialogue + pattern: dialogue wins."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        # Dialogue method
        state.add_message(Message(
            sender="Entity",
            content="METHOD: Catalog.browse(query: str) -> List",
            message_type=MessageType.PROPOSITION,
        ))
        # Pattern insight that would also produce a "browse" stub for "catalog"
        state.insights = ["catalog works like Pinterest — browse + preview"]

        methods, machines = engine._extract_methods_for_synthesis(state)

        # browse should appear once, from dialogue (has parameters)
        browse_methods = [m for m in methods if m["name"] == "browse"]
        assert len(browse_methods) == 1
        assert browse_methods[0]["source"] == "dialogue"
        assert browse_methods[0]["parameters"][0]["name"] == "query"

        # preview should also appear (from pattern transfer)
        preview_methods = [m for m in methods if m["name"] == "preview"]
        assert len(preview_methods) == 1
        assert preview_methods[0]["source"] == "pattern_transfer"


class TestSynthesisPromptIntegration:
    """Test that synthesis prompt includes SECTION 3 when methods extracted."""

    def test_synthesis_prompt_has_section_3(self):
        """When methods extracted, prompt contains SECTION 3."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.known["input"] = "Build a user authentication system"
        state.known["extracted_methods"] = [
            {
                "component": "AuditTrail",
                "name": "append_link",
                "parameters": [{"name": "source", "type_hint": "Any"}],
                "return_type": "DerivationLink",
                "derived_from": "METHOD: AuditTrail.append_link(source) -> DerivationLink",
                "source": "dialogue",
            }
        ]
        state.known["extracted_state_machines"] = []

        # Capture the prompt sent to synthesis agent
        captured_prompts = []

        def mock_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result(json.dumps({
                "components": [{"name": "AuditTrail", "type": "entity", "derived_from": "test"}],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            }))

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run

        blueprint, retries = engine._synthesize(state)
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "SECTION 3: EXTRACTED METHODS, STATE MACHINES & ALGORITHMS" in prompt
        assert "append_link" in prompt
        assert "AuditTrail" in prompt

    def test_synthesis_prompt_no_section_without_methods(self):
        """No methods extracted → no SECTION 3."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.known["input"] = "Build a simple app"

        captured_prompts = []

        def mock_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result(json.dumps({
                "components": [{"name": "App", "type": "entity", "derived_from": "test"}],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            }))

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run

        blueprint, retries = engine._synthesize(state)
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "SECTION 3: EXTRACTED METHODS" not in prompt

    def test_section_renumbering(self):
        """Pattern transfer is now SECTION 4 (not 3)."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.known["input"] = "Build something"
        state.insights = ["booking flow works like Stripe checkout — validate + charge"]
        state.known["extracted_methods"] = [
            {
                "component": "Booking",
                "name": "reserve",
                "parameters": [],
                "return_type": "None",
                "derived_from": "METHOD: Booking.reserve()",
                "source": "dialogue",
            }
        ]

        captured_prompts = []

        def mock_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result(json.dumps({
                "components": [{"name": "Booking", "type": "process", "derived_from": "test"}],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            }))

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run

        blueprint, retries = engine._synthesize(state)
        prompt = captured_prompts[0]

        # SECTION 3 should be methods
        assert "SECTION 3: EXTRACTED METHODS" in prompt
        # SECTION 4 should be pattern transfer
        assert "SECTION 4: PATTERN TRANSFER DIRECTIVES" in prompt
        # Old SECTION 3 for pattern transfer should NOT exist
        assert "SECTION 3: PATTERN TRANSFER" not in prompt


# =============================================================================
# Phase 12.4b: Component Name Normalization & Method Capping
# =============================================================================

class TestNormalizeMethodComponent:
    """Test _normalize_method_component static method."""

    def test_exact_alias_match(self):
        """'governor' normalizes to 'Governor Agent'."""
        result = MotherlabsEngine._normalize_method_component("governor")
        assert result == "Governor Agent"

    def test_prefix_match_with_suffix(self):
        """'Governor sequencing' normalizes via prefix to 'Governor Agent'."""
        result = MotherlabsEngine._normalize_method_component("Governor sequencing")
        assert result == "Governor Agent"

    def test_prefix_match_shared_state(self):
        """'SharedState management' normalizes via prefix to 'SharedState'."""
        result = MotherlabsEngine._normalize_method_component("SharedState management")
        assert result == "SharedState"

    def test_prefix_match_confidence(self):
        """'confidence tracking' normalizes to 'ConfidenceVector'."""
        result = MotherlabsEngine._normalize_method_component("confidence tracking")
        assert result == "ConfidenceVector"

    def test_no_match_returns_original(self):
        """Non-canonical names returned as-is."""
        result = MotherlabsEngine._normalize_method_component("AuditTrail logging")
        assert result == "AuditTrail logging"

    def test_exact_canonical_passthrough(self):
        """Already-canonical names pass through."""
        result = MotherlabsEngine._normalize_method_component("SharedState")
        assert result == "SharedState"

    def test_prefix_no_false_positive(self):
        """'processes' should NOT match 'process' prefix (no word boundary)."""
        result = MotherlabsEngine._normalize_method_component("processes data")
        assert result == "processes data"


class TestMethodCapping:
    """Test that pattern_transfer methods are capped at 5 per component."""

    def test_cap_at_five_per_component(self):
        """Pattern stubs capped at 5 per normalized component."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        # Create 8 pattern insights all targeting "Governor" variants
        state.insights = [
            "Governor sequencing works like Airflow — needs: poll, schedule, trigger",
            "Governor flow works like FSM — needs: transition, get_state, add_guard",
            "Governor updates work like Jira — needs: set_status, add_comment",
        ]

        methods, _ = engine._extract_methods_for_synthesis(state)

        # All should normalize to "Governor Agent"
        gov_methods = [m for m in methods if m["component"] == "Governor Agent"]
        assert len(gov_methods) == 5  # Capped at 5, not 8

    def test_dialogue_methods_not_capped(self):
        """Dialogue methods are not subject to the 5-per-component cap."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        # 6 dialogue methods for same component — should all be kept
        for i in range(6):
            state.add_message(Message(
                sender="Entity",
                content=f"METHOD: BigEntity.method_{i}(param: str) -> None",
                message_type=MessageType.PROPOSITION,
            ))
        state.insights = []

        methods, _ = engine._extract_methods_for_synthesis(state)
        big_methods = [m for m in methods if m["component"] == "BigEntity"]
        assert len(big_methods) == 6  # No cap on dialogue methods

    def test_dedup_normalizes_before_capping(self):
        """Dialogue 'GovernorAgent.x' and pattern 'Governor flow.x' dedup correctly."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        # Regex uses \w+ for component — use GovernorAgent (no space)
        state.add_message(Message(
            sender="Process",
            content="METHOD: GovernorAgent.poll(state: SharedState) -> Action",
            message_type=MessageType.PROPOSITION,
        ))
        # Pattern insight with 'poll' for Governor — should dedup with dialogue
        # "Governor flow" normalizes to "Governor Agent" via prefix match
        state.insights = [
            "Governor flow works like Airflow — needs: poll, schedule"
        ]

        methods, _ = engine._extract_methods_for_synthesis(state)

        # GovernorAgent normalizes to "Governor Agent" via alias,
        # Governor flow normalizes to "Governor Agent" via prefix — same key
        poll_methods = [m for m in methods if m["name"] == "poll"]
        assert len(poll_methods) == 1
        assert poll_methods[0]["source"] == "dialogue"  # Dialogue wins

        # schedule should still appear (not a duplicate)
        schedule_methods = [m for m in methods if m["name"] == "schedule"]
        assert len(schedule_methods) == 1
        assert schedule_methods[0]["source"] == "pattern_transfer"


class TestSection4NoDoubleInjection:
    """Test that Section 4 doesn't duplicate method enumeration when Section 3 has stubs."""

    def test_section4_scoped_down_when_stubs_exist(self):
        """When pattern stubs in Section 3, Section 4 should NOT say 'enumerate interface'."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.known["input"] = "Build a task system"
        state.insights = ["task flow works like Stripe checkout — validate + charge"]
        state.known["extracted_methods"] = [
            {
                "component": "task flow",
                "name": "validate",
                "parameters": [],
                "return_type": "None",
                "derived_from": "task flow works like Stripe checkout",
                "source": "pattern_transfer",
            }
        ]
        state.known["extracted_state_machines"] = []

        captured_prompts = []

        def mock_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result(json.dumps({
                "components": [{"name": "Task", "type": "process", "derived_from": "test"}],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            }))

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run

        blueprint, _ = engine._synthesize(state)
        prompt = captured_prompts[0]

        # Should NOT contain the generative "enumerate functional interface" instruction
        assert "Enumerate" not in prompt
        assert "known functional interface" not in prompt
        # Should contain the scoped-down instruction
        assert "Methods from these patterns are already listed in SECTION 3" in prompt

    def test_section4_full_when_no_stubs(self):
        """When no pattern stubs in Section 3, Section 4 uses full enumeration."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        engine = MotherlabsEngine(llm_client=client)

        state = SharedState()
        state.known["input"] = "Build a task system"
        state.insights = ["task flow works like Stripe checkout — validate + charge"]
        # No extracted methods at all
        state.known["extracted_methods"] = []
        state.known["extracted_state_machines"] = []

        captured_prompts = []

        def mock_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result(json.dumps({
                "components": [{"name": "Task", "type": "process", "derived_from": "test"}],
                "relationships": [],
                "constraints": [],
                "unresolved": [],
            }))

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run

        blueprint, _ = engine._synthesize(state)
        prompt = captured_prompts[0]

        # Should contain the generative instruction
        assert "Enumerate" in prompt or "known functional interface" in prompt
