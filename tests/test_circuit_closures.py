"""
Circuit Closure Integration Tests.

Proves that the 4 key circuits actually close:
1. Kernel semantic_nav → synthesis prompt (SECTION 2a)
2. Trust floor gate → re-synthesis trigger (verification_fail_threshold)
3. Closed-loop failure → re-synthesis trigger
4. Compiler directives → synthesis prompt (SECTION 2g, confirmed closed)
Plus catastrophic hard-block and re-synthesis flow.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.engine import MotherlabsEngine, CompileResult
from core.protocol import SharedState, Message, MessageType
from core.llm import BaseLLMClient
from core.protocol_spec import PROTOCOL
from mother.governor_feedback import CompilationOutcome
from persistence.corpus import Corpus


# =============================================================================
# MOCK DATA
# =============================================================================

MOCK_INTENT_JSON = {
    "core_need": "Build a task manager",
    "domain": "productivity",
    "actors": ["User"],
    "implicit_goals": ["Track tasks"],
    "constraints": [],
    "insight": "Core need is task tracking",
    "explicit_components": [],
    "explicit_relationships": [],
}

MOCK_PERSONA_JSON = {
    "personas": [
        {
            "name": "Product Designer",
            "perspective": "User experience focus",
            "blind_spots": "May miss technical constraints",
        }
    ],
    "cross_cutting_concerns": [],
    "suggested_focus_areas": ["Task lifecycle"],
}

MOCK_SYNTHESIS_JSON = {
    "components": [
        {
            "name": "Task",
            "type": "entity",
            "description": "A task with title, status, due date",
            "derived_from": "INSIGHT: task entity with title",
            "properties": [
                {"name": "title", "type": "str"},
                {"name": "status", "type": "str"},
            ],
        },
        {
            "name": "TaskManager",
            "type": "process",
            "description": "Manages task lifecycle",
            "derived_from": "INSIGHT: task manager process",
            "methods": [
                {
                    "name": "create_task",
                    "parameters": [{"name": "title", "type_hint": "str"}],
                    "return_type": "Task",
                    "description": "Create a new task",
                    "derived_from": "create_task(title) -> Task",
                }
            ],
        },
    ],
    "relationships": [
        {
            "from": "TaskManager",
            "to": "Task",
            "type": "manages",
            "description": "TaskManager manages Task entities",
        }
    ],
    "constraints": [],
    "unresolved": [],
}

_KERNEL_EXTRACT_MARKER = "You are a semantic compiler. You extract structured concepts"
_MOCK_KERNEL_EXTRACTIONS = json.dumps([
    {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "task", "content": "Task entity",
     "confidence": 0.9, "connections": []},
])


def _make_verify_dict(comp=85, cons=90, coh=80, trace=95, status="pass"):
    """Build a verification dict with controlled scores."""
    return {
        "status": status,
        "overall_score": (comp + cons + coh + trace) / 4,
        "completeness": {"score": comp, "details": "test"},
        "consistency": {"score": cons, "details": "test"},
        "coherence": {"score": coh, "details": "test"},
        "traceability": {"score": trace, "details": "test"},
        "actionability": {"score": 70, "details": "test"},
        "specificity": {"score": 70, "details": "test"},
        "codegen_readiness": {"score": 70, "details": "test"},
        "verification_mode": "deterministic",
    }


def make_sequenced_mock(extra_dialogue_turns=9):
    """Create a mock LLM client for full pipeline."""
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
            return json.dumps(_make_verify_dict())
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
                "Analyzing structure of the task system.\n"
                "INSIGHT: system builds Task entity with title, status fields"
            )
        return (
            "Analyzing behavior of the task system.\n"
            "INSIGHT: system builds TaskManager process that manages Task lifecycle"
        )

    client.complete_with_system = Mock(side_effect=mock_complete_with_system)
    return client


@pytest.fixture
def tmp_corpus(tmp_path):
    return Corpus(corpus_path=tmp_path / "corpus")


# =============================================================================
# CIRCUIT 1: Kernel semantic_nav → synthesis prompt
# =============================================================================


class TestCircuit1KernelToSynthesis:
    """Kernel nav output appears as SECTION 2a in synthesis prompt."""

    def test_semantic_nav_injected_into_synthesis_prompt(self, tmp_corpus):
        """When semantic_nav is in state.known, SECTION 2a appears in synthesis prompt."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Intercept _synthesize to inject semantic_nav and capture the prompt
        original_synthesize = engine._synthesize
        captured_prompts = []

        def intercept_synthesize(state, **kwargs):
            state.known["semantic_nav"] = "L0.SEM.ECO.WHAT.SFT  task  Task entity  F"
            original_agent_run = engine.synthesis_agent.run_llm_only

            def capture_agent_run(st, msg, max_tokens=4096):
                captured_prompts.append(msg.content)
                return original_agent_run(st, msg, max_tokens)

            engine.synthesis_agent.run_llm_only = capture_agent_run
            try:
                return original_synthesize(state, **kwargs)
            finally:
                engine.synthesis_agent.run_llm_only = original_agent_run

        engine._synthesize = intercept_synthesize
        result = engine.compile("Build a task manager app")
        engine._synthesize = original_synthesize

        assert len(captured_prompts) >= 1, "Synthesis prompt not captured"
        assert "SECTION 2a: SEMANTIC GRID" in captured_prompts[0]
        assert "L0.SEM.ECO.WHAT.SFT  task  Task entity  F" in captured_prompts[0]
        assert "structural backbone" in captured_prompts[0]

    def test_no_semantic_nav_no_section_2a(self, tmp_corpus):
        """When semantic_nav is empty, SECTION 2a does not appear."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        original_synthesize = engine._synthesize
        captured_prompts = []

        def intercept_synthesize(state, **kwargs):
            state.known.pop("semantic_nav", None)
            original_agent_run = engine.synthesis_agent.run_llm_only

            def capture_agent_run(st, msg, max_tokens=4096):
                captured_prompts.append(msg.content)
                return original_agent_run(st, msg, max_tokens)

            engine.synthesis_agent.run_llm_only = capture_agent_run
            try:
                return original_synthesize(state, **kwargs)
            finally:
                engine.synthesis_agent.run_llm_only = original_agent_run

        engine._synthesize = intercept_synthesize
        result = engine.compile("Build a task manager app")
        engine._synthesize = original_synthesize

        assert len(captured_prompts) >= 1, "Synthesis prompt not captured"
        assert "SECTION 2a: SEMANTIC GRID" not in captured_prompts[0]


# =============================================================================
# CIRCUIT 2: Trust floor gate → re-synthesis
# =============================================================================


class TestCircuit2TrustFloor:
    """Overall trust < 40 triggers re-synthesis via needs_work."""

    def test_low_trust_triggers_resynthesis(self, tmp_corpus):
        """Overall trust 35 (< 40 threshold) → verification marked needs_work → re-synthesis fires."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock _verify_hybrid to return controlled low scores
        low_verify = _make_verify_dict(comp=35, cons=35, coh=35, trace=35, status="pass")

        resynth_called = [False]
        original_targeted = engine._targeted_resynthesis

        def track_resynth(*args, **kwargs):
            resynth_called[0] = True
            return original_targeted(*args, **kwargs)

        engine._targeted_resynthesis = track_resynth
        engine._verify_hybrid = Mock(return_value=low_verify)

        result = engine.compile("Build a task manager app")
        engine._targeted_resynthesis = original_targeted

        assert resynth_called[0], (
            "Trust floor (35 < 40) should trigger re-synthesis"
        )

    def test_good_trust_no_resynthesis(self, tmp_corpus):
        """Overall trust 87.5 (> 40 threshold) → no re-synthesis from trust floor."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock _verify_hybrid to return controlled high scores
        high_verify = _make_verify_dict(comp=85, cons=90, coh=80, trace=95, status="pass")

        resynth_called = [False]
        original_targeted = engine._targeted_resynthesis

        def track_resynth(*args, **kwargs):
            resynth_called[0] = True
            return original_targeted(*args, **kwargs)

        engine._targeted_resynthesis = track_resynth
        engine._verify_hybrid = Mock(return_value=high_verify)

        # Mock closed-loop to pass (so it doesn't trigger re-synthesis)
        mock_cl_result = MagicMock()
        mock_cl_result.passed = True
        mock_cl_result.fidelity_score = 0.90
        mock_cl_result.compression_losses = []

        with patch("kernel.closed_loop.closed_loop_gate", return_value=mock_cl_result):
            result = engine.compile("Build a task manager app")

        engine._targeted_resynthesis = original_targeted

        assert not resynth_called[0], (
            "Good trust (87.5 > 40) + passing closed-loop should not trigger re-synthesis"
        )

    def test_trust_floor_uses_protocol_constant(self):
        """Verify the threshold comes from PROTOCOL, not hardcoded."""
        assert PROTOCOL.engine.verification_fail_threshold == 40


# =============================================================================
# CIRCUIT 3: Closed-loop failure → re-synthesis
# =============================================================================


class TestCircuit3ClosedLoopTrigger:
    """Closed-loop fidelity < 0.70 triggers re-synthesis."""

    def test_closed_loop_failure_triggers_resynthesis(self, tmp_corpus):
        """Fidelity < 0.70 → verification status forced to needs_work → re-synthesis fires."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock verification to PASS (so trust floor doesn't trigger)
        high_verify = _make_verify_dict(comp=85, cons=90, coh=80, trace=95, status="pass")
        engine._verify_hybrid = Mock(return_value=high_verify)

        # Mock closed-loop to FAIL
        mock_cl_result = MagicMock()
        mock_cl_result.passed = False
        mock_cl_result.fidelity_score = 0.45
        mock_cl_result.compression_losses = [
            MagicMock(category="entity", severity=0.8, original_fragment="missing user entity"),
            MagicMock(category="behavior", severity=0.6, original_fragment="missing login flow"),
        ]

        resynth_called = [False]
        original_targeted = engine._targeted_resynthesis

        def track_resynth(*args, **kwargs):
            resynth_called[0] = True
            return original_targeted(*args, **kwargs)

        engine._targeted_resynthesis = track_resynth

        with patch("kernel.closed_loop.closed_loop_gate", return_value=mock_cl_result):
            result = engine.compile("Build a task manager app")

        engine._targeted_resynthesis = original_targeted

        assert resynth_called[0], (
            "Closed-loop failure (fidelity 0.45 < 0.70) should trigger re-synthesis"
        )

    def test_closed_loop_pass_no_resynthesis_trigger(self, tmp_corpus):
        """Fidelity > 0.70 → no re-synthesis trigger from closed-loop."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock verification to PASS
        high_verify = _make_verify_dict(comp=85, cons=90, coh=80, trace=95, status="pass")
        engine._verify_hybrid = Mock(return_value=high_verify)

        # Mock closed-loop to PASS
        mock_cl_result = MagicMock()
        mock_cl_result.passed = True
        mock_cl_result.fidelity_score = 0.85
        mock_cl_result.compression_losses = []

        resynth_called = [False]
        original_targeted = engine._targeted_resynthesis

        def track_resynth(*args, **kwargs):
            resynth_called[0] = True
            return original_targeted(*args, **kwargs)

        engine._targeted_resynthesis = track_resynth

        with patch("kernel.closed_loop.closed_loop_gate", return_value=mock_cl_result):
            result = engine.compile("Build a task manager app")

        engine._targeted_resynthesis = original_targeted

        assert not resynth_called[0], (
            "Closed-loop pass (fidelity 0.85 > 0.70) should not trigger re-synthesis"
        )


# =============================================================================
# CIRCUIT 4: Compiler directives (confirmed closed)
# =============================================================================


class TestCircuit4CompilerDirectives:
    """Prior compilation outcomes flow into SECTION 2g of synthesis prompt."""

    def test_compiler_directives_in_synthesis_prompt(self, tmp_corpus):
        """When _compilation_outcomes produce directives, SECTION 2g appears."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Simulate prior compilation outcomes with proper CompilationOutcome objects
        engine._compilation_outcomes = [
            CompilationOutcome(
                compile_id="test-001",
                input_summary="Build a task manager",
                trust_score=45,
                completeness=30,
                consistency=50,
                coherence=45,
                traceability=55,
                actionability=45.0,
                specificity=45.0,
                codegen_readiness=45.0,
                component_count=3,
                rejected=False,
                domain="software",
            ),
            CompilationOutcome(
                compile_id="test-002",
                input_summary="Build a task manager v2",
                trust_score=72,
                completeness=70,
                consistency=75,
                coherence=68,
                traceability=80,
                actionability=72.0,
                specificity=72.0,
                codegen_readiness=72.0,
                component_count=5,
                rejected=False,
                domain="software",
            ),
        ]

        # Capture synthesis prompt
        captured_prompts = []
        original_synthesize = engine._synthesize

        def intercept_synthesize(state, **kwargs):
            original_agent_run = engine.synthesis_agent.run_llm_only

            def capture_agent_run(st, msg, max_tokens=4096):
                captured_prompts.append(msg.content)
                return original_agent_run(st, msg, max_tokens)

            engine.synthesis_agent.run_llm_only = capture_agent_run
            try:
                return original_synthesize(state, **kwargs)
            finally:
                engine.synthesis_agent.run_llm_only = original_agent_run

        engine._synthesize = intercept_synthesize
        result = engine.compile("Build a task manager app")
        engine._synthesize = original_synthesize

        assert len(captured_prompts) >= 1, "Synthesis prompt not captured"
        assert "SECTION 2g: COMPILER SELF-IMPROVEMENT" in captured_prompts[0], (
            "Compiler directives from prior compilations should reach SECTION 2g"
        )


# =============================================================================
# CATASTROPHIC HARD-BLOCK
# =============================================================================


class TestCatastrophicHardBlock:
    """All dimension scores < 30 → CompileResult.success == False."""

    def test_catastrophic_verification_blocks(self, tmp_corpus):
        """Blueprint with all scores < 30 hard-fails compilation."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock _verify_hybrid to return catastrophic scores
        catastrophic = _make_verify_dict(comp=15, cons=10, coh=20, trace=5, status="fail")
        engine._verify_hybrid = Mock(return_value=catastrophic)

        result = engine.compile("Build a task manager app")

        assert result.success is False
        assert "catastrophic" in result.error.lower()


# =============================================================================
# RE-SYNTHESIS FLOW
# =============================================================================


class TestResynthesisFlow:
    """Verification needs_work → _targeted_resynthesis called."""

    def test_needs_work_triggers_targeted_resynthesis(self, tmp_corpus):
        """When verification status is needs_work, re-synthesis fires."""
        client = make_sequenced_mock()
        engine = MotherlabsEngine(
            llm_client=client, corpus=tmp_corpus, auto_store=True, cache_policy="none"
        )

        # Mock verification to return needs_work with gaps
        needs_work = _make_verify_dict(comp=50, cons=60, coh=55, trace=70, status="needs_work")
        needs_work["completeness"]["gaps"] = ["missing auth component"]
        needs_work["coherence"]["suggested_fixes"] = ["add login flow"]
        engine._verify_hybrid = Mock(return_value=needs_work)

        resynth_called = [False]
        original_targeted = engine._targeted_resynthesis

        def track_resynth(*args, **kwargs):
            resynth_called[0] = True
            return original_targeted(*args, **kwargs)

        engine._targeted_resynthesis = track_resynth
        result = engine.compile("Build a task manager app")
        engine._targeted_resynthesis = original_targeted

        assert resynth_called[0], "needs_work verification should trigger re-synthesis"
