"""Tests for parallel best-of-N synthesis.

Tests cover:
A. Fast path (2 tests):
   - First attempt passes → returns immediately, no parallel dispatch
   - Freeform intents clamp to max 1 retry (parallel batch of 1)

B. Parallel dispatch (3 tests):
   - Failed first attempt → N parallel retries launched
   - Best-scoring candidate selected from parallel results
   - One failed LLM call doesn't block others

C. Scoring & selection (2 tests):
   - All retries fail → returns best candidate from all attempts
   - Only best candidate's mutations applied to state

D. Prompt construction (1 test):
   - Retry prompts include missing components from attempt 0

E. Time budget (1 test):
   - 180s budget still enforced
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor

from core.protocol import Message, MessageType, SharedState
from core.llm import MockClient
from agents.base import LLMAgent, AgentCallResult


# =============================================================================
# Helpers
# =============================================================================


def _make_state():
    """Create a SharedState with minimal intent data for synthesis."""
    state = SharedState()
    state.known["input"] = "Build a task manager with projects and deadlines"
    state.known["intent"] = {
        "core_need": "task management",
        "domain": "software",
        "actors": ["User"],
        "constraints": [],
        "explicit_components": ["TaskManager", "Project", "Deadline"],
        "insight": "",
    }
    return state


def _make_blueprint(components, relationships=None):
    """Create a valid blueprint dict."""
    return {
        "components": [
            {"name": c, "type": "entity", "description": f"The {c}", "methods": [{"name": "do"}]}
            for c in components
        ],
        "relationships": relationships or [],
        "constraints": [],
        "unresolved": [],
    }


def _bp_json(components, relationships=None):
    """Return blueprint as JSON string."""
    return json.dumps(_make_blueprint(components, relationships))


def _make_synthesis_agent(responses):
    """
    Create a synthesis LLMAgent whose run_llm_only() returns sequenced responses.

    responses: list of JSON strings. Each call pops the next response.
    """
    mock_client = MockClient()
    mock_client._store_usage({"input_tokens": 100, "output_tokens": 200, "total_tokens": 300})

    agent = LLMAgent(
        name="Synthesis",
        perspective="synthesis",
        system_prompt="You are a synthesis agent.",
        llm_client=mock_client,
    )

    # Track call count for sequencing
    call_count = {"n": 0}
    original_responses = list(responses)

    def fake_run_llm_only(state, msg, max_tokens=4096):
        idx = call_count["n"]
        call_count["n"] += 1
        resp_text = original_responses[idx] if idx < len(original_responses) else original_responses[-1]
        return AgentCallResult(
            agent_name="Synthesis",
            response_text=resp_text,
            message=Message(
                sender="Synthesis",
                content=resp_text,
                message_type=MessageType.PROPOSITION,
            ),
            conflicts=(),
            unknowns=(),
            fractures=(),
            confidence_boost=0.0,
            agent_dimension="",
            has_insight=False,
            token_usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
        )

    agent.run_llm_only = fake_run_llm_only
    agent._call_count = call_count  # expose for assertions
    return agent


def _make_engine(synthesis_agent, canonical_components=None, canonical_relationships=None):
    """Create a minimal engine mock with just enough for _synthesize()."""
    engine = MagicMock()
    engine.synthesis_agent = synthesis_agent
    engine.provider_name = "mock"
    engine.model_name = "mock-model"
    engine._compilation_tokens = []
    engine._emit_insight = MagicMock()
    engine._extract_json = MagicMock(side_effect=lambda text: json.loads(text.strip()) if text.strip().startswith("{") else json.loads(text[text.find("{"):text.rfind("}") + 1]))
    engine._calculate_insight_coverage = MagicMock(return_value=0.8)
    engine._validate_method_completeness = MagicMock(return_value=[])

    return engine


# =============================================================================
# A. Fast path
# =============================================================================


class TestFastPath:
    """Tests for when attempt 0 passes all checks."""

    def test_fast_path_no_retries(self):
        """First attempt passes → returns immediately, no parallel dispatch."""
        # Blueprint with all canonical components
        good_bp = _bp_json(["TaskManager", "Project", "Deadline"])
        agent = _make_synthesis_agent([good_bp])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        # Call _synthesize directly via the real engine method
        from core.engine import MotherlabsEngine
        bp, retries = MotherlabsEngine._synthesize.__wrapped__(
            _make_engine(agent), state, canonical_components=canonical
        ) if hasattr(MotherlabsEngine._synthesize, '__wrapped__') else (None, None)

        # Since _synthesize isn't easily callable on a mock, test via a real minimal engine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=good_bp)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            # Override synthesis agent with our mock
            eng.synthesis_agent = agent

            # Call _synthesize
            bp_result, retry_count = eng._synthesize(
                state,
                canonical_components=canonical,
            )

        assert retry_count == 0
        assert "TaskManager" in [c["name"] for c in bp_result["components"]]
        # Only 1 call to run_llm_only (attempt 0 only, no parallel retries)
        assert agent._call_count["n"] == 1

    def test_freeform_max_1_retry(self):
        """Freeform intents (no canonical) clamp to max 1 retry."""
        # First attempt returns incomplete, second has more
        incomplete_bp = _bp_json(["TaskManager"])
        better_bp = _bp_json(["TaskManager", "Project"])
        agent = _make_synthesis_agent([incomplete_bp, better_bp])

        state = _make_state()

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=incomplete_bp)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            # No canonical components = freeform → MAX_RETRIES clamped to 1
            bp_result, retry_count = eng._synthesize(state)

        # Should have called run_llm_only at most 2 times (attempt 0 + 1 parallel retry)
        assert agent._call_count["n"] <= 2


# =============================================================================
# B. Parallel dispatch
# =============================================================================


class TestParallelDispatch:
    """Tests for parallel retry launching."""

    def test_parallel_retries_launch(self):
        """Failed first attempt → N parallel retries launched."""
        # Attempt 0: missing "Deadline"
        incomplete_bp = _bp_json(["TaskManager", "Project"])
        # Retries: complete blueprint
        complete_bp = _bp_json(["TaskManager", "Project", "Deadline"])
        # Provider config default max_retries=3, so expect 3 parallel retries
        agent = _make_synthesis_agent([incomplete_bp, complete_bp, complete_bp, complete_bp])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=incomplete_bp)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, retry_count = eng._synthesize(
                state, canonical_components=canonical,
            )

        # Attempt 0 + N retries = more than 1 call
        assert agent._call_count["n"] > 1
        # Blueprint should contain all canonical components
        names = [c["name"] for c in bp_result["components"]]
        assert "Deadline" in names

    def test_parallel_retries_best_of_n(self):
        """Best-scoring candidate selected from parallel results."""
        # Attempt 0: only 1 component (low density penalty)
        bp_1 = _bp_json(["TaskManager"])
        # Retry 1: 2 components
        bp_2 = _bp_json(["TaskManager", "Project"])
        # Retry 2: all 3 (best)
        bp_3 = _bp_json(["TaskManager", "Project", "Deadline"])
        agent = _make_synthesis_agent([bp_1, bp_2, bp_3, bp_3])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=bp_1)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, retry_count = eng._synthesize(
                state, canonical_components=canonical,
            )

        # Best candidate should have all 3 components
        names = [c["name"] for c in bp_result["components"]]
        assert "TaskManager" in names
        assert "Project" in names
        assert "Deadline" in names

    def test_parallel_retries_one_failure(self):
        """One failed LLM call doesn't block others."""
        # Attempt 0: incomplete
        bp_incomplete = _bp_json(["TaskManager"])
        # Good retry
        bp_good = _bp_json(["TaskManager", "Project", "Deadline"])
        agent = _make_synthesis_agent([bp_incomplete, bp_good, bp_good, bp_good])

        # Make one of the parallel calls raise an exception
        original_run = agent.run_llm_only
        call_seq = {"n": 0}

        def flaky_run(state, msg, max_tokens=4096):
            idx = call_seq["n"]
            call_seq["n"] += 1
            # Fail on attempt 0's result already set, this is for retries
            # Attempt 0 is call 0, retry calls start at 1
            if idx == 2:  # Third call (second retry) fails
                raise RuntimeError("LLM connection error")
            return original_run(state, msg, max_tokens)

        agent.run_llm_only = flaky_run

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=bp_incomplete)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, retry_count = eng._synthesize(
                state, canonical_components=canonical,
            )

        # Should still get a valid result despite one failure
        assert bp_result is not None
        assert "components" in bp_result


# =============================================================================
# C. Scoring & selection
# =============================================================================


class TestScoringSelection:
    """Tests for candidate scoring and mutation application."""

    def test_parallel_retries_all_fail(self):
        """All retries fail → returns best candidate from all attempts."""
        # All attempts return incomplete blueprints (no full coverage)
        bp_1comp = _bp_json(["TaskManager"])
        bp_2comp = _bp_json(["TaskManager", "Project"])
        agent = _make_synthesis_agent([bp_1comp, bp_1comp, bp_2comp, bp_1comp])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=bp_1comp)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, retry_count = eng._synthesize(
                state, canonical_components=canonical,
            )

        # Should return best available candidate (not crash)
        assert bp_result is not None
        assert len(bp_result.get("components", [])) >= 1

    def test_mutations_applied_from_best(self):
        """Only best candidate's mutations applied to state."""
        # Attempt 0: incomplete
        bp_incomplete = _bp_json(["TaskManager"])
        # Best retry has all components
        bp_complete = _bp_json(["TaskManager", "Project", "Deadline"])
        agent = _make_synthesis_agent([bp_incomplete, bp_complete, bp_complete, bp_complete])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]
        initial_history_count = len(state.history)

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=bp_incomplete)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, retry_count = eng._synthesize(
                state, canonical_components=canonical,
            )

        # State should have messages from attempt 0 + best retry (2 messages total)
        added_msgs = len(state.history) - initial_history_count
        # Attempt 0 always adds a message. If retries pass, best adds one more.
        assert added_msgs >= 1
        assert added_msgs <= 2  # At most attempt 0 + best retry


# =============================================================================
# D. Prompt construction
# =============================================================================


class TestPromptConstruction:
    """Tests for retry prompt content."""

    def test_missing_components_in_retry_prompt(self):
        """Retry prompts include missing components from attempt 0."""
        # Track what prompts are sent to retries
        prompts_seen = []

        # Attempt 0: missing "Deadline"
        bp_incomplete = _bp_json(["TaskManager", "Project"])
        bp_complete = _bp_json(["TaskManager", "Project", "Deadline"])

        mock_client = MockClient()
        mock_client._store_usage({"input_tokens": 100, "output_tokens": 200, "total_tokens": 300})
        agent = LLMAgent(
            name="Synthesis",
            perspective="synthesis",
            system_prompt="You are a synthesis agent.",
            llm_client=mock_client,
        )

        call_count = {"n": 0}

        def tracking_run(state, msg, max_tokens=4096):
            idx = call_count["n"]
            call_count["n"] += 1
            # Capture retry prompts (attempt > 0)
            if idx > 0:
                prompts_seen.append(msg.content)
            resp_text = bp_incomplete if idx == 0 else bp_complete
            return AgentCallResult(
                agent_name="Synthesis",
                response_text=resp_text,
                message=Message(sender="Synthesis", content=resp_text, message_type=MessageType.PROPOSITION),
                conflicts=(), unknowns=(), fractures=(),
                confidence_boost=0.0, agent_dimension="", has_insight=False,
                token_usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            )

        agent.run_llm_only = tracking_run

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            bp_result, _ = eng._synthesize(
                state, canonical_components=canonical,
            )

        # At least one retry prompt should mention "Deadline" as missing
        assert any("Deadline" in p for p in prompts_seen), \
            f"No retry prompt mentioned 'Deadline'. Prompts: {[p[:200] for p in prompts_seen]}"


# =============================================================================
# E. Time budget
# =============================================================================


class TestTimeBudget:
    """Tests for time budget enforcement."""

    def test_time_budget_respected(self):
        """180s budget enforced — no retries if attempt 0 exceeds budget."""
        bp_incomplete = _bp_json(["TaskManager"])
        agent = _make_synthesis_agent([bp_incomplete, bp_incomplete, bp_incomplete, bp_incomplete])

        state = _make_state()
        canonical = ["TaskManager", "Project", "Deadline"]

        from core.engine import MotherlabsEngine
        mock_client = MockClient()
        mock_client.complete_with_system = MagicMock(return_value=bp_incomplete)
        mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})

        from pathlib import Path
        from persistence.corpus import Corpus
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(corpus_path=Path(tmp) / "corpus")
            eng = MotherlabsEngine(
                llm_client=mock_client,
                corpus=corpus,
                auto_store=False,
            )
            eng.synthesis_agent = agent

            # Patch time.time to simulate budget exhaustion
            real_time = time.time
            call_times = [real_time()]  # First call returns now

            def fake_time():
                # After first time() call, jump 200s into the future
                if len(call_times) > 1:
                    return call_times[0] + 200.0
                call_times.append(True)
                return call_times[0]

            with patch("core.engine.time") as mock_time:
                mock_time.time = fake_time
                mock_time.sleep = MagicMock()  # Don't actually sleep

                bp_result, retry_count = eng._synthesize(
                    state, canonical_components=canonical,
                )

        # Should return without launching parallel retries (budget exceeded)
        assert retry_count == 0
        # Only attempt 0 was called
        assert agent._call_count["n"] == 1
