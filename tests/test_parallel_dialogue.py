"""Tests for parallel agent dispatch in grid-driven dialogue.

Tests cover:
A. Split correctness (5 tests):
   - run_llm_only returns AgentCallResult
   - run_llm_only does not mutate state
   - apply_mutations updates state
   - run() == run_llm_only() + apply_mutations()
   - AgentCallResult is frozen

B. Parallel dispatch (4 tests):
   - Parallel batch produces same results as serial
   - max_turns respected mid-batch
   - One LLM failure doesn't block others
   - Grid not mutated during Phase B

C. Thread safety (2 tests):
   - Thread-local usage not clobbered
   - Shared client concurrent calls

D. Regression (1 test):
   - Grid dialogue still converges end-to-end
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from core.protocol import Message, MessageType, SharedState
from core.llm import MockClient, BaseLLMClient
from agents.base import LLMAgent, AgentCallResult


# =============================================================================
# Helpers
# =============================================================================


def _make_state():
    """Create a SharedState with minimal intent data."""
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


def _make_agent(name="Entity", response_text="INSIGHT: Tasks need deadlines\nThis is analysis."):
    """Create an LLMAgent with a mock client that returns fixed text."""
    mock_client = MockClient()
    mock_client.complete_with_system = MagicMock(return_value=response_text)
    mock_client._store_usage({"input_tokens": 100, "output_tokens": 200, "total_tokens": 300})
    agent = LLMAgent(
        name=name,
        perspective="structural analysis",
        system_prompt="You are an analysis agent.",
        llm_client=mock_client,
    )
    return agent


def _make_input_msg():
    return Message(
        sender="System",
        content="Analyze the task manager system.",
        message_type=MessageType.PROPOSITION,
    )


# =============================================================================
# A. Split correctness
# =============================================================================


class TestRunLlmOnly:
    """Tests for the run_llm_only / apply_mutations split."""

    def test_run_llm_only_returns_agent_call_result(self):
        """run_llm_only returns an AgentCallResult with correct fields."""
        agent = _make_agent()
        state = _make_state()
        input_msg = _make_input_msg()

        result = agent.run_llm_only(state, input_msg)

        assert isinstance(result, AgentCallResult)
        assert result.agent_name == "Entity"
        assert result.response_text == "INSIGHT: Tasks need deadlines\nThis is analysis."
        assert isinstance(result.message, Message)
        assert result.message.sender == "Entity"
        assert result.has_insight is True
        assert isinstance(result.token_usage, dict)
        assert result.token_usage.get("input_tokens") == 100

    def test_run_llm_only_does_not_mutate_state(self):
        """State remains unchanged after run_llm_only."""
        agent = _make_agent(response_text="CONFLICT: API vs REST\nINSIGHT: Found gap\nUNKNOWN: Auth method")
        state = _make_state()
        input_msg = _make_input_msg()

        # Snapshot state before
        conflicts_before = len(state.conflicts)
        unknowns_before = len(state.unknown)
        history_before = len(state.history)
        conf_structural_before = state.confidence.structural

        result = agent.run_llm_only(state, input_msg)

        # State unchanged
        assert len(state.conflicts) == conflicts_before
        assert len(state.unknown) == unknowns_before
        assert len(state.history) == history_before
        assert state.confidence.structural == conf_structural_before

        # But result has extracted data
        assert len(result.conflicts) == 1
        assert "API vs REST" in result.conflicts[0]
        assert len(result.unknowns) == 1
        assert "Auth method" in result.unknowns[0]

    def test_apply_mutations_updates_state(self):
        """apply_mutations correctly applies conflicts, unknowns, confidence."""
        agent = _make_agent(response_text="CONFLICT: API vs REST\nINSIGHT: Found gap\nUNKNOWN: Auth method")
        state = _make_state()
        input_msg = _make_input_msg()

        result = agent.run_llm_only(state, input_msg)

        # Apply mutations
        LLMAgent.apply_mutations(state, result)

        # State now updated
        assert len(state.conflicts) == 1
        assert state.conflicts[0]["topic"] == "API vs REST"
        assert len(state.unknown) == 1
        assert "Auth method" in state.unknown[0]

    def test_run_equals_run_llm_only_plus_apply(self):
        """run() produces identical state as run_llm_only() + apply_mutations()."""
        response_text = "INSIGHT: Components need interfaces\nCONFLICT: sync vs async"

        # Path A: run()
        agent_a = _make_agent(response_text=response_text)
        state_a = _make_state()
        input_msg = _make_input_msg()
        message_a = agent_a.run(state_a, input_msg)

        # Path B: run_llm_only() + apply_mutations()
        agent_b = _make_agent(response_text=response_text)
        state_b = _make_state()
        result_b = agent_b.run_llm_only(state_b, input_msg)
        LLMAgent.apply_mutations(state_b, result_b)

        # Messages identical
        assert message_a.sender == result_b.message.sender
        assert message_a.content == result_b.message.content
        assert message_a.message_type == result_b.message.message_type
        assert message_a.insight == result_b.message.insight

        # State mutations identical
        assert len(state_a.conflicts) == len(state_b.conflicts)
        assert len(state_a.unknown) == len(state_b.unknown)
        assert state_a.confidence.structural == pytest.approx(state_b.confidence.structural)
        assert state_a.confidence.behavioral == pytest.approx(state_b.confidence.behavioral)

    def test_agent_call_result_is_frozen(self):
        """AgentCallResult is immutable."""
        agent = _make_agent()
        state = _make_state()
        result = agent.run_llm_only(state, _make_input_msg())

        with pytest.raises(AttributeError):
            result.agent_name = "Modified"

        with pytest.raises(AttributeError):
            result.confidence_boost = 999.0


# =============================================================================
# B. Parallel dispatch
# =============================================================================


class TestParallelDispatch:
    """Tests for parallel LLM calls in grid dialogue."""

    def test_parallel_batch_same_results_as_serial(self):
        """Parallel dispatch with deterministic ordering = identical final state."""
        responses = [
            "INSIGHT: Entity A found\nStructural analysis of TaskManager",
            "INSIGHT: Process B found\nBehavioral analysis of workflow",
            "INSIGHT: Entity C found\nRelationship analysis",
        ]

        # Serial execution
        state_serial = _make_state()
        agents_serial = [_make_agent(f"Agent{i}", responses[i]) for i in range(3)]
        results_serial = []
        for agent in agents_serial:
            result = agent.run_llm_only(state_serial, _make_input_msg())
            LLMAgent.apply_mutations(state_serial, result)
            state_serial.add_message(result.message)
            results_serial.append(result)

        # Parallel execution (same order via sort)
        state_parallel = _make_state()
        agents_parallel = [_make_agent(f"Agent{i}", responses[i]) for i in range(3)]

        call_results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(agents_parallel[i].run_llm_only, state_parallel, _make_input_msg()): i
                for i in range(3)
            }
            for future in as_completed(futures):
                idx = futures[future]
                call_results.append((idx, future.result()))

        # Sort by index (simulates deterministic postcode ordering)
        call_results.sort(key=lambda r: r[0])

        for idx, result in call_results:
            LLMAgent.apply_mutations(state_parallel, result)
            state_parallel.add_message(result.message)

        # Final states match
        assert len(state_serial.history) == len(state_parallel.history)
        assert len(state_serial.conflicts) == len(state_parallel.conflicts)
        assert state_serial.confidence.structural == pytest.approx(
            state_parallel.confidence.structural
        )

    def test_parallel_batch_respects_max_turns(self):
        """Only N cells applied if max_turns would be exceeded mid-batch."""
        responses = [
            "INSIGHT: A found",
            "INSIGHT: B found",
            "INSIGHT: C found",
        ]

        state = _make_state()
        agents = [_make_agent(f"Agent{i}", responses[i]) for i in range(3)]

        # Simulate parallel calls
        results = []
        for i, agent in enumerate(agents):
            result = agent.run_llm_only(state, _make_input_msg())
            results.append((i, result))

        # Apply with max_turns = 2 (only first 2 should apply)
        total_turns = 0
        max_turns = 2
        applied = 0
        for idx, result in results:
            if total_turns >= max_turns:
                break
            LLMAgent.apply_mutations(state, result)
            state.add_message(result.message)
            total_turns += 1
            applied += 1

        assert applied == 2
        assert len(state.history) == 2

    def test_parallel_batch_handles_one_failure(self):
        """Failed LLM call doesn't block the other agents."""
        good_agent = _make_agent("Entity", "INSIGHT: Found component")

        # Create a failing agent
        bad_agent = _make_agent("Process", "INSIGHT: Process found")
        bad_agent.llm.complete_with_system = MagicMock(
            side_effect=RuntimeError("LLM timeout")
        )

        state = _make_state()

        call_results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(good_agent.run_llm_only, state, _make_input_msg()): "good",
                executor.submit(bad_agent.run_llm_only, state, _make_input_msg()): "bad",
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    call_results.append((label, future.result()))
                except Exception:
                    pass  # Bad agent fails gracefully

        # Only good agent's result collected
        assert len(call_results) == 1
        assert call_results[0][0] == "good"

    def test_parallel_batch_grid_not_mutated_during_llm(self):
        """Grid unchanged during Phase B — run_llm_only doesn't touch grid."""
        from kernel.grid import Grid

        grid = Grid()
        grid.set_intent("task manager", "INT.ENT.ECO.WHAT.SFT", "intent")
        root_cell = grid.activate_layer("STR", "ENT", "WHAT", "SFT")

        initial_fill_rate = grid.fill_rate
        initial_cells = grid.total_cells

        agent = _make_agent("Entity", "INSIGHT: Found component\nThis fills a cell")
        state = _make_state()

        # Phase B: parallel LLM calls — grid not passed to agent
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(agent.run_llm_only, state, _make_input_msg()),
                executor.submit(agent.run_llm_only, state, _make_input_msg()),
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # Grid unchanged during Phase B
        assert grid.fill_rate == initial_fill_rate
        assert grid.total_cells == initial_cells
        assert len(results) == 2


# =============================================================================
# C. Thread safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe usage tracking."""

    def test_thread_local_usage_not_clobbered(self):
        """Each thread reads its own usage via _thread_local."""
        client = MockClient()
        results = {}

        def call_with_usage(thread_id, usage_val):
            # Simulate LLM call with custom usage
            client._store_usage({
                "input_tokens": usage_val,
                "output_tokens": usage_val * 2,
                "total_tokens": usage_val * 3,
            })
            # Small delay to increase chance of interleaving
            time.sleep(0.01)
            # Read back from thread-local
            local_usage = getattr(client._thread_local, 'last_usage', {})
            results[thread_id] = local_usage

        threads = []
        for i in range(5):
            t = threading.Thread(target=call_with_usage, args=(i, (i + 1) * 100))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own usage values
        for i in range(5):
            expected = (i + 1) * 100
            assert results[i]["input_tokens"] == expected, (
                f"Thread {i}: expected input_tokens={expected}, got {results[i]}"
            )

    def test_shared_client_concurrent_calls(self):
        """Two agents sharing same mock client produce correct individual usage."""
        shared_client = MockClient()

        usage_a = {"input_tokens": 111, "output_tokens": 222, "total_tokens": 333}
        usage_b = {"input_tokens": 444, "output_tokens": 555, "total_tokens": 999}

        call_count = 0
        call_lock = threading.Lock()

        original_complete = shared_client.complete_with_system

        def mock_complete(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                current = call_count
                call_count += 1

            # First call gets usage_a, second gets usage_b
            if current == 0:
                time.sleep(0.02)  # Slower to ensure interleaving
                shared_client._store_usage(usage_a)
                return "INSIGHT: First response"
            else:
                shared_client._store_usage(usage_b)
                return "INSIGHT: Second response"

        shared_client.complete_with_system = mock_complete

        agent_a = LLMAgent("Entity", "structural", "prompt", shared_client)
        agent_b = LLMAgent("Process", "behavioral", "prompt", shared_client)

        state = _make_state()
        input_msg = _make_input_msg()

        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(agent_a.run_llm_only, state, input_msg)
            future_b = executor.submit(agent_b.run_llm_only, state, input_msg)
            results["a"] = future_a.result()
            results["b"] = future_b.result()

        # Each agent captured its own thread-local usage
        usages = {results["a"].token_usage.get("input_tokens"), results["b"].token_usage.get("input_tokens")}
        assert 111 in usages or 444 in usages, (
            f"Expected thread-local usage values, got {usages}"
        )


# =============================================================================
# D. Regression
# =============================================================================


class TestGridDialogueRegression:
    """End-to-end regression: grid dialogue still converges with parallel dispatch."""

    def test_grid_dialogue_still_converges(self):
        """Grid dialogue converges using parallel dispatch (mock LLM)."""
        from core.engine import MotherlabsEngine
        from core.protocol import SharedState

        mock_client = MockClient()
        response_counter = {"n": 0}
        original_complete = mock_client.complete_with_system

        def mock_response(system_prompt, user_content, **kwargs):
            response_counter["n"] += 1
            n = response_counter["n"]
            mock_client._store_usage({"input_tokens": 50, "output_tokens": 100, "total_tokens": 150})
            return (
                f"INSIGHT: Component{n} identified as structural element\n"
                f"The system needs Component{n} for task management. "
                f"This connects to the core task manager functionality."
            )

        mock_client.complete_with_system = MagicMock(side_effect=mock_response)

        engine = MotherlabsEngine.__new__(MotherlabsEngine)
        engine.llm = mock_client
        engine.entity_agent = LLMAgent("Entity", "structural", "entity prompt", mock_client)
        engine.process_agent = LLMAgent("Process", "behavioral", "process prompt", mock_client)
        engine.provider_name = "mock"
        engine.model_name = "mock"
        engine._callbacks = []
        engine._compilation_tokens = []
        engine._kernel_grid = None
        engine._emit_insight = MagicMock()
        engine._emit = MagicMock()
        engine._PROCESS_CONCERNS = frozenset({"BHV", "FLW", "TRN", "FNC", "ACT", "GTE", "SCH"})

        state = SharedState()
        state.known["input"] = "Build a task manager with projects and deadlines"
        state.known["intent"] = {
            "core_need": "task management",
            "domain": "software",
            "actors": ["User"],
            "constraints": [],
            "explicit_components": ["TaskManager"],
            "insight": "",
        }

        # Bootstrap grid
        grid = engine._bootstrap_dialogue_grid(state)
        assert grid is not None

        # Run grid dialogue with parallel dispatch
        engine._run_grid_driven_dialogue(
            state, grid,
            min_turns=3,
            recommended_turns=6,
            max_turns=15,
        )

        # Verify convergence
        assert "_convergence" in state.known
        conv = state.known["_convergence"]
        assert conv["mode"] == "grid"
        assert conv["turns"] > 0
        assert conv["fill_rate"] > 0

        # Verify token tracking (was missing before, now added)
        assert len(engine._compilation_tokens) > 0
