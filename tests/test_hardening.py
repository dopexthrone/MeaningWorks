"""
Hardening sprint tests — verify safety-critical invariants.

Covers:
1. Cost caps at documented $5/$50 levels
2. Post-synthesis cost check coverage
3. Code safety gate wired into AgentOrchestrator
4. STAGE_GATES isolation per engine instance
5. Verification parse failure logging
6. Silent except blocks now log
"""

import copy
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from core.protocol_spec import PROTOCOL
from core.engine import MotherlabsEngine, STAGE_GATES, StageGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    """Create a minimal engine with mock LLM for testing."""
    from persistence.corpus import Corpus
    client = Mock()
    client.provider_name = "mock"
    client.model_name = "mock-model"
    client.deterministic = True
    client.model = "mock-model"
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=Corpus(tmp_path / "corpus"),
        auto_store=False,
    )
    return engine


# ===========================================================================
# TestCostCaps
# ===========================================================================

class TestCostCaps:
    """Cost caps match documented $5/$50 limits."""

    def test_per_compilation_cap_is_5(self):
        assert PROTOCOL.cost.per_compilation_cap_usd == 5.0

    def test_session_cap_is_50(self):
        assert PROTOCOL.cost.session_cap_usd == 50.0

    def test_warning_threshold_below_cap(self):
        assert PROTOCOL.cost.per_compilation_warn_usd < PROTOCOL.cost.per_compilation_cap_usd

    def test_health_warn_below_cap(self):
        assert PROTOCOL.cost.health_cost_warn_threshold < PROTOCOL.cost.per_compilation_cap_usd

    def test_cost_cap_raises_on_exceed(self, tmp_path):
        """_check_cost_cap raises CostCapExceededError when exceeded."""
        from core.exceptions import CostCapExceededError
        from core.telemetry import TokenUsage

        engine = _make_engine(tmp_path)
        # Inject fake token usage that exceeds $5 cap
        # At Claude rates ($3/M input, $15/M output), 500K output tokens = $7.50
        engine._compilation_tokens = [
            TokenUsage(
                input_tokens=100_000,
                output_tokens=500_000,
                total_tokens=600_000,
                provider="claude",
                model="claude-sonnet-4",
            )
        ]
        # Cost: 100K input * $3/M + 500K output * $15/M = $0.30 + $7.50 = $7.80 > $5
        with pytest.raises(CostCapExceededError):
            engine._check_cost_cap()

    def test_cost_cap_doesnt_raise_under_limit(self, tmp_path):
        from core.telemetry import TokenUsage

        engine = _make_engine(tmp_path)
        # Small usage — well under $5
        engine._compilation_tokens = [
            TokenUsage(
                input_tokens=1_000,
                output_tokens=1_000,
                total_tokens=2_000,
                provider="claude",
                model="claude-sonnet-4",
            )
        ]
        # Should not raise
        engine._check_cost_cap()

    def test_cost_cap_noop_with_no_tokens(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._compilation_tokens = []
        engine._check_cost_cap()  # Should not raise


# ===========================================================================
# TestPostSynthesisCostChecks
# ===========================================================================

class TestPostSynthesisCostChecks:
    """Cost checks exist after verification, re-synthesis, and re-verification."""

    def test_cost_check_after_verification_exists(self):
        """Verify _check_cost_cap is called after verification in engine source."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        # Find verification collection and cost check
        lines = source.split('\n')
        found_verify_collect = False
        found_cost_check_after = False
        for i, line in enumerate(lines):
            if 'stage_timings["verification"]' in line:
                found_verify_collect = True
            if found_verify_collect and '_check_cost_cap' in line:
                found_cost_check_after = True
                break
        assert found_cost_check_after, "No _check_cost_cap after verification stage"

    def test_cost_check_after_resynthesis_exists(self):
        """Verify _check_cost_cap is called after re-synthesis."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        lines = source.split('\n')
        found_resynth_collect = False
        found_cost_check_after = False
        for i, line in enumerate(lines):
            if 'stage_timings["resynthesis"]' in line:
                found_resynth_collect = True
            if found_resynth_collect and '_check_cost_cap' in line:
                found_cost_check_after = True
                break
        assert found_cost_check_after, "No _check_cost_cap after re-synthesis stage"

    def test_cost_check_after_reverification_exists(self):
        """Verify _check_cost_cap is called after re-verification."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        lines = source.split('\n')
        found_reverify = False
        found_cost_check = False
        for i, line in enumerate(lines):
            if 're-verification' in line:
                found_reverify = True
            if found_reverify and '_check_cost_cap' in line:
                found_cost_check = True
                break
        assert found_cost_check, "No _check_cost_cap after re-verification"


# ===========================================================================
# TestCodeSafetyGate
# ===========================================================================

class TestCodeSafetyGate:
    """Code safety check wired into AgentOrchestrator before write_project."""

    def test_safety_check_in_orchestrator_source(self):
        """AgentOrchestrator.run() calls check_code_safety before write_project."""
        import inspect
        from core.agent_orchestrator import AgentOrchestrator
        source = inspect.getsource(AgentOrchestrator.run)
        # check_code_safety must appear before write_project
        safety_idx = source.find("check_code_safety")
        write_idx = source.find("write_project")
        assert safety_idx > 0, "check_code_safety not found in AgentOrchestrator.run"
        assert write_idx > 0, "write_project not found in AgentOrchestrator.run"
        assert safety_idx < write_idx, "check_code_safety must come before write_project"

    def test_check_code_safety_rejects_dangerous_code(self):
        from core.governor_validation import check_code_safety
        dangerous = {"evil_component": "import os\nos.system('rm -rf /')"}
        safe, warnings = check_code_safety(dangerous)
        assert safe is False
        assert len(warnings) > 0

    def test_check_code_safety_passes_clean_code(self):
        from core.governor_validation import check_code_safety
        clean = {"my_component": "class MyComponent:\n    def run(self):\n        return 42\n"}
        safe, warnings = check_code_safety(clean)
        assert safe is True

    def test_check_code_safety_rejects_exec(self):
        from core.governor_validation import check_code_safety
        code = {"comp": "result = exec('print(1)')"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_check_code_safety_rejects_eval(self):
        from core.governor_validation import check_code_safety
        code = {"comp": "result = eval('1+1')"}
        safe, warnings = check_code_safety(code)
        assert safe is False

    def test_check_code_safety_size_limit(self):
        from core.governor_validation import check_code_safety
        huge = {"comp": "x = 1\n" * 200_000}
        safe, warnings = check_code_safety(huge, max_size_bytes=1_000)
        assert safe is False
        assert any("size" in w.lower() for w in warnings)

    def test_check_code_safety_empty_code(self):
        from core.governor_validation import check_code_safety
        safe, warnings = check_code_safety({})
        assert safe is True
        assert warnings == []


# ===========================================================================
# TestStageGatesIsolation
# ===========================================================================

class TestStageGatesIsolation:
    """STAGE_GATES are deep-copied per engine instance — no cross-contamination."""

    def test_engine_has_instance_stage_gates(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert hasattr(engine, "_stage_gates")
        assert engine._stage_gates is not STAGE_GATES

    def test_two_engines_have_independent_gates(self, tmp_path):
        e1 = _make_engine(tmp_path)
        e2 = _make_engine(tmp_path / "corpus2")

        # Mutate e1's timeout
        e1._stage_gates["intent"].timeout_seconds = 9999
        # e2 should be unaffected
        assert e2._stage_gates["intent"].timeout_seconds != 9999

    def test_engine_mutation_doesnt_affect_module_level(self, tmp_path):
        original_timeout = STAGE_GATES["intent"].timeout_seconds
        engine = _make_engine(tmp_path)
        engine._stage_gates["intent"].timeout_seconds = 12345
        assert STAGE_GATES["intent"].timeout_seconds == original_timeout

    def test_all_stage_names_copied(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert set(engine._stage_gates.keys()) == set(STAGE_GATES.keys())

    def test_base_timeouts_from_instance_gates(self, tmp_path):
        engine = _make_engine(tmp_path)
        for name in engine._base_timeouts:
            assert engine._base_timeouts[name] == engine._stage_gates[name].timeout_seconds


# ===========================================================================
# TestVerificationParseLogging
# ===========================================================================

class TestVerificationParseLogging:
    """Verification parse failures produce log warnings."""

    def test_verify_llm_logs_on_parse_failure(self, tmp_path, caplog):
        engine = _make_engine(tmp_path)
        from core.protocol import SharedState, Message, MessageType

        state = SharedState()
        # Mock verify_agent to return unparseable content
        engine.verify_agent = Mock()
        engine.verify_agent.run.return_value = Message(
            sender="Verify",
            content="This is not JSON at all",
            message_type=MessageType.PROPOSITION,
        )

        with caplog.at_level(logging.WARNING, logger="motherlabs.engine"):
            result = engine._verify_llm({}, state)

        assert result["status"] == "needs_work"
        assert any("Verification LLM parse failed" in r.message for r in caplog.records)

    def test_verify_llm_focused_logs_on_parse_failure(self, tmp_path, caplog):
        engine = _make_engine(tmp_path)
        from core.protocol import SharedState, Message, MessageType

        state = SharedState()
        engine.verify_agent = Mock()
        engine.verify_agent.run.return_value = Message(
            sender="Verify",
            content="Not JSON either",
            message_type=MessageType.PROPOSITION,
        )

        with caplog.at_level(logging.WARNING, logger="motherlabs.engine"):
            result = engine._verify_llm_focused({}, state, ("completeness",))

        assert result["status"] == "needs_work"
        assert any("Focused verification parse failed" in r.message for r in caplog.records)


# ===========================================================================
# TestSilentExceptLogging
# ===========================================================================

class TestSilentExceptLogging:
    """Formerly silent except blocks now emit logger.debug messages."""

    def test_engine_compile_source_has_no_bare_pass_in_additive_blocks(self):
        """All additive except blocks should have logger.debug, not bare pass."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        lines = source.split('\n')

        # Find except blocks and check the next non-blank line
        bare_pass_blocks = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('except Exception'):
                # Look at next non-blank line
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line:
                        if next_line == 'pass':
                            bare_pass_blocks.append(i)
                        break

        assert len(bare_pass_blocks) == 0, (
            f"Found {len(bare_pass_blocks)} bare pass blocks in compile() at source lines: {bare_pass_blocks}"
        )

    def test_additive_blocks_use_logger_debug(self):
        """Verify logger.debug calls exist for additive system failures."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)

        expected_messages = [
            "Rejection log read skipped",
            "Governor feedback injection skipped",
            "Prior grid load skipped",
            "Kernel compilation skipped",
            "Closed-loop gate failed",
            "Observer skipped",
            "Grid persistence skipped",
            "Outcome recording skipped",
        ]
        for msg in expected_messages:
            assert msg in source, f"Missing logger.debug message: {msg}"


# ===========================================================================
# TestOutcomeStore
# ===========================================================================

class TestOutcomeStore:
    """Persistent CompilationOutcome storage."""

    def test_create_store(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        assert store.count() == 0
        store.close()

    def test_append_and_count(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        row_id = store.append(compile_id="test-1", trust_score=75.0)
        assert row_id >= 1
        assert store.count() == 1
        store.close()

    def test_recent_returns_newest_first(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        store.append(compile_id="first", trust_score=50.0)
        store.append(compile_id="second", trust_score=80.0)
        records = store.recent(limit=10)
        assert len(records) == 2
        assert records[0].compile_id == "second"
        assert records[1].compile_id == "first"
        store.close()

    def test_record_fields_roundtrip(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        store.append(
            compile_id="rt-1",
            input_summary="test input",
            trust_score=72.5,
            completeness=80.0,
            consistency=65.0,
            coherence=90.0,
            traceability=55.0,
            component_count=12,
            rejected=True,
            rejection_reason="low trust",
            domain="api",
        )
        rec = store.recent(limit=1)[0]
        assert rec.compile_id == "rt-1"
        assert rec.input_summary == "test input"
        assert rec.trust_score == 72.5
        assert rec.completeness == 80.0
        assert rec.consistency == 65.0
        assert rec.coherence == 90.0
        assert rec.traceability == 55.0
        assert rec.component_count == 12
        assert rec.rejected is True
        assert rec.rejection_reason == "low trust"
        assert rec.domain == "api"
        store.close()

    def test_rejection_rate(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        store.append(compile_id="a", rejected=True)
        store.append(compile_id="b", rejected=False)
        store.append(compile_id="c", rejected=False)
        store.append(compile_id="d", rejected=True)
        rate = store.rejection_rate(last_n=4)
        assert rate == 0.5
        store.close()

    def test_rejection_rate_empty(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store = OutcomeStore(db_dir=tmp_path)
        assert store.rejection_rate() == 0.0
        store.close()

    def test_store_survives_reopen(self, tmp_path):
        from core.outcome_store import OutcomeStore
        store1 = OutcomeStore(db_dir=tmp_path)
        store1.append(compile_id="persist-1", trust_score=42.0)
        store1.close()

        store2 = OutcomeStore(db_dir=tmp_path)
        assert store2.count() == 1
        rec = store2.recent(limit=1)[0]
        assert rec.compile_id == "persist-1"
        assert rec.trust_score == 42.0
        store2.close()

    def test_engine_initializes_outcome_store(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert hasattr(engine, "_outcome_store")

    def test_engine_outcome_store_writes(self, tmp_path):
        """OutcomeStore.append is present in engine source after outcome recording."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "outcome_store" in source
        assert "self._outcome_store.append" in source


# ===========================================================================
# TestCatastrophicGates
# ===========================================================================

class TestCatastrophicGates:
    """Hard stops on catastrophic failures."""

    def test_zero_components_hard_stop_in_source(self):
        """Engine source contains 0-component hard stop after synthesis."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Synthesis produced 0 components" in source

    def test_catastrophic_verification_hard_stop_in_source(self):
        """Engine source contains catastrophic verification hard stop."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Catastrophic verification failure" in source

    def test_catastrophic_gate_checks_all_dimensions(self):
        """All 4 dimension scores must be < 30 for catastrophic stop."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        # Must check all 4 dimensions
        for dim in ["completeness", "consistency", "coherence", "traceability"]:
            assert dim in source

    def test_catastrophic_gate_requires_nonzero(self):
        """Hard stop requires at least one score > 0 (verification actually ran)."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "any(s > 0 for s in _dim_scores)" in source

    def test_zero_components_returns_compile_result(self):
        """The 0-component hard stop returns CompileResult(success=False)."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        # Must return CompileResult, not raise
        idx_zero = source.find("Synthesis produced 0 components")
        assert idx_zero > 0
        # Find the return statement near it
        after_zero = source[idx_zero - 200:idx_zero + 200]
        assert "success=False" in after_zero
        assert "CompileResult" in after_zero

    def test_resynth_min_completeness_exists(self):
        """Protocol has resynth_min_completeness threshold."""
        assert hasattr(PROTOCOL.engine, "resynth_min_completeness")
        assert PROTOCOL.engine.resynth_min_completeness > 0
