"""
Tests for core/agent_orchestrator.py.

Phase 3 of Agent Ship: Agent Orchestrator tests.
"""

import os
import tempfile
import json
import pytest
from unittest.mock import MagicMock, patch

from core.agent_orchestrator import (
    AgentConfig,
    AgentResult,
    AgentOrchestrator,
    serialize_agent_result,
    _split_template_output,
)
from core.input_enrichment import EnrichmentResult
from core.project_writer import ProjectConfig, ProjectManifest
from core.llm import MockClient
from core.engine import MotherlabsEngine, CompileResult


# =============================================================================
# FIXTURES
# =============================================================================

def _make_engine():
    """Create engine with MockClient for testing."""
    return MotherlabsEngine(llm_client=MockClient(), auto_store=False)


RICH_DESCRIPTION = (
    "A task management system where team leads can create tasks, "
    "assign them to team members, set deadlines, and track progress. "
    "Members receive notifications when tasks are assigned or deadlines "
    "approach. Administrators can manage teams and view analytics dashboards. "
    "Similar to Trello or Asana in scope."
)

SPARSE_DESCRIPTION = "todo app"

SAMPLE_BLUEPRINT = {
    "domain": "Task Management",
    "core_need": "A task management system for teams with deadlines.",
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item"},
        {"name": "Team", "type": "entity", "description": "A group of members"},
        {"name": "TaskManager", "type": "process", "description": "Manages tasks"},
    ],
    "relationships": [
        {"from": "TaskManager", "to": "Task", "type": "manages"},
    ],
    "constraints": [],
}


def _mock_compile_result(success=True, blueprint=None):
    """Create a mock CompileResult for testing."""
    bp = blueprint or SAMPLE_BLUEPRINT
    return CompileResult(
        success=success,
        blueprint=bp,
        error=None if success else "Compilation failed",
    )


def _mock_emission_result():
    """Create a mock EmissionResult-like object."""
    mock = MagicMock()
    mock.generated_code = {
        "Task": 'class Task:\n    """A work item."""\n    def __init__(self):\n        self.title = ""\n',
        "Team": 'class Team:\n    """A group."""\n    def __init__(self):\n        self.members = []\n',
        "TaskManager": 'class TaskManager:\n    """Manages tasks."""\n    pass\n',
    }
    return mock


# =============================================================================
# FROZEN DATACLASS TESTS
# =============================================================================

class TestAgentConfig:
    def test_frozen(self):
        config = AgentConfig()
        with pytest.raises(AttributeError):
            config.codegen_mode = "template"

    def test_defaults(self):
        config = AgentConfig()
        assert config.codegen_mode == "llm"
        assert config.enrich_input is True
        assert config.write_project is True
        assert config.output_dir == "./output"
        assert config.language == "python"

    def test_custom(self):
        config = AgentConfig(codegen_mode="template", write_project=False)
        assert config.codegen_mode == "template"
        assert config.write_project is False


class TestAgentResult:
    def test_frozen(self):
        result = AgentResult(success=True)
        with pytest.raises(AttributeError):
            result.success = False

    def test_defaults(self):
        result = AgentResult(success=True)
        assert result.project_manifest is None
        assert result.blueprint == {}
        assert result.generated_code == {}
        assert result.enrichment is None
        assert result.compile_result is None
        assert result.quality_score == 0.0
        assert result.error is None
        assert result.timing == {}

    def test_with_all_fields(self):
        manifest = ProjectManifest(
            project_dir="/tmp/test",
            files_written=("main.py",),
            entry_point="main.py",
            total_lines=10,
        )
        enrichment = EnrichmentResult(
            original_input="test",
            enriched_input="expanded test",
            expansion_ratio=2.0,
        )
        result = AgentResult(
            success=True,
            project_manifest=manifest,
            blueprint={"domain": "test"},
            generated_code={"Widget": "class Widget: pass"},
            enrichment=enrichment,
            quality_score=0.85,
            timing={"compile": 1.5},
        )
        assert result.success is True
        assert result.project_manifest.project_dir == "/tmp/test"
        assert result.enrichment.expansion_ratio == 2.0


# =============================================================================
# SERIALIZATION
# =============================================================================

class TestSerializeAgentResult:
    def test_minimal(self):
        result = AgentResult(success=True)
        d = serialize_agent_result(result)
        assert d["success"] is True
        assert d["project_manifest"] is None
        assert d["enrichment"] is None
        assert isinstance(d["timing"], dict)

    def test_full(self):
        manifest = ProjectManifest(
            project_dir="/tmp/test",
            files_written=("main.py", "models.py"),
            entry_point="main.py",
            total_lines=42,
        )
        enrichment = EnrichmentResult(
            original_input="test",
            enriched_input="expanded test",
            expansion_ratio=2.0,
        )
        result = AgentResult(
            success=True,
            project_manifest=manifest,
            blueprint={"domain": "test"},
            generated_code={"Widget": "class Widget: pass"},
            enrichment=enrichment,
            quality_score=0.85,
            timing={"compile": 1.5},
        )
        d = serialize_agent_result(result)
        assert d["project_manifest"]["total_lines"] == 42
        assert d["enrichment"]["expansion_ratio"] == 2.0
        assert d["quality_score"] == 0.85
        # Ensure JSON-serializable
        json.dumps(d)

    def test_json_roundtrip(self):
        result = AgentResult(
            success=True,
            blueprint={"components": []},
            generated_code={"Foo": "class Foo: pass"},
            quality_score=0.5,
            timing={"compile": 2.0, "emit": 1.0},
        )
        d = serialize_agent_result(result)
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["success"] is True
        assert loaded["quality_score"] == 0.5


# =============================================================================
# SPLIT TEMPLATE OUTPUT
# =============================================================================

class TestSplitTemplateOutput:
    def test_splits_classes(self):
        code = (
            "class Task:\n    pass\n\n"
            "class TaskManager:\n    pass\n"
        )
        bp = {"components": [{"name": "Task"}, {"name": "TaskManager"}]}
        result = _split_template_output(code, bp)
        assert "Task" in result
        assert "TaskManager" in result

    def test_skips_boilerplate(self):
        code = (
            "class BaseAgent:\n    pass\n\n"
            "class Task:\n    pass\n"
        )
        bp = {"components": [{"name": "Task"}]}
        result = _split_template_output(code, bp)
        assert "Task" in result
        assert "BaseAgent" not in result

    def test_empty_code(self):
        result = _split_template_output("", {"components": []})
        assert result == {}

    def test_no_matching_components(self):
        code = "class Unrelated:\n    pass\n"
        bp = {"components": [{"name": "Task"}]}
        result = _split_template_output(code, bp)
        # Falls back to generic key
        assert "generated" in result


# =============================================================================
# ORCHESTRATOR — COMPILE FLOW
# =============================================================================

class TestOrchestratorRun:
    def test_run_with_rich_input(self):
        """Rich input → compile + emit (no enrichment)."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(output_dir=tmpdir, write_project=False)
            orch = AgentOrchestrator(engine, config)

            with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
                 patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
                result = orch.run(RICH_DESCRIPTION)

            assert result.success is True
            assert result.blueprint  # Non-empty
            assert result.quality_score > 0
            assert result.enrichment is None  # Rich input not enriched
            assert "compile" in result.timing

    def test_run_compile_failure_returns_graceful_result(self):
        """If compilation fails, return AgentResult with error."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', side_effect=Exception("LLM down")):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert "LLM down" in result.error

    def test_run_compile_result_failure(self):
        """Compile returns success=False → AgentResult with error."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result(success=False)):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert result.error

    def test_run_reject_empty_input(self):
        """Empty input → quality gate rejects."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)
        result = orch.run("")

        assert result.success is False
        assert "quality" in result.error.lower()

    def test_run_with_write_project(self):
        """Full flow with project writing to tmpdir."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(output_dir=tmpdir)
            orch = AgentOrchestrator(engine, config)

            with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
                 patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
                result = orch.run(RICH_DESCRIPTION)

            assert result.success is True
            assert result.project_manifest is not None
            assert os.path.isdir(result.project_manifest.project_dir)
            assert result.project_manifest.entry_point == "main.py"

    def test_run_no_write(self):
        """write_project=False → no disk writes."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(RICH_DESCRIPTION)

        assert result.project_manifest is None

    def test_run_template_mode(self):
        """Template codegen mode uses BlueprintCodeGenerator."""
        engine = _make_engine()
        config = AgentConfig(codegen_mode="template", write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result()):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is True
        assert result.generated_code  # Template mode should produce code

    def test_progress_callback(self):
        """on_progress callback receives phase messages."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        progress_calls = []
        def track_progress(phase, msg):
            progress_calls.append((phase, msg))

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(RICH_DESCRIPTION, on_progress=track_progress)

        phases = [p for p, _ in progress_calls]
        assert "quality" in phases
        assert "compile" in phases

    def test_timing_recorded(self):
        """Timing dict has phase durations."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(RICH_DESCRIPTION)

        assert "quality" in result.timing
        assert result.timing["quality"] >= 0


# =============================================================================
# ORCHESTRATOR — ENRICHMENT
# =============================================================================

class TestOrchestratorEnrichment:
    def test_enrichment_on_sparse_input(self):
        """Sparse input triggers enrichment when in hollow zone."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        # Use input that's above reject but below hollow threshold
        sparse = "a simple task management application"

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(sparse)

        # If quality was in hollow zone, enrichment fires
        if result.enrichment is not None:
            assert result.enrichment.original_input == sparse

    def test_no_enrichment_when_disabled(self):
        """enrich_input=False → no enrichment."""
        engine = _make_engine()
        config = AgentConfig(enrich_input=False, write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(RICH_DESCRIPTION)

        assert result.enrichment is None

    def test_enrichment_failure_falls_back(self):
        """If enrichment LLM call fails, uses original input."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        # Force quality to appear hollow so enrichment triggers
        from core.input_quality import QualityScore
        hollow_score = QualityScore(overall=0.22, suggestion="Add detail")

        with patch('core.agent_orchestrator.InputQualityAnalyzer') as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.analyze.return_value = hollow_score
            with patch.object(engine.llm, 'complete_with_system', side_effect=Exception("API error")), \
                 patch.object(engine, 'compile', return_value=_mock_compile_result()), \
                 patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
                result = orch.run("a simple app")
                # Should still succeed — enrichment failed silently
                assert result.success is True


# =============================================================================
# ORCHESTRATOR — EMIT FAILURE
# =============================================================================

class TestOrchestratorEmitFailure:
    def test_emit_failure_returns_error(self):
        """Code emission failure returns AgentResult with error."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', return_value=_mock_compile_result()), \
             patch.object(engine, 'emit_code', side_effect=Exception("Emission failed")):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert "Emission failed" in result.error


# =============================================================================
# TRUST INDICATORS WIRING
# =============================================================================

class TestTrustWiring:
    def test_agent_result_has_trust_field(self):
        """AgentResult accepts and stores trust parameter."""
        from core.trust import TrustIndicators
        trust = TrustIndicators(
            overall_score=75.0,
            provenance_depth=2,
            fidelity_scores={"completeness": 80, "consistency": 70},
            gap_report=(),
            dimensional_coverage={},
            verification_badge="partial",
            confidence_trajectory=(),
            silence_zones=(),
            derivation_chain_length=1.5,
            component_count=3,
            relationship_count=1,
            constraint_count=0,
            method_coverage=0.8,
        )
        result = AgentResult(success=True, trust=trust)
        assert result.trust is not None
        assert result.trust.overall_score == 75.0
        assert result.trust.verification_badge == "partial"

    def test_trust_computed_after_compile(self):
        """run() populates trust from compile_result data."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        cr = _mock_compile_result()
        with patch.object(engine, 'compile', return_value=cr), \
             patch.object(engine, 'emit_code', return_value=_mock_emission_result()):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is True
        assert result.trust is not None
        assert result.trust.component_count == len(SAMPLE_BLUEPRINT["components"])
        assert result.trust.relationship_count == len(SAMPLE_BLUEPRINT["relationships"])
        assert result.trust.verification_badge in ("verified", "partial", "unverified")

    def test_trust_serialized_in_agent_result(self):
        """serialize_agent_result() includes trust dict with all fields."""
        from core.trust import TrustIndicators
        trust = TrustIndicators(
            overall_score=60.0,
            provenance_depth=1,
            fidelity_scores={"completeness": 60, "consistency": 55, "coherence": 50, "traceability": 45},
            gap_report=("missing: auth",),
            dimensional_coverage={"complexity": 0.8},
            verification_badge="partial",
            confidence_trajectory=(0.5, 0.6),
            silence_zones=("security",),
            derivation_chain_length=1.2,
            component_count=3,
            relationship_count=1,
            constraint_count=0,
            method_coverage=0.67,
        )
        result = AgentResult(success=True, blueprint=SAMPLE_BLUEPRINT, trust=trust)
        d = serialize_agent_result(result)

        assert d["trust"] is not None
        assert d["trust"]["overall_score"] == 60.0
        assert d["trust"]["verification_badge"] == "partial"
        assert d["trust"]["fidelity_scores"]["completeness"] == 60
        assert d["trust"]["gap_report"] == ["missing: auth"]
        assert d["trust"]["silence_zones"] == ["security"]
        assert d["trust"]["component_count"] == 3
        # Ensure JSON-serializable
        json.dumps(d)

    def test_trust_none_on_compile_failure(self):
        """trust is None when compile fails."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(engine, 'compile', side_effect=Exception("LLM down")):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert result.trust is None


# =============================================================================
# COMPILATION RETRY TESTS
# =============================================================================


class TestCompilationRetry:
    """Tests for transient error retry in AgentOrchestrator.run()."""

    def test_retry_on_provider_error(self):
        """Orchestrator retries compile once on ProviderError."""
        from core.exceptions import ProviderError

        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        success_result = _mock_compile_result(success=True)
        call_count = {"n": 0}

        def compile_with_retry(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ProviderError("500 Internal Server Error", provider="test")
            return success_result

        with patch.object(engine, 'compile', side_effect=compile_with_retry), \
             patch("core.agent_orchestrator.time.sleep"):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is True
        assert call_count["n"] == 2  # 1 fail + 1 success

    def test_retry_on_timeout_error(self):
        """Orchestrator retries compile once on TimeoutError."""
        from core.exceptions import TimeoutError as MotherlabsTimeout

        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        success_result = _mock_compile_result(success=True)
        call_count = {"n": 0}

        def compile_with_retry(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise MotherlabsTimeout("request timed out")
            return success_result

        with patch.object(engine, 'compile', side_effect=compile_with_retry), \
             patch("core.agent_orchestrator.time.sleep"):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is True
        assert call_count["n"] == 2

    def test_retry_exhausted_returns_failure(self):
        """Orchestrator returns failure after retry exhausted."""
        from core.exceptions import ProviderError

        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(
            engine, 'compile',
            side_effect=ProviderError("always fails", provider="test")
        ), patch("core.agent_orchestrator.time.sleep"):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert "always fails" in result.error

    def test_non_retryable_error_fails_immediately(self):
        """Non-transient errors (ValueError, etc.) don't trigger retry."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        with patch.object(
            engine, 'compile',
            side_effect=ValueError("bad schema")
        ):
            result = orch.run(RICH_DESCRIPTION)

        assert result.success is False
        assert "bad schema" in result.error


# =============================================================================
# SYNTAX REPAIR
# =============================================================================

class TestSyntaxRepair:
    """Tests for post-emission syntax repair loop."""

    def test_no_repair_needed(self):
        """Clean code passes through without repair."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        code = {"Task": "class Task:\n    pass\n"}
        progress_calls = []
        result = orch._repair_syntax_errors(
            code, SAMPLE_BLUEPRINT,
            lambda phase, msg: progress_calls.append((phase, msg)),
        )
        assert result == code
        # No "repair" progress calls
        assert not any(p == "repair" for p, _ in progress_calls)

    def test_repair_syntax_error(self):
        """Broken code triggers LLM repair and produces fixed code."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        broken = {"Task": "class Task:\n    def foo(self\n"}
        fixed_response = "```python\nclass Task:\n    def foo(self):\n        pass\n```"

        with patch.object(
            engine.llm, 'complete_with_system',
            return_value=fixed_response,
        ):
            result = orch._repair_syntax_errors(
                broken, SAMPLE_BLUEPRINT,
                lambda phase, msg: None,
            )

        assert "def foo(self):" in result["Task"]

    def test_repair_max_attempts(self):
        """Repair stops after max_repair_attempts passes."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        # Return still-broken code each time
        still_broken = "```python\nclass Task:\n    def foo(self\n```"
        call_count = [0]

        def _mock_complete(**kwargs):
            call_count[0] += 1
            return still_broken

        with patch.object(
            engine.llm, 'complete_with_system',
            side_effect=_mock_complete,
        ):
            result = orch._repair_syntax_errors(
                {"Task": "class Task:\n    def foo(self\n"},
                SAMPLE_BLUEPRINT,
                lambda phase, msg: None,
                max_repair_attempts=2,
            )

        # Should have tried exactly 2 times
        assert call_count[0] == 2

    def test_repair_skips_non_python(self):
        """Non-Python formats skip syntax repair."""
        engine = _make_engine()
        # Use process adapter (YAML output)
        from adapters.process import PROCESS_ADAPTER
        engine_yaml = MotherlabsEngine(
            llm_client=MockClient(), auto_store=False,
            domain_adapter=PROCESS_ADAPTER,
        )
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine_yaml, config)

        # Invalid Python but valid-ish YAML
        code = {"Process": "steps:\n  - name: do thing\n"}
        result = orch._repair_syntax_errors(
            code, {}, lambda phase, msg: None,
        )
        # No repair attempted — YAML doesn't go through ast.parse
        assert result == code

    def test_repair_preserves_clean_components(self):
        """Only broken components get repaired; clean ones stay untouched."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        code = {
            "Task": "class Task:\n    pass\n",
            "Team": "class Team:\n    def foo(self\n",  # broken
        }
        fixed_response = "```python\nclass Team:\n    def foo(self):\n        pass\n```"

        with patch.object(
            engine.llm, 'complete_with_system',
            return_value=fixed_response,
        ):
            result = orch._repair_syntax_errors(
                code, SAMPLE_BLUEPRINT,
                lambda phase, msg: None,
            )

        # Task unchanged
        assert result["Task"] == "class Task:\n    pass\n"
        # Team fixed
        assert "def foo(self):" in result["Team"]

    def test_repair_llm_failure_is_safe(self):
        """If LLM call fails during repair, original code is kept."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        broken = {"Task": "class Task:\n    def foo(self\n"}

        with patch.object(
            engine.llm, 'complete_with_system',
            side_effect=Exception("LLM unavailable"),
        ):
            result = orch._repair_syntax_errors(
                broken, SAMPLE_BLUEPRINT,
                lambda phase, msg: None,
            )

        # Original code preserved (not replaced with empty)
        assert result["Task"] == broken["Task"]

    def test_repair_wired_in_run(self):
        """The run() method calls syntax repair after emission."""
        engine = _make_engine()
        config = AgentConfig(write_project=False)
        orch = AgentOrchestrator(engine, config)

        compile_result = CompileResult(
            success=True,
            blueprint=SAMPLE_BLUEPRINT,
            verification={"status": "pass"},
            context_graph={"keywords": ["task"]},
        )

        with patch.object(engine, 'compile', return_value=compile_result), \
             patch.object(orch, '_emit_code', return_value={"Task": "class Task:\n    pass\n"}), \
             patch.object(orch, '_repair_syntax_errors', wraps=orch._repair_syntax_errors) as mock_repair:
            result = orch.run(RICH_DESCRIPTION)

        assert result.success
        mock_repair.assert_called_once()
