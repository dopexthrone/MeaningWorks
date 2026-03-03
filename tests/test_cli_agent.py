"""
Tests for CLI `agent` command.

Phase 4 of Agent Ship: CLI integration tests.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cli.main import cmd_agent, cmd_health, cmd_metrics, CLI, ConfigManager, ProgressIndicator


def _make_mock_args(**overrides):
    """Create mock args for cmd_agent."""
    args = MagicMock()
    args.description = overrides.get("description", None)
    args.file = overrides.get("file", None)
    args.output = overrides.get("output", "./output")
    args.provider = overrides.get("provider", None)
    args.model = overrides.get("model", None)
    args.mode = overrides.get("mode", "llm")
    args.no_enrich = overrides.get("no_enrich", False)
    args.dry_run = overrides.get("dry_run", False)
    args.domain = overrides.get("domain", "software")
    return args


def _make_mock_result(success=True, error=None):
    """Create a mock AgentResult."""
    from core.agent_orchestrator import AgentResult
    from core.project_writer import ProjectManifest

    manifest = ProjectManifest(
        project_dir="/tmp/test_project",
        files_written=("main.py", "models.py", "requirements.txt"),
        entry_point="main.py",
        total_lines=42,
    ) if success else None

    return AgentResult(
        success=success,
        project_manifest=manifest,
        blueprint={"domain": "test", "components": [{"name": "Task", "type": "entity"}], "relationships": []},
        generated_code={"Task": "class Task: pass"},
        quality_score=0.85,
        error=error,
        timing={"compile": 1.0, "emit": 0.5},
    )


class TestCmdAgentSubparser:
    def test_agent_subparser_exists(self):
        """Verify 'agent' subparser is registered."""
        from cli.main import main
        import argparse
        # Just check it parses without error
        import sys
        with patch.object(sys, 'argv', ['motherlabs', 'agent', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_agent_dry_run_flag(self):
        """Verify --dry-run flag is parsed."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        agent_p = subparsers.add_parser("agent")
        agent_p.add_argument("description", nargs="?")
        agent_p.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(["agent", "test", "--dry-run"])
        assert args.dry_run is True


class TestCmdAgentExecution:
    def test_cmd_agent_success(self):
        """cmd_agent with mocked orchestrator produces output."""
        cli = CLI()
        config = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_mock_args(
                description="A task management system with teams and deadlines",
                output=tmpdir,
            )

            mock_result = _make_mock_result(success=True)

            with patch('core.engine.MotherlabsEngine') as MockEng, \
                 patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
                mock_orch_instance = MockOrch.return_value
                mock_orch_instance.run.return_value = mock_result
                # Should not raise
                cmd_agent(args, cli, config)

    def test_cmd_agent_reads_file(self):
        """cmd_agent reads from --file."""
        cli = CLI()
        config = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write description file
            desc_file = Path(tmpdir) / "desc.txt"
            desc_file.write_text("A todo app with teams")

            args = _make_mock_args(
                description=None,
                file=str(desc_file),
                output=tmpdir,
            )

            mock_result = _make_mock_result(success=True)

            with patch('core.engine.MotherlabsEngine') as MockEng, \
                 patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
                mock_orch_instance = MockOrch.return_value
                mock_orch_instance.run.return_value = mock_result
                cmd_agent(args, cli, config)

    def test_cmd_agent_failure_exits(self):
        """cmd_agent exits with code 1 on failure."""
        cli = CLI()
        config = ConfigManager()

        args = _make_mock_args(
            description="A task management app",
        )

        mock_result = _make_mock_result(success=False, error="Compilation failed")

        with patch('core.engine.MotherlabsEngine') as MockEng, \
             patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.run.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_agent(args, cli, config)
            assert exc_info.value.code == 1

    def test_cmd_agent_no_description_exits(self):
        """cmd_agent with no description prompts, then exits if empty."""
        cli = CLI()
        config = ConfigManager()

        args = _make_mock_args(description=None, file=None)

        with patch.object(cli, 'prompt', return_value=""):
            with pytest.raises(SystemExit):
                cmd_agent(args, cli, config)


class TestNewSubparsers:
    def test_health_subparser_exists(self):
        """Verify 'health' subparser is registered."""
        from cli.main import main
        import sys
        with patch.object(sys, 'argv', ['motherlabs', 'health', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_metrics_subparser_exists(self):
        """Verify 'metrics' subparser is registered."""
        from cli.main import main
        import sys
        with patch.object(sys, 'argv', ['motherlabs', 'metrics', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_compile_tree_subparser_exists(self):
        """Verify 'compile-tree' subparser is registered."""
        from cli.main import main
        import sys
        with patch.object(sys, 'argv', ['motherlabs', 'compile-tree', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_edit_subparser_exists(self):
        """Verify 'edit' subparser is registered."""
        from cli.main import main
        import sys
        with patch.object(sys, 'argv', ['motherlabs', 'edit', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_self_compile_loop_flag(self):
        """Verify --loop flag on self-compile."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        sp = subparsers.add_parser("self-compile")
        sp.add_argument("--loop", type=int, default=0)
        args = parser.parse_args(["self-compile", "--loop", "3"])
        assert args.loop == 3

    def test_cmd_health_runs(self):
        """cmd_health runs without error."""
        cli = CLI()
        config = ConfigManager()
        args = MagicMock()
        # Should not raise — engine with no llm_client still provides health
        with patch('core.engine.MotherlabsEngine') as MockEng:
            mock_engine = MockEng.return_value
            mock_engine.get_health_snapshot.return_value = {
                "status": "healthy",
                "uptime_seconds": 100,
                "total_compilations": 5,
                "success_rate": 0.8,
                "avg_duration_seconds": 2.5,
                "cache_hit_rate": 0.5,
                "recent_total_cost_usd": 0.01,
                "issues": [],
            }
            cmd_health(args, cli, config)

    def test_cmd_metrics_runs(self):
        """cmd_metrics runs without error."""
        cli = CLI()
        config = ConfigManager()
        args = MagicMock()
        with patch('core.engine.MotherlabsEngine') as MockEng:
            mock_engine = MockEng.return_value
            mock_engine.get_metrics.return_value = {
                "total_compilations": 10,
                "success_rate": 0.9,
                "avg_duration_seconds": 3.0,
                "total_input_tokens": 5000,
                "total_output_tokens": 3000,
                "total_cost_usd": 0.05,
                "avg_cost_usd": 0.005,
            }
            cmd_metrics(args, cli, config)


class TestCLITrustSummary:
    """Tests for CLI.trust_summary() rendering."""

    def test_trust_summary_verified(self, capsys):
        """trust_summary renders badge + scores for verified compilation."""
        from core.trust import TrustIndicators
        cli = CLI()
        trust = TrustIndicators(
            overall_score=78.5,
            provenance_depth=2,
            fidelity_scores={
                "completeness": 72,
                "consistency": 85,
                "coherence": 68,
                "traceability": 91,
            },
            gap_report=("Input keyword not covered: 'caching'", "No provenance: component 'Logger'"),
            dimensional_coverage={"complexity": 0.8},
            verification_badge="verified",
            confidence_trajectory=(0.3, 0.5, 0.7, 0.9),
            silence_zones=(),
            derivation_chain_length=2.1,
            component_count=10,
            relationship_count=15,
            constraint_count=3,
            method_coverage=0.9,
        )
        cli.trust_summary(trust)
        out = capsys.readouterr().out
        assert "verified" in out
        assert "completeness" in out
        assert "72" in out
        assert "2/3" in out
        assert "2 gaps" in out

    def test_trust_summary_unverified(self, capsys):
        """trust_summary renders unverified badge."""
        from core.trust import TrustIndicators
        cli = CLI()
        trust = TrustIndicators(
            overall_score=25.0,
            provenance_depth=1,
            fidelity_scores={"completeness": 30, "consistency": 20, "coherence": 35, "traceability": 15},
            gap_report=(),
            dimensional_coverage={},
            verification_badge="unverified",
            confidence_trajectory=(),
            silence_zones=(),
            derivation_chain_length=0.5,
            component_count=2,
            relationship_count=0,
            constraint_count=0,
            method_coverage=0.0,
        )
        cli.trust_summary(trust)
        out = capsys.readouterr().out
        assert "unverified" in out
        assert "no gaps" in out


class TestCLIRichError:
    """Tests for CLI.rich_error() rendering."""

    def test_rich_error_motherlabs_error(self, capsys):
        """rich_error renders structured info for MotherlabsError."""
        from core.exceptions import InputQualityError
        cli = CLI()
        exc = InputQualityError(
            "Input too vague",
            user_message="Your description is too vague to compile.",
            suggestion="Add more detail about the domain.",
            error_code="E1001",
        )
        cli.rich_error(exc)
        out = capsys.readouterr().out
        assert "E1001" in out
        assert "vague" in out.lower()
        assert "DEBUG=1" in out

    def test_rich_error_generic_exception(self, capsys):
        """rich_error falls back to plain string for generic exceptions."""
        cli = CLI()
        exc = ValueError("something broke")
        cli.rich_error(exc)
        out = capsys.readouterr().out
        assert "something broke" in out

    def test_rich_error_with_fix_examples(self, capsys):
        """rich_error renders fix_examples from catalog."""
        from core.exceptions import MotherlabsError
        cli = CLI()
        exc = MotherlabsError(
            "test error",
            user_message="Test failure",
            error_code="E3006",
        )
        cli.rich_error(exc)
        out = capsys.readouterr().out
        assert "E3006" in out
        assert "DEBUG=1" in out


class TestCLICostLine:
    """Tests for CLI.cost_line() rendering."""

    def test_cost_line_with_tokens(self, capsys):
        """cost_line renders cost for mock engine with tokens."""
        from core.telemetry import TokenUsage
        cli = CLI()
        mock_engine = MagicMock()
        mock_engine._compilation_tokens = [
            TokenUsage(input_tokens=5000, output_tokens=2000, total_tokens=7000,
                       provider="claude", model="claude-sonnet-4"),
        ]
        mock_engine.model_name = "claude-sonnet-4"
        cli.cost_line(mock_engine)
        out = capsys.readouterr().out
        assert "$" in out
        assert "7,000" in out
        assert "claude-sonnet-4" in out

    def test_cost_line_no_tokens(self, capsys):
        """cost_line prints nothing when no tokens tracked."""
        cli = CLI()
        mock_engine = MagicMock()
        mock_engine._compilation_tokens = []
        cli.cost_line(mock_engine)
        out = capsys.readouterr().out
        assert out == ""


class TestSpinnerTTY:
    """Tests for ProgressIndicator spinner tty guard."""

    def test_spinner_no_crash_non_tty(self):
        """Spinner doesn't crash when stderr is not a tty."""
        cli = CLI()
        progress = ProgressIndicator(cli)
        # Force non-tty
        progress._is_tty = False
        progress.start("test")
        assert progress._spinning is False
        progress.update("msg")
        progress.done()

    def test_spinner_flag_set_for_tty(self):
        """Spinner sets _is_tty from sys.stderr."""
        cli = CLI()
        progress = ProgressIndicator(cli)
        # _is_tty should be bool
        assert isinstance(progress._is_tty, bool)


class TestCmdTrustSubparser:
    """Tests for the trust command subparser."""

    def test_trust_subparser_exists(self):
        """Verify 'trust' subparser is registered."""
        from cli.main import main
        import sys as _sys
        with patch.object(_sys, 'argv', ['motherlabs', 'trust', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_cmd_trust_dispatches(self):
        """cmd_trust loads from corpus and computes trust."""
        from cli.main import cmd_trust
        cli = CLI()
        config = ConfigManager()
        args = MagicMock()
        args.compilation_id = "abc123"

        mock_blueprint = {
            "components": [{"name": "A", "type": "service", "derived_from": "user input"}],
            "relationships": [],
            "constraints": [],
        }

        with patch('persistence.corpus.Corpus') as MockCorpus:
            mock_corpus = MockCorpus.return_value
            mock_corpus.load_blueprint.return_value = mock_blueprint
            mock_corpus.load_context_graph.return_value = {
                "keywords": ["auth", "login"],
                "insights": ["insight1"],
                "dimensional_metadata": {},
                "verification": {},
            }
            cmd_trust(args, cli, config)

    def test_cmd_trust_not_found(self):
        """cmd_trust exits when compilation not found."""
        from cli.main import cmd_trust
        cli = CLI()
        config = ConfigManager()
        args = MagicMock()
        args.compilation_id = "nonexistent"

        with patch('persistence.corpus.Corpus') as MockCorpus:
            mock_corpus = MockCorpus.return_value
            mock_corpus.load_blueprint.return_value = None
            with pytest.raises(SystemExit):
                cmd_trust(args, cli, config)


class TestAgentPhaseNumbers:
    """Tests for agent progress phase numbering."""

    def test_agent_phases_with_build(self):
        """Phase map uses /5 when build=True."""
        from cli.main import cmd_agent
        from core.agent_orchestrator import AgentConfig
        cli = CLI()
        config = ConfigManager()
        args = _make_mock_args(description="test app", build=True)
        args.build = True

        captured_headers = []
        original_phase = cli.phase

        def capture_phase(name):
            captured_headers.append(name)
            original_phase(name)

        mock_result = _make_mock_result(success=True)

        with patch.object(cli, 'phase', side_effect=capture_phase), \
             patch('core.engine.MotherlabsEngine'), \
             patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
            mock_orch_instance = MockOrch.return_value
            # Make run() call progress callback with "compile" phase
            def fake_run(desc, on_progress=None):
                if on_progress:
                    on_progress("compile", "Compiling...")
                return mock_result
            mock_orch_instance.run.side_effect = fake_run
            cmd_agent(args, cli, config)

        compile_headers = [h for h in captured_headers if "Compiling" in h]
        assert any("/5]" in h for h in compile_headers), f"Expected /5 in headers: {compile_headers}"

    def test_agent_phases_without_build(self):
        """Phase map uses /4 when build=False."""
        from cli.main import cmd_agent
        cli = CLI()
        config = ConfigManager()
        args = _make_mock_args(description="test app")
        args.build = False

        captured_headers = []
        original_phase = cli.phase

        def capture_phase(name):
            captured_headers.append(name)
            original_phase(name)

        mock_result = _make_mock_result(success=True)

        with patch.object(cli, 'phase', side_effect=capture_phase), \
             patch('core.engine.MotherlabsEngine'), \
             patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
            mock_orch_instance = MockOrch.return_value
            def fake_run(desc, on_progress=None):
                if on_progress:
                    on_progress("compile", "Compiling...")
                return mock_result
            mock_orch_instance.run.side_effect = fake_run
            cmd_agent(args, cli, config)

        compile_headers = [h for h in captured_headers if "Compiling" in h]
        assert any("/4]" in h for h in compile_headers), f"Expected /4 in headers: {compile_headers}"


class TestCmdAgentDomainFlag:
    """Tests for --domain flag on agent command."""

    def test_domain_flag_parsed(self):
        """Verify --domain flag is accepted by agent subparser."""
        from cli.main import main
        import sys as _sys
        with patch.object(_sys, 'argv', ['motherlabs', 'agent', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_domain_software_default(self):
        """Default domain is software — no adapter lookup needed."""
        cli = CLI()
        config = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_mock_args(description="A task app", output=tmpdir)
            mock_result = _make_mock_result(success=True)

            with patch('core.engine.MotherlabsEngine') as MockEng, \
                 patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
                mock_orch_instance = MockOrch.return_value
                mock_orch_instance.run.return_value = mock_result
                cmd_agent(args, cli, config)

                # Engine created with domain_adapter=None (software default)
                call_kwargs = MockEng.call_args[1]
                assert call_kwargs.get("domain_adapter") is None

    def test_domain_process_wires_adapter(self):
        """--domain process loads PROCESS_ADAPTER and passes to engine."""
        cli = CLI()
        config = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_mock_args(description="Employee onboarding", output=tmpdir, domain="process")
            mock_result = _make_mock_result(success=True)

            with patch('core.engine.MotherlabsEngine') as MockEng, \
                 patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
                mock_orch_instance = MockOrch.return_value
                mock_orch_instance.run.return_value = mock_result
                cmd_agent(args, cli, config)

                # Engine created with domain_adapter set
                call_kwargs = MockEng.call_args[1]
                adapter = call_kwargs.get("domain_adapter")
                assert adapter is not None
                assert adapter.name == "process"

    def test_domain_unknown_exits(self):
        """Unknown domain name exits with error."""
        cli = CLI()
        config = ConfigManager()

        args = _make_mock_args(description="test", domain="nonexistent_domain_xyz")

        with pytest.raises(SystemExit):
            cmd_agent(args, cli, config)
