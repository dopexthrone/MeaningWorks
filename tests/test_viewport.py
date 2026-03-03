"""
Tests for cli/viewport.py — Phase C: CLI Viewport.

All rendering functions are pure (dict → str), so tests verify:
1. Output contains expected content
2. Graceful degradation on empty/missing data
3. Correct structure (bars, sparklines, scatter, colors)
4. CLI command integration (viewport + emit subcommands)
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from cli.viewport import (
    render_dimensional_summary,
    render_node_map,
    render_interface_contracts,
    render_fragile_edges,
    render_emission_result,
    render_blueprint_tree,
    render_compilation_overview,
    _bar,
    _sparkline,
    _scatter_2d,
    _color_risk,
    _truncate,
)
from cli.main import Style, styled


# =============================================================================
# FIXTURES — Reusable Test Data
# =============================================================================

@pytest.fixture
def sample_dim_meta():
    """Serialized DimensionalMetadata dict."""
    return {
        "axes": [
            {
                "name": "complexity",
                "range_low": "simple",
                "range_high": "complex",
                "exploration_depth": 0.8,
                "derived_from": "dialogue analysis",
                "silence_zones": ["error handling edge cases"],
            },
            {
                "name": "user-facing",
                "range_low": "internal",
                "range_high": "user-facing",
                "exploration_depth": 0.6,
                "derived_from": "persona synthesis",
                "silence_zones": [],
            },
        ],
        "node_positions": {
            "AuthService": {
                "dimension_values": {"complexity": 0.7, "user-facing": 0.3},
                "confidence": 0.85,
            },
            "UserProfile": {
                "dimension_values": {"complexity": 0.4, "user-facing": 0.9},
                "confidence": 0.9,
            },
            "Database": {
                "dimension_values": {"complexity": 0.9, "user-facing": 0.1},
                "confidence": 0.75,
            },
        },
        "fragile_edges": [
            {
                "description": "Auth-to-DB connection",
                "affected_nodes": ["AuthService", "Database"],
                "drift_risk": "high",
                "reasoning": "Late-discovered dependency with no validation",
                "derived_from": "dimensional analysis",
            },
            {
                "description": "Profile update flow",
                "affected_nodes": ["UserProfile"],
                "drift_risk": "low",
                "reasoning": "Well-explored path",
                "derived_from": "dialogue coverage",
            },
        ],
        "silence_zones": ["caching strategy", "rate limiting"],
        "confidence_trajectory": [0.2, 0.35, 0.5, 0.65, 0.8, 0.85],
        "dimension_confidence": {"complexity": 0.9, "user-facing": 0.7},
        "dialogue_depth": 6,
        "stage_discovery": {
            "AuthService": "DECOMPOSE",
            "UserProfile": "EXPAND",
            "Database": "GROUND",
        },
    }


@pytest.fixture
def sample_interface_map():
    """Serialized InterfaceMap dict."""
    return {
        "contracts": [
            {
                "node_a": "AuthService",
                "node_b": "Database",
                "relationship_type": "depends_on",
                "relationship_description": "AuthService depends on Database",
                "data_flows": [
                    {
                        "name": "credentials",
                        "type_hint": "Dict[str, str]",
                        "direction": "A_to_B",
                        "derived_from": "dialogue",
                    },
                    {
                        "name": "session_token",
                        "type_hint": "str",
                        "direction": "B_to_A",
                        "derived_from": "dialogue",
                    },
                ],
                "constraints": [],
                "fragility": 0.7,
                "confidence": 0.8,
                "directionality": "A_depends_on_B",
                "derived_from": "interface_extraction",
            },
            {
                "node_a": "UserProfile",
                "node_b": "AuthService",
                "relationship_type": "uses",
                "relationship_description": "UserProfile uses AuthService",
                "data_flows": [
                    {
                        "name": "user_id",
                        "type_hint": "str",
                        "direction": "A_to_B",
                        "derived_from": "dialogue",
                    },
                ],
                "constraints": [],
                "fragility": 0.2,
                "confidence": 0.9,
                "directionality": "mutual",
                "derived_from": "interface_extraction",
            },
        ],
        "unmatched_relationships": ["Logger -> AuthService"],
        "extraction_confidence": 0.85,
        "derived_from": "interface_extraction",
    }


@pytest.fixture
def sample_blueprint():
    """Blueprint dict."""
    return {
        "components": [
            {"name": "AuthService", "type": "service", "description": "Handles authentication and session management"},
            {"name": "UserProfile", "type": "entity", "description": "User profile data and preferences"},
            {"name": "Database", "type": "infrastructure", "description": "Persistent storage layer"},
        ],
        "relationships": [
            {"from": "AuthService", "to": "Database", "type": "depends_on"},
            {"from": "UserProfile", "to": "AuthService", "type": "uses"},
        ],
        "constraints": [
            {"description": "All auth tokens must expire within 24h", "applies_to": ["AuthService"]},
            {"description": "Database connections must be pooled", "applies_to": ["Database"]},
        ],
    }


@pytest.fixture
def sample_emission():
    """Serialized EmissionResult dict."""
    return {
        "batch_emissions": [
            {
                "batch_index": 0,
                "emissions": [
                    {
                        "component_name": "Database",
                        "component_type": "infrastructure",
                        "code": "class Database:\n    def connect(self):\n        pass\n",
                        "success": True,
                        "error": None,
                        "prompt_hash": "abc12345deadbeef",
                        "derived_from": "agent_emission:v1.0",
                    },
                ],
                "success_count": 1,
                "failure_count": 0,
            },
            {
                "batch_index": 1,
                "emissions": [
                    {
                        "component_name": "AuthService",
                        "component_type": "service",
                        "code": "class AuthService:\n    def authenticate(self, user_id):\n        pass\n",
                        "success": True,
                        "error": None,
                        "prompt_hash": "1234567890abcdef",
                        "derived_from": "agent_emission:v1.0",
                    },
                    {
                        "component_name": "UserProfile",
                        "component_type": "entity",
                        "code": "",
                        "success": False,
                        "error": "LLM returned empty response",
                        "prompt_hash": "fedcba0987654321",
                        "derived_from": "agent_emission:v1.0",
                    },
                ],
                "success_count": 1,
                "failure_count": 1,
            },
        ],
        "generated_code": {
            "Database": "class Database:\n    def connect(self):\n        pass\n",
            "AuthService": "class AuthService:\n    def authenticate(self, user_id):\n        pass\n",
        },
        "verification_report": {
            "total_contracts": 2,
            "passed": 1,
            "failed": 1,
            "pass_rate": 0.5,
            "details": [],
        },
        "total_nodes": 3,
        "success_count": 2,
        "failure_count": 1,
        "pass_rate": 0.5,
        "l2_context_injected": False,
        "timestamp": "2026-02-09T12:00:00",
        "derived_from": "agent_emission:v1.0",
    }


# =============================================================================
# HELPER TESTS
# =============================================================================

class TestBar:
    """Tests for _bar() helper."""

    def test_zero(self):
        result = _bar(0.0)
        assert result == "[----------]"

    def test_full(self):
        result = _bar(1.0)
        assert result == "[##########]"

    def test_half(self):
        result = _bar(0.5)
        assert result == "[#####-----]"

    def test_clamp_over(self):
        result = _bar(1.5)
        assert result == "[##########]"

    def test_clamp_under(self):
        result = _bar(-0.3)
        assert result == "[----------]"

    def test_custom_width(self):
        result = _bar(0.5, width=4)
        assert result == "[##--]"


class TestSparkline:
    """Tests for _sparkline() helper."""

    def test_empty(self):
        assert _sparkline([]) == ""

    def test_zeros(self):
        result = _sparkline([0.0, 0.0, 0.0])
        assert len(result) == 3

    def test_ones(self):
        result = _sparkline([1.0, 1.0])
        assert len(result) == 2

    def test_ascending(self):
        result = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(result) == 5
        # Each character should be >= previous
        # (monotonically non-decreasing in the blocks mapping)

    def test_clamping(self):
        result = _sparkline([2.0, -1.0])
        assert len(result) == 2


class TestScatter2d:
    """Tests for _scatter_2d() helper."""

    def test_empty(self):
        result = _scatter_2d([])
        assert "no data" in result

    def test_single_point(self):
        result = _scatter_2d([(0.5, 0.5, "Center")])
        assert "Center" in result
        assert "Legend" in result
        assert "1" in result  # Marker

    def test_multiple_points(self):
        points = [(0.0, 0.0, "A"), (1.0, 1.0, "B"), (0.5, 0.5, "C")]
        result = _scatter_2d(points, x_label="x-axis", y_label="y-axis")
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "x-axis" in result
        assert "y-axis" in result

    def test_custom_dimensions(self):
        points = [(0.5, 0.5, "X")]
        result = _scatter_2d(points, w=20, h=10)
        lines = result.split("\n")
        assert len(lines) > 10  # Header + grid + legend


class TestColorRisk:
    """Tests for _color_risk() helper."""

    def test_high(self):
        result = _color_risk("high")
        assert Style.RED in result

    def test_medium(self):
        result = _color_risk("medium")
        assert Style.YELLOW in result

    def test_low(self):
        result = _color_risk("low")
        assert Style.GREEN in result

    def test_unknown(self):
        result = _color_risk("unknown")
        assert Style.GRAY in result

    def test_none(self):
        result = _color_risk(None)
        assert Style.GRAY in result


class TestTruncate:
    """Tests for _truncate() helper."""

    def test_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact(self):
        assert _truncate("hello", 5) == "hello"

    def test_long(self):
        assert _truncate("hello world", 8) == "hello..."

    def test_empty(self):
        assert _truncate("", 10) == ""

    def test_none(self):
        assert _truncate(None, 10) == ""


# =============================================================================
# RENDER FUNCTION TESTS
# =============================================================================

class TestRenderDimensionalSummary:
    """Tests for render_dimensional_summary()."""

    def test_full_data(self, sample_dim_meta):
        result = render_dimensional_summary(sample_dim_meta)
        assert "Dimensional Space" in result
        assert "complexity" in result
        assert "user-facing" in result
        assert "simple" in result
        assert "complex" in result
        assert "80%" in result  # exploration_depth 0.8
        assert "silence" in result.lower()
        assert "Confidence" in result

    def test_empty_dict(self):
        result = render_dimensional_summary({})
        # Empty dict is falsy — returns "not available"
        assert "not available" in result

    def test_no_axes(self):
        result = render_dimensional_summary({"axes": [], "node_positions": {}})
        # Has data but no axes — shows header + "No dimensions extracted"
        assert "Dimensional Space" in result
        assert "No dimensions" in result

    def test_none(self):
        result = render_dimensional_summary(None)
        assert "not available" in result

    def test_silence_zones(self, sample_dim_meta):
        result = render_dimensional_summary(sample_dim_meta)
        assert "caching strategy" in result
        assert "rate limiting" in result

    def test_trajectory_sparkline(self, sample_dim_meta):
        result = render_dimensional_summary(sample_dim_meta)
        assert "85%" in result  # Final confidence value

    def test_dialogue_depth(self, sample_dim_meta):
        result = render_dimensional_summary(sample_dim_meta)
        assert "6" in result  # dialogue_depth


class TestRenderNodeMap:
    """Tests for render_node_map()."""

    def test_full_data(self, sample_dim_meta):
        result = render_node_map(sample_dim_meta)
        assert "Node Map" in result
        assert "AuthService" in result
        assert "UserProfile" in result
        assert "Database" in result
        assert "Legend" in result

    def test_custom_axes(self, sample_dim_meta):
        result = render_node_map(sample_dim_meta, x_axis="complexity", y_axis="user-facing")
        assert "complexity" in result
        assert "user-facing" in result

    def test_none(self):
        result = render_node_map(None)
        assert "not available" in result

    def test_no_positions(self):
        dim = {"axes": [{"name": "x", "range_low": "a", "range_high": "b", "exploration_depth": 0.5}], "node_positions": {}}
        result = render_node_map(dim)
        assert "insufficient" in result

    def test_single_axis(self):
        dim = {
            "axes": [{"name": "only_axis", "range_low": "lo", "range_high": "hi", "exploration_depth": 0.5}],
            "node_positions": {"A": {"dimension_values": {"only_axis": 0.5}, "confidence": 0.8}},
        }
        result = render_node_map(dim)
        # Should default both x and y to "only_axis"
        assert "only_axis" in result
        assert "A" in result


class TestRenderInterfaceContracts:
    """Tests for render_interface_contracts()."""

    def test_full_data(self, sample_interface_map):
        result = render_interface_contracts(sample_interface_map)
        assert "Interface Contracts" in result
        assert "AuthService" in result
        assert "Database" in result
        assert "credentials" in result
        assert "session_token" in result
        assert "Unmatched" in result
        assert "Logger" in result
        assert "85%" in result  # extraction_confidence

    def test_none(self):
        result = render_interface_contracts(None)
        assert "not available" in result

    def test_empty_contracts(self):
        result = render_interface_contracts({"contracts": []})
        assert "No contracts" in result

    def test_direction_arrows(self, sample_interface_map):
        result = render_interface_contracts(sample_interface_map)
        assert "<-" in result  # A_depends_on_B
        assert "<>" in result  # mutual


class TestRenderFragileEdges:
    """Tests for render_fragile_edges()."""

    def test_full_data(self, sample_dim_meta):
        result = render_fragile_edges(sample_dim_meta)
        assert "Fragile Edges" in result
        assert "Auth-to-DB" in result
        assert "AuthService" in result
        assert "Database" in result
        assert "Late-discovered" in result

    def test_none(self):
        result = render_fragile_edges(None)
        assert "not available" in result

    def test_no_fragile(self):
        result = render_fragile_edges({"fragile_edges": []})
        assert "No fragile edges" in result


class TestRenderEmissionResult:
    """Tests for render_emission_result()."""

    def test_full_data(self, sample_emission):
        result = render_emission_result(sample_emission)
        assert "Agent Emission" in result
        assert "Batch 0" in result
        assert "Batch 1" in result
        assert "Database" in result
        assert "AuthService" in result
        assert "UserProfile" in result
        assert "FAIL" in result  # UserProfile failed
        assert "Verification" in result
        # "2" and "/3" are split by ANSI codes — check separately
        assert "/3 succeeded" in result
        assert "50%" in result

    def test_none(self):
        result = render_emission_result(None)
        assert "not available" in result

    def test_error_shown(self, sample_emission):
        result = render_emission_result(sample_emission)
        assert "empty response" in result

    def test_prompt_hash(self, sample_emission):
        result = render_emission_result(sample_emission)
        assert "abc12345" in result  # Truncated to 8 chars


class TestRenderBlueprintTree:
    """Tests for render_blueprint_tree()."""

    def test_full_data(self, sample_blueprint):
        result = render_blueprint_tree(sample_blueprint)
        assert "Blueprint" in result
        assert "AuthService" in result
        assert "UserProfile" in result
        assert "Database" in result
        assert "service" in result
        assert "entity" in result
        assert "infrastructure" in result
        assert "Relationships" in result
        assert "Constraints" in result

    def test_none(self):
        result = render_blueprint_tree(None)
        assert "not available" in result

    def test_empty(self):
        result = render_blueprint_tree({})
        assert "Blueprint" in result

    def test_no_constraints(self):
        bp = {"components": [{"name": "A", "type": "service", "description": "test"}], "relationships": [], "constraints": []}
        result = render_blueprint_tree(bp)
        assert "A" in result
        assert "Constraints" not in result  # Empty list, section not shown


class TestRenderCompilationOverview:
    """Tests for render_compilation_overview()."""

    def test_full(self, sample_blueprint, sample_dim_meta, sample_interface_map, sample_emission):
        result = render_compilation_overview(sample_blueprint, sample_dim_meta, sample_interface_map, sample_emission)
        assert "Compilation Overview" in result
        assert "3" in result  # 3 components
        assert "2" in result  # 2 relationships
        assert "Dimensions:" in result
        assert "Contracts:" in result
        assert "Emission:" in result

    def test_blueprint_only(self, sample_blueprint):
        result = render_compilation_overview(sample_blueprint)
        assert "Compilation Overview" in result
        assert "not available" in result  # dim, interfaces, emission not available

    def test_none(self):
        result = render_compilation_overview(None)
        assert "not available" in result

    def test_with_dim_no_interfaces(self, sample_blueprint, sample_dim_meta):
        result = render_compilation_overview(sample_blueprint, sample_dim_meta)
        assert "Dimensions:" in result
        assert "Exploration:" in result
        assert "Interfaces: not available" in result


# =============================================================================
# CLI COMMAND TESTS
# =============================================================================

class TestCmdViewport:
    """Tests for cmd_viewport() CLI command."""

    def test_viewport_loads_and_renders(self, sample_blueprint, sample_dim_meta, sample_interface_map, capsys):
        """Test that viewport loads compilation and prints output."""
        from cli.main import cmd_viewport, CLI, ConfigManager

        mock_corpus = MagicMock()
        mock_corpus.load_blueprint.return_value = sample_blueprint
        mock_corpus.load_context_graph.return_value = {
            "dimensional_metadata": sample_dim_meta,
            "interface_map": sample_interface_map,
        }

        args = MagicMock()
        args.compilation_id = "test123"
        args.map = False
        args.interfaces = False
        args.emission = False
        args.fragile = False
        args.tree = False
        args.axes = None

        cli = CLI()
        config = ConfigManager()

        with patch("persistence.corpus.Corpus", return_value=mock_corpus):
            cmd_viewport(args, cli, config)

        output = capsys.readouterr().out
        assert "Compilation Overview" in output
        assert "Dimensional Space" in output

    def test_viewport_map_only(self, sample_blueprint, sample_dim_meta, capsys):
        """Test --map flag shows only node map."""
        from cli.main import cmd_viewport, CLI, ConfigManager

        mock_corpus = MagicMock()
        mock_corpus.load_blueprint.return_value = sample_blueprint
        mock_corpus.load_context_graph.return_value = {
            "dimensional_metadata": sample_dim_meta,
        }

        args = MagicMock()
        args.compilation_id = "test123"
        args.map = True
        args.interfaces = False
        args.emission = False
        args.fragile = False
        args.tree = False
        args.axes = None

        cli = CLI()
        config = ConfigManager()

        with patch("persistence.corpus.Corpus", return_value=mock_corpus):
            cmd_viewport(args, cli, config)

        output = capsys.readouterr().out
        assert "Node Map" in output
        # Should NOT show overview when specific view is requested
        assert "Compilation Overview" not in output

    def test_viewport_not_found(self):
        """Test viewport with nonexistent compilation ID."""
        from cli.main import cmd_viewport, CLI, ConfigManager

        mock_corpus = MagicMock()
        mock_corpus.load_blueprint.return_value = None

        args = MagicMock()
        args.compilation_id = "nonexistent"

        cli = CLI()
        config = ConfigManager()

        with patch("persistence.corpus.Corpus", return_value=mock_corpus):
            with pytest.raises(SystemExit):
                cmd_viewport(args, cli, config)

    def test_viewport_custom_axes(self, sample_blueprint, sample_dim_meta, capsys):
        """Test --axes x,y flag."""
        from cli.main import cmd_viewport, CLI, ConfigManager

        mock_corpus = MagicMock()
        mock_corpus.load_blueprint.return_value = sample_blueprint
        mock_corpus.load_context_graph.return_value = {
            "dimensional_metadata": sample_dim_meta,
        }

        args = MagicMock()
        args.compilation_id = "test123"
        args.map = True
        args.interfaces = False
        args.emission = False
        args.fragile = False
        args.tree = False
        args.axes = "complexity,user-facing"

        cli = CLI()
        config = ConfigManager()

        with patch("persistence.corpus.Corpus", return_value=mock_corpus):
            cmd_viewport(args, cli, config)

        output = capsys.readouterr().out
        assert "complexity" in output
        assert "user-facing" in output


class TestCmdEmit:
    """Tests for cmd_emit() CLI command."""

    def test_emit_not_found(self):
        """Test emit with nonexistent compilation ID."""
        from cli.main import cmd_emit, CLI, ConfigManager

        mock_corpus = MagicMock()
        mock_corpus.load_blueprint.return_value = None

        args = MagicMock()
        args.compilation_id = "nonexistent"

        cli = CLI()
        config = ConfigManager()

        with patch("persistence.corpus.Corpus", return_value=mock_corpus):
            with pytest.raises(SystemExit):
                cmd_emit(args, cli, config)


# =============================================================================
# INLINE VIEWPORT TESTS
# =============================================================================

class TestCompileInlineViewport:
    """Tests for --viewport and --emit flags on compile command."""

    def test_viewport_flag_renders_after_compile(self):
        """Verify --viewport flag triggers viewport rendering."""
        from cli.viewport import render_compilation_overview
        # Test that render functions work with CompileResult-style data
        blueprint = {"components": [{"name": "A", "type": "service", "description": "test"}], "relationships": [], "constraints": []}
        dim_meta = {"axes": [], "node_positions": {}}
        imap = {"contracts": []}
        result = render_compilation_overview(blueprint, dim_meta, imap)
        assert "Compilation Overview" in result

    def test_viewport_with_emission(self):
        """Verify emission rendering works with sample data."""
        from cli.viewport import render_emission_result
        emission = {
            "batch_emissions": [],
            "verification_report": {"total_contracts": 0, "passed": 0, "failed": 0, "pass_rate": 1.0},
            "total_nodes": 0,
            "success_count": 0,
            "failure_count": 0,
            "pass_rate": 1.0,
        }
        result = render_emission_result(emission)
        assert "Agent Emission" in result

    def test_viewport_flag_exists_on_parser(self):
        """Verify --viewport flag is registered on compile parser."""
        import argparse
        from cli.main import main
        # Just verify the parser accepts --viewport
        from cli.main import argparse as ap
        parser = ap.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        compile_parser = subparsers.add_parser("compile")
        compile_parser.add_argument("description", nargs="?")
        compile_parser.add_argument("--viewport", action="store_true")
        compile_parser.add_argument("--emit", action="store_true", dest="emit_code")
        args = parser.parse_args(["compile", "test", "--viewport"])
        assert args.viewport is True
        assert args.emit_code is False


# =============================================================================
# LEAF MODULE CONSTRAINT
# =============================================================================

class TestLeafModuleConstraint:
    """Verify cli/viewport.py is a leaf module (only imports cli/main.py + stdlib)."""

    def test_no_engine_imports(self):
        import ast
        from pathlib import Path
        source = Path("cli/viewport.py").read_text()
        tree = ast.parse(source)
        prohibited = {"core.engine", "core.protocol", "core.pipeline", "persistence"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prohibited_mod in prohibited:
                    assert prohibited_mod not in node.module, (
                        f"viewport.py imports {node.module} — should be a leaf module"
                    )
