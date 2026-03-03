"""Tests for mother/self_build_planner.py — self-build prompt translator."""

import os
import tempfile
from pathlib import Path

import pytest

from mother.self_build_planner import (
    SelfBuildSpec,
    _BOUNDARY_RULES,
    _CATEGORY_TO_ACTION,
    _CONCERN_DESCRIPTIONS,
    _CONCERN_TO_PATTERN,
    _LAYER_DESCRIPTIONS,
    _LAYER_TO_MODULE,
    _PROTECTED_FILES,
    _TEST_COMMAND,
    _describe_postcodes,
    assemble_self_build_prompt,
    blueprint_to_build_context,
    goal_to_build_intent,
    infer_target_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal repo-like directory structure."""
    for mod in ("core", "kernel", "mother", "agents"):
        (tmp_path / mod).mkdir()
    # Create some .py files
    (tmp_path / "core" / "engine.py").write_text("# engine")
    (tmp_path / "core" / "pipeline.py").write_text("# pipeline")
    (tmp_path / "core" / "llm.py").write_text("# llm")
    (tmp_path / "kernel" / "cell.py").write_text("# cell")
    (tmp_path / "kernel" / "grid.py").write_text("# grid")
    (tmp_path / "kernel" / "store.py").write_text("# store")
    (tmp_path / "kernel" / "memory.py").write_text("# memory")
    (tmp_path / "kernel" / "observer.py").write_text("# observer")
    (tmp_path / "kernel" / "agents.py").write_text("# agents")
    (tmp_path / "kernel" / "entity_model.py").write_text("# entity model")
    (tmp_path / "core" / "schema.py").write_text("# schema")
    (tmp_path / "mother" / "bridge.py").write_text("# bridge")
    (tmp_path / "mother" / "context.py").write_text("# context")
    (tmp_path / "mother" / "persona.py").write_text("# persona")
    (tmp_path / "mother" / "goal_generator.py").write_text("# goals")
    (tmp_path / "agents" / "base.py").write_text("# base")
    (tmp_path / "agents" / "spec_agents.py").write_text("# spec agents")
    return tmp_path


@pytest.fixture
def sample_goal():
    return {
        "description": "3 cells below 30% confidence need immediate attention.",
        "category": "confidence",
        "target_postcodes": ("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"),
        "source": "grid:critical_confidence",
    }


@pytest.fixture
def sample_blueprint():
    return {
        "components": [
            {
                "name": "SemanticParser",
                "description": "Parses input text into semantic structures",
                "constraints": [
                    {"name": "idempotent", "description": "Same input always produces same output"},
                ],
                "methods": [
                    {"name": "parse"},
                    {"name": "validate"},
                ],
            },
            {
                "name": "GridCompiler",
                "description": "Compiles parsed semantics into grid cells",
                "constraints": ["must preserve provenance"],
                "properties": [
                    {"name": "compile"},
                ],
            },
        ],
        "relationships": [
            {"source": "SemanticParser", "target": "GridCompiler", "type": "feeds"},
        ],
    }


# ---------------------------------------------------------------------------
# SelfBuildSpec type
# ---------------------------------------------------------------------------

class TestSelfBuildSpec:
    def test_frozen(self):
        spec = SelfBuildSpec(
            goal_description="test",
            build_intent="test intent",
            target_postcodes=(),
            target_files=(),
            protected_files=_PROTECTED_FILES,
            boundary_rules=_BOUNDARY_RULES,
            test_command=_TEST_COMMAND,
            blueprint_context="",
            prompt="test prompt",
        )
        with pytest.raises(AttributeError):
            spec.prompt = "changed"

    def test_fields_present(self):
        spec = SelfBuildSpec(
            goal_description="g",
            build_intent="b",
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            target_files=("kernel/cell.py",),
            protected_files=_PROTECTED_FILES,
            boundary_rules=_BOUNDARY_RULES,
            test_command=_TEST_COMMAND,
            blueprint_context="ctx",
            prompt="full prompt",
        )
        assert spec.goal_description == "g"
        assert spec.build_intent == "b"
        assert len(spec.target_postcodes) == 1
        assert len(spec.target_files) == 1
        assert len(spec.protected_files) == 3
        assert len(spec.boundary_rules) == 6
        assert spec.test_command == _TEST_COMMAND


# ---------------------------------------------------------------------------
# Mapping tables coverage
# ---------------------------------------------------------------------------

class TestMappingTables:
    def test_layer_to_module_covers_all_layers(self):
        from kernel.cell import LAYERS
        for layer in LAYERS:
            assert layer in _LAYER_TO_MODULE, f"Missing layer mapping: {layer}"

    def test_concern_to_pattern_has_entries(self):
        assert len(_CONCERN_TO_PATTERN) >= 15

    def test_category_to_action_covers_core_categories(self):
        for cat in ("confidence", "coverage", "quality", "resilience"):
            assert cat in _CATEGORY_TO_ACTION

    def test_layer_descriptions_match_layer_to_module(self):
        for layer in _LAYER_TO_MODULE:
            assert layer in _LAYER_DESCRIPTIONS, f"Missing description for {layer}"

    def test_protected_files_are_correct(self):
        assert "mother/context.py" in _PROTECTED_FILES
        assert "mother/persona.py" in _PROTECTED_FILES
        assert "mother/senses.py" in _PROTECTED_FILES


# ---------------------------------------------------------------------------
# goal_to_build_intent
# ---------------------------------------------------------------------------

class TestGoalToBuildIntent:
    def test_confidence_category(self, sample_goal):
        intent = goal_to_build_intent(sample_goal)
        assert "strengthen" in intent.lower()

    def test_coverage_category(self):
        goal = {"description": "Core layers not mapped", "category": "coverage"}
        intent = goal_to_build_intent(goal)
        assert "implement" in intent.lower() or "missing" in intent.lower()

    def test_quality_category(self):
        goal = {"description": "Output quality low", "category": "quality"}
        intent = goal_to_build_intent(goal)
        assert "improve" in intent.lower() or "quality" in intent.lower()

    def test_resilience_category(self):
        goal = {"description": "Reliability issues", "category": "resilience"}
        intent = goal_to_build_intent(goal)
        assert "fix" in intent.lower() or "reliab" in intent.lower()

    def test_unknown_category_falls_back(self):
        goal = {"description": "Something unusual", "category": "mystery"}
        intent = goal_to_build_intent(goal)
        assert len(intent) > 0

    def test_with_postcodes_expands_territory(self, sample_goal):
        intent = goal_to_build_intent(sample_goal)
        # Should mention layer or concern descriptions
        assert "Semantic" in intent or "Intent" in intent or "layers" in intent

    def test_with_grid_cells_includes_cell_details(self, sample_goal):
        cells = [
            ("SEM.ENT.DOM.WHAT.SFT", 0.25, "P", "entity-model"),
            ("COG.BHV.APP.HOW.SFT", 0.10, "P", "behavior-flow"),
        ]
        intent = goal_to_build_intent(sample_goal, grid_cells=cells)
        assert "entity-model" in intent
        assert "behavior-flow" in intent

    def test_with_blueprint_includes_component_names(self, sample_goal, sample_blueprint):
        intent = goal_to_build_intent(sample_goal, blueprint=sample_blueprint)
        assert "SemanticParser" in intent

    def test_empty_goal_returns_something(self):
        intent = goal_to_build_intent({})
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_no_postcodes_uses_description(self):
        goal = {"description": "Fix the pipeline", "category": "resilience"}
        intent = goal_to_build_intent(goal)
        assert "pipeline" in intent.lower() or "fix" in intent.lower()


# ---------------------------------------------------------------------------
# blueprint_to_build_context
# ---------------------------------------------------------------------------

class TestBlueprintToBuildContext:
    def test_extracts_component_names(self, sample_blueprint):
        ctx = blueprint_to_build_context(sample_blueprint)
        assert "SemanticParser" in ctx
        assert "GridCompiler" in ctx

    def test_extracts_constraints(self, sample_blueprint):
        ctx = blueprint_to_build_context(sample_blueprint)
        assert "idempotent" in ctx or "provenance" in ctx

    def test_extracts_methods(self, sample_blueprint):
        ctx = blueprint_to_build_context(sample_blueprint)
        assert "parse" in ctx

    def test_extracts_relationships(self, sample_blueprint):
        ctx = blueprint_to_build_context(sample_blueprint)
        assert "SemanticParser" in ctx and "GridCompiler" in ctx

    def test_empty_blueprint(self):
        ctx = blueprint_to_build_context({})
        assert ctx == ""

    def test_max_components_limit(self):
        bp = {
            "components": [
                {"name": f"Comp{i}", "description": f"desc {i}"}
                for i in range(20)
            ]
        }
        ctx = blueprint_to_build_context(bp, max_components=3)
        assert "Comp0" in ctx
        assert "Comp2" in ctx
        assert "Comp3" not in ctx

    def test_string_constraints(self):
        bp = {"components": [{"name": "A", "constraints": ["no nulls", "immutable"]}]}
        ctx = blueprint_to_build_context(bp)
        assert "no nulls" in ctx


# ---------------------------------------------------------------------------
# infer_target_files
# ---------------------------------------------------------------------------

class TestInferTargetFiles:
    def test_layer_maps_to_module(self, tmp_repo):
        files = infer_target_files(("SEM.ENT.DOM.WHAT.SFT",), tmp_repo)
        # SEM → kernel, ENT → entity patterns
        # kernel/ has cell.py, grid.py, store.py, memory.py, etc.
        assert any("kernel/" in f for f in files)

    def test_concern_filters_files(self, tmp_repo):
        files = infer_target_files(("OBS.MEM.APP.WHAT.SFT",), tmp_repo)
        # MEM concern → "memory", "store", "cache", "recall"
        # OBS layer → kernel
        matched_patterns = [f for f in files if "memory" in f or "store" in f]
        assert len(matched_patterns) > 0

    def test_empty_postcodes(self, tmp_repo):
        files = infer_target_files((), tmp_repo)
        assert files == ()

    def test_unknown_layer_returns_empty(self, tmp_repo):
        files = infer_target_files(("ZZZ.ENT.DOM.WHAT.SFT",), tmp_repo)
        # ZZZ not in _LAYER_TO_MODULE, but ENT has patterns — no module dirs to search
        assert isinstance(files, tuple)

    def test_multiple_postcodes_union(self, tmp_repo):
        files = infer_target_files(
            ("SEM.ENT.DOM.WHAT.SFT", "AGN.AGT.APP.HOW.SFT"),
            tmp_repo,
        )
        # SEM→kernel, AGN→agents
        has_kernel = any("kernel/" in f for f in files)
        has_agents = any("agents/" in f for f in files)
        assert has_kernel or has_agents

    def test_capped_at_20(self, tmp_repo):
        # Create many files in core/
        for i in range(25):
            (tmp_repo / "core" / f"entity_{i}.py").write_text(f"# {i}")
        files = infer_target_files(("INT.ENT.DOM.WHAT.SFT",), tmp_repo)
        assert len(files) <= 20

    def test_skips_private_files(self, tmp_repo):
        (tmp_repo / "kernel" / "_internal.py").write_text("# private")
        files = infer_target_files(("SEM.ENT.DOM.WHAT.SFT",), tmp_repo)
        assert not any("_internal.py" in f for f in files)


# ---------------------------------------------------------------------------
# assemble_self_build_prompt
# ---------------------------------------------------------------------------

class TestAssembleSelfBuildPrompt:
    def test_returns_self_build_spec(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Strengthen entity modeling in kernel",
            repo_dir=tmp_repo,
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
        )
        assert isinstance(spec, SelfBuildSpec)

    def test_prompt_contains_intent(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Fix reliability in core pipeline",
            repo_dir=tmp_repo,
        )
        assert "Fix reliability in core pipeline" in spec.prompt

    def test_prompt_contains_architectural_rules(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
        )
        assert "Avoid modifying these core files" in spec.prompt
        assert "bridge.py" in spec.prompt

    def test_prompt_contains_test_command(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
        )
        assert _TEST_COMMAND in spec.prompt

    def test_prompt_contains_target_files(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
            target_files=("kernel/cell.py", "kernel/grid.py"),
        )
        assert "kernel/cell.py" in spec.prompt
        assert "kernel/grid.py" in spec.prompt

    def test_prompt_contains_blueprint_context(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
            blueprint_context="## SemanticParser\n  Parses input",
        )
        assert "BLUEPRINT CONTEXT" in spec.prompt
        assert "SemanticParser" in spec.prompt

    def test_prompt_contains_learning_context(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
            learning_context="Previous build failed due to import cycle",
        )
        assert "LEARNING CONTEXT" in spec.prompt
        assert "import cycle" in spec.prompt

    def test_infers_files_from_postcodes(self, tmp_repo):
        spec = assemble_self_build_prompt(
            build_intent="Build something",
            repo_dir=tmp_repo,
            target_postcodes=("SEM.MEM.APP.WHAT.SFT",),
        )
        # SEM→kernel, MEM→memory/store/cache/recall
        assert any("kernel/" in f for f in spec.target_files)


# ---------------------------------------------------------------------------
# _describe_postcodes
# ---------------------------------------------------------------------------

class TestDescribePostcodes:
    def test_single_postcode(self):
        desc = _describe_postcodes(("SEM.ENT.DOM.WHAT.SFT",))
        assert "Semantic" in desc
        assert "Entity" in desc

    def test_multiple_postcodes_union(self):
        desc = _describe_postcodes(("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"))
        assert "Cognitive" in desc or "COG" in desc
        assert "Semantic" in desc or "SEM" in desc

    def test_empty_postcodes(self):
        desc = _describe_postcodes(())
        assert desc == ""

    def test_unknown_axes_pass_through(self):
        desc = _describe_postcodes(("ZZZ.XXX.DOM.WHAT.SFT",))
        # Unknown codes pass through as-is
        assert "ZZZ" in desc or "XXX" in desc or desc == ""
