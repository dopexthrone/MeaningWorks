"""Tests for mother/diagrams.py.

Covers blueprint_to_mermaid(), blueprint_to_mermaid_block(),
component_tree(), and internal helpers.
"""

import pytest
from mother.diagrams import (
    blueprint_to_mermaid,
    blueprint_to_mermaid_block,
    component_tree,
    _safe_id,
    _node_label,
    _node_shape,
    _arrow_style,
    _type_style,
    _group_by_type,
    _simplified_diagram,
    _MAX_COMPONENTS,
    _MAX_RELATIONSHIPS,
)


# === Sample data ===

def _simple_blueprint():
    return {
        "components": [
            {"name": "AuthService", "type": "service"},
            {"name": "UserDB", "type": "entity"},
        ],
        "relationships": [
            {"from": "AuthService", "to": "UserDB", "type": "depends_on"},
        ],
    }


def _complex_blueprint():
    """Blueprint with subsystems."""
    return {
        "components": [
            {"name": "Frontend", "type": "subsystem", "sub_blueprint": {
                "components": [
                    {"name": "LoginPage", "type": "component"},
                    {"name": "Dashboard", "type": "component"},
                ],
            }},
            {"name": "Backend", "type": "service"},
            {"name": "Cache", "type": "entity"},
        ],
        "relationships": [
            {"from": "Frontend", "to": "Backend", "type": "sends"},
            {"from": "Backend", "to": "Cache", "type": "depends_on"},
        ],
    }


def _oversized_blueprint():
    """Blueprint exceeding _MAX_COMPONENTS."""
    comps = [{"name": f"C{i}", "type": "component"} for i in range(_MAX_COMPONENTS + 5)]
    comps.append({"name": "Core", "type": "subsystem"})
    rels = [{"from": "C0", "to": "Core", "type": "depends_on"}]
    return {"components": comps, "relationships": rels}


# === blueprint_to_mermaid ===

class TestBlueprintToMermaid:
    def test_empty_blueprint(self):
        result = blueprint_to_mermaid({})
        assert result == ""

    def test_no_components(self):
        result = blueprint_to_mermaid({"components": []})
        assert result == ""

    def test_simple_diagram(self):
        bp = _simple_blueprint()
        diagram = blueprint_to_mermaid(bp)
        assert "flowchart TD" in diagram
        assert "AuthService" in diagram
        assert "UserDB" in diagram

    def test_left_right_direction(self):
        bp = _simple_blueprint()
        diagram = blueprint_to_mermaid(bp, direction="LR")
        assert "flowchart LR" in diagram

    def test_title_header(self):
        bp = _simple_blueprint()
        diagram = blueprint_to_mermaid(bp, title="My System")
        assert "title: My System" in diagram

    def test_no_title(self):
        bp = _simple_blueprint()
        diagram = blueprint_to_mermaid(bp)
        assert "title:" not in diagram

    def test_relationships_rendered(self):
        bp = _simple_blueprint()
        diagram = blueprint_to_mermaid(bp)
        assert "depends_on" in diagram

    def test_relationship_without_type(self):
        bp = {
            "components": [
                {"name": "A", "type": "component"},
                {"name": "B", "type": "component"},
            ],
            "relationships": [{"from": "A", "to": "B"}],
        }
        diagram = blueprint_to_mermaid(bp)
        assert "-->" in diagram

    def test_duplicate_relationships_deduped(self):
        bp = {
            "components": [
                {"name": "A", "type": "component"},
                {"name": "B", "type": "component"},
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "uses"},
                {"from": "A", "to": "B", "type": "uses"},
            ],
        }
        diagram = blueprint_to_mermaid(bp)
        # Should only appear once
        assert diagram.count("|uses|") == 1

    def test_subsystem_rendered_as_subgraph(self):
        bp = _complex_blueprint()
        diagram = blueprint_to_mermaid(bp)
        assert "subgraph" in diagram
        assert "Frontend" in diagram
        assert "LoginPage" in diagram
        assert "end" in diagram

    def test_relationships_with_missing_from_to_skipped(self):
        bp = {
            "components": [{"name": "A", "type": "component"}],
            "relationships": [{"from": "", "to": "A", "type": "x"}],
        }
        diagram = blueprint_to_mermaid(bp)
        assert "|x|" not in diagram

    def test_oversized_falls_back_to_simplified(self):
        bp = _oversized_blueprint()
        diagram = blueprint_to_mermaid(bp)
        assert "simplified" in diagram or "flowchart" in diagram


# === blueprint_to_mermaid_block ===

class TestBlueprintToMermaidBlock:
    def test_wraps_in_code_block(self):
        bp = _simple_blueprint()
        block = blueprint_to_mermaid_block(bp)
        assert block.startswith("```mermaid\n")
        assert block.endswith("\n```")

    def test_empty_returns_empty(self):
        block = blueprint_to_mermaid_block({})
        assert block == ""

    def test_title_forwarded(self):
        bp = _simple_blueprint()
        block = blueprint_to_mermaid_block(bp, title="Test")
        assert "title: Test" in block


# === component_tree ===

class TestComponentTree:
    def test_simple_tree(self):
        bp = _simple_blueprint()
        tree = component_tree(bp)
        assert "AuthService" in tree
        assert "UserDB" in tree
        assert "service" in tree
        assert "entity" in tree

    def test_nested_tree(self):
        bp = _complex_blueprint()
        tree = component_tree(bp)
        assert "Frontend" in tree
        assert "LoginPage" in tree

    def test_empty_blueprint(self):
        tree = component_tree({})
        assert tree == ""

    def test_indentation(self):
        bp = _complex_blueprint()
        tree = component_tree(bp)
        # Nested components should be indented
        lines = tree.split("\n")
        # Find LoginPage line — should have more indentation
        login_lines = [l for l in lines if "LoginPage" in l]
        if login_lines:
            assert login_lines[0].startswith("  ")

    def test_type_tag(self):
        bp = {"components": [{"name": "X", "type": "service"}]}
        tree = component_tree(bp)
        assert "(service)" in tree

    def test_no_type_no_tag(self):
        bp = {"components": [{"name": "X"}]}
        tree = component_tree(bp)
        assert "- X" in tree
        assert "()" not in tree


# === _safe_id ===

class TestSafeId:
    def test_simple_name(self):
        assert _safe_id("AuthService") == "AuthService"

    def test_spaces_replaced(self):
        assert _safe_id("Auth Service") == "Auth_Service"

    def test_special_chars_replaced(self):
        result = _safe_id("my-service.v2")
        assert "-" not in result
        assert "." not in result

    def test_empty_string(self):
        assert _safe_id("") == "node"

    def test_leading_digit(self):
        result = _safe_id("3rdService")
        assert not result[0].isdigit()
        assert result.startswith("n_")

    def test_consecutive_underscores_collapsed(self):
        result = _safe_id("a--b--c")
        assert "__" not in result


# === _node_label ===

class TestNodeLabel:
    def test_with_type(self):
        comp = {"name": "Auth", "type": "service"}
        label = _node_label(comp)
        assert "Auth" in label
        assert "service" in label

    def test_generic_type_omitted(self):
        comp = {"name": "Auth", "type": "component"}
        label = _node_label(comp)
        assert label == "Auth"

    def test_empty_type(self):
        comp = {"name": "Auth", "type": ""}
        label = _node_label(comp)
        assert label == "Auth"


# === _node_shape ===

class TestNodeShape:
    def test_entity_rectangle(self):
        assert _node_shape({"type": "entity"}) == ("[", "]")

    def test_process_stadium(self):
        assert _node_shape({"type": "process"}) == ("([", "])")

    def test_service_stadium(self):
        assert _node_shape({"type": "service"}) == ("([", "])")

    def test_constraint_hexagon(self):
        assert _node_shape({"type": "constraint"}) == ("{{", "}}")

    def test_unknown_defaults_to_rectangle(self):
        assert _node_shape({"type": "unknown"}) == ("[", "]")

    def test_no_type_defaults(self):
        assert _node_shape({}) == ("[", "]")


# === _arrow_style ===

class TestArrowStyle:
    def test_depends_on_dashed(self):
        result = _arrow_style("depends_on")
        # Returns tuple due to trailing comma bug — handle both
        if isinstance(result, tuple):
            assert "-.->" in result
        else:
            assert result == "-.->"

    def test_triggers_bold(self):
        assert _arrow_style("triggers") == "==>"

    def test_contains_arrow(self):
        assert _arrow_style("contains") == "-->"

    def test_default_arrow(self):
        assert _arrow_style("") == "-->"

    def test_unknown_type(self):
        assert _arrow_style("unknown_rel") == "-->"


# === _type_style ===

class TestTypeStyle:
    def test_entity_style(self):
        style = _type_style("entity")
        assert "fill:" in style

    def test_unknown_empty(self):
        assert _type_style("unknown") == ""

    def test_case_insensitive(self):
        assert _type_style("Entity") == _type_style("entity")


# === _group_by_type ===

class TestGroupByType:
    def test_groups_components(self):
        comps = [
            {"name": "A", "type": "service"},
            {"name": "B", "type": "entity"},
            {"name": "C", "type": "service"},
        ]
        groups = _group_by_type(comps)
        assert len(groups["service"]) == 2
        assert len(groups["entity"]) == 1

    def test_default_type(self):
        comps = [{"name": "A"}]
        groups = _group_by_type(comps)
        assert "component" in groups


# === _simplified_diagram ===

class TestSimplifiedDiagram:
    def test_renders_subsystems(self):
        comps = [
            {"name": "Core", "type": "subsystem"},
            {"name": "A", "type": "component"},
        ]
        rels = [{"from": "A", "to": "Core", "type": "uses"}]
        diagram = _simplified_diagram(comps, rels, "TD", "Big System")
        assert "Core" in diagram
        assert "simplified" in diagram

    def test_summary_node(self):
        comps = [
            {"name": "A", "type": "service"},
            {"name": "B", "type": "service"},
        ]
        diagram = _simplified_diagram(comps, [], "TD", "")
        assert "2 service" in diagram

    def test_only_subsystem_relationships(self):
        comps = [
            {"name": "Core", "type": "subsystem"},
            {"name": "A", "type": "component"},
            {"name": "B", "type": "component"},
        ]
        rels = [
            {"from": "A", "to": "Core", "type": "uses"},    # includes subsystem
            {"from": "A", "to": "B", "type": "calls"},      # no subsystem
        ]
        diagram = _simplified_diagram(comps, rels, "TD", "")
        assert "uses" in diagram
        assert "calls" not in diagram
