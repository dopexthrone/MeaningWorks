"""
Phase 17.1: Graph Safety — Cycle-Tolerant Materialization Tests.

Tests for:
- GraphError exception (replaces ValueError on cycles)
- topological_sort_tolerant (breaks cycles gracefully)
- build_materialization_plan fallback on cycles
- E7001 catalog entry
"""

import pytest
from core.exceptions import GraphError, MotherlabsError
from core.error_catalog import get_entry, get_code_for_exception
from core.materialization import (
    DependencyGraph,
    topological_sort,
    topological_sort_tolerant,
    build_materialization_plan,
    build_dependency_graph,
)
from core.interface_schema import InterfaceMap, InterfaceContract


# =============================================================================
# Helpers
# =============================================================================

def _make_graph(nodes, edges):
    """Build a DependencyGraph from node list and edge list."""
    adjacency = {n: set() for n in nodes}
    reverse = {n: set() for n in nodes}
    for dep, dependent in edges:
        adjacency[dependent].add(dep)
        reverse[dep].add(dependent)
    return DependencyGraph(
        nodes=sorted(nodes),
        edges=edges,
        adjacency=adjacency,
        reverse=reverse,
    )


def _make_contract(node_a, node_b, directionality="A_depends_on_B"):
    """Build an InterfaceContract with required fields."""
    return InterfaceContract(
        node_a=node_a,
        node_b=node_b,
        relationship_type="depends_on",
        relationship_description="test dep",
        data_flows=(),
        constraints=(),
        fragility=0.5,
        confidence=0.9,
        directionality=directionality,
        derived_from="test",
    )


def _make_imap(contracts):
    """Build an InterfaceMap from a tuple of contracts."""
    return InterfaceMap(
        contracts=contracts,
        unmatched_relationships=(),
        extraction_confidence=0.9,
        derived_from="test",
    )


def _make_cyclic_interface_map():
    """Build an InterfaceMap that produces a cycle: A depends_on B depends_on A."""
    contracts = (
        _make_contract("A", "B", "A_depends_on_B"),
        _make_contract("B", "A", "A_depends_on_B"),
    )
    return _make_imap(contracts)


def _make_acyclic_interface_map():
    """Build an InterfaceMap: A -> B -> C (no cycle)."""
    contracts = (
        _make_contract("A", "B", "B_depends_on_A"),
        _make_contract("B", "C", "B_depends_on_A"),
    )
    return _make_imap(contracts)


# =============================================================================
# GraphError exception tests
# =============================================================================

class TestGraphError:
    def test_inherits_from_motherlabs_error(self):
        assert issubclass(GraphError, MotherlabsError)

    def test_default_error_code(self):
        err = GraphError("cycle found")
        assert err.error_code == "E7001"

    def test_cycle_nodes_attribute(self):
        nodes = {"A", "B"}
        err = GraphError("cycle found", cycle_nodes=nodes)
        assert err.cycle_nodes == nodes

    def test_cycle_nodes_defaults_empty(self):
        err = GraphError("cycle found")
        assert err.cycle_nodes == set()

    def test_user_message_populated(self):
        err = GraphError("cycle found")
        assert "circular dependencies" in err.user_message

    def test_to_user_dict(self):
        err = GraphError("cycle found", cycle_nodes={"A", "B"})
        d = err.to_user_dict()
        assert d["error_code"] == "E7001"
        assert "GraphError" in d["error_type"]


# =============================================================================
# topological_sort raises GraphError (not ValueError)
# =============================================================================

class TestTopologicalSortGraphError:
    def test_raises_graph_error_on_cycle(self):
        graph = _make_graph(["A", "B"], [("A", "B"), ("B", "A")])
        with pytest.raises(GraphError) as exc_info:
            topological_sort(graph)
        assert exc_info.value.error_code == "E7001"
        assert len(exc_info.value.cycle_nodes) > 0

    def test_does_not_raise_value_error(self):
        graph = _make_graph(["A", "B"], [("A", "B"), ("B", "A")])
        with pytest.raises(GraphError):
            topological_sort(graph)
        # Confirm it's NOT a bare ValueError
        try:
            topological_sort(graph)
        except GraphError:
            pass
        except ValueError:
            pytest.fail("Should raise GraphError, not ValueError")

    def test_acyclic_succeeds(self):
        graph = _make_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        result = topological_sort(graph)
        assert result == ["A", "B", "C"]


# =============================================================================
# topological_sort_tolerant
# =============================================================================

class TestTopologicalSortTolerant:
    def test_acyclic_graph_no_broken_edges(self):
        graph = _make_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        sorted_nodes, broken = topological_sort_tolerant(graph)
        assert sorted_nodes == ["A", "B", "C"]
        assert broken == []

    def test_simple_cycle_a_b(self):
        graph = _make_graph(["A", "B"], [("A", "B"), ("B", "A")])
        sorted_nodes, broken = topological_sort_tolerant(graph)
        assert set(sorted_nodes) == {"A", "B"}
        assert len(broken) == 1

    def test_multi_node_cycle_a_b_c(self):
        graph = _make_graph(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C"), ("C", "A")]
        )
        sorted_nodes, broken = topological_sort_tolerant(graph)
        assert set(sorted_nodes) == {"A", "B", "C"}
        assert len(sorted_nodes) == 3
        assert len(broken) >= 1

    def test_self_loop(self):
        graph = _make_graph(["A"], [("A", "A")])
        sorted_nodes, broken = topological_sort_tolerant(graph)
        assert sorted_nodes == ["A"]
        assert len(broken) == 1
        assert broken[0] == ("A", "A")

    def test_returns_valid_ordering(self):
        """After breaking edges, the returned order should be a valid topo sort."""
        graph = _make_graph(
            ["A", "B", "C", "D"],
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]
        )
        sorted_nodes, broken = topological_sort_tolerant(graph)
        assert len(sorted_nodes) == 4
        assert set(sorted_nodes) == {"A", "B", "C", "D"}


# =============================================================================
# build_materialization_plan with cycles
# =============================================================================

class TestBuildPlanCycleFallback:
    def test_cyclic_plan_returns_with_warnings(self):
        imap = _make_cyclic_interface_map()
        blueprint = {
            "components": [
                {"name": "A", "type": "entity", "description": "test"},
                {"name": "B", "type": "entity", "description": "test"},
            ],
            "relationships": [],
            "constraints": [],
        }
        plan = build_materialization_plan(blueprint, imap)
        assert plan.total_nodes == 2
        assert len(plan.warnings) >= 1
        assert any("cycle" in w.lower() for w in plan.warnings)

    def test_acyclic_plan_no_warnings(self):
        imap = _make_acyclic_interface_map()
        blueprint = {
            "components": [
                {"name": "A", "type": "entity", "description": "test"},
                {"name": "B", "type": "entity", "description": "test"},
                {"name": "C", "type": "entity", "description": "test"},
            ],
            "relationships": [],
            "constraints": [],
        }
        plan = build_materialization_plan(blueprint, imap)
        assert plan.warnings == []


# =============================================================================
# E7001 catalog entry
# =============================================================================

class TestE7001Catalog:
    def test_entry_exists(self):
        entry = get_entry("E7001")
        assert entry is not None
        assert entry.code == "E7001"
        assert "cycle" in entry.title.lower()

    def test_graph_error_code_inference(self):
        code = get_code_for_exception("GraphError")
        assert code == "E7001"
