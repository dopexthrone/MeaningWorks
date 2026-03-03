"""
Tests for core/materialization.py — parallel agent dispatch for code generation.

Phase B.2: Parallel Agent Materialization
~30 tests — dependency graph, topological sort, parallel batching,
node prompt building, materialization plan, interface verification.
"""

import pytest

from core.interface_schema import (
    InterfaceMap, InterfaceContract, DataFlow, InterfaceConstraint,
)
from core.dimensional import (
    DimensionalMetadata, DimensionAxis, NodePosition,
)
from core.materialization import (
    DependencyGraph,
    build_dependency_graph,
    topological_sort,
    MaterializationBatch,
    find_parallel_batches,
    NodePrompt,
    build_node_prompt,
    MaterializationPlan,
    build_materialization_plan,
    verify_interfaces,
)
from core.naming import to_snake as _to_snake_case


# =============================================================================
# Helpers
# =============================================================================

def _contract(a: str, b: str, direction: str = "mutual", rel_type: str = "manages") -> InterfaceContract:
    """Build a minimal InterfaceContract."""
    return InterfaceContract(
        node_a=a,
        node_b=b,
        relationship_type=rel_type,
        relationship_description=f"{a} {rel_type} {b}",
        data_flows=(
            DataFlow(name="state", type_hint="Dict", direction="A_to_B", derived_from="test"),
        ),
        constraints=(),
        fragility=0.5,
        confidence=0.8,
        directionality=direction,
        derived_from="test",
    )


def _imap(*contracts: InterfaceContract) -> InterfaceMap:
    """Build an InterfaceMap from contracts."""
    return InterfaceMap(
        contracts=tuple(contracts),
        unmatched_relationships=(),
        extraction_confidence=0.8,
        derived_from="test",
    )


def _blueprint(*component_names: str, constraints=None) -> dict:
    """Build a minimal blueprint dict."""
    components = [
        {"name": n, "type": "entity", "description": f"The {n} component", "methods": []}
        for n in component_names
    ]
    bp = {"components": components, "constraints": constraints or []}
    return bp


def _dim_meta(*names: str) -> DimensionalMetadata:
    """Build minimal DimensionalMetadata with positions."""
    axes = (
        DimensionAxis(
            name="complexity", range_low="simple", range_high="complex",
            exploration_depth=0.8, derived_from="test",
        ),
    )
    node_positions = tuple(
        (n, NodePosition(
            dimension_values=(("complexity", 0.1 * i),),
            confidence=0.8,
        ))
        for i, n in enumerate(names)
    )
    return DimensionalMetadata(
        dimensions=axes,
        node_positions=node_positions,
        fragile_edges=(),
        silence_zones=(),
        confidence_trajectory=(0.8,),
        dimension_confidence=(("complexity", 0.8),),
        dialogue_depth=1,
        stage_discovery=(),
    )


# =============================================================================
# Dependency Graph Tests
# =============================================================================

class TestBuildDependencyGraph:
    def test_empty_interface_map(self):
        graph = build_dependency_graph(_imap())
        assert graph.nodes == []
        assert graph.edges == []

    def test_mutual_no_edges(self):
        """Mutual direction creates no ordering edges."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "mutual"),
        ))
        assert set(graph.nodes) == {"A", "B"}
        assert graph.edges == []

    def test_a_depends_on_b_edge(self):
        """A_depends_on_B means A drives B → edge from A to B."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
        ))
        assert ("A", "B") in graph.edges

    def test_b_depends_on_a_edge(self):
        """B_depends_on_A means B serves A → edge from B to A."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "B_depends_on_A"),
        ))
        assert ("B", "A") in graph.edges

    def test_adjacency_populated(self):
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
        ))
        assert "A" in graph.adjacency["B"]   # B depends on A
        assert "B" not in graph.adjacency["A"]  # A has no deps

    def test_reverse_adjacency(self):
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
        ))
        assert "B" in graph.reverse["A"]  # A is depended on by B

    def test_three_node_chain(self):
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
            _contract("B", "C", "B_depends_on_A"),
        ))
        # A → B (from first contract)
        # B → C becomes C → B (B_depends_on_A means B depends on A, so edge B to A... wait)
        # B_depends_on_A: B needs A first → edge from A to B. But node_a="B", node_b="C"
        # So B_depends_on_A means: B serves A → edge from node_b to node_a = C to B
        # Wait, re-read: contract("B", "C", "B_depends_on_A") → B=node_a, C=node_b
        # "B_depends_on_A" direction → edge from (node_b, node_a) = (C, B)
        # But that means "B depends on A" is confusing. Let me check the code.
        # From code: elif contract.directionality == "B_depends_on_A":
        #   edges.append((contract.node_b, contract.node_a))
        # So for contract(B, C, B_depends_on_A) → edge (C, B) → C must come before B
        assert set(graph.nodes) == {"A", "B", "C"}


# =============================================================================
# Topological Sort Tests
# =============================================================================

class TestTopologicalSort:
    def test_no_edges(self):
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "mutual"),
            _contract("B", "C", "mutual"),
        ))
        result = topological_sort(graph)
        assert set(result) == {"A", "B", "C"}
        assert len(result) == 3

    def test_linear_chain(self):
        """A → B → C: A first, then B, then C."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),   # edge A → B
            _contract("B", "C", "A_depends_on_B"),   # edge B → C
        ))
        result = topological_sort(graph)
        assert result.index("A") < result.index("B")
        assert result.index("B") < result.index("C")

    def test_diamond(self):
        """A → B, A → C, B → D, C → D."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
            _contract("A", "C", "A_depends_on_B"),
            _contract("B", "D", "A_depends_on_B"),
            _contract("C", "D", "A_depends_on_B"),
        ))
        result = topological_sort(graph)
        assert result.index("A") < result.index("B")
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_cycle_raises(self):
        """Cycle detection: A → B → A."""
        # Create a cycle manually
        graph = DependencyGraph(
            nodes=["A", "B"],
            edges=[("A", "B"), ("B", "A")],
            adjacency={"A": {"B"}, "B": {"A"}},
            reverse={"A": {"B"}, "B": {"A"}},
        )
        from core.exceptions import GraphError
        with pytest.raises(GraphError):
            topological_sort(graph)

    def test_single_node(self):
        # Single node in mutual
        graph = build_dependency_graph(_imap(
            _contract("A", "A", "mutual"),
        ))
        # A appears in both node_a and node_b, but only once
        result = topological_sort(graph)
        assert result == ["A"]


# =============================================================================
# Parallel Batching Tests
# =============================================================================

class TestFindParallelBatches:
    def test_all_mutual_single_batch(self):
        """All mutual = all in batch 0."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "mutual"),
            _contract("B", "C", "mutual"),
        ))
        batches = find_parallel_batches(graph)
        assert len(batches) == 1
        assert set(batches[0].nodes) == {"A", "B", "C"}
        assert batches[0].batch_index == 0

    def test_linear_chain_sequential_batches(self):
        """A → B → C = 3 batches, 1 node each."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
            _contract("B", "C", "A_depends_on_B"),
        ))
        batches = find_parallel_batches(graph)
        assert len(batches) == 3
        assert batches[0].nodes == ("A",)
        assert batches[1].nodes == ("B",)
        assert batches[2].nodes == ("C",)

    def test_diamond_two_batches(self):
        """A → {B, C} → D = 3 batches."""
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
            _contract("A", "C", "A_depends_on_B"),
            _contract("B", "D", "A_depends_on_B"),
            _contract("C", "D", "A_depends_on_B"),
        ))
        batches = find_parallel_batches(graph)
        assert len(batches) == 3
        # Batch 0: A (root)
        assert batches[0].nodes == ("A",)
        # Batch 1: B and C in parallel
        assert set(batches[1].nodes) == {"B", "C"}
        # Batch 2: D (depends on B and C)
        assert batches[2].nodes == ("D",)

    def test_batch_index_sequential(self):
        graph = build_dependency_graph(_imap(
            _contract("A", "B", "A_depends_on_B"),
        ))
        batches = find_parallel_batches(graph)
        for i, batch in enumerate(batches):
            assert batch.batch_index == i

    def test_empty_graph(self):
        graph = build_dependency_graph(_imap())
        batches = find_parallel_batches(graph)
        assert batches == []

    def test_batch_is_frozen(self):
        batch = MaterializationBatch(batch_index=0, nodes=("A",), dependency_count=0)
        with pytest.raises(AttributeError):
            batch.batch_index = 1


# =============================================================================
# Node Prompt Building Tests
# =============================================================================

class TestBuildNodePrompt:
    def test_basic_prompt(self):
        component = {"name": "SharedState", "type": "entity", "description": "Stores state", "methods": []}
        blueprint = _blueprint("SharedState")
        imap = _imap()
        prompt = build_node_prompt(component, blueprint, imap)
        assert prompt.component_name == "SharedState"
        assert prompt.component_type == "entity"
        assert "SharedState" in prompt.prompt_text
        assert "Materialize" in prompt.prompt_text

    def test_prompt_includes_interfaces(self):
        component = {"name": "Governor", "type": "agent", "description": "Orchestrates", "methods": []}
        contract = _contract("Governor", "SharedState", "A_depends_on_B")
        imap = _imap(contract)
        blueprint = _blueprint("Governor", "SharedState")
        prompt = build_node_prompt(component, blueprint, imap)
        assert len(prompt.interfaces) == 1
        assert prompt.interfaces[0]["adjacent_node"] == "SharedState"
        assert "Interfaces" in prompt.prompt_text

    def test_prompt_includes_dimensional_position(self):
        component = {"name": "Governor", "type": "agent", "description": "Orchestrates", "methods": []}
        blueprint = _blueprint("Governor")
        imap = _imap()
        dim = _dim_meta("Governor")
        prompt = build_node_prompt(component, blueprint, imap, dim)
        assert len(prompt.dimensional_position) > 0
        assert "Dimensional position" in prompt.prompt_text

    def test_prompt_includes_constraints(self):
        component = {"name": "Auth", "type": "entity", "description": "Authentication", "methods": []}
        blueprint = {
            "components": [component],
            "constraints": [
                {"description": "Auth must use JWT tokens", "applies_to": ["Auth"]},
                {"description": "Unrelated constraint", "applies_to": ["Database"]},
            ],
        }
        imap = _imap()
        prompt = build_node_prompt(component, blueprint, imap)
        assert "JWT tokens" in prompt.prompt_text
        assert len(prompt.constraints) == 1

    def test_prompt_includes_methods(self):
        component = {
            "name": "UserService",
            "type": "agent",
            "description": "User management",
            "methods": [
                {"name": "create_user", "parameters": [{"name": "name", "type_hint": "str"}], "return_type": "User"},
            ],
        }
        blueprint = _blueprint("UserService")
        imap = _imap()
        prompt = build_node_prompt(component, blueprint, imap)
        assert "create_user" in prompt.prompt_text
        assert len(prompt.methods) == 1

    def test_prompt_no_dim_meta(self):
        component = {"name": "X", "type": "entity", "description": "Test", "methods": []}
        prompt = build_node_prompt(component, _blueprint("X"), _imap())
        assert prompt.dimensional_position == {}
        assert "Dimensional position" not in prompt.prompt_text


# =============================================================================
# Materialization Plan Tests
# =============================================================================

class TestBuildMaterializationPlan:
    def test_basic_plan(self):
        imap = _imap(
            _contract("A", "B", "A_depends_on_B"),
        )
        blueprint = _blueprint("A", "B")
        plan = build_materialization_plan(blueprint, imap)
        assert plan.total_nodes == 2
        assert len(plan.batches) == 2
        assert plan.max_parallelism >= 1
        assert plan.estimated_serial_steps == 2

    def test_plan_has_prompts(self):
        imap = _imap(_contract("A", "B", "mutual"))
        blueprint = _blueprint("A", "B")
        plan = build_materialization_plan(blueprint, imap)
        assert "A" in plan.node_prompts
        assert "B" in plan.node_prompts

    def test_plan_with_dim_meta(self):
        imap = _imap(_contract("A", "B", "mutual"))
        blueprint = _blueprint("A", "B")
        dim = _dim_meta("A", "B")
        plan = build_materialization_plan(blueprint, imap, dim)
        assert plan.node_prompts["A"].dimensional_position != {}

    def test_plan_max_parallelism(self):
        """Three independent nodes → max parallelism = 3."""
        imap = _imap(
            _contract("A", "B", "mutual"),
            _contract("B", "C", "mutual"),
        )
        blueprint = _blueprint("A", "B", "C")
        plan = build_materialization_plan(blueprint, imap)
        assert plan.max_parallelism == 3  # All in one batch

    def test_empty_blueprint(self):
        plan = build_materialization_plan({"components": []}, _imap())
        assert plan.total_nodes == 0
        assert plan.max_parallelism == 0


# =============================================================================
# Interface Verification Tests
# =============================================================================

class TestVerifyInterfaces:
    def test_both_reference(self):
        imap = _imap(_contract("Auth", "Database", "A_depends_on_B"))
        code = {
            "Auth": "class Auth:\n    def __init__(self, database: Database): pass",
            "Database": "class Database:\n    # used by Auth\n    pass",
        }
        report = verify_interfaces(code, imap)
        assert report["total_contracts"] == 1
        assert report["passed"] == 1
        assert report["pass_rate"] == 1.0

    def test_missing_reference_fails(self):
        imap = _imap(_contract("Auth", "Database", "A_depends_on_B"))
        code = {
            "Auth": "class Auth:\n    pass",
            "Database": "class Database:\n    pass",
        }
        report = verify_interfaces(code, imap)
        assert report["failed"] == 1

    def test_partial_reference_passes(self):
        """At least one direction referenced = pass."""
        imap = _imap(_contract("Auth", "Database", "A_depends_on_B"))
        code = {
            "Auth": "class Auth:\n    db = Database()",
            "Database": "class Database:\n    pass",
        }
        report = verify_interfaces(code, imap)
        assert report["passed"] == 1

    def test_empty_code(self):
        imap = _imap(_contract("A", "B", "mutual"))
        report = verify_interfaces({}, imap)
        assert report["failed"] == 1

    def test_no_contracts(self):
        report = verify_interfaces({"A": "code"}, _imap())
        assert report["total_contracts"] == 0
        assert report["pass_rate"] == 1.0

    def test_flow_method_detection(self):
        imap = _imap(_contract("A", "B", "mutual"))
        # The default data flow name is "state"
        code = {
            "A": "class A:\n    def get_state(self, b: B): pass",
            "B": "class B:\n    pass",
        }
        report = verify_interfaces(code, imap)
        details = report["details"][0]
        assert details["flow_methods_found"] >= 1


# =============================================================================
# ENRICHED PROMPT TESTS
# =============================================================================


def _contract_with_constraints(a, b, frag=0.5, constraints=()):
    """Build a contract with explicit constraints and fragility."""
    return InterfaceContract(
        node_a=a,
        node_b=b,
        relationship_type="manages",
        relationship_description=f"{a} manages {b}",
        data_flows=(
            DataFlow(name="state", type_hint="Dict", direction="A_to_B", derived_from="test"),
        ),
        constraints=tuple(
            InterfaceConstraint(description=c, constraint_type="custom", derived_from="test")
            for c in constraints
        ),
        fragility=frag,
        confidence=0.8,
        directionality="A_depends_on_B",
        derived_from="test",
    )


class TestEnrichedPrompts:
    """Tests for enriched build_node_prompt()."""

    def test_prompt_interface_import_directive(self):
        """from .x import X in prompt_text."""
        contract = _contract("TaskManager", "Task", "A_depends_on_B")
        imap = _imap(contract)
        component = {"name": "TaskManager", "type": "process", "description": "Manages tasks"}
        blueprint = {"components": [component, {"name": "Task", "type": "entity"}], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "from .task import Task" in prompt.prompt_text

    def test_prompt_interface_constraints(self):
        """Constraint descriptions rendered."""
        contract = _contract_with_constraints(
            "TaskManager", "Task",
            constraints=["Must validate task before assignment"],
        )
        imap = _imap(contract)
        component = {"name": "TaskManager", "type": "process", "description": "Manages tasks"}
        blueprint = {"components": [component, {"name": "Task", "type": "entity"}], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "Must validate task before assignment" in prompt.prompt_text

    def test_prompt_fragility_warning(self):
        """HIGH FRAGILITY when fragility > 0.5."""
        contract = _contract_with_constraints("A", "B", frag=0.8)
        imap = _imap(contract)
        component = {"name": "A", "type": "entity", "description": "test"}
        blueprint = {"components": [component, {"name": "B", "type": "entity"}], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "HIGH FRAGILITY" in prompt.prompt_text

    def test_prompt_no_fragility_low(self):
        """No warning when fragility <= 0.5."""
        contract = _contract_with_constraints("A", "B", frag=0.3)
        imap = _imap(contract)
        component = {"name": "A", "type": "entity", "description": "test"}
        blueprint = {"components": [component, {"name": "B", "type": "entity"}], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "HIGH FRAGILITY" not in prompt.prompt_text

    def test_prompt_adjacent_methods(self):
        """Adjacent node methods appear."""
        contract = _contract("TaskManager", "Task", "A_depends_on_B")
        imap = _imap(contract)
        task_comp = {
            "name": "Task",
            "type": "entity",
            "methods": [
                {"name": "validate", "parameters": [{"name": "data", "type_hint": "Dict"}], "return_type": "bool"},
            ],
        }
        component = {"name": "TaskManager", "type": "process", "description": "Manages tasks"}
        blueprint = {"components": [component, task_comp], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "Adjacent Node Methods" in prompt.prompt_text
        assert "Task.validate" in prompt.prompt_text

    def test_prompt_no_adjacent_when_empty(self):
        """Section omitted when adjacent has no methods."""
        contract = _contract("A", "B", "mutual")
        imap = _imap(contract)
        component = {"name": "A", "type": "entity", "description": "test"}
        blueprint = {"components": [component, {"name": "B", "type": "entity"}], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "Adjacent Node Methods" not in prompt.prompt_text

    def test_prompt_algorithms_section(self):
        """Algorithm steps rendered in prompt."""
        imap = _imap()
        component = {
            "name": "Router",
            "type": "process",
            "description": "Routes requests",
            "algorithms": [
                {
                    "method_name": "dispatch",
                    "steps": ["1. Parse request", "2. Match route", "3. Return handler"],
                    "preconditions": ["request is not None"],
                    "postconditions": ["handler is assigned"],
                },
            ],
        }
        blueprint = {"components": [component], "constraints": []}
        prompt = build_node_prompt(component, blueprint, imap)
        assert "Algorithm: dispatch" in prompt.prompt_text
        assert "Parse request" in prompt.prompt_text
        assert "PRE: request is not None" in prompt.prompt_text

    def test_to_snake_case(self):
        """Helper conversions correct."""
        assert _to_snake_case("PayloadAggregator") == "payload_aggregator"
        assert _to_snake_case("Task") == "task"
        assert _to_snake_case("TaskManager") == "task_manager"
        assert _to_snake_case("my component") == "my_component"
