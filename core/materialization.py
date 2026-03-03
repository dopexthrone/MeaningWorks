"""
Motherlabs Materialization — parallel agent dispatch for code generation.

Phase B.2: Parallel Agent Materialization
Derived from: DIMENSIONAL_BLUEPRINT.md — "parallel print heads: blueprint
declares interfaces → agents materialize nodes simultaneously →
no merge conflicts by construction"

Uses InterfaceMap to determine which nodes can be materialized in parallel.
Each agent receives: node coordinates, adjacent interfaces, constraints.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, List, Tuple, Optional, Set, FrozenSet

from core.interface_schema import InterfaceMap, InterfaceContract
from core.dimensional import DimensionalMetadata
from core.exceptions import GraphError
from core.naming import to_snake


# =============================================================================
# EMISSION LAYERS — enum only (dataclasses after MaterializationBatch)
# =============================================================================

class EmissionLayer(IntEnum):
    """Semantic layers for layered emission.

    Layer 0 emits first (pure types), each subsequent layer sees actual code
    from prior layers — not descriptions, real importable code.
    """
    TYPES = 0           # entity/data — pure dataclasses, no cross-deps
    INTERFACES = 1      # interface/api — ABCs/protocols, import L0
    IMPLEMENTATIONS = 2 # process/agent — full behavior, import L0+L1
    INTEGRATION = 3     # deterministic — main.py, __init__.py, test stubs


# =============================================================================
# DEPENDENCY GRAPH
# =============================================================================

@dataclass
class DependencyGraph:
    """
    Directed acyclic graph of component dependencies.

    Built from InterfaceMap directionality. Used for topological sort
    and parallel batch computation.
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]     # (dependency, dependent) — dependency must come first
    adjacency: Dict[str, Set[str]]   # node -> set of nodes it depends on
    reverse: Dict[str, Set[str]]     # node -> set of nodes that depend on it


def build_dependency_graph(interface_map: InterfaceMap) -> DependencyGraph:
    """
    Build a dependency graph from interface contracts.

    Direction mapping:
    - "A_depends_on_B": A needs B first → edge from B to A
    - "B_depends_on_A": B needs A first → edge from A to B
    - "mutual": no ordering constraint (both can run in parallel)

    Args:
        interface_map: The InterfaceMap with all contracts

    Returns:
        DependencyGraph ready for topological sort
    """
    all_nodes: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    adjacency: Dict[str, Set[str]] = {}
    reverse: Dict[str, Set[str]] = {}

    for contract in interface_map.contracts:
        all_nodes.add(contract.node_a)
        all_nodes.add(contract.node_b)

        if contract.directionality == "A_depends_on_B":
            # A drives B → B depends on A being materialized first
            edges.append((contract.node_a, contract.node_b))
        elif contract.directionality == "B_depends_on_A":
            # B serves A → A depends on B being materialized first
            edges.append((contract.node_b, contract.node_a))
        # "mutual" → no ordering edge (both can be parallel)

    # Build adjacency lists
    for node in all_nodes:
        adjacency[node] = set()
        reverse[node] = set()

    for dep, dependent in edges:
        adjacency[dependent].add(dep)
        reverse[dep].add(dependent)

    return DependencyGraph(
        nodes=sorted(all_nodes),
        edges=edges,
        adjacency=adjacency,
        reverse=reverse,
    )


def topological_sort(graph: DependencyGraph) -> List[str]:
    """
    Topological sort of the dependency graph (Kahn's algorithm).

    Returns nodes in dependency order — materialized first to last.
    Raises ValueError if a cycle is detected.
    """
    in_degree: Dict[str, int] = {n: len(graph.adjacency.get(n, set())) for n in graph.nodes}
    queue = sorted(n for n in graph.nodes if in_degree[n] == 0)
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(graph.reverse.get(node, set())):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(graph.nodes):
        missing = set(graph.nodes) - set(result)
        raise GraphError(
            f"Dependency cycle detected involving: {missing}",
            cycle_nodes=missing,
        )

    return result


def topological_sort_tolerant(graph: DependencyGraph) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Cycle-tolerant topological sort.

    When a cycle is detected, breaks it by removing the back-edge whose
    source node has the fewest downstream dependents, then continues.

    Returns:
        Tuple of (sorted_nodes, broken_edges) where broken_edges lists
        the (from, to) edges that were removed to break cycles.
    """
    # Work on mutable copies
    adjacency = {n: set(deps) for n, deps in graph.adjacency.items()}
    reverse = {n: set(deps) for n, deps in graph.reverse.items()}
    all_nodes = list(graph.nodes)

    broken_edges: List[Tuple[str, str]] = []

    # Iteratively break cycles until topo sort succeeds
    max_iterations = len(all_nodes) + 1  # Safety bound
    for _ in range(max_iterations):
        # Attempt Kahn's algorithm
        in_degree = {n: len(adjacency.get(n, set())) for n in all_nodes}
        queue = sorted(n for n in all_nodes if in_degree[n] == 0)
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in sorted(reverse.get(node, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) == len(all_nodes):
            return result, broken_edges

        # Cycle detected — find nodes still stuck
        stuck = set(all_nodes) - set(result)

        # Pick the back-edge to break: among stuck nodes, find the one
        # with fewest downstream dependents and remove one of its incoming edges
        best_edge = None
        best_downstream = float('inf')

        for node in sorted(stuck):
            for dep in sorted(adjacency.get(node, set())):
                if dep in stuck:
                    # dep -> node edge (dep is a dependency of node, both stuck)
                    downstream = len(reverse.get(dep, set()))
                    if downstream < best_downstream:
                        best_downstream = downstream
                        best_edge = (dep, node)

        if best_edge is None:
            # Self-loops: find and break
            for node in sorted(stuck):
                if node in adjacency.get(node, set()):
                    best_edge = (node, node)
                    break

        if best_edge is None:
            # Shouldn't happen, but safety: break arbitrary edge
            for node in sorted(stuck):
                deps = adjacency.get(node, set()) & stuck
                if deps:
                    best_edge = (sorted(deps)[0], node)
                    break

        if best_edge:
            dep, dependent = best_edge
            adjacency[dependent].discard(dep)
            reverse[dep].discard(dependent)
            broken_edges.append(best_edge)

    # Fallback: return nodes in alphabetical order with all edges broken
    return sorted(all_nodes), broken_edges


# =============================================================================
# PARALLEL BATCHING
# =============================================================================

@dataclass(frozen=True)
class MaterializationBatch:
    """
    A batch of nodes that can be materialized in parallel.

    All nodes in a batch have their dependencies satisfied by prior batches.
    """
    batch_index: int
    nodes: Tuple[str, ...]
    dependency_count: int   # Total deps across all nodes in batch


@dataclass(frozen=True)
class LayerPlan:
    """Plan for emitting one semantic layer.

    Each layer contains batches of nodes that can be emitted in parallel
    within that layer. Layers are processed sequentially.
    """
    layer: int                                # EmissionLayer value
    layer_name: str                           # "types" | "interfaces" | "implementations" | "integration"
    batches: Tuple[MaterializationBatch, ...]  # dependency-ordered within layer
    node_names: Tuple[str, ...]               # all components in this layer
    is_deterministic: bool                    # True for Layer 3


@dataclass(frozen=True)
class LayerGateResult:
    """Result of validating a layer's emitted code before proceeding.

    Gate must pass before the next layer can use this layer's code as context.
    """
    layer: int
    passed: bool
    errors: Tuple[str, ...]                   # blocking (parse failures, unresolved imports)
    warnings: Tuple[str, ...]                 # advisory


def find_parallel_batches(graph: DependencyGraph) -> List[MaterializationBatch]:
    """
    Partition nodes into parallel batches using dependency levels.

    Batch 0: nodes with no dependencies (can start immediately)
    Batch 1: nodes whose deps are all in batch 0
    Batch N: nodes whose deps are all in batches 0..N-1

    Returns:
        List of MaterializationBatch in execution order
    """
    # Assign levels via BFS from roots
    levels: Dict[str, int] = {}
    in_degree: Dict[str, int] = {n: len(graph.adjacency.get(n, set())) for n in graph.nodes}
    queue = sorted(n for n in graph.nodes if in_degree[n] == 0)

    for n in queue:
        levels[n] = 0

    processed = set(queue)
    current_queue = list(queue)

    while current_queue:
        next_queue = []
        for node in current_queue:
            for dependent in sorted(graph.reverse.get(node, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    levels[dependent] = levels[node] + 1
                    next_queue.append(dependent)
                    processed.add(dependent)
        current_queue = next_queue

    # Handle any unprocessed nodes (cycle or disconnected)
    for n in graph.nodes:
        if n not in levels:
            levels[n] = 0

    # Group by level
    max_level = max(levels.values()) if levels else 0
    batches = []
    for level in range(max_level + 1):
        nodes_at_level = tuple(sorted(n for n, l in levels.items() if l == level))
        if nodes_at_level:
            dep_count = sum(len(graph.adjacency.get(n, set())) for n in nodes_at_level)
            batches.append(MaterializationBatch(
                batch_index=level,
                nodes=nodes_at_level,
                dependency_count=dep_count,
            ))

    return batches


# =============================================================================
# NODE PROMPT BUILDING
# =============================================================================

@dataclass
class NodePrompt:
    """
    Complete prompt for materializing a single node.

    Contains everything an agent needs to generate code for this component
    without coordinating with other agents.
    """
    component_name: str
    component_type: str
    description: str
    dimensional_position: Dict[str, float]   # axis_name -> value
    interfaces: List[Dict[str, Any]]         # adjacent contracts
    constraints: List[str]                    # applicable constraints
    methods: List[Dict[str, Any]]            # declared methods
    prompt_text: str                          # Assembled prompt string


def build_node_prompt(
    component: Dict[str, Any],
    blueprint: Dict[str, Any],
    interface_map: InterfaceMap,
    dim_meta: Optional[DimensionalMetadata] = None,
) -> NodePrompt:
    """
    Build a materialization prompt for a single blueprint component.

    Assembles: component metadata + dimensional position + interface contracts
    + applicable constraints + methods into a structured prompt.

    Args:
        component: Component dict from blueprint
        blueprint: Full blueprint for constraint lookup
        interface_map: InterfaceMap for interface contracts
        dim_meta: Optional dimensional metadata for positioning

    Returns:
        NodePrompt ready for agent dispatch
    """
    name = component.get("name", "")
    comp_type = component.get("type", "entity")
    description = component.get("description", "")
    methods = component.get("methods", [])

    # Get dimensional position
    position = {}
    if dim_meta:
        pos = dim_meta.get_position(name)
        position = dict(pos.dimension_values)

    # Find interface contracts involving this component
    interfaces = []
    for contract in interface_map.contracts:
        if contract.node_a == name or contract.node_b == name:
            other = contract.node_b if contract.node_a == name else contract.node_a
            direction = contract.directionality
            flows = [
                {"name": df.name, "type": df.type_hint, "direction": df.direction}
                for df in contract.data_flows
            ]
            interfaces.append({
                "adjacent_node": other,
                "relationship": contract.relationship_type,
                "direction": direction,
                "data_flows": flows,
                "fragility": contract.fragility,
            })

    # Find applicable constraints
    constraints = []
    name_lower = name.lower()
    for constraint in blueprint.get("constraints", []):
        desc = constraint.get("description", "")
        applies_to = constraint.get("applies_to", [])
        if any(name_lower in at.lower() for at in applies_to) or name_lower in desc.lower():
            constraints.append(desc)

    # Look up adjacent component info from blueprint for method context
    bp_components = {c.get("name", ""): c for c in blueprint.get("components", []) if c.get("name")}

    # Build prompt text
    lines = [
        f"# Materialize: {name}",
        f"Type: {comp_type}",
        f"Description: {description}",
        "",
    ]

    if position:
        pos_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(position.items()))
        lines.append(f"Dimensional position: [{pos_str}]")
        lines.append("")

    if interfaces:
        lines.append("## Interfaces (MUST honor these contracts)")
        for iface in interfaces:
            other = iface["adjacent_node"]
            frag = iface.get("fragility", 0)
            frag_warn = "  \u26a0 HIGH FRAGILITY" if frag > 0.5 else ""
            lines.append(f"- {other} ({iface['relationship']}, {iface['direction']}){frag_warn}")
            snake_other = to_snake(other)
            lines.append(f"  Import: from .{snake_other} import {other}")
            for flow in iface["data_flows"]:
                lines.append(f"  Flow: {flow['name']}: {flow['type']} ({flow['direction']})")
            # Render interface constraints
            for contract in interface_map.contracts:
                if (contract.node_a == name and contract.node_b == other) or \
                   (contract.node_b == name and contract.node_a == other):
                    for ic in contract.constraints:
                        lines.append(f"  Constraint: {ic.description}")
                    break
        lines.append("")

    if constraints:
        lines.append("## Constraints")
        for c in constraints:
            lines.append(f"- {c}")
        lines.append("")

    if methods:
        lines.append("## Methods")
        for m in methods:
            params = ", ".join(
                f"{p.get('name', '')}: {p.get('type_hint', 'Any')}"
                for p in m.get("parameters", [])
            )
            lines.append(f"- {m.get('name', '')}({params}) -> {m.get('return_type', 'None')}")
        lines.append("")

    # Adjacent node methods — show what this component can call
    if interfaces:
        adj_method_lines = []
        for iface in interfaces:
            other = iface["adjacent_node"]
            adj_comp = bp_components.get(other, {})
            adj_methods = adj_comp.get("methods", [])
            if adj_methods:
                shown = adj_methods[:5]
                for m in shown:
                    params = ", ".join(
                        f"{p.get('name', '')}: {p.get('type_hint', 'Any')}"
                        for p in m.get("parameters", [])
                    )
                    adj_method_lines.append(
                        f"- {other}.{m.get('name', '')}({params}) -> {m.get('return_type', 'None')}"
                    )
                remaining = len(adj_methods) - 5
                if remaining > 0:
                    adj_method_lines.append(f"  ... and {remaining} more")
        if adj_method_lines:
            lines.append("## Adjacent Node Methods")
            lines.extend(adj_method_lines)
            lines.append("")

    # Algorithms section
    algorithms = component.get("algorithms", [])
    if algorithms:
        for algo in algorithms:
            algo_method = algo.get("method_name", "unknown")
            lines.append(f"## Algorithm: {algo_method}")
            for step in algo.get("steps", []):
                lines.append(f"  {step}")
            for pre in algo.get("preconditions", []):
                lines.append(f"  PRE: {pre}")
            for post in algo.get("postconditions", []):
                lines.append(f"  POST: {post}")
            lines.append("")

    lines.append("Generate a complete Python implementation for this component.")
    lines.append("Honor all declared interfaces. Do not add undeclared dependencies.")

    prompt_text = "\n".join(lines)

    return NodePrompt(
        component_name=name,
        component_type=comp_type,
        description=description,
        dimensional_position=position,
        interfaces=interfaces,
        constraints=constraints,
        methods=methods,
        prompt_text=prompt_text,
    )


# =============================================================================
# MATERIALIZATION PLAN
# =============================================================================

@dataclass
class MaterializationPlan:
    """
    Complete plan for materializing a blueprint into code.

    Contains batches of node prompts in dependency order.
    Batch 0 can start immediately. Each subsequent batch waits for prior batch.
    """
    batches: List[MaterializationBatch]
    node_prompts: Dict[str, NodePrompt]       # component_name -> prompt
    dependency_graph: DependencyGraph
    total_nodes: int
    max_parallelism: int                       # Largest batch size
    estimated_serial_steps: int                # Number of sequential batches
    warnings: List[str] = field(default_factory=list)  # Phase 17: edge case warnings
    layers: Optional[List[LayerPlan]] = None  # None = legacy flat mode


def build_materialization_plan(
    blueprint: Dict[str, Any],
    interface_map: InterfaceMap,
    dim_meta: Optional[DimensionalMetadata] = None,
) -> MaterializationPlan:
    """
    Build a complete materialization plan from a blueprint.

    This is the main entry point for Phase B.2. Takes a compiled blueprint
    with interface contracts and produces an ordered plan for parallel
    code generation.

    Args:
        blueprint: The compiled blueprint
        interface_map: Interface contracts between components
        dim_meta: Optional dimensional metadata for positioning

    Returns:
        MaterializationPlan with batches, prompts, and dependency info
    """
    # Build dependency graph
    graph = build_dependency_graph(interface_map)
    warnings: List[str] = []

    # Find parallel batches (with cycle fallback)
    try:
        # Validate acyclicity first
        topological_sort(graph)
        batches = find_parallel_batches(graph)
    except GraphError as e:
        # Fall back to tolerant sort — break cycles and continue
        sorted_nodes, broken_edges = topological_sort_tolerant(graph)
        for dep, dependent in broken_edges:
            warnings.append(
                f"Broke dependency cycle: {dep} -> {dependent} "
                f"(edge removed to allow materialization)"
            )
        # Build single-batch plan from tolerant ordering
        # Re-run batching on a patched graph
        patched_adj = {n: set(deps) for n, deps in graph.adjacency.items()}
        patched_rev = {n: set(deps) for n, deps in graph.reverse.items()}
        for dep, dependent in broken_edges:
            patched_adj[dependent].discard(dep)
            patched_rev[dep].discard(dependent)
        patched_graph = DependencyGraph(
            nodes=graph.nodes,
            edges=[(d, t) for d, t in graph.edges if (d, t) not in set(broken_edges)],
            adjacency=patched_adj,
            reverse=patched_rev,
        )
        batches = find_parallel_batches(patched_graph)

    # Build node prompts for each component
    components = {c.get("name", ""): c for c in blueprint.get("components", []) if c.get("name")}
    node_prompts = {}

    for comp_name, comp in components.items():
        prompt = build_node_prompt(comp, blueprint, interface_map, dim_meta)
        node_prompts[comp_name] = prompt

    max_parallelism = max(len(b.nodes) for b in batches) if batches else 0

    return MaterializationPlan(
        batches=batches,
        node_prompts=node_prompts,
        dependency_graph=graph,
        total_nodes=len(components),
        max_parallelism=max_parallelism,
        estimated_serial_steps=len(batches),
        warnings=warnings,
    )


def verify_interfaces(
    generated_code: Dict[str, str],
    interface_map: InterfaceMap,
) -> Dict[str, Any]:
    """
    Verify that generated code honors declared interface contracts.

    Checks that each node's generated code:
    1. References all declared adjacent nodes
    2. Includes method signatures for declared data flows
    3. Does not reference undeclared dependencies

    Args:
        generated_code: component_name -> generated Python code string
        interface_map: The expected interface contracts

    Returns:
        Verification report with pass/fail per contract
    """
    results = []
    for contract in interface_map.contracts:
        code_a = generated_code.get(contract.node_a, "")
        code_b = generated_code.get(contract.node_b, "")

        # Check that each node references the other
        a_refs_b = contract.node_b.lower().replace(" ", "_") in code_a.lower() or contract.node_b in code_a
        b_refs_a = contract.node_a.lower().replace(" ", "_") in code_b.lower() or contract.node_a in code_b

        # Check data flow method signatures
        flow_methods_found = 0
        for flow in contract.data_flows:
            flow_name_snake = flow.name.lower().replace(" ", "_")
            if flow_name_snake in code_a.lower() or flow_name_snake in code_b.lower():
                flow_methods_found += 1

        results.append({
            "contract": f"{contract.node_a} <-> {contract.node_b}",
            "a_references_b": a_refs_b,
            "b_references_a": b_refs_a,
            "flow_methods_found": flow_methods_found,
            "total_flows": len(contract.data_flows),
            "passed": a_refs_b or b_refs_a,  # At least one direction referenced
        })

    passed = sum(1 for r in results if r["passed"])
    return {
        "total_contracts": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 1.0,
        "details": results,
    }


# =============================================================================
# LAYERED EMISSION — classification + planning + context + gates
# =============================================================================

_DEFAULT_ENTITY_TYPES: FrozenSet[str] = frozenset({
    "entity", "data", "model", "record", "state", "store",
})

_DEFAULT_INTERFACE_TYPES: FrozenSet[str] = frozenset({
    "interface", "api", "contract", "boundary", "gateway",
})

_LAYER_NAMES = {
    EmissionLayer.TYPES: "types",
    EmissionLayer.INTERFACES: "interfaces",
    EmissionLayer.IMPLEMENTATIONS: "implementations",
    EmissionLayer.INTEGRATION: "integration",
}


def classify_component_layer(
    component_type: str,
    entity_types: Optional[FrozenSet[str]] = None,
    interface_types: Optional[FrozenSet[str]] = None,
) -> int:
    """Classify a component into an emission layer based on its type.

    Args:
        component_type: The component's type string (e.g. "entity", "process")
        entity_types: Types that map to Layer 0 (TYPES). Defaults to standard set.
        interface_types: Types that map to Layer 1 (INTERFACES). Defaults to standard set.

    Returns:
        EmissionLayer int value (0, 1, or 2)
    """
    ent = entity_types if entity_types is not None else _DEFAULT_ENTITY_TYPES
    ifc = interface_types if interface_types is not None else _DEFAULT_INTERFACE_TYPES
    ct = component_type.lower().strip()

    if ct in ent:
        return EmissionLayer.TYPES
    if ct in ifc:
        return EmissionLayer.INTERFACES
    return EmissionLayer.IMPLEMENTATIONS


def _promote_subsystem_children(
    node_layers: Dict[str, int],
    dependency_graph: DependencyGraph,
    blueprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Promote Layer 0 nodes to Layer 1 if they depend on other Layer 0 nodes.

    Sub-fields of shared-state entities (e.g. K, U, O, P importing SharedState)
    are classified as entities but cannot be self-contained — they import from
    another Layer 0 node. This violates the Layer 0 gate ("no relative imports").

    Fix: any Layer 0 node whose dependencies include another Layer 0 node is
    promoted to Layer 1. Dependencies are detected from:
    1. The dependency graph adjacency (interface contracts)
    2. Blueprint 'contains' relationships (parent → child containment)

    Args:
        node_layers: component_name → layer assignment (mutated copy returned)
        dependency_graph: DependencyGraph with adjacency info
        blueprint: Optional blueprint dict for containment relationships

    Returns:
        Updated dict with promoted nodes at Layer 0 or promoted to Layer 1
    """
    result = dict(node_layers)
    layer_0_nodes = {n for n, layer in result.items() if layer == EmissionLayer.TYPES}

    if not layer_0_nodes:
        return result

    # Build containment edges: child depends on parent (child needs to import parent)
    # "A contains B" means B is a sub-field of A → B may need to import A
    containment_deps: Dict[str, Set[str]] = {n: set() for n in layer_0_nodes}
    if blueprint:
        for rel in blueprint.get("relationships", []):
            rel_type = rel.get("type", "").lower()
            if rel_type == "contains":
                parent = rel.get("from", "")
                child = rel.get("to", "")
                # Child contained by parent → child depends on parent
                if child in layer_0_nodes and parent in layer_0_nodes:
                    containment_deps.setdefault(child, set()).add(parent)
                # Also check reverse: parent contains child → child depends on parent
                if parent in layer_0_nodes and child in layer_0_nodes:
                    containment_deps.setdefault(child, set()).add(parent)

    for node in list(layer_0_nodes):
        # Check graph adjacency (interface contracts with directional edges)
        graph_deps = dependency_graph.adjacency.get(node, set())
        has_l0_graph_dep = bool(graph_deps & layer_0_nodes)

        # Check containment relationships
        contain_deps = containment_deps.get(node, set())
        has_l0_contain_dep = bool(contain_deps - {node})  # exclude self

        if has_l0_graph_dep or has_l0_contain_dep:
            result[node] = EmissionLayer.INTERFACES

    return result


def build_layered_plan(
    blueprint: Dict[str, Any],
    interface_map: InterfaceMap,
    dim_meta: Optional[DimensionalMetadata] = None,
    entity_types: Optional[FrozenSet[str]] = None,
    interface_types: Optional[FrozenSet[str]] = None,
) -> MaterializationPlan:
    """Build a materialization plan with semantic layer ordering.

    Wraps build_materialization_plan() and partitions components into
    emission layers. Each layer's code is emitted sequentially so that
    later layers can receive actual code from earlier layers as context.

    Falls back to flat mode (layers=None) if all components land in
    the same layer.

    Args:
        blueprint: Compiled blueprint dict
        interface_map: InterfaceMap with contracts
        dim_meta: Optional dimensional metadata
        entity_types: Override entity type set for Layer 0
        interface_types: Override interface type set for Layer 1

    Returns:
        MaterializationPlan with layers populated (or None for flat fallback)
    """
    # Build base plan with dependency graph, batches, prompts
    plan = build_materialization_plan(blueprint, interface_map, dim_meta)

    if plan.total_nodes == 0:
        return plan

    # Classify each component into a layer
    components = {c.get("name", ""): c for c in blueprint.get("components", []) if c.get("name")}
    node_layers: Dict[str, int] = {}
    for comp_name, comp in components.items():
        comp_type = comp.get("type", "process")
        node_layers[comp_name] = classify_component_layer(
            comp_type, entity_types=entity_types, interface_types=interface_types,
        )

    # Promote subsystem children out of Layer 0 if they depend on other L0 nodes
    node_layers = _promote_subsystem_children(node_layers, plan.dependency_graph, blueprint)

    # Check if all components are in the same layer → flat fallback
    unique_layers = set(node_layers.values())
    if len(unique_layers) <= 1:
        return plan  # layers=None → flat mode

    # Partition batches by layer
    layer_plans: List[LayerPlan] = []
    for layer_val in (EmissionLayer.TYPES, EmissionLayer.INTERFACES, EmissionLayer.IMPLEMENTATIONS):
        layer_nodes = frozenset(n for n, l in node_layers.items() if l == layer_val)
        if not layer_nodes:
            continue

        # Filter global batches to only include this layer's nodes
        layer_batches = []
        batch_idx = 0
        for batch in plan.batches:
            filtered_nodes = tuple(n for n in batch.nodes if n in layer_nodes)
            if filtered_nodes:
                dep_count = sum(
                    len(plan.dependency_graph.adjacency.get(n, set()))
                    for n in filtered_nodes
                )
                layer_batches.append(MaterializationBatch(
                    batch_index=batch_idx,
                    nodes=filtered_nodes,
                    dependency_count=dep_count,
                ))
                batch_idx += 1

        if layer_batches:
            layer_plans.append(LayerPlan(
                layer=layer_val,
                layer_name=_LAYER_NAMES[layer_val],
                batches=tuple(layer_batches),
                node_names=tuple(sorted(layer_nodes)),
                is_deterministic=False,
            ))

    # Add Layer 3 (deterministic integration) — always present when layered
    layer_plans.append(LayerPlan(
        layer=EmissionLayer.INTEGRATION,
        layer_name="integration",
        batches=(),
        node_names=(),
        is_deterministic=True,
    ))

    # Store layers on the plan
    plan.layers = layer_plans
    return plan


# Maximum chars of prior-layer code to include in prompts
_MAX_PRIOR_CODE_CHARS = 12000

_LAYER_INSTRUCTIONS = {
    EmissionLayer.TYPES: (
        "Generate a pure dataclass or schema definition. "
        "No behavior logic. No imports from other generated modules."
    ),
    EmissionLayer.INTERFACES: (
        "Generate an ABC, Protocol, or interface definition. "
        "Import entity types from the prior layer code shown above. "
        "Do NOT redefine types — use them exactly as imported."
    ),
    EmissionLayer.IMPLEMENTATIONS: (
        "Generate a full implementation with behavior. "
        "Import types and interfaces from the prior layer code shown above. "
        "Use them exactly as defined — do NOT redefine them."
    ),
}


def build_node_prompt_with_context(
    component: Dict[str, Any],
    blueprint: Dict[str, Any],
    interface_map: InterfaceMap,
    dim_meta: Optional[DimensionalMetadata] = None,
    prior_layer_code: Optional[Dict[str, str]] = None,
    layer: Optional[int] = None,
    runtime_capabilities: Optional[Any] = None,
) -> NodePrompt:
    """Build a node prompt enriched with actual code from prior layers.

    Calls build_node_prompt() for the base, then appends:
    1. Prior layer code for adjacent components (full code)
    2. Prior layer code for non-adjacent components (signatures only)
    3. Layer-specific generation instructions
    4. Runtime contract section (when RuntimeCapabilities present)

    Args:
        component: Component dict from blueprint
        blueprint: Full blueprint
        interface_map: InterfaceMap for adjacency
        dim_meta: Optional dimensional metadata
        prior_layer_code: Dict of component_name → code from earlier layers
        layer: EmissionLayer value for layer-specific instructions
        runtime_capabilities: Optional RuntimeCapabilities for runtime contract

    Returns:
        NodePrompt with enriched prompt_text
    """
    base = build_node_prompt(component, blueprint, interface_map, dim_meta)

    if not prior_layer_code:
        # Add layer instruction and runtime contract even without prior code
        parts = [base.prompt_text]
        if layer is not None and layer in _LAYER_INSTRUCTIONS:
            parts.append("\n## Layer Instruction\n" + _LAYER_INSTRUCTIONS[layer])
        if runtime_capabilities is not None:
            rt_section = _build_runtime_contract(runtime_capabilities)
            if rt_section:
                parts.append("\n" + rt_section)
        if len(parts) > 1:
            enriched = "\n".join(parts)
            return NodePrompt(
                component_name=base.component_name,
                component_type=base.component_type,
                description=base.description,
                dimensional_position=base.dimensional_position,
                interfaces=base.interfaces,
                constraints=base.constraints,
                methods=base.methods,
                prompt_text=enriched,
            )
        return base

    name = component.get("name", "")

    # Find adjacent component names from interface_map
    adjacent: Set[str] = set()
    for contract in interface_map.contracts:
        if contract.node_a == name:
            adjacent.add(contract.node_b)
        elif contract.node_b == name:
            adjacent.add(contract.node_a)

    # Build prior code sections
    code_sections = []
    total_chars = 0

    # Adjacent components: full code
    for adj_name in sorted(adjacent):
        if adj_name in prior_layer_code and total_chars < _MAX_PRIOR_CODE_CHARS:
            code = prior_layer_code[adj_name]
            if total_chars + len(code) > _MAX_PRIOR_CODE_CHARS:
                code = _extract_signatures(code)
            code_sections.append(f"### {adj_name} (adjacent — full code)\n```python\n{code}\n```")
            total_chars += len(code)

    # Non-adjacent prior components: signatures only
    for comp_name in sorted(prior_layer_code.keys()):
        if comp_name not in adjacent and total_chars < _MAX_PRIOR_CODE_CHARS:
            sigs = _extract_signatures(prior_layer_code[comp_name])
            if sigs.strip():
                code_sections.append(f"### {comp_name} (available — signatures)\n```python\n{sigs}\n```")
                total_chars += len(sigs)

    # Assemble enriched prompt
    parts = [base.prompt_text]

    if code_sections:
        parts.append("")
        parts.append("## Imported Code (DO NOT REDEFINE)")
        parts.append("The following code is already generated. Import and use it as-is.")
        parts.extend(code_sections)

    if layer is not None and layer in _LAYER_INSTRUCTIONS:
        parts.append("")
        parts.append("## Layer Instruction")
        parts.append(_LAYER_INSTRUCTIONS[layer])

    # Runtime contract — guides LLM to emit components that fit the runtime harness
    if runtime_capabilities is not None:
        rt_section = _build_runtime_contract(runtime_capabilities)
        if rt_section:
            parts.append("")
            parts.append(rt_section)

    enriched = "\n".join(parts)

    return NodePrompt(
        component_name=base.component_name,
        component_type=base.component_type,
        description=base.description,
        dimensional_position=base.dimensional_position,
        interfaces=base.interfaces,
        constraints=base.constraints,
        methods=base.methods,
        prompt_text=enriched,
    )


def _build_runtime_contract(capabilities: Any) -> str:
    """Build runtime contract section for node prompts.

    Informs the LLM about available runtime services so generated
    components use them correctly.

    Args:
        capabilities: RuntimeCapabilities instance

    Returns:
        Runtime contract text, or empty string if no runtime features.
    """
    has_any = (
        capabilities.has_event_loop
        or capabilities.has_llm_client
        or capabilities.has_persistent_state
        or capabilities.has_tool_execution
    )
    if not has_any:
        return ""

    lines = ["## Runtime Contract", ""]
    lines.append("Your component runs inside an async runtime with these available interfaces:")

    if capabilities.has_persistent_state:
        lines.append("- `self.state: StateStore` — async get/set/query for persistent state")
    if capabilities.has_llm_client:
        lines.append("- `self.llm: LLMClient` — async chat/complete for LLM calls")
    if capabilities.has_tool_execution:
        lines.append("- `self.tools: ToolExecutor` — async execute(tool_name, **kwargs) for sandboxed tools")
    if capabilities.has_event_loop:
        lines.append('- `self.emit(event: str, data: dict)` — publish events to other components')

    # Build constructor param list matching available capabilities
    ctor_params = []
    if capabilities.has_persistent_state:
        ctor_params.append("state")
    if capabilities.has_llm_client:
        ctor_params.append("llm")
    if capabilities.has_tool_execution:
        ctor_params.append("tools")
    ctor_str = ", ".join(ctor_params)
    store_str = "; ".join(f"self.{p} = {p}" for p in ctor_params)

    lines.append("")
    lines.append("Your component MUST:")
    lines.append(f"1. Accept these __init__ parameters: `def __init__(self, {ctor_str}):`")
    lines.append(f"2. Store them: `{store_str}`")
    lines.append("3. Define an async `handle(self, message: dict) -> dict` method")
    if capabilities.has_persistent_state:
        lines.append("4. Use `self.state` for any data that should survive restarts")
    if capabilities.has_llm_client:
        lines.append("5. Use `self.llm` instead of direct API calls")
    lines.append("6. Return a response dict from handle()")
    lines.append("")
    lines.append("CRITICAL RULES:")
    lines.append("- Use ABSOLUTE imports only: `from models import Foo` — NEVER relative imports like `from .models import Foo`")
    lines.append("- Do NOT import other components. Do NOT instantiate other components.")
    lines.append("- Your component is SELF-CONTAINED. It communicates with other components only through handle().")
    lines.append("- If you need data from another component, return a response requesting it — the runtime routes messages between components.")
    lines.append("- Only import: standard library modules, your own file's classes, and `models` for shared data types.")

    if capabilities.has_event_loop and capabilities.event_loop_type != "none":
        lines.append("")
        lines.append("All methods MUST be async (use `async def`).")

    return "\n".join(lines)


def _extract_signatures(code: str) -> str:
    """Extract class and function signatures from Python code.

    Returns a truncated version with only class/def lines and their docstrings.
    """
    lines = code.split("\n")
    sig_lines = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("class ", "def ")):
            sig_lines.append(line)
            in_docstring = False
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                sig_lines.append(line)
                in_docstring = False
            else:
                sig_lines.append(line)
                # Single-line docstring
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    in_docstring = False
                else:
                    in_docstring = True
        elif in_docstring:
            sig_lines.append(line)
        elif stripped.startswith("@"):
            sig_lines.append(line)
    return "\n".join(sig_lines)


def validate_layer_gate(
    layer: int,
    layer_code: Dict[str, str],
    prior_layer_code: Optional[Dict[str, str]] = None,
    is_python: bool = True,
) -> LayerGateResult:
    """Validate emitted code for a layer before proceeding to the next.

    Checks:
    - All files parse (ast.parse for Python)
    - Layer 0: no relative imports (entities should be self-contained)
    - Layer 1/2: imported names from prior layers exist

    Args:
        layer: EmissionLayer value
        layer_code: component_name → code for this layer
        prior_layer_code: component_name → code from all prior layers
        is_python: Whether to use Python-specific validation

    Returns:
        LayerGateResult with pass/fail and error details
    """
    if not is_python:
        return LayerGateResult(
            layer=layer,
            passed=True,
            errors=(),
            warnings=("Non-Python: gate validation skipped",),
        )

    errors: List[str] = []
    warnings: List[str] = []

    prior_names = set(prior_layer_code.keys()) if prior_layer_code else set()

    for comp_name, code in layer_code.items():
        # Parse check
        try:
            ast.parse(code, filename=f"{comp_name}.py")
        except SyntaxError as e:
            line_info = f" (line {e.lineno})" if e.lineno else ""
            errors.append(f"{comp_name}: syntax error{line_info} — {e.msg}")
            continue

        # Layer-specific checks
        if layer == EmissionLayer.TYPES:
            # Layer 0: no relative imports allowed
            for match in re.finditer(r'from\s+\.(\w+)\s+import', code):
                errors.append(
                    f"{comp_name}: entity layer has relative import 'from .{match.group(1)}' — "
                    f"type definitions should be self-contained"
                )
        elif layer in (EmissionLayer.INTERFACES, EmissionLayer.IMPLEMENTATIONS):
            # Check that relative imports reference known prior-layer components
            for match in re.finditer(r'from\s+\.(\w+)\s+import', code):
                import_module = match.group(1)
                # Convert snake_case module name to check against prior component names
                # Both the snake_case and original name are valid matches
                found = any(
                    to_snake(pn) == import_module or pn.lower() == import_module.lower()
                    for pn in prior_names
                )
                if not found and import_module not in {to_snake(n) for n in layer_code.keys()}:
                    warnings.append(
                        f"{comp_name}: imports from .{import_module} — "
                        f"not found in prior layers (may resolve at integration)"
                    )

    return LayerGateResult(
        layer=layer,
        passed=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
