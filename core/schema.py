"""
Motherlabs Schema - Formal constraints for Blueprint structure.

Derived from: Engineering research - "Use AST-based representations to guarantee well-formed outputs"

This module defines the formal schema for all Motherlabs artifacts:
- Blueprint: The output specification structure
- Component: Individual spec components with required fields
- Relationship: Links between components
- Constraint: Rules that apply to components

These schemas serve as the "AST" for our semantic compiler,
constraining the output space and enabling verification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Literal, Tuple
from enum import Enum
import re


# =============================================================================
# BLUEPRINT VERSION - Phase 5.3: API Stabilization
# =============================================================================

BLUEPRINT_VERSION = "3.0"  # Phase A: Dimensional blueprint output


def add_version(blueprint: dict) -> dict:
    """
    Add version field to blueprint output.

    Phase 5.3: API Stabilization - blueprints now include version.
    Phase A: Version 3.0 includes dimensional metadata layer.

    Args:
        blueprint: The blueprint dict to version

    Returns:
        Blueprint with version field added
    """
    blueprint["version"] = BLUEPRINT_VERSION
    return blueprint


# =============================================================================
# COMPONENT TYPES
# Derived from: Dogfood blueprints - observed component type patterns
# =============================================================================

class ComponentType(Enum):
    """Valid component types in a blueprint."""
    ENTITY = "entity"          # Static structure (nouns)
    PROCESS = "process"        # Dynamic behavior (verbs)
    INTERFACE = "interface"    # Boundary/contract
    EVENT = "event"            # Trigger/signal
    CONSTRAINT = "constraint"  # Rule/invariant
    SUBSYSTEM = "subsystem"    # Contains nested blueprint


class RelationshipType(Enum):
    """Valid relationship types between components."""
    CONTAINS = "contains"           # Composition (A contains B)
    TRIGGERS = "triggers"           # Causation (A triggers B)
    ACCESSES = "accesses"           # Data flow (A accesses B)
    DEPENDS_ON = "depends_on"       # Dependency (A depends on B)
    FLOWS_TO = "flows_to"           # Sequential flow
    GENERATES = "generates"         # Production (A generates B)
    SNAPSHOTS = "snapshots"         # State capture
    PROPAGATES = "propagates"       # Information propagation
    CONSTRAINED_BY = "constrained_by"  # Rule application
    BIDIRECTIONAL = "bidirectional"    # Two-way relationship


# =============================================================================
# SCHEMA DATACLASSES
# =============================================================================

# -----------------------------------------------------------------------------
# Phase 2: Blueprint Depth - Method and State Machine extraction
# Derived from: ROADMAP.md Phase 2 - treat natural language as source code
# -----------------------------------------------------------------------------

@dataclass
class Parameter:
    """
    Method parameter with optional type hint.

    Derived from: Pattern matching on method signatures like 'add_known(name: str, value: Any)'
    """
    name: str
    type_hint: str = "Any"
    default: Optional[str] = None
    derived_from: str = ""


@dataclass
class MethodSpec:
    """
    Extracted method signature from natural language specification.

    Core principle: Extract only what is explicitly stated, never invent.
    Every MethodSpec MUST have a non-empty derived_from tracing back to input text.
    """
    name: str
    parameters: List["Parameter"] = field(default_factory=list)
    return_type: str = "None"
    description: str = ""
    derived_from: str = ""  # REQUIRED: exact text this was extracted from

    def validate(self) -> List[str]:
        """Return list of validation errors."""
        errors = []
        if not self.name:
            errors.append("MethodSpec missing name")
        if not self.derived_from:
            errors.append(f"MethodSpec '{self.name}' missing derived_from (violates excavation principle)")
        return errors


@dataclass
class Transition:
    """
    State machine transition for process components.

    Extracted from patterns like: INIT -> ACTIVE -> HALTED
    """
    from_state: str
    to_state: str
    trigger: str = ""  # Optional: what causes the transition
    derived_from: str = ""


@dataclass
class StateSpec:
    """
    State machine specification for process components.

    Extracted from natural language describing state transitions.
    Example: "DialogueProtocol: INIT -> ACTIVE -> CONVERGING -> HALTED"
    """
    states: List[str] = field(default_factory=list)
    initial_state: str = ""
    transitions: List[Transition] = field(default_factory=list)
    derived_from: str = ""

    def validate(self) -> List[str]:
        """Return list of validation errors including completeness checks."""
        errors = []
        if not self.states:
            errors.append("StateSpec has no states defined")
            return errors
        if self.initial_state and self.initial_state not in self.states:
            errors.append(f"StateSpec initial_state '{self.initial_state}' not in states list")
        if not self.initial_state:
            errors.append("StateSpec missing initial_state")
        if not self.derived_from:
            errors.append("StateSpec missing derived_from")

        # Completeness checks
        if self.states and self.transitions:
            # Check all transition states reference declared states
            for t in self.transitions:
                if t.from_state not in self.states:
                    errors.append(f"Transition from undeclared state '{t.from_state}'")
                if t.to_state not in self.states:
                    errors.append(f"Transition to undeclared state '{t.to_state}'")

            # Reachability: all states reachable from initial_state
            if self.initial_state:
                reachable = {self.initial_state}
                changed = True
                while changed:
                    changed = False
                    for t in self.transitions:
                        if t.from_state in reachable and t.to_state not in reachable:
                            reachable.add(t.to_state)
                            changed = True
                unreachable = set(self.states) - reachable
                for s in unreachable:
                    errors.append(f"State '{s}' unreachable from initial state '{self.initial_state}'")

            # Dead-end states: states with no outbound transitions (terminal states)
            # This is a warning, not an error — terminal states are valid
            states_with_outbound = {t.from_state for t in self.transitions}
            terminal_states = set(self.states) - states_with_outbound
            # At least one terminal state should exist (otherwise infinite loop)
            if not terminal_states and len(self.states) > 1:
                errors.append("No terminal states — state machine has no completion path")

        elif self.states and not self.transitions:
            errors.append("StateSpec has states but no transitions defined")

        return errors


# -----------------------------------------------------------------------------

@dataclass
class ComponentSchema:
    """
    Schema for a blueprint component.

    Required fields enforce traceability (C005: Derivation Complete).
    For subsystem components, sub_blueprint contains nested blueprint structure.

    Phase 2 additions:
    - methods: Extracted method signatures (optional)
    - state_machine: State transitions for process components (optional)
    - validation_rules: Formalized constraints (optional)
    """
    name: str                                    # Unique identifier
    type: ComponentType                          # Must be valid type
    description: str                             # What it is/does
    derived_from: str                            # REQUIRED: Source in input/dialogue
    attributes: Dict[str, Any] = field(default_factory=dict)  # Optional properties
    sub_blueprint: Optional["BlueprintSchema"] = None  # Nested blueprint for subsystems
    # Phase 2: Blueprint Depth fields
    methods: List[MethodSpec] = field(default_factory=list)  # Extracted method signatures
    state_machine: Optional[StateSpec] = None  # State transitions for processes
    validation_rules: List[str] = field(default_factory=list)  # Formalized constraints

    def validate(self) -> List[str]:
        """Return list of validation errors."""
        errors = []
        if not self.name:
            errors.append("Component name is required")
        if not self.description:
            errors.append(f"Component '{self.name}' missing description")
        if not self.derived_from:
            errors.append(f"Component '{self.name}' missing derived_from (violates C005)")
        if len(self.derived_from) < 10:
            errors.append(f"Component '{self.name}' has weak derivation: '{self.derived_from}'")

        # Validate sub_blueprint if present (recursive)
        if self.sub_blueprint:
            if self.type != ComponentType.SUBSYSTEM:
                errors.append(f"Component '{self.name}' has sub_blueprint but type is not SUBSYSTEM")
            sub_result = self.sub_blueprint.validate()
            for err in sub_result.get("errors", []):
                errors.append(f"{self.name}.{err}")

        # Phase 2: Validate methods
        for method in self.methods:
            method_errors = method.validate()
            for err in method_errors:
                errors.append(f"{self.name}.{err}")

        # Phase 2: Validate state_machine
        if self.state_machine:
            state_errors = self.state_machine.validate()
            for err in state_errors:
                errors.append(f"{self.name}.{err}")

        return errors


def _fuzzy_match_component(name: str, component_names: Set[str]) -> bool:
    """
    Check if a component name matches any known component.

    Handles:
    - Exact match (case-insensitive)
    - Plural/singular ('Messages' -> 'Message')
    - Dotted notation ('SharedState.History' -> 'SharedState')
    - Substring match where the name is a significant part of a component name

    Does NOT accept:
    - Generic names like 'system', 'all', 'components' (these hide real errors)
    - Broad substring matches like 'agent' matching any agent
    """
    if not name or not name.strip():
        return False

    name_lower = name.lower().strip()

    # Exact match
    if name in component_names:
        return True

    # Case-insensitive match
    component_names_lower = {c.lower() for c in component_names}
    if name_lower in component_names_lower:
        return True

    # Plural/singular match (Messages -> Message)
    if name_lower.endswith('s') and name_lower[:-1] in component_names_lower:
        return True
    if name_lower + 's' in component_names_lower:
        return True

    # Dotted notation (SharedState.History -> SharedState or History)
    if '.' in name:
        parts = name.split('.')
        for part in parts:
            if part.lower() in component_names_lower:
                return True

    # Substring: name is a significant suffix/prefix of a real component
    # e.g. "Booking" matches "BookingService", "UserBooking"
    # Only if the name is >= 4 chars (avoid matching "a", "id", etc.)
    if len(name_lower) >= 4:
        for comp_lower in component_names_lower:
            # name is a word-boundary substring of a component name
            if comp_lower.startswith(name_lower) or comp_lower.endswith(name_lower):
                return True
            # component name is a word-boundary substring of the reference
            if name_lower.startswith(comp_lower) or name_lower.endswith(comp_lower):
                return True

    return False


def _infer_cardinality(rel: 'RelationshipSchema') -> str:
    """Infer cardinality from relationship type when not explicitly set.

    Rules:
    - contains: 1:N (a container holds many items)
    - triggers/generates/propagates: 1:N (one source, many targets possible)
    - accesses/depends_on: N:1 (many consumers, one provider)
    - flows_to/bidirectional: 1:1 (default for sequential/symmetric)
    - snapshots/constrained_by: 1:1
    """
    type_map = {
        "contains": "1:N",
        "triggers": "1:N",
        "generates": "1:N",
        "propagates": "1:N",
        "accesses": "N:1",
        "depends_on": "N:1",
        "flows_to": "1:1",
        "bidirectional": "1:1",
        "snapshots": "1:1",
        "constrained_by": "1:1",
    }
    rel_type = rel.type.value if hasattr(rel.type, 'value') else str(rel.type)
    return type_map.get(rel_type, "1:1")


@dataclass
class RelationshipSchema:
    """
    Schema for a relationship between components.
    """
    from_component: str                          # Source component name
    to_component: str                            # Target component name
    type: RelationshipType                       # Must be valid type
    description: str                             # Nature of relationship
    derived_from: str = ""                       # Optional derivation
    cardinality: str = ""                        # "1:1", "1:N", "N:1", "N:M" or "" if unspecified

    def validate(self, component_names: Set[str]) -> List[str]:
        """Return list of validation errors with fuzzy matching."""
        errors = []
        warnings = []

        if not _fuzzy_match_component(self.from_component, component_names):
            errors.append(f"Relationship references unknown component: '{self.from_component}'")
        if not _fuzzy_match_component(self.to_component, component_names):
            errors.append(f"Relationship references unknown component: '{self.to_component}'")
        if not self.description:
            errors.append(f"Relationship {self.from_component}->{self.to_component} missing description")
        return errors


@dataclass
class ConstraintSchema:
    """
    Schema for a blueprint constraint.
    """
    description: str                             # The constraint rule
    applies_to: List[str]                        # Component names this applies to
    derived_from: str                            # Source in input/dialogue

    def validate(self, component_names: Set[str]) -> List[str]:
        """Return list of validation errors with fuzzy matching."""
        errors = []
        if not self.description:
            errors.append("Constraint missing description")
        if not self.applies_to:
            errors.append("Constraint must apply to at least one component")
        for comp in self.applies_to:
            if not _fuzzy_match_component(comp, component_names):
                errors.append(f"Constraint references unknown component: '{comp}'")
        if not self.derived_from:
            errors.append(f"Constraint missing derived_from: '{self.description[:50]}...'")
        return errors


@dataclass
class BlueprintSchema:
    """
    Formal schema for a Motherlabs blueprint.

    This is the "AST" of our semantic compiler - constraining
    the output space to guarantee well-formed specifications.

    Derived from: Engineering research on AST-based constraints
    """
    components: List[ComponentSchema] = field(default_factory=list)
    relationships: List[RelationshipSchema] = field(default_factory=list)
    constraints: List[ConstraintSchema] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    semantic_gates: List[Dict[str, Any]] = field(default_factory=list)
    semantic_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> Dict[str, Any]:
        """
        Validate blueprint against schema.

        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "stats": {component_count, relationship_count, etc.}
            }
        """
        errors = []
        warnings = []

        # Collect component names
        component_names = {c.name for c in self.components}

        # Validate components
        for comp in self.components:
            errors.extend(comp.validate())

        # Check for duplicate component names
        names_seen = set()
        for comp in self.components:
            if comp.name in names_seen:
                errors.append(f"Duplicate component name: '{comp.name}'")
            names_seen.add(comp.name)

        # Validate relationships
        for rel in self.relationships:
            errors.extend(rel.validate(component_names))

        # Validate constraints
        for const in self.constraints:
            errors.extend(const.validate(component_names))

        # Warnings for potential issues
        if len(self.components) < 3:
            warnings.append("Blueprint has fewer than 3 components - may be incomplete")

        if len(self.unresolved) > len(self.components) // 2:
            warnings.append("Many unresolved items - consider re-dialogue")

        # Check for orphan components (no relationships)
        components_in_relationships = set()
        for rel in self.relationships:
            components_in_relationships.add(rel.from_component)
            components_in_relationships.add(rel.to_component)

        orphans = component_names - components_in_relationships
        for orphan in orphans:
            warnings.append(f"Orphan component (no relationships): '{orphan}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "component_count": len(self.components),
                "relationship_count": len(self.relationships),
                "constraint_count": len(self.constraints),
                "unresolved_count": len(self.unresolved),
                "orphan_count": len(orphans)
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlueprintSchema":
        """
        Parse a blueprint dict into validated schema.

        Converts raw JSON blueprint to typed schema objects.
        """
        components = []
        for c in data.get("components", []):
            try:
                comp_type = ComponentType(c.get("type", "entity"))
            except ValueError:
                comp_type = ComponentType.ENTITY  # Default

            # Recursively parse sub_blueprint for subsystem components
            sub_bp = None
            if c.get("sub_blueprint"):
                sub_bp = cls.from_dict(c["sub_blueprint"])

            # Phase 2: Parse methods
            methods = []
            for m in c.get("methods", []):
                if isinstance(m, str):
                    # LLM sometimes returns method names as bare strings
                    methods.append(MethodSpec(
                        name=m, parameters=[], return_type="None",
                        description="", derived_from="",
                    ))
                    continue
                raw_params = m.get("parameters", []) if isinstance(m, dict) else []
                params = []
                for p in raw_params:
                    if isinstance(p, dict):
                        params.append(Parameter(
                            name=p.get("name", ""),
                            type_hint=p.get("type_hint", "Any"),
                            default=p.get("default"),
                            derived_from=p.get("derived_from", ""),
                        ))
                    elif isinstance(p, str):
                        params.append(Parameter(name=p, type_hint="Any", default=None, derived_from=""))
                methods.append(MethodSpec(
                    name=m.get("name", ""),
                    parameters=params,
                    return_type=m.get("return_type", "None"),
                    description=m.get("description", ""),
                    derived_from=m.get("derived_from", "")
                ))

            # Phase 2: Parse state_machine
            state_machine = None
            if c.get("state_machine"):
                sm = c["state_machine"]
                if isinstance(sm, dict):
                    raw_transitions = sm.get("transitions", [])
                else:
                    raw_transitions = []
                    sm = {}
                transitions = []
                for t in raw_transitions:
                    if isinstance(t, dict):
                        transitions.append(Transition(
                            from_state=t.get("from_state", ""),
                            to_state=t.get("to_state", ""),
                            trigger=t.get("trigger", ""),
                            derived_from=t.get("derived_from", ""),
                        ))
                state_machine = StateSpec(
                    states=sm.get("states", []),
                    initial_state=sm.get("initial_state", ""),
                    transitions=transitions,
                    derived_from=sm.get("derived_from", "")
                )

            components.append(ComponentSchema(
                name=c.get("name", ""),
                type=comp_type,
                description=c.get("description", ""),
                derived_from=c.get("derived_from", ""),
                attributes=c.get("attributes", {}),
                sub_blueprint=sub_bp,
                methods=methods,
                state_machine=state_machine,
                validation_rules=c.get("validation_rules", [])
            ))

        relationships = []
        for r in data.get("relationships", []):
            try:
                rel_type = RelationshipType(r.get("type", "depends_on"))
            except ValueError:
                rel_type = RelationshipType.DEPENDS_ON  # Default

            relationships.append(RelationshipSchema(
                from_component=r.get("from", "") or r.get("from_component", ""),
                to_component=r.get("to", "") or r.get("to_component", ""),
                type=rel_type,
                description=r.get("description", ""),
                derived_from=r.get("derived_from", ""),
                cardinality=r.get("cardinality", ""),
            ))

        constraints = []
        for c in data.get("constraints", []):
            constraints.append(ConstraintSchema(
                description=c.get("description", ""),
                applies_to=c.get("applies_to", []),
                derived_from=c.get("derived_from", "")
            ))

        return cls(
            components=components,
            relationships=relationships,
            constraints=constraints,
            unresolved=data.get("unresolved", []),
            semantic_gates=data.get("semantic_gates", []),
            semantic_nodes=data.get("semantic_nodes", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export to JSON-serializable dict."""
        components_list = []
        for c in self.components:
            comp_dict = {
                "name": c.name,
                "type": c.type.value,
                "description": c.description,
                "derived_from": c.derived_from,
                "attributes": c.attributes,
                # Phase 2: Include methods if present
                "methods": [
                    {
                        "name": m.name,
                        "parameters": [
                            {
                                "name": p.name,
                                "type_hint": p.type_hint,
                                "default": p.default,
                                "derived_from": p.derived_from
                            }
                            for p in m.parameters
                        ],
                        "return_type": m.return_type,
                        "description": m.description,
                        "derived_from": m.derived_from
                    }
                    for m in c.methods
                ] if c.methods else [],
                # Phase 2: Include state_machine if present
                "state_machine": {
                    "states": c.state_machine.states,
                    "initial_state": c.state_machine.initial_state,
                    "transitions": [
                        {
                            "from_state": t.from_state,
                            "to_state": t.to_state,
                            "trigger": t.trigger,
                            "derived_from": t.derived_from
                        }
                        for t in c.state_machine.transitions
                    ],
                    "derived_from": c.state_machine.derived_from
                } if c.state_machine else None,
                # Phase 2: Include validation_rules if present
                "validation_rules": c.validation_rules if c.validation_rules else []
            }
            # Include sub_blueprint if present
            if c.sub_blueprint:
                comp_dict["sub_blueprint"] = c.sub_blueprint.to_dict()
            components_list.append(comp_dict)

        return {
            "components": components_list,
            "relationships": [
                {
                    "from": r.from_component,
                    "to": r.to_component,
                    "type": r.type.value,
                    "description": r.description,
                    "cardinality": r.cardinality or _infer_cardinality(r),
                }
                for r in self.relationships
            ],
            "constraints": [
                {
                    "description": c.description,
                    "applies_to": c.applies_to,
                    "derived_from": c.derived_from
                }
                for c in self.constraints
            ],
            "unresolved": self.unresolved,
            "semantic_gates": self.semantic_gates,
            "semantic_nodes": self.semantic_nodes,
        }


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def validate_blueprint(blueprint_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a blueprint dict against the formal schema.

    This is the main entry point for blueprint validation.

    Args:
        blueprint_dict: Raw blueprint JSON

    Returns:
        Validation result with errors, warnings, and stats
    """
    schema = BlueprintSchema.from_dict(blueprint_dict)
    return schema.validate()


# =============================================================================
# CANONICAL COMPONENTS
# These MUST be present in any valid Motherlabs self-compile blueprint
# Derived from: Self-description in engine.py
# =============================================================================

CANONICAL_COMPONENTS = {
    # Agents (required)
    "Intent Agent": "agent",
    "Persona Agent": "agent",
    "Entity Agent": "agent",
    "Process Agent": "agent",
    "Synthesis Agent": "agent",
    "Verify Agent": "agent",
    "Governor Agent": "agent",
    # Core entities (required)
    "SharedState": "entity",
    "ConfidenceVector": "entity",
    "ConflictOracle": "entity",
    "Message": "entity",
    "DialogueProtocol": "entity",
    "Corpus": "entity",
}

# Aliases for canonical components (models may use different names)
COMPONENT_ALIASES = {
    "intent": "Intent Agent",
    "intent agent": "Intent Agent",
    "intentagent": "Intent Agent",
    "persona": "Persona Agent",
    "persona agent": "Persona Agent",
    "personaagent": "Persona Agent",
    "entity": "Entity Agent",
    "entity agent": "Entity Agent",
    "entityagent": "Entity Agent",
    "process": "Process Agent",
    "process agent": "Process Agent",
    "processagent": "Process Agent",
    "synthesis": "Synthesis Agent",
    "synthesis agent": "Synthesis Agent",
    "synthesisagent": "Synthesis Agent",
    "verify": "Verify Agent",
    "verify agent": "Verify Agent",
    "verifyagent": "Verify Agent",
    "governor": "Governor Agent",
    "governor agent": "Governor Agent",
    "governoragent": "Governor Agent",
    "sharedstate": "SharedState",
    "shared state": "SharedState",
    "shared_state": "SharedState",
    "confidencevector": "ConfidenceVector",
    "confidence vector": "ConfidenceVector",
    "confidence_vector": "ConfidenceVector",
    "confidence": "ConfidenceVector",
    "conflictoracle": "ConflictOracle",
    "conflict oracle": "ConflictOracle",
    "conflict_oracle": "ConflictOracle",
    "dialogueprotocol": "DialogueProtocol",
    "dialogue protocol": "DialogueProtocol",
    "dialogue_protocol": "DialogueProtocol",
}

# =============================================================================
# CANONICAL RELATIONSHIPS
# Required relationships between canonical components
# Derived from: Self-description pipeline and architecture
# =============================================================================

CANONICAL_RELATIONSHIPS = [
    # Governor triggers all agents in sequence
    ("Governor Agent", "Intent Agent", "triggers"),
    ("Governor Agent", "Persona Agent", "triggers"),
    ("Governor Agent", "Entity Agent", "triggers"),
    ("Governor Agent", "Process Agent", "triggers"),
    ("Governor Agent", "Synthesis Agent", "triggers"),
    ("Governor Agent", "Verify Agent", "triggers"),

    # All agents access SharedState
    ("Intent Agent", "SharedState", "accesses"),
    ("Persona Agent", "SharedState", "accesses"),
    ("Entity Agent", "SharedState", "accesses"),
    ("Process Agent", "SharedState", "accesses"),
    ("Synthesis Agent", "SharedState", "accesses"),
    ("Verify Agent", "SharedState", "accesses"),

    # ConflictOracle monitors ConfidenceVector
    ("ConflictOracle", "ConfidenceVector", "monitors"),

    # ConflictOracle triggers Governor on issues
    ("ConflictOracle", "Governor Agent", "triggers"),

    # DialogueProtocol constrains dialogue
    ("DialogueProtocol", "Entity Agent", "constrains"),
    ("DialogueProtocol", "Process Agent", "constrains"),

    # Corpus stores compilations
    ("Corpus", "SharedState", "snapshots"),
]


def check_canonical_relationships(
    blueprint: Dict[str, Any],
    canonical_relationships: List[tuple] = None
) -> Dict[str, Any]:
    """
    Check how many canonical relationships are present in a blueprint.

    Uses fuzzy matching for component names and relationship types.

    Args:
        blueprint: The blueprint to check
        canonical_relationships: Custom list of required relationships (default: CANONICAL_RELATIONSHIPS)

    Returns:
        {
            "coverage": float (0-1),
            "found": List[tuple],
            "missing": List[tuple],
            "extra_count": int
        }
    """
    required_rels = canonical_relationships or CANONICAL_RELATIONSHIPS

    # Build set of relationships in blueprint (normalized)
    bp_rels = set()
    for r in blueprint.get("relationships", []):
        from_comp = normalize_component_name(r.get("from", ""))
        to_comp = normalize_component_name(r.get("to", ""))
        rel_type = r.get("type", "").lower()
        bp_rels.add((from_comp, to_comp, rel_type))

        # Also add with generic type matching
        bp_rels.add((from_comp, to_comp, "*"))

    found = []
    missing = []

    for (from_c, to_c, rel_type) in required_rels:
        # Check exact match or wildcard match
        if (from_c, to_c, rel_type) in bp_rels or (from_c, to_c, "*") in bp_rels:
            found.append((from_c, to_c, rel_type))
        else:
            # Check reverse direction for bidirectional relationships
            if (to_c, from_c, rel_type) in bp_rels or (to_c, from_c, "*") in bp_rels:
                found.append((from_c, to_c, rel_type))
            else:
                missing.append((from_c, to_c, rel_type))

    total = len(required_rels)
    return {
        "coverage": len(found) / total if total else 1.0,
        "found": found,
        "missing": missing,
        "found_count": len(found),
        "missing_count": len(missing),
        "total_required": total
    }


def normalize_component_name(name: str) -> str:
    """Normalize a component name to its canonical form."""
    lower = name.lower().strip()
    return COMPONENT_ALIASES.get(lower, name)


def check_canonical_coverage(
    blueprint: Dict[str, Any],
    canonical_components: List[str] = None
) -> Dict[str, Any]:
    """
    Check how many canonical components are present in a blueprint.

    Args:
        blueprint: The blueprint to check
        canonical_components: Custom list of required components (default: CANONICAL_COMPONENTS keys)

    Returns:
        {
            "coverage": float (0-1),
            "found": List[str],
            "missing": List[str],
            "extra": List[str]  # Components not in canonical set
        }
    """
    bp_components = set()
    bp_names_raw = []  # Keep raw names for better matching
    for c in blueprint.get("components", []):
        name = c.get("name", "")
        bp_names_raw.append(name)
        normalized = normalize_component_name(name)
        bp_components.add(normalized)
        # Also add the original name lowercased for matching
        bp_components.add(name.lower().strip())

    # Use custom canonical set or default
    if canonical_components:
        canonical_set = set(canonical_components)
    else:
        canonical_set = set(CANONICAL_COMPONENTS.keys())

    # Find matches with fuzzy matching
    found = []
    for canon in canonical_set:
        canon_lower = canon.lower().strip()
        # Check normalized match
        if canon in bp_components or canon_lower in bp_components:
            found.append(canon)
        # Check if any blueprint component contains the canonical name
        elif any(canon_lower in raw.lower() for raw in bp_names_raw):
            found.append(canon)
        # Check if canonical name contains any blueprint component
        elif any(raw.lower() in canon_lower for raw in bp_names_raw if len(raw) > 3):
            found.append(canon)

    found_set = set(found)
    missing = canonical_set - found_set
    extra = set(bp_names_raw) - canonical_set

    return {
        "coverage": len(found) / len(canonical_set) if canonical_set else 1.0,
        "found": sorted(list(found)),
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra))
    }


def compare_blueprints(bp1: Dict[str, Any], bp2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two blueprints for DDC-style verification.

    Derived from: Engineering research - "Diverse Double-Compiling"

    Used to verify that different providers produce equivalent outputs.

    Args:
        bp1: First blueprint
        bp2: Second blueprint

    Returns:
        {
            "equivalent": bool,
            "component_overlap": float (0-1),
            "missing_in_bp1": List[str],
            "missing_in_bp2": List[str],
            "type_mismatches": List[str]
        }
    """
    # Extract component names and types
    components1 = {c["name"]: c.get("type", "") for c in bp1.get("components", [])}
    components2 = {c["name"]: c.get("type", "") for c in bp2.get("components", [])}

    names1 = set(components1.keys())
    names2 = set(components2.keys())

    # Calculate overlap
    common = names1 & names2
    all_names = names1 | names2
    overlap = len(common) / len(all_names) if all_names else 1.0

    # Find differences
    missing_in_bp1 = list(names2 - names1)
    missing_in_bp2 = list(names1 - names2)

    # Check type mismatches for common components
    type_mismatches = []
    for name in common:
        if components1[name] != components2[name]:
            type_mismatches.append(f"{name}: {components1[name]} vs {components2[name]}")

    # Check relationship coverage
    rels1 = {(r["from"], r["to"]) for r in bp1.get("relationships", [])}
    rels2 = {(r["from"], r["to"]) for r in bp2.get("relationships", [])}
    rel_overlap = len(rels1 & rels2) / len(rels1 | rels2) if (rels1 | rels2) else 1.0

    return {
        "equivalent": overlap >= 0.8 and len(type_mismatches) == 0,
        "component_overlap": overlap,
        "relationship_overlap": rel_overlap,
        "missing_in_bp1": missing_in_bp1,
        "missing_in_bp2": missing_in_bp2,
        "type_mismatches": type_mismatches,
        "common_components": len(common),
        "total_unique_components": len(all_names)
    }


# =============================================================================
# GRAPH VALIDATION
# Derived from: NEXT-STEPS.md "Relationship Graph Validation"
# =============================================================================

def _build_graph(blueprint: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build adjacency list from blueprint relationships.

    Returns:
        Dict mapping component -> set of components it connects to
    """
    graph = {}

    # Initialize all components
    for c in blueprint.get("components", []):
        name = c.get("name", "")
        if name:
            graph[name] = set()

    # Add edges from relationships
    for r in blueprint.get("relationships", []):
        from_comp = r.get("from", "")
        to_comp = r.get("to", "")
        if from_comp and to_comp:
            if from_comp not in graph:
                graph[from_comp] = set()
            graph[from_comp].add(to_comp)

    return graph


def _build_undirected_graph(blueprint: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build undirected adjacency list from blueprint relationships.

    For reachability, we treat all relationships as bidirectional.
    """
    graph = {}

    # Initialize all components
    for c in blueprint.get("components", []):
        name = c.get("name", "")
        if name:
            graph[name] = set()

    # Add bidirectional edges
    for r in blueprint.get("relationships", []):
        from_comp = r.get("from", "")
        to_comp = r.get("to", "")
        if from_comp and to_comp:
            if from_comp not in graph:
                graph[from_comp] = set()
            if to_comp not in graph:
                graph[to_comp] = set()
            graph[from_comp].add(to_comp)
            graph[to_comp].add(from_comp)

    return graph


def check_reachability(
    blueprint: Dict[str, Any],
    root_component: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check if all components are reachable from a root component.

    Uses BFS to find all reachable components from root.
    If no root specified, uses the first component or "Governor Agent".

    Args:
        blueprint: The blueprint to check
        root_component: Starting component (default: auto-detect)

    Returns:
        {
            "all_reachable": bool,
            "reachable": List[str],
            "unreachable": List[str],
            "root": str,
            "coverage": float (0-1)
        }
    """
    graph = _build_undirected_graph(blueprint)

    if not graph:
        return {
            "all_reachable": True,
            "reachable": [],
            "unreachable": [],
            "root": None,
            "coverage": 1.0
        }

    # Determine root
    if root_component and root_component in graph:
        root = root_component
    elif "Governor Agent" in graph:
        root = "Governor Agent"
    elif "Governor" in graph:
        root = "Governor"
    else:
        # Use first component
        root = next(iter(graph.keys()))

    # BFS from root
    visited = set()
    queue = [root]
    visited.add(root)

    while queue:
        current = queue.pop(0)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    all_components = set(graph.keys())
    unreachable = all_components - visited

    return {
        "all_reachable": len(unreachable) == 0,
        "reachable": sorted(list(visited)),
        "unreachable": sorted(list(unreachable)),
        "root": root,
        "coverage": len(visited) / len(all_components) if all_components else 1.0
    }


def find_orphan_components(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find components with no relationships (orphans).

    An orphan is a component that:
    - Is not the source of any relationship
    - Is not the target of any relationship

    Args:
        blueprint: The blueprint to check

    Returns:
        {
            "has_orphans": bool,
            "orphans": List[str],
            "connected": List[str],
            "orphan_ratio": float (0-1)
        }
    """
    all_components = {c.get("name") for c in blueprint.get("components", []) if c.get("name")}
    connected = set()

    for r in blueprint.get("relationships", []):
        from_comp = r.get("from", "")
        to_comp = r.get("to", "")
        if from_comp:
            connected.add(from_comp)
        if to_comp:
            connected.add(to_comp)

    orphans = all_components - connected

    return {
        "has_orphans": len(orphans) > 0,
        "orphans": sorted(list(orphans)),
        "connected": sorted(list(connected)),
        "orphan_ratio": len(orphans) / len(all_components) if all_components else 0.0
    }


def validate_relationship_types(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that relationships use recognized types.

    Args:
        blueprint: The blueprint to check

    Returns:
        {
            "all_valid": bool,
            "valid_types": List[str],
            "invalid_types": List[tuple],  # (from, to, invalid_type)
            "type_distribution": Dict[str, int]
        }
    """
    valid_types = {rt.value for rt in RelationshipType}
    # Also accept common aliases
    valid_types.update({"monitors", "uses", "reads", "writes", "calls", "creates", "updates"})

    invalid = []
    type_counts = {}

    for r in blueprint.get("relationships", []):
        rel_type = r.get("type", "").lower()
        from_comp = r.get("from", "")
        to_comp = r.get("to", "")

        type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        if rel_type and rel_type not in valid_types:
            invalid.append((from_comp, to_comp, rel_type))

    return {
        "all_valid": len(invalid) == 0,
        "valid_types": sorted(list(valid_types)),
        "invalid_types": invalid,
        "type_distribution": type_counts
    }


def detect_cycles(
    blueprint: Dict[str, Any],
    cycle_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect cycles in the relationship graph.

    Only considers relationships of specified types for cycle detection.
    By default, checks "depends_on" relationships for dependency cycles.

    Args:
        blueprint: The blueprint to check
        cycle_types: Relationship types to check for cycles (default: ["depends_on"])

    Returns:
        {
            "has_cycles": bool,
            "cycles": List[List[str]],  # List of component cycles found
            "cycle_count": int
        }
    """
    if cycle_types is None:
        cycle_types = ["depends_on", "triggers", "flows_to"]

    # Build directed graph for specified relationship types
    graph = {}
    for c in blueprint.get("components", []):
        name = c.get("name", "")
        if name:
            graph[name] = set()

    for r in blueprint.get("relationships", []):
        rel_type = r.get("type", "").lower()
        if rel_type in cycle_types:
            from_comp = r.get("from", "")
            to_comp = r.get("to", "")
            if from_comp and to_comp:
                if from_comp not in graph:
                    graph[from_comp] = set()
                graph[from_comp].add(to_comp)

    # Detect cycles using DFS with coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    cycles = []

    def dfs(node: str, path: List[str]) -> bool:
        """Return True if cycle found."""
        color[node] = GRAY
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                # Found cycle - extract it from path
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                return True
            elif color[neighbor] == WHITE:
                if dfs(neighbor, path):
                    return True

        path.pop()
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            dfs(node, [])

    return {
        "has_cycles": len(cycles) > 0,
        "cycles": cycles,
        "cycle_count": len(cycles)
    }


def validate_graph(blueprint: Dict[str, Any], root: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive graph validation combining all checks.

    Derived from: NEXT-STEPS.md "Relationship Graph Validation"

    Args:
        blueprint: The blueprint to validate
        root: Optional root component for reachability check

    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "reachability": Dict,
            "orphans": Dict,
            "relationship_types": Dict,
            "cycles": Dict
        }
    """
    errors = []
    warnings = []

    # 1. Reachability check
    reachability = check_reachability(blueprint, root)
    if not reachability["all_reachable"]:
        for comp in reachability["unreachable"]:
            warnings.append(f"Component '{comp}' is not reachable from root '{reachability['root']}'")

    # 2. Orphan detection
    orphans = find_orphan_components(blueprint)
    if orphans["has_orphans"]:
        for comp in orphans["orphans"]:
            warnings.append(f"Orphan component (no relationships): '{comp}'")

    # 3. Relationship type validation
    rel_types = validate_relationship_types(blueprint)
    if not rel_types["all_valid"]:
        for (from_c, to_c, invalid_type) in rel_types["invalid_types"]:
            warnings.append(f"Unknown relationship type '{invalid_type}' in {from_c}->{to_c}")

    # 4. Cycle detection
    cycles = detect_cycles(blueprint)
    if cycles["has_cycles"]:
        for cycle in cycles["cycles"][:3]:  # Report up to 3 cycles
            cycle_str = " -> ".join(cycle)
            errors.append(f"Dependency cycle detected: {cycle_str}")

    # Determine overall validity
    # Errors: cycles in dependencies (can cause issues)
    # Warnings: orphans, unreachable, unknown types (may be intentional)
    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "reachability": reachability,
        "orphans": orphans,
        "relationship_types": rel_types,
        "cycles": cycles
    }


# =============================================================================
# CONSTRAINT PARSING AND EXECUTION
# Phase 6.3: Generate runnable Python validators from constraints
# Derived from: Plan Phase 6.3 - Constraint Execution
# =============================================================================

class ConstraintType(Enum):
    """Types of formal constraints that can be parsed and validated."""
    RANGE = "range"              # Value must be in [min, max]
    NOT_NULL = "not_null"        # Value must not be None
    NOT_EMPTY = "not_empty"      # Value must not be empty (string, list, dict)
    REGEX = "regex"              # Value must match regex pattern
    TYPE = "type"                # Value must be specific type
    ENUM = "enum"                # Value must be one of allowed values
    LENGTH = "length"            # Value length must be in range
    POSITIVE = "positive"        # Number must be > 0
    NON_NEGATIVE = "non_negative"  # Number must be >= 0
    UNIQUE = "unique"            # Values must be unique
    CUSTOM = "custom"            # Arbitrary Python expression


@dataclass
class FormalConstraint:
    """
    A structured, machine-readable constraint.

    Phase 6.3: Enables code generation of validators from constraints.

    Example:
        FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="confidence",
            params={"min": 0.0, "max": 1.0},
            applies_to=["Message", "ConfidenceVector"],
            description="Confidence must be between 0 and 1",
            derived_from="confidence: float in range [0, 1]"
        )
    """
    constraint_type: ConstraintType
    field: str
    params: Dict[str, Any] = field(default_factory=dict)
    applies_to: List[str] = field(default_factory=list)
    description: str = ""
    derived_from: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            "type": self.constraint_type.value,
            "field": self.field,
            "params": self.params,
            "applies_to": self.applies_to,
            "description": self.description,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormalConstraint":
        """Parse from dictionary."""
        try:
            ctype = ConstraintType(data.get("type", "custom"))
        except ValueError:
            ctype = ConstraintType.CUSTOM

        return cls(
            constraint_type=ctype,
            field=data.get("field", ""),
            params=data.get("params", {}),
            applies_to=data.get("applies_to", []),
            description=data.get("description", ""),
            derived_from=data.get("derived_from", ""),
        )


# Regex patterns for constraint parsing
_RANGE_PATTERNS = [
    # "value in range [0, 100]" or "value in [0, 100]"
    re.compile(r"(\w+)\s+(?:in\s+)?(?:range\s+)?\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]", re.IGNORECASE),
    # "value between 0 and 100" - but NOT "length between" which is handled separately
    re.compile(r"(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    # "0 <= value <= 100" or "0 < value < 100" (supports both <= and single < or ≤)
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:<=?|≤)\s*(\w+)\s*(?:<=?|≤)\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    # "value >= 0 and value <= 100"
    re.compile(r"(\w+)\s*>=?\s*(\d+(?:\.\d+)?)\s+(?:and|,)\s*\1\s*<=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
]

_NOT_NULL_PATTERNS = [
    # "field cannot be none/null" - must be first to avoid greedy match
    re.compile(r"(\w+)\s+cannot\s+be\s+(?:null|none)", re.IGNORECASE),
    re.compile(r"(\w+)\s+(?:must\s+)?(?:be\s+)?(?:not\s+)?(?:be\s+)?(?:null|none|empty)", re.IGNORECASE),
    re.compile(r"(\w+)\s+(?:is\s+)?required", re.IGNORECASE),
]

_REGEX_PATTERNS = [
    # "field matches /pattern/" or "field matches 'pattern'"
    re.compile(r"(\w+)\s+matches?\s+[/'\"](.+?)[/'\"]", re.IGNORECASE),
    # "field must match pattern"
    re.compile(r"(\w+)\s+must\s+match\s+[/'\"](.+?)[/'\"]", re.IGNORECASE),
]

_ENUM_PATTERNS = [
    # "field must be one of: a, b, c"
    re.compile(r"(\w+)\s+(?:must\s+be\s+)?(?:one\s+of|in)\s*:\s*(.+)", re.IGNORECASE),
    # "field in {a, b, c}" or "field in [a, b, c]"
    re.compile(r"(\w+)\s+in\s+[\[{](.+?)[\]}]", re.IGNORECASE),
]

_POSITIVE_PATTERNS = [
    re.compile(r"(\w+)\s+(?:must\s+be\s+)?positive", re.IGNORECASE),
    re.compile(r"(\w+)\s*>\s*0", re.IGNORECASE),
]

_NON_NEGATIVE_PATTERNS = [
    re.compile(r"(\w+)\s+(?:must\s+be\s+)?non[_-]?negative", re.IGNORECASE),
    re.compile(r"(\w+)\s*>=\s*0", re.IGNORECASE),
]

_LENGTH_PATTERNS = [
    # "field length between 1 and 100"
    re.compile(r"(\w+)\s+length\s+(?:between\s+)?(\d+)\s+(?:and|to)\s+(\d+)", re.IGNORECASE),
    # "field max length 100" or "field max 100 chars"
    re.compile(r"(\w+)\s+(?:max(?:imum)?\s+)?length\s*(?:<=?|:)?\s*(\d+)", re.IGNORECASE),
    # "field min length 1"
    re.compile(r"(\w+)\s+min(?:imum)?\s+length\s*(?:>=?|:)?\s*(\d+)", re.IGNORECASE),
]

_UNIQUE_PATTERNS = [
    re.compile(r"(\w+)\s+(?:must\s+be\s+)?unique", re.IGNORECASE),
    re.compile(r"(?:no\s+)?duplicate(?:s)?\s+(?:in\s+)?(\w+)", re.IGNORECASE),
]


def parse_constraint(text: str, applies_to: Optional[List[str]] = None) -> Optional[FormalConstraint]:
    """
    Parse natural language constraint into structured FormalConstraint.

    Phase 6.3: Constraint Execution - parse constraints for code generation.

    Args:
        text: Natural language constraint description
        applies_to: Optional list of component names this applies to

    Returns:
        FormalConstraint if parseable, None otherwise

    Examples:
        >>> parse_constraint("confidence in range [0, 1]")
        FormalConstraint(type=RANGE, field="confidence", params={"min": 0, "max": 1})

        >>> parse_constraint("name must not be null")
        FormalConstraint(type=NOT_NULL, field="name")

        >>> parse_constraint("email matches /^[a-zA-Z0-9._%+-]+@/")
        FormalConstraint(type=REGEX, field="email", params={"pattern": "^[a-zA-Z0-9._%+-]+@"})
    """
    text = text.strip()
    applies_to = applies_to or []

    # Check for length constraints FIRST (before range, since "length between" matches range pattern)
    if "length" in text.lower():
        for pattern in _LENGTH_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                field_name = groups[0]
                params = {}
                if len(groups) == 3:
                    params["min"] = int(groups[1])
                    params["max"] = int(groups[2])
                elif "max" in text.lower():
                    params["max"] = int(groups[1])
                else:
                    params["min"] = int(groups[1])

                return FormalConstraint(
                    constraint_type=ConstraintType.LENGTH,
                    field=field_name,
                    params=params,
                    applies_to=applies_to,
                    description=text,
                    derived_from=text,
                )

    # Try range patterns first
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            # Handle different pattern formats
            if len(groups) == 3:
                # Check if first group is number (0 <= val <= 100 format)
                try:
                    float(groups[0])
                    # Pattern: min, field, max
                    field_name, min_val, max_val = groups[1], groups[0], groups[2]
                except ValueError:
                    # Pattern: field, min, max
                    field_name, min_val, max_val = groups

                return FormalConstraint(
                    constraint_type=ConstraintType.RANGE,
                    field=field_name,
                    params={"min": float(min_val), "max": float(max_val)},
                    applies_to=applies_to,
                    description=text,
                    derived_from=text,
                )

    # Try not null patterns
    for pattern in _NOT_NULL_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name = match.group(1)
            return FormalConstraint(
                constraint_type=ConstraintType.NOT_NULL,
                field=field_name,
                params={},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try regex patterns
    for pattern in _REGEX_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name, regex_pattern = match.groups()
            return FormalConstraint(
                constraint_type=ConstraintType.REGEX,
                field=field_name,
                params={"pattern": regex_pattern},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try enum patterns
    for pattern in _ENUM_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name, values_str = match.groups()
            # Parse values: "a, b, c" -> ["a", "b", "c"]
            values = [v.strip().strip('"\'') for v in values_str.split(",")]
            return FormalConstraint(
                constraint_type=ConstraintType.ENUM,
                field=field_name,
                params={"values": values},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try positive patterns
    for pattern in _POSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name = match.group(1)
            return FormalConstraint(
                constraint_type=ConstraintType.POSITIVE,
                field=field_name,
                params={},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try non-negative patterns
    for pattern in _NON_NEGATIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name = match.group(1)
            return FormalConstraint(
                constraint_type=ConstraintType.NON_NEGATIVE,
                field=field_name,
                params={},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try length patterns
    for pattern in _LENGTH_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            field_name = groups[0]
            params = {}
            if len(groups) == 3:
                params["min"] = int(groups[1])
                params["max"] = int(groups[2])
            elif "max" in text.lower():
                params["max"] = int(groups[1])
            else:
                params["min"] = int(groups[1])

            return FormalConstraint(
                constraint_type=ConstraintType.LENGTH,
                field=field_name,
                params=params,
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Try unique patterns
    for pattern in _UNIQUE_PATTERNS:
        match = pattern.search(text)
        if match:
            field_name = match.group(1)
            return FormalConstraint(
                constraint_type=ConstraintType.UNIQUE,
                field=field_name,
                params={},
                applies_to=applies_to,
                description=text,
                derived_from=text,
            )

    # Fallback: return as custom constraint
    return FormalConstraint(
        constraint_type=ConstraintType.CUSTOM,
        field="",
        params={"expression": text},
        applies_to=applies_to,
        description=text,
        derived_from=text,
    )


def parse_blueprint_constraints(blueprint: Dict[str, Any]) -> List[FormalConstraint]:
    """
    Parse all constraints in a blueprint into FormalConstraints.

    Args:
        blueprint: The blueprint dict containing constraints

    Returns:
        List of parsed FormalConstraints
    """
    formal_constraints = []

    for constraint in blueprint.get("constraints", []):
        description = constraint.get("description", "")
        applies_to = constraint.get("applies_to", [])

        if description:
            parsed = parse_constraint(description, applies_to)
            if parsed:
                formal_constraints.append(parsed)

    return formal_constraints


# =============================================================================
# CONSTRAINT CONTRADICTION DETECTION
# Phase 17.4: Deterministic detection of incompatible constraints
# =============================================================================

@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction between two constraints."""
    constraint_a: str       # description of first constraint
    constraint_b: str       # description of second constraint
    field: str              # the field they conflict on
    contradiction_type: str # "range_conflict", "enum_disjoint", "polarity_conflict"
    description: str        # human-readable explanation


def detect_contradictions(constraints: list) -> List[Contradiction]:
    """
    Detect contradictory constraints (deterministic, no LLM).

    Rules:
    - Range conflict: two RANGE on same field with non-overlapping intervals
    - Enum disjoint: two ENUM on same field with empty intersection
    - Polarity conflict: POSITIVE + RANGE [max < 0] on same field

    Args:
        constraints: List of constraint dicts from blueprint, or FormalConstraints

    Returns:
        List of Contradiction objects found
    """
    if not constraints:
        return []

    # Parse constraints into FormalConstraints if needed
    parsed: List[FormalConstraint] = []
    for c in constraints:
        if isinstance(c, FormalConstraint):
            parsed.append(c)
        elif isinstance(c, dict):
            desc = c.get("description", "")
            applies_to = c.get("applies_to", [])
            if desc:
                fc = parse_constraint(desc, applies_to)
                if fc:
                    parsed.append(fc)
        # else: skip unparseable

    if len(parsed) < 2:
        return []

    contradictions: List[Contradiction] = []

    # Group by field
    by_field: Dict[str, List[FormalConstraint]] = {}
    for fc in parsed:
        if fc.field:
            by_field.setdefault(fc.field, []).append(fc)

    for field_name, field_constraints in by_field.items():
        ranges = [fc for fc in field_constraints if fc.constraint_type == ConstraintType.RANGE]
        enums = [fc for fc in field_constraints if fc.constraint_type == ConstraintType.ENUM]
        positives = [fc for fc in field_constraints if fc.constraint_type == ConstraintType.POSITIVE]

        # Rule 1: Range conflict — non-overlapping intervals
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                a = ranges[i]
                b = ranges[j]
                a_min = a.params.get("min", float("-inf"))
                a_max = a.params.get("max", float("inf"))
                b_min = b.params.get("min", float("-inf"))
                b_max = b.params.get("max", float("inf"))

                # Non-overlapping: a_max < b_min or b_max < a_min
                if a_max < b_min or b_max < a_min:
                    contradictions.append(Contradiction(
                        constraint_a=a.description,
                        constraint_b=b.description,
                        field=field_name,
                        contradiction_type="range_conflict",
                        description=(
                            f"Field '{field_name}': range [{a_min}, {a_max}] "
                            f"does not overlap with [{b_min}, {b_max}]"
                        ),
                    ))

        # Rule 2: Enum disjoint — empty intersection
        for i in range(len(enums)):
            for j in range(i + 1, len(enums)):
                a = enums[i]
                b = enums[j]
                a_vals = set(a.params.get("values", []))
                b_vals = set(b.params.get("values", []))
                if a_vals and b_vals and not (a_vals & b_vals):
                    contradictions.append(Contradiction(
                        constraint_a=a.description,
                        constraint_b=b.description,
                        field=field_name,
                        contradiction_type="enum_disjoint",
                        description=(
                            f"Field '{field_name}': enum values {sorted(a_vals)} "
                            f"and {sorted(b_vals)} have no overlap"
                        ),
                    ))

        # Rule 3: Polarity conflict — POSITIVE + RANGE [max < 0]
        for pos in positives:
            for rng in ranges:
                r_max = rng.params.get("max", float("inf"))
                if r_max < 0:
                    contradictions.append(Contradiction(
                        constraint_a=pos.description,
                        constraint_b=rng.description,
                        field=field_name,
                        contradiction_type="polarity_conflict",
                        description=(
                            f"Field '{field_name}': must be positive but "
                            f"range max is {r_max} (< 0)"
                        ),
                    ))

    return contradictions


def generate_validator_code(constraint: FormalConstraint) -> str:
    """
    Generate Python validation code for a FormalConstraint.

    Phase 6.3: Code generation from parsed constraints.
    Phase 20: Error messages include actual field value via f-string.

    Args:
        constraint: The formal constraint to generate code for

    Returns:
        Python code string that validates the constraint

    Examples:
        >>> c = FormalConstraint(type=RANGE, field="confidence", params={"min": 0, "max": 1})
        >>> generate_validator_code(c)
        'assert 0 <= self.confidence <= 1, f"confidence must be in range [0, 1], got {self.confidence}"'
    """
    field = constraint.field
    params = constraint.params
    ctype = constraint.constraint_type

    if ctype == ConstraintType.RANGE:
        min_val = params.get("min", 0)
        max_val = params.get("max", 100)
        return f'assert {min_val} <= self.{field} <= {max_val}, f"{field} must be in range [{min_val}, {max_val}], got {{self.{field}}}"'

    elif ctype == ConstraintType.NOT_NULL:
        return f'assert self.{field} is not None, "{field} must not be None"'

    elif ctype == ConstraintType.NOT_EMPTY:
        return f'assert self.{field}, f"{field} must not be empty, got {{self.{field}!r}}"'

    elif ctype == ConstraintType.REGEX:
        pattern = params.get("pattern", ".*")
        # Escape quotes in pattern
        escaped_pattern = pattern.replace("'", "\\'")
        return f"import re; assert re.match(r'{escaped_pattern}', self.{field}), f\"{field} must match pattern, got {{self.{field}!r}}\""

    elif ctype == ConstraintType.ENUM:
        values = params.get("values", [])
        values_str = ", ".join(repr(v) for v in values)
        return f'assert self.{field} in [{values_str}], f"{field} must be one of: {values_str}, got {{self.{field}!r}}"'

    elif ctype == ConstraintType.POSITIVE:
        return f'assert self.{field} > 0, f"{field} must be positive, got {{self.{field}}}"'

    elif ctype == ConstraintType.NON_NEGATIVE:
        return f'assert self.{field} >= 0, f"{field} must be non-negative, got {{self.{field}}}"'

    elif ctype == ConstraintType.LENGTH:
        min_len = params.get("min")
        max_len = params.get("max")
        if min_len is not None and max_len is not None:
            return f'assert {min_len} <= len(self.{field}) <= {max_len}, f"{field} length must be between {min_len} and {max_len}, got {{len(self.{field})}}"'
        elif max_len is not None:
            return f'assert len(self.{field}) <= {max_len}, f"{field} length must be at most {max_len}, got {{len(self.{field})}}"'
        elif min_len is not None:
            return f'assert len(self.{field}) >= {min_len}, f"{field} length must be at least {min_len}, got {{len(self.{field})}}"'

    elif ctype == ConstraintType.UNIQUE:
        return f'assert len(self.{field}) == len(set(self.{field})), f"{field} must contain unique values, got {{len(self.{field}) - len(set(self.{field}))}} duplicates"'

    elif ctype == ConstraintType.TYPE:
        expected_type = params.get("type", "object")
        return f'assert isinstance(self.{field}, {expected_type}), f"{field} must be of type {expected_type}, got {{type(self.{field}).__name__}}"'

    # Custom - return expression as-is or wrap
    expression = params.get("expression", "True")
    return f'assert {expression}, "{constraint.description}"'


def generate_validate_method(constraints: List[FormalConstraint], class_name: str = "") -> str:
    """
    Generate a complete validate() method from multiple constraints.

    Args:
        constraints: List of FormalConstraints
        class_name: Optional class name for better error messages

    Returns:
        Python code for a validate() method
    """
    if not constraints:
        return "def validate(self) -> bool:\n    return True"

    lines = [
        "def validate(self) -> bool:",
        '    """Validate all constraints."""',
        "    errors = []",
        "    try:",
    ]

    for constraint in constraints:
        code = generate_validator_code(constraint)
        # Wrap in try/except for better error collection
        lines.append(f"        {code}")

    lines.extend([
        "    except AssertionError as e:",
        "        errors.append(str(e))",
        "    if errors:",
        f'        raise ValueError(f"{class_name or "Validation"} failed: {{errors}}")',
        "    return True",
    ])

    return "\n".join(lines)


# =============================================================================
# NESTED BLUEPRINT SUPPORT
# Derived from: NEXT-STEPS.md "Nested Blueprint Support"
# =============================================================================

def check_nesting_depth(
    blueprint: Dict[str, Any],
    current_depth: int = 0,
    max_depth: int = 3
) -> List[str]:
    """
    Check that nested blueprints don't exceed maximum depth.

    Args:
        blueprint: The blueprint to check
        current_depth: Current nesting level (0 = root)
        max_depth: Maximum allowed depth (default: 3)

    Returns:
        List of error messages for depth violations
    """
    errors = []

    if current_depth > max_depth:
        return [f"Nesting depth {current_depth} exceeds max {max_depth}"]

    for comp in blueprint.get("components", []):
        if comp.get("sub_blueprint"):
            sub_errors = check_nesting_depth(
                comp["sub_blueprint"],
                current_depth + 1,
                max_depth
            )
            for err in sub_errors:
                errors.append(f"{comp.get('name', 'Unknown')}.{err}")

    return errors


def resolve_component_path(
    blueprint: Dict[str, Any],
    path: str
) -> Optional[Dict[str, Any]]:
    """
    Resolve a dotted path to a component in a nested blueprint.

    Dot notation format: "ParentSubsystem.ChildSubsystem.Component"

    Args:
        blueprint: The root blueprint
        path: Dotted path to component (e.g., "UserService.User")

    Returns:
        The component dict if found, None otherwise
    """
    if not path:
        return None

    parts = path.split(".")
    current = blueprint

    # Navigate through subsystems
    for part in parts[:-1]:
        found = False
        for comp in current.get("components", []):
            if comp.get("name") == part and comp.get("sub_blueprint"):
                current = comp["sub_blueprint"]
                found = True
                break
        if not found:
            return None  # Path not found

    # Find final component
    final_name = parts[-1]
    for comp in current.get("components", []):
        if comp.get("name") == final_name:
            return comp

    return None


def validate_cross_references(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate cross-subsystem relationships using dot notation.

    Checks that all dotted component references resolve to actual components.

    Args:
        blueprint: The blueprint to validate

    Returns:
        {
            "valid": bool,
            "invalid_refs": List[tuple],  # (from, to, which_invalid)
            "valid_refs": List[tuple]
        }
    """
    invalid_refs = []
    valid_refs = []

    for rel in blueprint.get("relationships", []):
        from_comp = rel.get("from", "")
        to_comp = rel.get("to", "")

        # Check if using dot notation (cross-subsystem reference)
        from_valid = True
        to_valid = True

        if "." in from_comp:
            if resolve_component_path(blueprint, from_comp) is None:
                from_valid = False
        if "." in to_comp:
            if resolve_component_path(blueprint, to_comp) is None:
                to_valid = False

        if not from_valid or not to_valid:
            which = []
            if not from_valid:
                which.append(f"from:{from_comp}")
            if not to_valid:
                which.append(f"to:{to_comp}")
            invalid_refs.append((from_comp, to_comp, which))
        else:
            valid_refs.append((from_comp, to_comp))

    return {
        "valid": len(invalid_refs) == 0,
        "invalid_refs": invalid_refs,
        "valid_refs": valid_refs
    }


def validate_nested_blueprint(
    blueprint: Dict[str, Any],
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Comprehensive validation for nested blueprints.

    Combines depth checking, cross-reference validation, and recursive validation.

    Args:
        blueprint: The blueprint to validate
        max_depth: Maximum nesting depth allowed

    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "depth_errors": List[str],
            "cross_ref_errors": List[str],
            "subsystem_validations": Dict[str, Dict]
        }
    """
    errors = []
    warnings = []
    subsystem_validations = {}

    # 1. Check nesting depth
    depth_errors = check_nesting_depth(blueprint, 0, max_depth)
    errors.extend(depth_errors)

    # 2. Validate cross-references
    cross_refs = validate_cross_references(blueprint)
    if not cross_refs["valid"]:
        for (from_c, to_c, which) in cross_refs["invalid_refs"]:
            errors.append(f"Invalid cross-reference: {from_c} -> {to_c} ({', '.join(which)})")

    # 3. Recursively validate sub-blueprints
    for comp in blueprint.get("components", []):
        if comp.get("sub_blueprint"):
            comp_name = comp.get("name", "Unknown")
            sub_validation = validate_nested_blueprint(
                comp["sub_blueprint"],
                max_depth
            )
            subsystem_validations[comp_name] = sub_validation

            # Propagate errors with prefix
            for err in sub_validation.get("errors", []):
                errors.append(f"{comp_name}.{err}")
            for warn in sub_validation.get("warnings", []):
                warnings.append(f"{comp_name}.{warn}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "depth_errors": depth_errors,
        "cross_ref_errors": [str(r) for r in cross_refs.get("invalid_refs", [])],
        "subsystem_validations": subsystem_validations
    }


# =============================================================================
# BLUEPRINT DEDUPLICATION
# Phase 8.2: Compilation Quality - Remove duplicate components and relationships
# =============================================================================

def deduplicate_blueprint(blueprint: Dict[str, Any]) -> tuple:
    """
    Remove duplicate components and relationships from a blueprint.

    Phase 8.2: Eliminates confusion from duplicate components.

    Three dedup phases:
    1. Name dedup: If two components share normalized name, keep the richer one
    2. Containment dedup: If component appears top-level AND inside a subsystem, remove from top-level
    3. Relationship dedup: Remove duplicate (from, to, type) triples

    Args:
        blueprint: The blueprint dict to deduplicate

    Returns:
        Tuple of (cleaned_blueprint, dedup_report)
        dedup_report: {
            "name_dupes_removed": List[str],
            "containment_dupes_removed": List[str],
            "relationship_dupes_removed": int,
            "total_removed": int,
        }
    """
    report = {
        "name_dupes_removed": [],
        "containment_dupes_removed": [],
        "relationship_dupes_removed": 0,
        "total_removed": 0,
    }

    components = list(blueprint.get("components", []))
    relationships = list(blueprint.get("relationships", []))

    # Phase 1: Name dedup — if two components share normalized name, keep richer one
    seen_names: Dict[str, int] = {}  # normalized_name -> index of best component
    dedup_components = []
    for i, comp in enumerate(components):
        name = comp.get("name", "")
        norm = normalize_component_name(name).lower().strip()
        if not norm:
            dedup_components.append(comp)
            continue

        if norm in seen_names:
            # Compare: keep the one with longer description
            existing_idx = seen_names[norm]
            existing = dedup_components[existing_idx]
            existing_richness = len(existing.get("description", "")) + len(str(existing.get("methods", [])))
            new_richness = len(comp.get("description", "")) + len(str(comp.get("methods", [])))

            if new_richness > existing_richness:
                # Replace existing with richer version
                report["name_dupes_removed"].append(existing.get("name", ""))
                dedup_components[existing_idx] = comp
            else:
                report["name_dupes_removed"].append(name)
        else:
            seen_names[norm] = len(dedup_components)
            dedup_components.append(comp)

    # Remove None placeholders (shouldn't exist but safety)
    dedup_components = [c for c in dedup_components if c is not None]

    # Phase 2: Containment dedup — if component in subsystem's sub_blueprint, remove from top-level
    # Uses fuzzy normalization: strips type suffixes, collapses whitespace/casing
    import re

    def _norm_containment(n: str) -> str:
        s = n.lower().strip()
        s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
        s = re.sub(r'\s+(service|manager|handler|controller|module|component|engine)$', '', s)
        return s

    subsystem_component_names = set()
    for comp in dedup_components:
        if comp.get("type") == "subsystem" and comp.get("sub_blueprint"):
            sub_bp = comp["sub_blueprint"]
            for sub_comp in sub_bp.get("components", []):
                sub_name = sub_comp.get("name", "")
                norm = _norm_containment(sub_name)
                if norm:
                    subsystem_component_names.add(norm)

    if subsystem_component_names:
        filtered = []
        for comp in dedup_components:
            norm = _norm_containment(comp.get("name", ""))
            if norm in subsystem_component_names and comp.get("type") != "subsystem":
                report["containment_dupes_removed"].append(comp.get("name", ""))
            else:
                filtered.append(comp)
        dedup_components = filtered

    # Phase 3: Relationship dedup — remove duplicate (from, to, type) triples
    seen_rels = set()
    dedup_relationships = []
    for rel in relationships:
        triple = (
            rel.get("from", "").lower().strip(),
            rel.get("to", "").lower().strip(),
            rel.get("type", "").lower().strip(),
        )
        if triple not in seen_rels:
            seen_rels.add(triple)
            dedup_relationships.append(rel)
        else:
            report["relationship_dupes_removed"] += 1

    # Phase 3b: Pair-level dedup — same (from, to) with different types → keep most descriptive
    seen_pairs: Dict[Tuple[str, str], int] = {}
    pair_dedup_relationships: list = []
    pair_dupes_removed = 0
    for rel in dedup_relationships:
        pair = (
            rel.get("from", "").lower().strip(),
            rel.get("to", "").lower().strip(),
        )
        if pair in seen_pairs:
            existing_idx = seen_pairs[pair]
            existing = pair_dedup_relationships[existing_idx]
            existing_len = len(existing.get("description", ""))
            new_len = len(rel.get("description", ""))
            if new_len > existing_len:
                pair_dedup_relationships[existing_idx] = rel
            pair_dupes_removed += 1
        else:
            seen_pairs[pair] = len(pair_dedup_relationships)
            pair_dedup_relationships.append(rel)

    dedup_relationships = pair_dedup_relationships
    report["relationship_dupes_removed"] += pair_dupes_removed

    report["total_removed"] = (
        len(report["name_dupes_removed"])
        + len(report["containment_dupes_removed"])
        + report["relationship_dupes_removed"]
    )

    cleaned = dict(blueprint)
    cleaned["components"] = dedup_components
    cleaned["relationships"] = dedup_relationships

    return cleaned, report


def normalize_blueprint_elements(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce string elements in blueprint arrays to proper dicts.

    LLM synthesis sometimes returns plain strings instead of structured dicts
    (e.g. ["AuthService"] instead of [{"name": "AuthService", ...}]).
    Every downstream function calls .get() assuming dicts, so strings crash.

    This gate normalizes before any .get() calls happen.
    Dict elements pass through unchanged.
    """
    result = dict(blueprint)

    # Components: str → {"name": str, "type": "component", "description": ""}
    components = result.get("components", [])
    if isinstance(components, list):
        result["components"] = [
            {"name": c, "type": "component", "description": ""} if isinstance(c, str)
            else c
            for c in components
        ]

    # Relationships: str → {"from": "", "to": "", "type": str, "description": str}
    relationships = result.get("relationships", [])
    if isinstance(relationships, list):
        result["relationships"] = [
            {"from": "", "to": "", "type": r, "description": r} if isinstance(r, str)
            else r
            for r in relationships
        ]

    # Constraints: str → {"description": str, "applies_to": []}
    constraints = result.get("constraints", [])
    if isinstance(constraints, list):
        result["constraints"] = [
            {"description": c, "applies_to": []} if isinstance(c, str)
            else c
            for c in constraints
        ]

    # Unresolved: str → {"description": str}
    unresolved = result.get("unresolved", [])
    if isinstance(unresolved, list):
        result["unresolved"] = [
            {"description": u} if isinstance(u, str)
            else u
            for u in unresolved
        ]

    return result
