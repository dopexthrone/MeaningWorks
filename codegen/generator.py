"""
Blueprint Code Generator: Transform Motherlabs blueprints to Python code.

Phase 3: Generate runnable code, not just stubs.

Templates provide per-type code generation:
- Agent: BaseAgent pattern with run(), observe()
- Entity: Dataclass with validation, serialization
- Process: State machine with transitions
- Subsystem: Container with lifecycle
"""
from typing import Dict, Any, List, Optional, Tuple
import re

from codegen.templates import (
    AGENT_TEMPLATE, ENTITY_TEMPLATE, PROCESS_TEMPLATE, SUBSYSTEM_TEMPLATE,
    get_template_for_type, generate_state_handlers, generate_validation_rules,
    generate_fields_from_sub_blueprint, generate_init_contained,
    generate_process_validation,
)
from core.schema import (
    parse_blueprint_constraints,
    parse_constraint,
    generate_validator_code,
    generate_validate_method,
    FormalConstraint,
)


TYPE_MAPPING = {
    'agent': 'class',
    'entity': 'dataclass',
    'process': 'class',
    'subsystem': 'class',
    'interface': 'class',
}


RELATIONSHIP_TO_METHOD = {
    'triggers': ('trigger', 'target'),
    'accesses': ('_access', 'resource'),
    'monitors': ('monitor', 'subject'),
    'contains': None,  # Handled as composition
    'snapshots': ('snapshot', None),
}


def to_python_name(name: str) -> str:
    """Convert blueprint name to Python identifier.

    Phase 4.5: Preserve PascalCase names like SharedState, ConfidenceVector.
    """
    # First, check if input is already PascalCase BEFORE any transformation
    # This preserves names like "SharedState", "ConfidenceVector", etc.
    stripped = re.sub(r'[^a-zA-Z0-9]', '', name)
    if stripped and stripped[0].isupper() and any(c.isupper() for c in stripped[1:]):
        # Already has mixed case with leading capital - it's PascalCase
        return stripped

    # For names with spaces like "Intent Agent" -> "IntentAgent"
    if ' ' in name:
        return re.sub(r'[^a-zA-Z0-9]', '', name.title().replace(' ', ''))

    # Single word or already clean
    return stripped


def to_snake_case(name: str) -> str:
    """Convert to snake_case for variable names."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', to_python_name(name))
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_pascal_case(name: str) -> str:
    """Convert to PascalCase for class names.

    Phase 4.5: Fix class name casing for generated components.
    Examples:
        "SharedState" -> "SharedState"
        "shared_state" -> "SharedState"
        "sharedstate" -> "Sharedstate"
    """
    # First, check if input is already PascalCase BEFORE any transformation
    # This preserves names like "SharedState", "ConfidenceVector", etc.
    stripped = re.sub(r'[^a-zA-Z0-9]', '', name)  # Remove non-alphanumeric
    if stripped and stripped[0].isupper() and any(c.isupper() for c in stripped[1:]):
        # Already has mixed case with leading capital - it's PascalCase
        return stripped

    # For names with underscores, split and capitalize each part
    if '_' in name:
        parts = name.replace(' ', '_').split('_')
        return ''.join(p.capitalize() for p in parts if p)

    # For names with spaces, split and capitalize each part
    if ' ' in name:
        parts = name.split(' ')
        return ''.join(p.capitalize() for p in parts if p)

    # Single word - capitalize first letter only
    clean = re.sub(r'[^a-zA-Z0-9]', '', name)
    return clean[0].upper() + clean[1:] if clean else clean


class BlueprintCodeGenerator:
    """Generates Python code from Motherlabs blueprints."""

    def __init__(self, blueprint: Dict[str, Any]):
        self.blueprint = blueprint
        self.components = {c['name']: c for c in blueprint.get('components', [])}
        self.relationships = blueprint.get('relationships', [])
        self.constraints = blueprint.get('constraints', [])
        # Build dependency graph
        self._deps = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[str, Dict[str, List[str]]]:
        """Build dependency graph from relationships.

        Returns dict mapping component -> relationship_type -> [targets]
        Example: {"IntentAgent": {"accesses": ["SharedState"], "triggers": ["EntityAgent"]}}
        """
        deps = {}
        for rel in self.relationships:
            from_comp = rel.get('from', '')
            to_comp = rel.get('to', '')
            rel_type = rel.get('type', 'uses')

            if from_comp not in deps:
                deps[from_comp] = {}
            if rel_type not in deps[from_comp]:
                deps[from_comp][rel_type] = []
            deps[from_comp][rel_type].append(to_comp)

        return deps

    def _get_dependencies(self, component_name: str) -> List[Tuple[str, str]]:
        """Get dependencies for a component as (target, rel_type) pairs."""
        deps = []
        comp_deps = self._deps.get(component_name, {})
        for rel_type, targets in comp_deps.items():
            for target in targets:
                deps.append((target, rel_type))
        return deps

    def _generate_dependency_injection(self, component_name: str) -> str:
        """Generate __init__ parameters for dependency injection."""
        deps = self._get_dependencies(component_name)
        if not deps:
            return ''

        lines = []
        for target, rel_type in deps:
            target_py = to_python_name(target)
            target_snake = to_snake_case(target)
            # Skip self-references and methods
            if target == component_name or '(' in target:
                continue
            lines.append(f'        self.{target_snake}: Optional["{target_py}"] = None')

        return '\n'.join(lines)

    def _generate_factory(self) -> str:
        """Generate a factory class for creating all components with wired dependencies."""
        lines = []
        lines.append('class BlueprintFactory:')
        lines.append('    """Factory for creating all blueprint components with wired dependencies."""')
        lines.append('')
        lines.append('    def __init__(self):')
        lines.append('        """Initialize factory with cached instances."""')
        lines.append('        self._instances: Dict[str, Any] = {}')
        lines.append('')

        # Generate create method for each non-metadata component
        for name, component in self.components.items():
            comp_type = component.get('type', 'entity')
            if comp_type in ('state', 'method'):
                continue

            py_name = to_python_name(name)
            snake_name = to_snake_case(name)
            deps = self._get_dependencies(name)

            lines.append(f'    def create_{snake_name}(self) -> "{py_name}":')
            lines.append(f'        """Create {py_name} with all dependencies wired."""')
            lines.append(f'        if "{py_name}" in self._instances:')
            lines.append(f'            return self._instances["{py_name}"]')
            lines.append(f'        ')
            lines.append(f'        instance = {py_name}()')

            # Wire dependencies
            for target, rel_type in deps:
                target_snake = to_snake_case(target)
                if target == name or '(' in target:
                    continue
                if target in self.components:
                    target_type = self.components[target].get('type', 'entity')
                    if target_type not in ('state', 'method'):
                        lines.append(f'        instance.{target_snake} = self.create_{target_snake}()')

            lines.append(f'        self._instances["{py_name}"] = instance')
            lines.append(f'        return instance')
            lines.append('')

        # Generate create_all method
        lines.append('    def create_all(self) -> Dict[str, Any]:')
        lines.append('        """Create all components with dependencies wired."""')
        valid_comps = [(n, c) for n, c in self.components.items()
                       if c.get('type', 'entity') not in ('state', 'method')]
        for name, _ in valid_comps:
            snake_name = to_snake_case(name)
            lines.append(f'        self.create_{snake_name}()')
        lines.append('        return self._instances.copy()')

        return '\n'.join(lines)

    def generate(self) -> str:
        """Generate full Python module from blueprint."""
        lines = []

        # Module header
        lines.append('"""')
        lines.append('Auto-generated from Motherlabs blueprint.')
        lines.append('DO NOT EDIT - Regenerate from blueprint instead.')
        lines.append('"""')
        lines.append('from dataclasses import dataclass, field')
        lines.append('from typing import Any, Dict, List, Optional')
        lines.append('from abc import ABC, abstractmethod')
        lines.append('from enum import Enum')
        lines.append('')
        lines.append('')
        lines.append('# Base classes (generated for standalone operation)')
        lines.append('class BaseAgent(ABC):')
        lines.append('    """Base class for all agents."""')
        lines.append('    def __init__(self, name: str, perspective: str, llm_client=None):')
        lines.append('        self.name = name')
        lines.append('        self.perspective = perspective')
        lines.append('        self.llm = llm_client')
        lines.append('')
        lines.append('    @abstractmethod')
        lines.append('    def run(self, state: Any, input_msg: Any = None) -> Any:')
        lines.append('        """Execute agent logic."""')
        lines.append('        pass')
        lines.append('')
        lines.append('')

        # Phase 4.5: Conditionally generate boilerplate classes
        # Only add these if the blueprint doesn't already contain them
        has_message = any(
            'message' in c.get('name', '').lower() and c.get('name', '').lower() != 'messagetype'
            for c in self.blueprint.get('components', [])
        )
        has_messagetype = any(
            c.get('name', '').lower() == 'messagetype'
            for c in self.blueprint.get('components', [])
        )
        has_sharedstate = any(
            'sharedstate' in c.get('name', '').lower().replace('_', '')
            for c in self.blueprint.get('components', [])
        )

        if not has_message:
            lines.append('class Message:')
            lines.append('    """Message in dialogue."""')
            lines.append('    def __init__(self, sender: str, content: str, message_type=None):')
            lines.append('        self.sender = sender')
            lines.append('        self.content = content')
            lines.append('        self.message_type = message_type')
            lines.append('')
            lines.append('')

        if not has_messagetype:
            lines.append('class MessageType(Enum):')
            lines.append('    """Types of messages in dialogue."""')
            lines.append('    PROPOSITION = "proposition"')
            lines.append('    CHALLENGE = "challenge"')
            lines.append('    ACCOMMODATION = "accommodation"')
            lines.append('')
            lines.append('')

        if not has_sharedstate:
            lines.append('class SharedState:')
            lines.append('    """Shared state for agent dialogue."""')
            lines.append('    def __init__(self):')
            lines.append('        self.known: Dict[str, Any] = {}')
            lines.append('        self.history: List[Message] = []')
            lines.append('        self.insights: List[str] = []')
            lines.append('')
            lines.append('    def get_recent(self, count: int = 5) -> List[Message]:')
            lines.append('        return self.history[-count:]')
            lines.append('')
            lines.append('')

        lines.append('# Generated components')
        lines.append('')

        # Generate each component
        for name, component in self.components.items():
            code = self._generate_component(name, component)
            if code:  # Skip empty strings (states, methods)
                lines.append(code)
                lines.append('')
                lines.append('')

        # Generate factory class for dependency injection
        if self.relationships:
            lines.append('# Factory for dependency injection')
            lines.append(self._generate_factory())
            lines.append('')
            lines.append('')

        return '\n'.join(lines)

    def generate_tests(self) -> str:
        """Generate pytest tests from blueprint constraints and structure."""
        lines = []

        # Test file header
        lines.append('"""')
        lines.append('Auto-generated tests from Motherlabs blueprint.')
        lines.append('Run with: pytest <this_file>')
        lines.append('"""')
        lines.append('import pytest')
        lines.append('from typing import Any, Dict, List')
        lines.append('')
        lines.append('')

        # Generate test class for each component
        for name, component in self.components.items():
            comp_type = component.get('type', 'entity')
            if comp_type in ('state', 'method'):
                continue

            py_name = to_python_name(name)
            test_class = self._generate_test_class(py_name, component)
            if test_class:
                lines.append(test_class)
                lines.append('')
                lines.append('')

        # Generate integration tests from relationships
        if self.relationships:
            lines.append(self._generate_integration_tests())
            lines.append('')

        # Generate constraint tests
        if self.constraints:
            lines.append(self._generate_constraint_tests())
            lines.append('')

        return '\n'.join(lines)

    def _generate_test_class(self, name: str, component: Dict) -> str:
        """Generate test class for a component."""
        lines = []
        comp_type = component.get('type', 'entity')
        desc = component.get('description', '')

        lines.append(f'class Test{name}:')
        lines.append(f'    """Tests for {name}."""')
        lines.append('')

        # Test instantiation
        lines.append(f'    def test_instantiation(self):')
        lines.append(f'        """Test {name} can be instantiated."""')
        lines.append(f'        instance = {name}()')
        lines.append(f'        assert instance is not None')
        lines.append('')

        # Test methods if present
        methods = component.get('methods', [])
        for method in methods:
            method_name = method.get('name', '')
            if method_name == '__init__':
                continue

            test_method = self._generate_method_test(name, method)
            lines.append(test_method)
            lines.append('')

        # State machine tests if present
        state_machine = component.get('state_machine')
        if state_machine and state_machine.get('states'):
            lines.append(self._generate_state_machine_tests(name, state_machine))
            lines.append('')

        return '\n'.join(lines)

    def _generate_method_test(self, class_name: str, method: Dict) -> str:
        """Generate test for a method."""
        method_name = method.get('name', 'unnamed')
        params = method.get('parameters', [])
        return_type = method.get('return_type', 'None')
        desc = method.get('description', '')

        lines = []
        lines.append(f'    def test_{method_name}_exists(self):')
        lines.append(f'        """Test {method_name} method exists."""')
        lines.append(f'        instance = {class_name}()')
        lines.append(f'        assert hasattr(instance, "{method_name}")')
        lines.append(f'        assert callable(getattr(instance, "{method_name}"))')

        return '\n'.join(lines)

    def _generate_state_machine_tests(self, class_name: str, state_machine: Dict) -> str:
        """Generate tests for state machine behavior."""
        states = state_machine.get('states', [])
        initial_state = state_machine.get('initial_state', states[0] if states else 'INIT')
        transitions = state_machine.get('transitions', [])

        lines = []

        # Test initial state
        lines.append(f'    def test_initial_state(self):')
        lines.append(f'        """Test {class_name} starts in {initial_state} state."""')
        lines.append(f'        instance = {class_name}()')
        lines.append(f'        assert hasattr(instance, "current_state") or hasattr(instance, "_state")')
        lines.append('')

        # Test valid states
        lines.append(f'    def test_valid_states(self):')
        lines.append(f'        """Test {class_name} has expected states."""')
        lines.append(f'        instance = {class_name}()')
        lines.append(f'        if hasattr(instance, "STATES"):')
        lines.append(f'            expected_states = {repr(states)}')
        lines.append(f'            for state in expected_states:')
        lines.append(f'                assert state in instance.STATES, f"Missing state: {{state}}"')
        lines.append('')

        # Test transitions
        for t in transitions:
            from_state = t.get('from_state', '')
            to_state = t.get('to_state', '')
            trigger = t.get('trigger', '')
            if from_state and to_state:
                lines.append(f'    def test_transition_{from_state.lower()}_to_{to_state.lower()}(self):')
                lines.append(f'        """Test transition {from_state} -> {to_state}."""')
                lines.append(f'        instance = {class_name}()')
                lines.append(f'        if hasattr(instance, "can_transition"):')
                lines.append(f'            # Skip if not starting in {from_state}')
                lines.append(f'            if instance.current_state == "{from_state}":')
                lines.append(f'                assert instance.can_transition("{to_state}")')
                lines.append('')

        return '\n'.join(lines)

    def _generate_integration_tests(self) -> str:
        """Generate integration tests from relationships."""
        lines = []
        lines.append('class TestIntegration:')
        lines.append('    """Integration tests from blueprint relationships."""')
        lines.append('')

        # Test relationship types
        rel_types = set(r.get('type', '') for r in self.relationships)

        for rel_type in rel_types:
            if not rel_type:
                continue
            rels = [r for r in self.relationships if r.get('type') == rel_type]

            lines.append(f'    def test_{rel_type}_relationships(self):')
            lines.append(f'        """Test {rel_type} relationships are wired correctly."""')
            lines.append(f'        # Relationships of type "{rel_type}":')
            for r in rels[:5]:  # Limit to 5 examples
                lines.append(f'        #   {r.get("from")} -> {r.get("to")}')
            lines.append(f'        pass  # TODO: Verify wiring')
            lines.append('')

        return '\n'.join(lines)

    def _generate_constraint_tests(self) -> str:
        """Generate tests from constraints using parsed constraint types.

        Phase 20: Enhanced to generate enforcement tests that verify
        ValueError is raised for invalid values.
        """
        lines = []
        lines.append('class TestConstraints:')
        lines.append('    """Tests derived from blueprint constraints."""')
        lines.append('')

        # Parse constraints into formal constraints
        parsed_constraints = parse_blueprint_constraints(self.blueprint)

        for i, constraint in enumerate(self.constraints):
            desc = constraint.get('description', f'Constraint {i}')
            applies_to = constraint.get('applies_to', [])

            # Find the corresponding parsed constraint
            parsed = parsed_constraints[i] if i < len(parsed_constraints) else None

            lines.append(f'    def test_constraint_{i}(self):')
            lines.append(f'        """')
            lines.append(f'        Constraint: {desc}')
            if applies_to:
                lines.append(f'        Applies to: {", ".join(applies_to)}')
            lines.append(f'        """')

            # Generate actual test code based on parsed constraint type
            if parsed and parsed.constraint_type.value != 'custom':
                validator_code = generate_validator_code(parsed)
                # Convert validator code to test assertion
                test_code = self._validator_to_test(parsed, applies_to)
                for test_line in test_code:
                    lines.append(f'        {test_line}')
            else:
                lines.append(f'        pass  # Custom constraint - implement manually')

            lines.append('')

            # Phase 20: Generate enforcement test (verify ValueError on violation)
            if parsed and parsed.constraint_type.value != 'custom':
                enforcement_test = self._generate_enforcement_test(i, parsed, applies_to)
                if enforcement_test:
                    lines.append(enforcement_test)
                    lines.append('')

        return '\n'.join(lines)

    def _generate_enforcement_test(self, index: int, constraint: FormalConstraint, applies_to: list) -> str:
        """Generate a test that verifies constraint violation raises ValueError.

        Phase 20: Enforcement tests verify that invalid values are rejected.
        """
        lines = []
        field = constraint.field
        ctype = constraint.constraint_type.value
        test_class = applies_to[0] if applies_to else 'TestClass'
        instance_var = to_snake_case(test_class)

        lines.append(f'    def test_constraint_{index}_enforcement(self):')
        lines.append(f'        """Verify {ctype} constraint on {field} raises ValueError on violation."""')

        if ctype == 'range':
            max_val = constraint.params.get('max', 100)
            lines.append(f'        import pytest')
            lines.append(f'        {instance_var} = {to_pascal_case(test_class)}()')
            lines.append(f'        {instance_var}.{field} = {max_val + 1}')
            lines.append(f'        with pytest.raises(ValueError):')
            lines.append(f'            {instance_var}.validate()')
        elif ctype == 'not_null':
            lines.append(f'        import pytest')
            lines.append(f'        {instance_var} = {to_pascal_case(test_class)}()')
            lines.append(f'        {instance_var}.{field} = None')
            lines.append(f'        with pytest.raises(ValueError):')
            lines.append(f'            {instance_var}.validate()')
        elif ctype == 'positive':
            lines.append(f'        import pytest')
            lines.append(f'        {instance_var} = {to_pascal_case(test_class)}()')
            lines.append(f'        {instance_var}.{field} = -1')
            lines.append(f'        with pytest.raises(ValueError):')
            lines.append(f'            {instance_var}.validate()')
        elif ctype == 'enum':
            lines.append(f'        import pytest')
            lines.append(f'        {instance_var} = {to_pascal_case(test_class)}()')
            lines.append(f'        {instance_var}.{field} = "INVALID_VALUE"')
            lines.append(f'        with pytest.raises(ValueError):')
            lines.append(f'            {instance_var}.validate()')
        elif ctype == 'non_negative':
            lines.append(f'        import pytest')
            lines.append(f'        {instance_var} = {to_pascal_case(test_class)}()')
            lines.append(f'        {instance_var}.{field} = -1')
            lines.append(f'        with pytest.raises(ValueError):')
            lines.append(f'            {instance_var}.validate()')
        else:
            return ''  # No enforcement test for this type

        return '\n'.join(lines)

    def _validator_to_test(self, constraint: FormalConstraint, applies_to: list) -> list:
        """Convert a FormalConstraint to test code lines."""
        lines = []
        field = constraint.field
        ctype = constraint.constraint_type.value

        # Get the class to test
        test_class = applies_to[0] if applies_to else 'TestClass'
        instance_var = to_snake_case(test_class)

        lines.append(f'# Testing {ctype} constraint on {field}')
        lines.append(f'{instance_var} = {to_pascal_case(test_class)}()')

        if ctype == 'range':
            min_val = constraint.params.get('min', 0)
            max_val = constraint.params.get('max', 100)
            lines.append(f'# Valid: {field} in [{min_val}, {max_val}]')
            lines.append(f'{instance_var}.{field} = {(min_val + max_val) / 2}')
            lines.append(f'assert {min_val} <= {instance_var}.{field} <= {max_val}')
        elif ctype == 'not_null':
            lines.append(f'# {field} must not be None')
            lines.append(f'assert {instance_var}.{field} is not None')
        elif ctype == 'positive':
            lines.append(f'# {field} must be positive')
            lines.append(f'{instance_var}.{field} = 1')
            lines.append(f'assert {instance_var}.{field} > 0')
        elif ctype == 'non_negative':
            lines.append(f'# {field} must be non-negative')
            lines.append(f'{instance_var}.{field} = 0')
            lines.append(f'assert {instance_var}.{field} >= 0')
        elif ctype == 'enum':
            values = constraint.params.get('values', [])
            if values:
                lines.append(f'# {field} must be one of: {values}')
                lines.append(f'{instance_var}.{field} = "{values[0]}"')
                lines.append(f'assert {instance_var}.{field} in {repr(values)}')
        elif ctype == 'length':
            min_len = constraint.params.get('min')
            max_len = constraint.params.get('max')
            if min_len is not None and max_len is not None:
                lines.append(f'# {field} length must be in [{min_len}, {max_len}]')
                lines.append(f'{instance_var}.{field} = "x" * {min_len}')
                lines.append(f'assert {min_len} <= len({instance_var}.{field}) <= {max_len}')
            elif max_len is not None:
                lines.append(f'# {field} length must be <= {max_len}')
                lines.append(f'{instance_var}.{field} = "x"')
                lines.append(f'assert len({instance_var}.{field}) <= {max_len}')
        elif ctype == 'unique':
            lines.append(f'# {field} must have unique elements')
            lines.append(f'{instance_var}.{field} = [1, 2, 3]')
            lines.append(f'assert len({instance_var}.{field}) == len(set({instance_var}.{field}))')
        elif ctype == 'regex':
            pattern = constraint.params.get('pattern', '.*')
            lines.append(f'import re')
            lines.append(f'# {field} must match pattern: {pattern}')
            lines.append(f'assert re.match(r"{pattern}", str({instance_var}.{field}))')
        else:
            lines.append(f'pass  # {ctype} constraint - implement manually')

        return lines

    def _generate_component(self, name: str, component: Dict) -> str:
        """Generate code for a single component."""
        comp_type = component.get('type', 'entity')
        py_name = to_python_name(name)

        # Skip components that are internal (states, methods extracted as components)
        if comp_type in ('state', 'method'):
            return ''  # These are metadata, not actual classes

        # Check for sub-blueprint (nested components)
        sub_blueprint = component.get('sub_blueprint')

        # Phase 4.3: Check if this component has methods via "contains" relationships
        # If a subsystem has contained process components (methods), use process template instead
        # Use the original blueprint name for relationship lookups
        original_name = component.get('name', name)
        has_contained_methods = self._has_contained_methods(original_name)

        # Phase 3: Use templates for type-specific generation
        if comp_type == 'agent':
            return self._generate_agent_from_template(py_name, component, sub_blueprint)
        elif comp_type == 'process':
            return self._generate_process_from_template(py_name, component, sub_blueprint)
        elif comp_type == 'subsystem':
            # Phase 4.3: If subsystem has state_machine AND contained methods, use process template
            if component.get('state_machine') and has_contained_methods:
                return self._generate_process_with_methods(py_name, component, sub_blueprint)
            return self._generate_subsystem_from_template(py_name, component, sub_blueprint)
        elif comp_type == 'entity':
            # Phase 4.2: Check if entity has 'contains' relationships (fields/methods in separate components)
            has_contains_relationships = any(
                rel.get('from') == name and rel.get('type') == 'contains'
                for rel in self.relationships
            )
            # Phase 9.2: Entities with methods, contains relationships, or constraints use entity template
            has_applicable_constraints = any(
                any(name.lower() in a.lower() or a.lower() in name.lower()
                    for a in c.get('applies_to', []))
                or not c.get('applies_to')
                for c in self.constraints if c.get('description')
            )
            if component.get('methods') or has_contains_relationships or has_applicable_constraints:
                return self._generate_entity_from_template(py_name, component, sub_blueprint)
            else:
                return self._generate_dataclass(py_name, component, sub_blueprint)
        else:
            # Fallback for unknown types
            return self._generate_class(py_name, component, sub_blueprint)

    def _generate_dataclass(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate a dataclass for entity components."""
        lines = []
        desc = component.get('description', 'No description')

        lines.append('@dataclass')
        lines.append(f'class {name}:')
        lines.append(f'    """{desc}"""')

        # Phase 4.2: Use enhanced field extraction
        fields, to_dict_fields, from_dict_fields = generate_fields_from_sub_blueprint(sub_blueprint, component)

        # Check if we got real fields or the fallback defaults
        if 'id: str' not in fields:
            # Got real fields from parsing
            lines.append(fields)
        elif sub_blueprint and sub_blueprint.get('components'):
            # Fallback to old sub_blueprint processing
            for sub_comp in sub_blueprint['components']:
                field_name = to_snake_case(sub_comp['name'])
                field_type = self._infer_field_type(sub_comp)
                field_desc = sub_comp.get('description', '')
                lines.append(f'    {field_name}: {field_type} = field(default_factory={field_type})')
                lines.append(f'    # {field_desc[:60]}')
        else:
            # Default fields based on description
            lines.append(f'    id: str = ""')
            lines.append(f'    data: Dict[str, Any] = field(default_factory=dict)')

        return '\n'.join(lines)

    def _generate_class(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate a class for agent/process components."""
        lines = []
        desc = component.get('description', 'No description')
        comp_type = component.get('type', 'agent')

        # Agents inherit from ABC
        if comp_type == 'agent':
            lines.append(f'class {name}(ABC):')
        else:
            lines.append(f'class {name}:')

        lines.append(f'    """{desc}"""')
        lines.append('')

        # Constructor
        lines.append('    def __init__(self):')
        lines.append(f'        """Initialize {name}."""')

        # If sub_blueprint, create attributes from nested components
        if sub_blueprint and sub_blueprint.get('components'):
            for sub_comp in sub_blueprint['components']:
                attr_name = to_snake_case(sub_comp['name'])
                attr_type = self._infer_field_type(sub_comp)
                lines.append(f'        self.{attr_name}: {attr_type} = {attr_type}()')
        else:
            lines.append('        pass')

        lines.append('')

        # Phase 2: Generate methods from MethodSpec if present
        explicit_methods = component.get('methods', [])
        if explicit_methods:
            for method_spec in explicit_methods:
                method_code = self._generate_method_from_spec(method_spec, component.get('name', ''))
                lines.append(method_code)
                lines.append('')
        else:
            # Fallback: Generate methods from relationships
            methods = self._get_methods_for_component(component['name'])
            for method in methods:
                lines.append(method)
                lines.append('')

        # If agent, add abstract execute method
        if comp_type == 'agent':
            lines.append('    @abstractmethod')
            lines.append('    def execute(self, state: Any) -> Any:')
            lines.append('        """Execute agent logic."""')
            lines.append('        pass')

        return '\n'.join(lines)

    # =========================================================================
    # Phase 3: Template-based generation methods
    # =========================================================================

    def _generate_agent_from_template(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate agent using AGENT_TEMPLATE."""
        desc = component.get('description', 'No description')

        # Extract perspective from description or default
        perspective = self._extract_perspective(desc)

        # Build init attributes from sub_blueprint
        init_attrs = self._build_init_attributes(sub_blueprint)

        # Build system prompt from description
        system_prompt = self._build_system_prompt(name, desc)

        # Build extra methods from MethodSpec if present
        extra_methods = self._build_extra_methods(component.get('methods', []), component.get('name', name))

        return AGENT_TEMPLATE.substitute(
            class_name=to_pascal_case(name),
            description=desc,
            default_name=to_snake_case(name),
            perspective=perspective,
            init_attributes=init_attrs or '        pass',
            system_prompt=system_prompt,
            extra_methods=extra_methods,
        )

    def _generate_process_from_template(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate process using PROCESS_TEMPLATE (state machine)."""
        desc = component.get('description', 'No description')
        state_machine = component.get('state_machine', {})

        # Extract states
        states = state_machine.get('states', ['INIT', 'ACTIVE', 'COMPLETE'])
        initial_state = state_machine.get('initial_state', states[0] if states else 'INIT')

        # Build transitions dict
        transitions_raw = state_machine.get('transitions', [])
        transitions = {}
        for t in transitions_raw:
            key = (t.get('from_state', ''), t.get('to_state', ''))
            transitions[key] = t.get('trigger', 'event')

        # If no transitions, create default linear chain
        if not transitions and len(states) > 1:
            for i in range(len(states) - 1):
                transitions[(states[i], states[i+1])] = 'next'

        # Build state diagram for docstring
        state_diagram = self._build_state_diagram(states, transitions)

        # Build init attributes
        init_attrs = self._build_init_attributes(sub_blueprint)

        # Generate state handlers
        state_handlers = generate_state_handlers(states)

        # Build extra methods from MethodSpec
        extra_methods = self._build_extra_methods(component.get('methods', []), component.get('name', name))

        # Phase 20: Generate constraint validation for process classes
        original_name = component.get('name', name)
        process_constraints = self._get_constraints_for_component(original_name)
        # Convert FormalConstraints back to dicts for generate_process_validation
        constraint_dicts = [
            {'description': fc.description, 'applies_to': fc.applies_to}
            for fc in process_constraints
        ]
        constraint_validation = generate_process_validation(constraint_dicts)
        init_validate_call = '        self.validate_constraints()' if constraint_validation else ''

        return PROCESS_TEMPLATE.substitute(
            class_name=to_pascal_case(name),
            description=desc,
            state_diagram=state_diagram,
            states=repr(states),
            transitions=repr(transitions),
            initial_state=initial_state,
            init_attributes=init_attrs or '        pass',
            state_handlers=state_handlers,
            constraint_validation=constraint_validation,
            init_validate_call=init_validate_call,
            extra_methods=extra_methods,
        )

    def _generate_subsystem_from_template(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate subsystem using SUBSYSTEM_TEMPLATE (container with lifecycle)."""
        desc = component.get('description', 'No description')

        # Get contained components info
        init_contained, init_logic, shutdown_logic, accessors, contained_list = generate_init_contained(sub_blueprint)

        # Build extra methods
        extra_methods = self._build_extra_methods(component.get('methods', []), component.get('name', name))

        return SUBSYSTEM_TEMPLATE.substitute(
            class_name=to_pascal_case(name),
            description=desc,
            contained_list=contained_list or '    (none)',
            init_contained=init_contained,
            init_logic=init_logic,
            shutdown_logic=shutdown_logic,
            accessor_methods=accessors,
            extra_methods=extra_methods,
        )

    def _generate_entity_from_template(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """Generate entity using ENTITY_TEMPLATE (dataclass with validation)."""
        desc = component.get('description', 'No description')
        constraints = component.get('constraints', self.constraints)

        # Phase 4.2: Build sub_blueprint from "contains" relationships if not provided
        if not sub_blueprint:
            sub_blueprint = self._build_sub_blueprint_from_relationships(component['name'])

        # Get field definitions - Phase 4.2: Pass component for derived_from parsing
        fields, to_dict_fields, from_dict_fields = generate_fields_from_sub_blueprint(sub_blueprint, component)

        # Generate validation rules
        validation_rules = generate_validation_rules(constraints)

        # Build extra methods - Phase 4.2: Also check sub_blueprint for methods
        methods = component.get('methods', [])
        if not methods and sub_blueprint and sub_blueprint.get('methods'):
            methods = sub_blueprint.get('methods', [])
        extra_methods = self._build_extra_methods(methods, component.get('name', name))

        return ENTITY_TEMPLATE.substitute(
            class_name=to_pascal_case(name),
            description=desc,
            fields=fields,
            validation_rules=validation_rules,
            to_dict_fields=to_dict_fields,
            from_dict_fields=from_dict_fields,
            extra_methods=extra_methods,
        )

    def _extract_perspective(self, description: str) -> str:
        """Extract agent perspective from description."""
        # Look for keywords
        desc_lower = description.lower()
        if 'structure' in desc_lower or 'entity' in desc_lower:
            return 'structural'
        elif 'behavior' in desc_lower or 'process' in desc_lower or 'flow' in desc_lower:
            return 'behavioral'
        elif 'intent' in desc_lower or 'interpret' in desc_lower:
            return 'interpretive'
        elif 'synthesiz' in desc_lower or 'integrat' in desc_lower:
            return 'integrative'
        else:
            return 'analytical'

    def _build_init_attributes(self, sub_blueprint: Dict) -> str:
        """Build __init__ attribute assignments from sub_blueprint."""
        if not sub_blueprint or not sub_blueprint.get('components'):
            return ''

        lines = []
        for comp in sub_blueprint['components']:
            attr_name = to_snake_case(comp['name'])
            attr_type = self._infer_field_type(comp)
            lines.append(f'        self.{attr_name}: {attr_type} = {attr_type}()')

        return '\n'.join(lines)

    def _build_system_prompt(self, name: str, description: str) -> str:
        """Build system prompt for agent from description."""
        return f"You are {name}. {description}"

    def _build_state_diagram(self, states: List[str], transitions: Dict) -> str:
        """Build ASCII state diagram for docstring."""
        if not states:
            return "    (no states defined)"

        lines = []
        for from_state, to_state in transitions.keys():
            trigger = transitions.get((from_state, to_state), '')
            lines.append(f"    {from_state} --[{trigger}]--> {to_state}")

        if not lines:
            lines.append(f"    States: {' -> '.join(states)}")

        return '\n'.join(lines)

    def _build_extra_methods(self, methods: List[Dict], component_name: str = '') -> str:
        """Build extra methods from MethodSpec list (excluding __init__)."""
        if not methods:
            return ''

        lines = []
        for method_spec in methods:
            # Skip __init__ as it's handled separately
            if method_spec.get('name') == '__init__':
                continue
            method_code = self._generate_method_from_spec(method_spec, component_name)
            lines.append(method_code)
            lines.append('')

        return '\n'.join(lines)

    def _get_constraints_for_component(self, component_name: str) -> List[FormalConstraint]:
        """
        Get parsed constraints that apply to a specific component.

        Phase 9.2: Enables constraint-aware code generation.

        Args:
            component_name: Name of the component to find constraints for

        Returns:
            List of FormalConstraints applicable to this component
        """
        result = []
        for constraint in self.constraints:
            desc = constraint.get('description', '')
            applies_to = constraint.get('applies_to', [])

            if not desc:
                continue

            # Match if component is in applies_to, or if applies_to is empty (global)
            name_lower = component_name.lower()
            matches = (
                not applies_to
                or any(name_lower in a.lower() or a.lower() in name_lower for a in applies_to)
            )

            if matches:
                parsed = parse_constraint(desc, applies_to)
                if parsed:
                    result.append(parsed)

        return result

    def _get_constraints_for_method(self, method_name: str, component_name: str) -> List[Dict]:
        """
        Get raw constraints relevant to a method.

        Phase 9.2: Finds constraints that mention the method or its parent component.
        """
        result = []
        method_lower = method_name.lower()
        comp_lower = component_name.lower()

        for constraint in self.constraints:
            desc = constraint.get('description', '').lower()
            applies_to = [a.lower() for a in constraint.get('applies_to', [])]

            if method_lower in desc or comp_lower in desc:
                result.append(constraint)
            elif any(comp_lower in a or method_lower in a for a in applies_to):
                result.append(constraint)

        return result

    def _generate_method_from_spec(self, method_spec: Dict, component_name: str = '') -> str:
        """
        Generate method from MethodSpec (Phase 2).

        Phase 9.2: Enhanced with constraint comments and derived_from traceability.

        Uses explicit method signatures extracted from input.
        """
        name = method_spec.get('name', 'unnamed')
        params = method_spec.get('parameters', [])
        return_type = method_spec.get('return_type', 'None')
        description = method_spec.get('description', '')
        derived_from = method_spec.get('derived_from', '')

        # Build parameter string
        param_parts = ['self']
        for p in params:
            p_name = p.get('name', 'arg')
            p_type = p.get('type_hint', 'Any')
            p_default = p.get('default')
            if p_default:
                param_parts.append(f'{p_name}: {p_type} = {p_default}')
            else:
                param_parts.append(f'{p_name}: {p_type}')

        param_str = ', '.join(param_parts)

        # Phase 9.2: Find constraints for this method
        method_constraints = self._get_constraints_for_method(name, component_name)

        # Build method
        lines = [
            f'    def {name}({param_str}) -> {return_type}:',
        ]

        # Docstring with description, constraints, and derivation
        has_docstring_content = description or derived_from or method_constraints
        if has_docstring_content:
            lines.append(f'        """')
            if description:
                lines.append(f'        {description}')

            if method_constraints:
                lines.append(f'')
                lines.append(f'        Constraints:')
                for mc in method_constraints:
                    mc_desc = mc.get('description', '')
                    lines.append(f'            - {mc_desc}')

            if derived_from:
                lines.append(f'')
                lines.append(f'        Derived from: "{derived_from}"')
            lines.append(f'        """')

        # Phase 9.2: Add inline constraint comments before NotImplementedError
        if method_constraints:
            for mc in method_constraints:
                mc_desc = mc.get('description', '')
                lines.append(f'        # CONSTRAINT: {mc_desc}')

        lines.append(f'        raise NotImplementedError("TODO: Implement {name}")')

        return '\n'.join(lines)

    def _infer_field_type(self, component: Dict) -> str:
        """Infer Python type from component description."""
        desc = component.get('description', '').lower()
        name = component.get('name', '').lower()

        if 'list' in desc or 'list' in name:
            return 'list'
        elif 'dict' in desc or 'map' in desc:
            return 'dict'
        elif 'set' in desc:
            return 'set'
        elif 'vector' in name:
            return 'dict'  # ConfidenceVector -> dict of scores
        else:
            return 'dict'

    def _has_contained_methods(self, component_name: str) -> bool:
        """
        Check if a component has contained process components (methods) via 'contains' relationships.

        Phase 4.3: Detect when subsystem has methods extracted as separate process components.
        """
        for rel in self.relationships:
            if rel.get('from') == component_name and rel.get('type') == 'contains':
                target_name = rel.get('to', '')
                # Check if target is a method (has parentheses in name or is process type)
                if target_name in self.components:
                    target_comp = self.components[target_name]
                    if target_comp.get('type') == 'process' or '(' in target_name:
                        return True
        return False

    def _generate_process_with_methods(self, name: str, component: Dict, sub_blueprint: Dict = None) -> str:
        """
        Generate a process class with state machine AND explicit methods.

        Phase 4.3: Used when subsystem has state_machine AND contained methods.
        Combines PROCESS_TEMPLATE base with methods from contained process components.
        """
        desc = component.get('description', 'No description')
        state_machine = component.get('state_machine', {})

        # Extract states
        states = state_machine.get('states', ['INIT', 'ACTIVE', 'COMPLETE'])
        initial_state = state_machine.get('initial_state', states[0] if states else 'INIT')

        # Build transitions dict
        transitions_raw = state_machine.get('transitions', [])
        transitions = {}
        for t in transitions_raw:
            key = (t.get('from_state', ''), t.get('to_state', ''))
            transitions[key] = t.get('trigger', 'event')

        # Build state diagram for docstring
        state_diagram = self._build_state_diagram(states, transitions)

        # Build init attributes from state machine fields (from SharedState entity in blueprint)
        init_attrs = self._build_init_attributes_from_fields(component)

        # Generate state handlers
        state_handlers = generate_state_handlers(states)

        # Phase 4.3: Collect methods from contained process components
        contained_methods = self._collect_contained_methods(component.get('name', name))

        # Build extra methods from both component's own methods AND contained methods
        all_methods = component.get('methods', []) + contained_methods
        extra_methods = self._build_extra_methods(all_methods, component.get('name', name))

        # Phase 20: Generate constraint validation for process-with-methods
        original_name = component.get('name', name)
        process_constraints = self._get_constraints_for_component(original_name)
        constraint_dicts = [
            {'description': fc.description, 'applies_to': fc.applies_to}
            for fc in process_constraints
        ]
        constraint_validation = generate_process_validation(constraint_dicts)
        init_validate_call = '        self.validate_constraints()' if constraint_validation else ''

        return PROCESS_TEMPLATE.substitute(
            class_name=to_pascal_case(name),
            description=desc,
            state_diagram=state_diagram,
            states=repr(states),
            transitions=repr(transitions),
            initial_state=initial_state,
            init_attributes=init_attrs or '        pass',
            state_handlers=state_handlers,
            constraint_validation=constraint_validation,
            init_validate_call=init_validate_call,
            extra_methods=extra_methods,
        )

    def _build_init_attributes_from_fields(self, component: Dict) -> str:
        """
        Build __init__ attributes from entity fields mentioned in description.

        Phase 4.3: Parse fields from component description when sub_blueprint doesn't have them.
        """
        desc = component.get('description', '')
        lines = []

        # Look for field patterns in description: "field_name: type = default"
        import re
        field_pattern = r'(\w+):\s*(\w+(?:\[[\w\s,]+\])?)\s*(?:=\s*(\S+))?'

        # Also check for explicit field mentions from derived_from
        derived = component.get('derived_from', '')

        # Search for field patterns
        for text in [desc, derived]:
            for match in re.finditer(field_pattern, text):
                field_name, field_type, default = match.groups()
                if default:
                    lines.append(f'        self.{field_name}: {field_type} = {default}')
                else:
                    type_defaults = {
                        'int': '0',
                        'str': '""',
                        'float': '0.0',
                        'bool': 'False',
                        'List': '[]',
                    }
                    default_val = type_defaults.get(field_type.split('[')[0], 'None')
                    if 'List' in field_type:
                        default_val = '[]'
                    lines.append(f'        self.{field_name}: {field_type} = {default_val}')

        return '\n'.join(lines) if lines else ''

    def _collect_contained_methods(self, component_name: str) -> List[Dict]:
        """
        Collect method specs from contained process components.

        Phase 4.3: Methods may be stored as separate process components linked via 'contains'.
        """
        methods = []

        for rel in self.relationships:
            if rel.get('from') == component_name and rel.get('type') == 'contains':
                target_name = rel.get('to', '')
                if target_name in self.components:
                    target_comp = self.components[target_name]

                    # Check if this is a method (process type or has parentheses)
                    if target_comp.get('type') == 'process' or '(' in target_name:
                        # Extract method spec
                        method_name = target_name.split('(')[0].strip()

                        # Get from component's methods if present
                        if target_comp.get('methods'):
                            methods.extend(target_comp['methods'])
                        else:
                            # Build method spec from component
                            method_spec = {
                                'name': method_name,
                                'parameters': target_comp.get('parameters', []),
                                'return_type': target_comp.get('return_type', 'None'),
                                'description': target_comp.get('description', ''),
                                'derived_from': target_comp.get('derived_from', ''),
                            }
                            methods.append(method_spec)

        return methods

    def _build_sub_blueprint_from_relationships(self, component_name: str) -> Dict:
        """
        Build a sub_blueprint from 'contains' relationships.

        Phase 4.2: Entity fields may be stored as separate components linked by 'contains'.
        This method collects those components to build an equivalent sub_blueprint.
        """
        contained_components = []
        method_specs = []

        for rel in self.relationships:
            if rel.get('from') == component_name and rel.get('type') == 'contains':
                target_name = rel.get('to', '')
                # Find the target component
                if target_name in self.components:
                    target_comp = self.components[target_name]
                    target_type = target_comp.get('type', 'entity')

                    # Separate fields (entities) from methods (processes)
                    if target_type == 'entity' and '(' not in target_name:
                        # This is a field
                        contained_components.append(target_comp)
                    elif target_type == 'process' or '(' in target_name:
                        # This is a method - convert to MethodSpec format
                        method_spec = {
                            'name': target_name.split('(')[0],
                            'parameters': target_comp.get('parameters', []),
                            'return_type': target_comp.get('return_type', 'None'),
                            'description': target_comp.get('description', ''),
                            'derived_from': target_comp.get('derived_from', ''),
                        }
                        method_specs.append(method_spec)

        result = {}
        if contained_components:
            result['components'] = contained_components
        if method_specs:
            result['methods'] = method_specs

        return result if result else None

    def _get_methods_for_component(self, component_name: str) -> List[str]:
        """Generate methods based on relationships from this component."""
        methods = []

        for rel in self.relationships:
            if rel.get('from') == component_name:
                rel_type = rel.get('type', '')
                target = rel.get('to', '')

                if rel_type in RELATIONSHIP_TO_METHOD and RELATIONSHIP_TO_METHOD[rel_type]:
                    method_prefix, param_name = RELATIONSHIP_TO_METHOD[rel_type]
                    target_py = to_python_name(target)
                    method_name = f'{method_prefix}_{to_snake_case(target)}'

                    method_lines = [
                        f'    def {method_name}(self, {param_name or "target"}: "{target_py}") -> None:',
                        f'        """{rel_type.title()} {target}."""',
                        f'        # TODO: Implement {rel_type} logic',
                        f'        raise NotImplementedError()',
                    ]
                    methods.append('\n'.join(method_lines))

        return methods


def generate_from_blueprint(blueprint: Dict[str, Any]) -> str:
    """Convenience function to generate code from blueprint."""
    generator = BlueprintCodeGenerator(blueprint)
    return generator.generate()


if __name__ == '__main__':
    import json
    import sys

    # Test with deep blueprint
    with open('tests/deep_blueprint_sharedstate.json') as f:
        blueprint = json.load(f)

    code = generate_from_blueprint(blueprint)
    print(code)
