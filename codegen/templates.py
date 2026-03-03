"""
Code Templates for Motherlabs CodeGen.

Phase 3: Generate runnable code, not just stubs.

Templates provide:
- Agent: BaseAgent pattern with run(), observe(), _build_context()
- Entity: Dataclass with validation, serialization, factory
- Process: State machine with transitions, execute(), current_state
- Subsystem: Container with lifecycle, init(), shutdown()
"""
from typing import Dict, Any, List, Optional
from string import Template

from core.schema import parse_constraint, generate_validator_code


# =============================================================================
# AGENT TEMPLATE
# Derived from: core/protocol.py BaseAgent pattern
# =============================================================================

AGENT_TEMPLATE = Template('''class ${class_name}(BaseAgent):
    """${description}"""

    def __init__(self, name: str = "${default_name}", llm_client=None):
        """Initialize ${class_name}."""
        super().__init__(name=name, perspective="${perspective}", llm_client=llm_client)
${init_attributes}

    def run(self, state: SharedState, input_msg: Optional[Message] = None) -> Message:
        """Execute agent logic."""
        if not self.llm:
            raise ValueError(f"Agent {self.name} has no LLM client")

        context = self._build_context(state)

        if input_msg:
            user_content = f"[{input_msg.sender}]: {input_msg.content}"
        else:
            user_content = "Begin analysis."

        response = self.llm.complete_with_system(
            system_prompt=self._get_system_prompt() + "\\n\\n" + context,
            user_content=user_content
        )

        return Message(
            sender=self.name,
            content=response,
            message_type=MessageType.PROPOSITION
        )

    def _get_system_prompt(self) -> str:
        """Return system prompt for this agent."""
        return """${system_prompt}"""

    def _build_context(self, state: SharedState) -> str:
        """Build context string from shared state."""
        recent = state.get_recent(5)
        history_str = "\\n".join([
            f"[{m.sender}] {m.content[:200]}"
            for m in recent
        ])
        return f"""
PERSPECTIVE: {self.perspective}

RECENT DIALOGUE:
{history_str or "Starting fresh"}

KNOWN:
{list(state.known.keys()) or "Nothing yet"}
"""
${extra_methods}
''')


# =============================================================================
# ENTITY TEMPLATE
# Derived from: core/protocol.py Message, ConfidenceVector patterns
# =============================================================================

ENTITY_TEMPLATE = Template('''@dataclass
class ${class_name}:
    """${description}"""
${fields}

    def __post_init__(self):
        """Validate constraints on construction."""
        self.validate()

    def validate(self) -> bool:
        """Validate entity state. Raises ValueError on constraint violation."""
${validation_rules}
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
${to_dict_fields}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "${class_name}":
        """Deserialize from dictionary."""
        return cls(
${from_dict_fields}
        )
${extra_methods}
''')


# =============================================================================
# PROCESS TEMPLATE
# Derived from: DialogueProtocol state machine pattern
# =============================================================================

PROCESS_TEMPLATE = Template('''class ${class_name}:
    """
    ${description}

    State Machine:
${state_diagram}
    """

    # Valid states
    STATES = ${states}

    # Valid transitions: (from_state, to_state) -> trigger
    TRANSITIONS = ${transitions}

    def __init__(self):
        """Initialize ${class_name}."""
        self._state: str = "${initial_state}"
${init_attributes}
${init_validate_call}

    @property
    def current_state(self) -> str:
        """Get current state."""
        return self._state

    def can_transition(self, to_state: str) -> bool:
        """Check if transition to target state is valid."""
        return (self._state, to_state) in self.TRANSITIONS

    def transition(self, to_state: str) -> bool:
        """
        Attempt state transition.

        Returns True if successful, False if invalid.
        """
        if not self.can_transition(to_state):
            return False

        # Execute transition
        trigger = self.TRANSITIONS[(self._state, to_state)]
        self._on_transition(self._state, to_state, trigger)
        self._state = to_state
        return True

    def _on_transition(self, from_state: str, to_state: str, trigger: str):
        """Hook for transition side effects. Override in subclass."""
        pass

    def execute(self) -> Any:
        """Execute current state logic."""
        handler = getattr(self, f"_handle_{self._state.lower()}", None)
        if handler:
            return handler()
        return None
${state_handlers}
${constraint_validation}
${extra_methods}
''')


# =============================================================================
# SUBSYSTEM TEMPLATE
# Derived from: SharedState container pattern
# =============================================================================

SUBSYSTEM_TEMPLATE = Template('''class ${class_name}:
    """
    ${description}

    Subsystem containing:
${contained_list}
    """

    def __init__(self):
        """Initialize ${class_name} and all contained components."""
${init_contained}
        self._initialized = False

    def init(self) -> bool:
        """Initialize subsystem. Returns True on success."""
        if self._initialized:
            return True

        try:
${init_logic}
            self._initialized = True
            return True
        except Exception as e:
            return False

    def shutdown(self):
        """Clean shutdown of subsystem."""
        if not self._initialized:
            return

${shutdown_logic}
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if subsystem is initialized."""
        return self._initialized
${accessor_methods}
${extra_methods}
''')


# =============================================================================
# TEMPLATE SELECTOR
# =============================================================================

def get_template_for_type(component_type: str) -> Template:
    """Get appropriate template for component type."""
    templates = {
        'agent': AGENT_TEMPLATE,
        'entity': ENTITY_TEMPLATE,
        'process': PROCESS_TEMPLATE,
        'subsystem': SUBSYSTEM_TEMPLATE,
        'interface': AGENT_TEMPLATE,  # Interfaces are agent-like
    }
    return templates.get(component_type, ENTITY_TEMPLATE)


# =============================================================================
# TEMPLATE HELPERS
# =============================================================================

def generate_state_handlers(states: List[str]) -> str:
    """Generate stub handlers for each state."""
    lines = []
    for state in states:
        lines.append(f'''
    def _handle_{state.lower()}(self) -> Any:
        """Handle {state} state."""
        # TODO: Implement {state} logic
        pass''')
    return '\n'.join(lines)


def generate_validation_rules(constraints: List[Dict]) -> str:
    """Generate validation code from constraints using formal constraint parsing.

    Phase 6.3: Uses parse_constraint() and generate_validator_code() from core/schema
    to handle all constraint types (range, not_null, enum, length, positive, etc.)
    instead of only simple "bounded [x, y]" patterns.
    """
    if not constraints:
        return '        # No validation rules defined'

    lines = []
    for c in constraints:
        desc = c.get('description', '')
        applies_to = c.get('applies_to', [])

        if not desc:
            continue

        parsed = parse_constraint(desc, applies_to)
        if parsed and parsed.constraint_type.value != 'custom':
            # Generate validator code (e.g. 'assert 0 <= self.confidence <= 1, "..."')
            validator = generate_validator_code(parsed)
            # Handle import prefix (e.g. "import re; assert ...")
            if validator.startswith('import re; '):
                lines.append('        import re')
                validator = validator[len('import re; '):]
            # Convert assert-style to if-raise-ValueError style for validate() method
            if validator.startswith('assert '):
                rest = validator[len('assert '):]
                # Split condition from message — handle both regular and f-string messages
                # Try f-string first: 'COND, f"MSG"'
                parts_f = rest.rsplit(', f"', 1)
                parts_plain = rest.rsplit(', "', 1)
                if len(parts_f) == 2 and (len(parts_plain) != 2 or len(parts_f[0]) < len(parts_plain[0])):
                    # f-string message
                    condition = parts_f[0]
                    message = parts_f[1].rstrip('"')
                    lines.append(f'        if not ({condition}):')
                    lines.append(f'            raise ValueError(f"{message}")')
                elif len(parts_plain) == 2:
                    condition = parts_plain[0]
                    message = parts_plain[1].rstrip('"')
                    lines.append(f'        if not ({condition}):')
                    lines.append(f'            raise ValueError("{message}")')
                else:
                    lines.append(f'        if not ({rest}):')
                    lines.append(f'            raise ValueError("{desc}")')
            else:
                lines.append(f'        # {desc}')
        else:
            # Custom constraint - add as comment
            lines.append(f'        # TODO: {desc}')

    return '\n'.join(lines) if lines else '        # No validation rules defined'


def generate_process_validation(constraints: List[Dict]) -> str:
    """Generate validate_constraints() method for process classes.

    Phase 20: Process classes get constraint enforcement too.
    Uses the same constraint parsing as entity validation but wrapped
    in a validate_constraints() method instead of validate().
    """
    if not constraints:
        return ''

    lines = []
    has_rules = False
    rule_lines = []

    for c in constraints:
        desc = c.get('description', '')
        applies_to = c.get('applies_to', [])

        if not desc:
            continue

        parsed = parse_constraint(desc, applies_to)
        if parsed and parsed.constraint_type.value != 'custom':
            validator = generate_validator_code(parsed)
            if validator.startswith('import re; '):
                rule_lines.append('        import re')
                validator = validator[len('import re; '):]
            if validator.startswith('assert '):
                rest = validator[len('assert '):]
                parts_f = rest.rsplit(', f"', 1)
                parts_plain = rest.rsplit(', "', 1)
                if len(parts_f) == 2 and (len(parts_plain) != 2 or len(parts_f[0]) < len(parts_plain[0])):
                    condition = parts_f[0]
                    message = parts_f[1].rstrip('"')
                    rule_lines.append(f'        if not ({condition}):')
                    rule_lines.append(f'            raise ValueError(f"{message}")')
                elif len(parts_plain) == 2:
                    condition = parts_plain[0]
                    message = parts_plain[1].rstrip('"')
                    rule_lines.append(f'        if not ({condition}):')
                    rule_lines.append(f'            raise ValueError("{message}")')
                else:
                    rule_lines.append(f'        if not ({rest}):')
                    rule_lines.append(f'            raise ValueError("{desc}")')
                has_rules = True
            else:
                rule_lines.append(f'        # {desc}')
        else:
            rule_lines.append(f'        # TODO: {desc}')

    if not has_rules:
        return ''

    lines.append('    def validate_constraints(self) -> bool:')
    lines.append('        """Validate process constraints. Raises ValueError on violation."""')
    lines.extend(rule_lines)
    lines.append('        return True')

    return '\n'.join(lines)


def _get_default_for_type(type_hint: str) -> str:
    """Get default value for a type hint."""
    # Handle Optional types
    if type_hint.startswith('Optional['):
        return 'None'

    # Handle List types
    if type_hint.startswith('List['):
        return '[]'

    # Handle Dict types
    if type_hint.startswith('Dict['):
        return '{}'

    # Handle Set types
    if type_hint.startswith('Set['):
        return 'set()'

    # Basic types
    defaults = {
        'str': '""',
        'int': '0',
        'float': '0.0',
        'bool': 'False',
        'list': '[]',
        'dict': '{}',
        'set': 'set()',
        'Any': 'None',
    }
    return defaults.get(type_hint, 'None')


def parse_field_from_description(description: str) -> Optional[Dict[str, Any]]:
    """
    Parse field definition from description like "float = 0.0 - Entity Agent's confidence".

    Phase 4.2: Enhanced field parsing from blueprint descriptions.

    Returns dict with: name (optional), type, default, description
    """
    import re

    # Pattern: "name: type = default - description" or "type = default - description"
    # Examples:
    #   "structural: float = 0.0 - Entity Agent's confidence (0.0-1.0)"
    #   "float = 0.0 - Entity Agent's confidence"

    patterns = [
        # Full pattern with name: "name: type = default"
        r'^(\w+):\s*(\w+)\s*=\s*([^\s-]+)\s*(?:-\s*(.*))?$',
        # Type with default: "type = default - desc"
        r'^(\w+)\s*=\s*([^\s-]+)\s*(?:-\s*(.*))?$',
    ]

    for pattern in patterns:
        match = re.match(pattern, description.strip())
        if match:
            groups = match.groups()
            if len(groups) == 4:  # Full pattern
                return {
                    'name': groups[0],
                    'type': groups[1],
                    'default': groups[2],
                    'description': groups[3] or ''
                }
            elif len(groups) == 3:  # Type + default
                return {
                    'type': groups[0],
                    'default': groups[1],
                    'description': groups[2] or ''
                }

    return None


def generate_fields_from_sub_blueprint(sub_blueprint: Dict, component: Dict = None) -> tuple:
    """
    Generate field definitions from sub-blueprint components or component description.

    Phase 4.2: Enhanced to parse field definitions from descriptions.
    Phase 4.4: Now uses type_hint and default_value from blueprint components.

    Priority order:
    1. sub_blueprint components (from 'contains' relationships) - most reliable
    2. Parsing derived_from patterns - fallback
    3. Default fields - last resort
    """
    import re

    fields = []
    to_dict = []
    from_dict = []

    # First try: Use sub_blueprint components (most reliable)
    # These come from _build_sub_blueprint_from_relationships which correctly
    # identifies contained entity components as fields
    if sub_blueprint and sub_blueprint.get('components'):
        for comp in sub_blueprint['components']:
            name = comp['name'].lower().replace(' ', '_')
            desc = comp.get('description', '')

            # Phase 4.4: Check for explicit type_hint from blueprint
            type_hint = comp.get('type_hint')
            default_value = comp.get('default_value')

            # Try to parse field type from description
            parsed = parse_field_from_description(desc)

            # Phase 9.2: Add inline description comment if available
            derived_from = comp.get('derived_from', '')
            field_comment = ''
            if desc and not desc.startswith(('float', 'int', 'str', 'bool', 'List', 'Dict')):
                # Description is actual text, not just a type hint
                field_comment = f'  # {desc[:80]}'
            elif derived_from:
                field_comment = f'  # Derived from: {derived_from[:60]}'

            # Phase 4.4: If we have explicit type_hint, use it
            if type_hint:
                py_type = type_hint
                default = default_value if default_value else _get_default_for_type(type_hint)

                fields.append(f'    {name}: {py_type} = {default}{field_comment}')
                to_dict.append(f'            "{name}": self.{name},')
                from_dict.append(f'            {name}=data.get("{name}", {default}),')
                continue

            if parsed:
                typ = parsed.get('type', 'Any')
                default = parsed.get('default', 'None')

                # Map types
                type_map = {
                    'float': 'float',
                    'int': 'int',
                    'str': 'str',
                    'bool': 'bool',
                    'List': 'List[Any]',
                    'Dict': 'Dict[str, Any]',
                }
                py_type = type_map.get(typ, 'Any')

                if py_type == 'float':
                    fields.append(f'    {name}: {py_type} = {default}{field_comment}')
                    to_dict.append(f'            "{name}": self.{name},')
                    from_dict.append(f'            {name}=data.get("{name}", {default}),')
                else:
                    fields.append(f'    {name}: {py_type} = {default}{field_comment}')
                    to_dict.append(f'            "{name}": self.{name},')
                    from_dict.append(f'            {name}=data.get("{name}"),')
            else:
                # Fallback: Infer type from description keywords
                if 'list' in desc.lower():
                    typ = 'List[Any]'
                    default = 'field(default_factory=list)'
                    to_dict.append(f'            "{name}": self.{name},')
                    from_dict.append(f'            {name}=data.get("{name}", []),')
                elif 'dict' in desc.lower() or 'map' in desc.lower():
                    typ = 'Dict[str, Any]'
                    default = 'field(default_factory=dict)'
                    to_dict.append(f'            "{name}": self.{name},')
                    from_dict.append(f'            {name}=data.get("{name}", {{}}),')
                elif 'set' in desc.lower():
                    typ = 'Set[Any]'
                    default = 'field(default_factory=set)'
                    to_dict.append(f'            "{name}": list(self.{name}),')
                    from_dict.append(f'            {name}=set(data.get("{name}", [])),')
                else:
                    typ = 'Any'
                    default = 'None'
                    to_dict.append(f'            "{name}": self.{name},')
                    from_dict.append(f'            {name}=data.get("{name}"),')

                fields.append(f'    {name}: {typ} = {default}{field_comment}')

        if fields:
            return ('\n'.join(fields), '\n'.join(to_dict), '\n'.join(from_dict))

    # Fallback: Default fields
    return (
        '    id: str = ""\n    data: Dict[str, Any] = field(default_factory=dict)',
        '            "id": self.id,\n            "data": self.data,',
        '            id=data.get("id", ""),\n            data=data.get("data", {}),',
    )


def generate_init_contained(sub_blueprint: Dict) -> tuple:
    """Generate initialization for contained components."""
    if not sub_blueprint or not sub_blueprint.get('components'):
        return ('        pass', '            pass', '        pass', '')

    init_lines = []
    init_logic = []
    shutdown_logic = []
    accessors = []

    for comp in sub_blueprint['components']:
        name = comp['name'].lower().replace(' ', '_')
        init_lines.append(f'        self._{name} = {{}}')
        init_logic.append(f'            # Initialize {name}')
        shutdown_logic.append(f'        self._{name}.clear()')

        # Property accessor
        accessors.append(f'''
    @property
    def {name}(self):
        """Access {comp['name']}."""
        return self._{name}''')

    contained_list = '\n'.join([f"    - {c['name']}" for c in sub_blueprint['components']])

    return (
        '\n'.join(init_lines),
        '\n'.join(init_logic) or '            pass',
        '\n'.join(shutdown_logic) or '        pass',
        '\n'.join(accessors),
        contained_list,
    )
