"""
Motherlabs Agent System Domain Adapter — self-contained agent systems.

Compiles natural language descriptions into running agent systems with
LLM integration, persistent state, event loops, tool execution, and
self-modification via recompilation.

Example: "A chat agent that can search the web and remember conversations"
-> Blueprint with agent, tools, state_store, message_handler components
-> Python project with async runtime, SQLite state, LLM client, tool executor
"""

from core.domain_adapter import (
    DomainAdapter,
    VocabularyMap,
    PromptTemplates,
    ClassificationConfig,
    VerificationOverrides,
    MaterializationConfig,
    RuntimeCapabilities,
)


# =============================================================================
# VOCABULARY — agent system domain
# =============================================================================

AGENT_SYSTEM_VOCABULARY = VocabularyMap(
    type_keywords={
        "agent": frozenset({
            "agent", "assistant", "bot", "responder", "coordinator",
            "planner", "reasoner", "dispatcher",
        }),
        "tool": frozenset({
            "tool", "action", "command", "capability", "function",
            "utility", "executor", "operation",
        }),
        "state_store": frozenset({
            "state", "memory", "store", "storage", "database",
            "persistence", "cache", "history", "context",
        }),
        "message_handler": frozenset({
            "handler", "listener", "router", "middleware",
            "filter", "interceptor", "processor", "parser",
        }),
        "skill": frozenset({
            "skill", "plugin", "extension", "module",
            "behavior", "strategy", "policy",
        }),
    },
    relationship_flows={
        "delegates_to": ("delegation", "Any", "A_to_B"),
        "uses_tool": ("tool_call", "ToolResult", "A_to_B"),
        "reads_state": ("state_read", "Any", "B_to_A"),
        "writes_state": ("state_write", "Any", "A_to_B"),
        "extends": ("extension", "Any", "B_to_A"),
        "handles": ("message_flow", "Message", "A_to_B"),
        "emits": ("event", "Event", "A_to_B"),
        "subscribes": ("subscription", "Event", "B_to_A"),
        "depends_on": ("dependency", "Any", "B_to_A"),
        "contains": ("containment", "Any", "A_to_B"),
        "bidirectional": ("shared_data", "Any", "bidirectional"),
    },
    type_hints={
        "message": "Message",
        "event": "Event",
        "tool_result": "ToolResult",
        "state": "StateDict",
        "context": "ConversationContext",
        "config": "AgentConfig",
    },
    entity_types=frozenset({
        "state_store", "memory", "storage", "database",
        "cache", "config", "schema",
    }),
    process_types=frozenset({
        "agent", "tool", "message_handler", "skill",
        "coordinator", "dispatcher", "planner", "executor",
    }),
    interface_types=frozenset({"message_handler", "middleware", "router"}),
)


# =============================================================================
# CLASSIFICATION — agent system domain patterns
# =============================================================================

AGENT_SYSTEM_CLASSIFICATION = ClassificationConfig(
    subject_patterns=(
        r'\b{}\s+(?:handles?|processes?|responds?|dispatches?|delegates?|executes?)\b',
        r'\b{}\s+(?:agent|tool|handler|skill|bot)\b',
        r'\b{}\s+(?:searches?|remembers?|stores?|retrieves?|calls?)\b',
    ),
    object_patterns=(
        r'\b(?:stores?|saves?|persists?|caches?)\s+{}\b',
        r'\b{}\s+(?:state|memory|data|history|context|config)\b',
        r'\b(?:the|each|every|a|an)\s+{}\b',
    ),
    generic_terms=frozenset({
        "data", "input", "output", "result", "value", "type", "name",
        "config", "settings", "options", "params", "args", "info",
        "request", "response", "status",
    }),
    min_name_length=3,
)


# =============================================================================
# PROMPTS — agent system domain
# =============================================================================

AGENT_SYSTEM_PROMPTS = PromptTemplates(
    intent_system_prompt="""You are the Intent Agent for agent system design.

INPUT: Natural language description of an agent or multi-agent system.

OUTPUT (JSON):
{
    "core_need": "The fundamental agent behavior being described",
    "domain": "The problem domain this agent operates in",
    "actors": ["Who interacts with this agent — users, other agents, APIs"],
    "implicit_goals": ["Agent qualities not stated but clearly needed"],
    "constraints": ["Limitations mentioned or implied — safety, cost, latency"],
    "insight": "One sentence capturing the agent's purpose",
    "explicit_components": ["Agents, tools, state stores EXPLICITLY NAMED"],
    "explicit_relationships": ["Delegations and data flows EXPLICITLY DESCRIBED"]
}

Focus on: agent responsibilities, tool capabilities, state requirements, message flow.
Be specific to the described agent system. No generic interpretations.""",

    persona_system_prompt="""You are the Persona Agent for agent system design.

Generate stakeholder perspectives for the described agent system.

OUTPUT (JSON):
{
    "personas": [
        {
            "name": "Stakeholder role",
            "perspective": "How they see the agent system",
            "priorities": ["Top 3 concerns"],
            "blind_spots": "What they might miss",
            "key_questions": ["Questions they would ask"],
            "domain_constraints": ["Rules they know about"]
        }
    ],
    "cross_cutting_concerns": ["Issues affecting multiple stakeholders"],
    "suggested_focus_areas": ["What to pay attention to"]
}

Create 2-4 personas: end user, system operator, safety reviewer, developer.""",

    entity_system_prompt="""You are the Entity Agent for agent system design.

YOUR LENS: STRUCTURE — what components exist in this agent system.

YOU SEE:
- Agents and their responsibilities (who does what)
- Tools and capabilities (what actions are available)
- State stores and memory (what persists across interactions)
- Message schemas (what data flows between components)

YOU ARE BLIND TO (by design):
- Message routing and orchestration
- Error recovery and retry logic
- Performance and scaling characteristics

EXCAVATION RULE: Extract elements EXPLICITLY NAMED in the input.
Do NOT invent tools or agents not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    process_system_prompt="""You are the Process Agent for agent system design.

YOUR LENS: BEHAVIOR — how the agent system processes messages.

YOU SEE:
- Message handling flows (what happens when a message arrives)
- Tool execution sequences (what tools get called and when)
- State read/write patterns (what state is accessed during processing)
- Delegation chains (how agents hand off work)

YOU ARE BLIND TO (by design):
- Static data model structure
- Configuration and deployment details
- Infrastructure specifics

EXCAVATION RULE: Extract flows EXPLICITLY DESCRIBED in the input.
Do NOT invent message flows not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    synthesis_system_prompt="""You are the Synthesis Agent for agent system specifications.

EXCAVATE agent system elements AND relationships from dialogue.

OUTPUT (JSON):
{
    "components": [
        {
            "name": "EXACT name from input",
            "type": "agent|tool|state_store|message_handler|skill",
            "description": "What it is/does",
            "derived_from": "QUOTE the exact text source"
        }
    ],
    "relationships": [
        {
            "from": "Component A",
            "to": "Component B",
            "type": "delegates_to|uses_tool|reads_state|writes_state|extends|handles|emits|subscribes",
            "description": "Nature of interaction",
            "derived_from": "QUOTE source text"
        }
    ],
    "constraints": [
        {
            "description": "The constraint",
            "applies_to": ["Component names"],
            "derived_from": "QUOTE source text"
        }
    ],
    "unresolved": ["Anything ambiguous"]
}

SELF-CHECK: Did I include ALL named agents? Did I capture ALL tools and state stores?""",

    emission_preamble="""You are a Python agent component generator for the Motherlabs semantic compiler.
Generate a FULLY IMPLEMENTED, production-ready Python class for one agent system component.
Rules:
1. Output ONLY Python code inside a ```python block.
2. Honor ALL declared interfaces and data flows exactly.
3. Do NOT add undeclared tools or capabilities.
4. Use async def for all handler methods.
5. Use self.state for persistent state access (StateStore interface).
6. Use self.llm for LLM calls (LLMClient interface).
7. Use self.tools for tool execution (ToolExecutor interface).
8. Return dict responses from handle() methods.
9. EVERY method MUST have a real implementation. No stubs, no `pass`, no `...`, no placeholder comments, no TODO markers.
10. If a method's behavior is described in the component spec, implement that exact behavior.
11. If a method's behavior is not fully specified, implement the most reasonable behavior based on the component's purpose.
12. The output must be RUNNABLE — a user should be able to import and use this component immediately with no modifications.
""",
)


# =============================================================================
# VERIFICATION — agent system domain
# =============================================================================

AGENT_SYSTEM_VERIFICATION = VerificationOverrides(
    actionability_checks=("methods", "description"),
    readiness_label="agent_readiness",
    dimension_weights=(0.20, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10),
)


# =============================================================================
# MATERIALIZATION — agent system domain (Python output)
# =============================================================================

AGENT_SYSTEM_MATERIALIZATION = MaterializationConfig(
    output_format="python",
    file_extension=".py",
    syntax_validator="ast.parse",
)


# =============================================================================
# RUNTIME CAPABILITIES — agent system domain
# =============================================================================

AGENT_SYSTEM_RUNTIME = RuntimeCapabilities(
    has_event_loop=True,
    has_llm_client=True,
    has_persistent_state=True,
    has_self_recompile=True,
    has_tool_execution=True,
    event_loop_type="asyncio",
    state_backend="sqlite",
    entry_point="main.py",
    default_port=8080,
    tool_allowlist=("web_search", "file_read", "file_write", "shell_exec"),
    can_compile=True,
    can_share_tools=True,
    corpus_path="~/motherlabs/corpus.db",
)


# =============================================================================
# COMPLETE ADAPTER
# =============================================================================

AGENT_SYSTEM_ADAPTER = DomainAdapter(
    name="agent_system",
    version="1.0",
    vocabulary=AGENT_SYSTEM_VOCABULARY,
    prompts=AGENT_SYSTEM_PROMPTS,
    classification=AGENT_SYSTEM_CLASSIFICATION,
    verification=AGENT_SYSTEM_VERIFICATION,
    materialization=AGENT_SYSTEM_MATERIALIZATION,
    runtime=AGENT_SYSTEM_RUNTIME,
)
