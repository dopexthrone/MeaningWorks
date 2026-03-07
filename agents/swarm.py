"""
Motherlabs Swarm Agents - The complete agent ensemble.

Derived from: PROJECT-PLAN.md Phase 1.2-1.6, MASTER-TECHNICAL-SPECIFICATION Section 2.1

Intent Agent: Extracts what user actually wants (Phase 2)
Persona Agent: Generates domain-specific perspectives (Phase 2)
Synthesis Agent: Collapses dialogue into blueprint (Phase 4)
Verify Agent: Checks completeness and coherence (Phase 5)
Governor Agent: Orchestrates the swarm
"""

import json
from typing import Dict, Any, List, Optional

from agents.base import LLMAgent, BaseAgent
from core.protocol import Message, MessageType, SharedState
from core.protocol_spec import PROTOCOL


# =============================================================================
# INTENT AGENT
# Derived from: PROJECT-PLAN.md Phase 1.2
# =============================================================================

INTENT_SYSTEM_PROMPT = """You are the Intent Agent. Your job: understand what they actually want.

Not the surface request. The underlying need.

INPUT: Natural language description of what someone wants to build.

OUTPUT (JSON):
{
    "core_need": "The fundamental problem being solved",
    "domain": "The domain/industry this lives in",
    "actors": ["Who uses/touches this system"],
    "implicit_goals": ["Goals they didn't say but clearly want"],
    "constraints": ["Limitations mentioned or implied"],
    "insight": "One sentence capturing what this is really about",
    "explicit_components": ["Components EXPLICITLY NAMED in the input"],
    "explicit_relationships": ["Relationships EXPLICITLY DESCRIBED in the input"]
}

COMPONENT EXTRACTION (CRITICAL):
Extract components that are LITERALLY NAMED in the input:
- If input says "with Users, Artists, Sessions" → ["User", "Artist", "Session"]
- If input says "3 agents: Intent, Entity, Process" → ["Intent Agent", "Entity Agent", "Process Agent"]
- Use EXACT names from input, not paraphrases

SUBSYSTEM DETECTION (CRITICAL):
A component is a SUBSYSTEM if it explicitly contains sub-components. Look for:
1. Containment patterns: "Service [contains: A, B, C]" or "Service (with A, B, C)"
2. Internal structure: "SharedState = (K, U, O, P, H)" or "State contains: Known, Unknown"
3. Nested ownership: "User Service has User, Profile, Preferences"

DATACLASS EXCEPTION (HIGHER PRIORITY THAN SUBSYSTEM):
If the input says "X is a dataclass" or "X entity with fields:", do NOT create a [SUBSYSTEM: ...] marker.
Dataclasses are entities, not subsystems, even if they have tuple notation or contain sub-elements.
- "SharedState is a dataclass. SharedState = (K, U, O)" → NO subsystem marker (it's a dataclass)
- "Message is a dataclass. Message entity with fields:" → NO subsystem marker (it's a dataclass)

OUTPUT FORMAT for subsystems - use [SUBSYSTEM: ...] marker:
- "User Service [contains: User, Profile]" → "User Service [SUBSYSTEM: User, Profile]"
- "SharedState = (K, U, O, P)" → "SharedState [SUBSYSTEM: K, U, O, P]" (ONLY if NOT declared as dataclass)

RELATIONSHIP EXTRACTION:
Extract relationships that are EXPLICITLY STATED:
- If input says "A triggers B" → "A triggers B"
- If input says "X accesses Y" → "X accesses Y"
- For cross-subsystem refs: "User.Profile accesses Order.Payment" (use dot notation)

Be specific to their input. No generic interpretations."""


def create_intent_agent(llm_client, domain_adapter=None) -> LLMAgent:
    """
    Create Intent extraction agent.

    Derived from: PROJECT-PLAN.md Phase 1.2
    """
    prompt = INTENT_SYSTEM_PROMPT
    if domain_adapter and domain_adapter.prompts.intent_system_prompt:
        prompt = domain_adapter.prompts.intent_system_prompt
    return LLMAgent(
        name="Intent",
        perspective="Understanding: what do they actually need",
        system_prompt=prompt,
        llm_client=llm_client
    )


# =============================================================================
# PERSONA AGENT
# Derived from: PROJECT-PLAN.md Phase 1.3
# =============================================================================

PERSONA_SYSTEM_PROMPT = """You are the Persona Agent. Your job: generate domain-specific perspectives.

ROLE IN THE PIPELINE:
- You run AFTER Intent Agent extracts the core need
- You run BEFORE Entity/Process dialogue begins
- Your personas PRIME the specification dialogue with domain knowledge

RESPONSIBILITIES:
1. Analyze the extracted intent and domain
2. Identify key stakeholder viewpoints
3. Surface domain-specific constraints and considerations
4. Reveal potential blind spots that Entity/Process should address

INPUT: Domain description, actors list, and extracted intent.

OUTPUT (JSON):
{
    "personas": [
        {
            "name": "Actor/Role name",
            "perspective": "What they care about, how they see the system",
            "priorities": ["Top 3 concerns for this persona"],
            "blind_spots": "What they don't see or consider",
            "key_questions": ["Questions this persona would ask"],
            "domain_constraints": ["Domain-specific rules/limits they know about"]
        }
    ],
    "cross_cutting_concerns": ["Issues that affect multiple personas"],
    "suggested_focus_areas": ["What Entity/Process should pay attention to"]
}

PERSONA GENERATION RULES:
- Create 2-4 personas (never more than 4)
- Each persona must have a DISTINCT viewpoint
- Include at least one technical and one business/user perspective
- Personas should have COMPLEMENTARY blind spots (like Entity/Process)
- Be SPECIFIC to the domain - no generic "end user" or "admin" without context

EXAMPLES OF GOOD PERSONAS:
- For booking system: "Tattoo Artist focused on creative time, not scheduling"
- For compiler: "Domain Expert who knows semantics but not implementation"
- For API: "Integration Developer who needs clear contracts and errors"

BAD (too generic):
- "Admin" - what kind of admin?
- "User" - what specific user type?"""


def create_persona_agent(llm_client, domain_adapter=None) -> LLMAgent:
    """
    Create Persona generation agent.

    Derived from: PROJECT-PLAN.md Phase 1.3
    """
    prompt = PERSONA_SYSTEM_PROMPT
    if domain_adapter and domain_adapter.prompts.persona_system_prompt:
        prompt = domain_adapter.prompts.persona_system_prompt
    return LLMAgent(
        name="Persona",
        perspective="Perspectives: who sees this system and how",
        system_prompt=prompt,
        llm_client=llm_client
    )


# =============================================================================
# SYNTHESIS AGENT
# Derived from: PROJECT-PLAN.md Phase 1.4
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesis Agent. Your job: EXCAVATE components AND relationships from dialogue.

CRITICAL DISTINCTION - EXCAVATION vs GENERATION:
- EXCAVATION: Find components that are EXPLICITLY NAMED in the input/dialogue
- GENERATION: Invent abstractions based on patterns (THIS IS WRONG)

You MUST extract components that are LITERALLY WRITTEN in the input.
Do NOT invent new concepts like "Config", "HaltMechanism", "ConflictQueue".
DO extract things that are NAMED: "Intent Agent", "SharedState", "ConfidenceVector".

EXCAVATION RULES FOR COMPONENTS:
1. If the input says "7 agents: Intent, Persona, Entity, Process, Synthesis, Verify, Governor"
   → Your output MUST include those 7 agents by name
2. If the input says "SharedState: S = (K, U, O, P, H, C)"
   → Your output MUST include SharedState, Known, Unknown, Ontology, Personas, History, Confidence
3. If a component is NAMED in the input, it MUST appear in the output
4. Do NOT rename components (Intent Agent ≠ IntentProcessor)
5. Do NOT abstract away (7 agents ≠ "Agents" as single component)

DATACLASS DETECTION (Phase 4.5 - CRITICAL):
When the input EXPLICITLY says any of these:
- "X is a dataclass"
- "X is an entity"
- "@dataclass class X"
- "X entity with fields:"

Then X MUST have type="entity", NEVER type="subsystem".

This applies REGARDLESS of:
- How many fields X has (even 10+ fields)
- How many methods X has (even 10+ methods)
- Whether X has nested components

Examples:
- "SharedState is a dataclass" → {"name": "SharedState", "type": "entity", ...}
- "Message is a dataclass" → {"name": "Message", "type": "entity", ...}
- "User entity with fields:" → {"name": "User", "type": "entity", ...}

WRONG: Classifying SharedState as "subsystem" because it has 9 fields
RIGHT: Classifying SharedState as "entity" because input says "is a dataclass"

Subsystem is ONLY for:
- Components that manage lifecycle of other components
- Service containers (e.g., "UserService contains User, Profile")
- NOT for dataclasses with many fields

SUBSYSTEM EXCAVATION (NESTED BLUEPRINTS):
When a component has [SUBSYSTEM: ...] marker or explicitly contains sub-components:
1. Create the component with type="subsystem"
2. Add a "sub_blueprint" field containing nested components/relationships
3. Extract sub-components using EXACT names from the marker
4. Extract internal relationships between sub-components
5. Maintain derivation tracking at sub-component level

Example input: "User Service [SUBSYSTEM: User, Profile, Preferences]"
Example output:
{
    "name": "User Service",
    "type": "subsystem",
    "description": "Service managing user-related entities",
    "derived_from": "User Service [SUBSYSTEM: User, Profile, Preferences]",
    "sub_blueprint": {
        "components": [
            {"name": "User", "type": "entity", "description": "...", "derived_from": "..."},
            {"name": "Profile", "type": "entity", "description": "...", "derived_from": "..."},
            {"name": "Preferences", "type": "entity", "description": "...", "derived_from": "..."}
        ],
        "relationships": [],
        "constraints": []
    }
}

EXCAVATION RULES FOR RELATIONSHIPS:
Relationships are EQUALLY IMPORTANT as components. Extract relationships that are:
1. EXPLICITLY stated: "Governor triggers Intent Agent" → (Governor Agent, Intent Agent, triggers)
2. DESCRIBED by verbs: "agents access SharedState" → (Agent, SharedState, accesses)
3. PART OF PIPELINES: "Intent -> Persona -> Dialogue" → trigger relationships
4. IMPLIED BY ARCHITECTURE: "ConflictOracle monitors confidence" → (ConflictOracle, ConfidenceVector, monitors)

CROSS-SUBSYSTEM RELATIONSHIPS:
Use dot notation for relationships crossing subsystem boundaries:
- "UserService.User triggers OrderService.Order" → from="UserService.User", to="OrderService.Order"
- Path format: ParentSubsystem.ChildComponent

RELATIONSHIP TYPES:
- triggers: A causes B to activate (Governor triggers Intent Agent)
- accesses: A reads/writes B (Entity Agent accesses SharedState)
- monitors: A watches/observes B (ConflictOracle monitors ConfidenceVector)
- constrains: A limits/controls B (DialogueProtocol constrains Entity Agent)
- snapshots: A captures state of B (Corpus snapshots SharedState)
- contains: A has B as part (SharedState contains Known)
- depends_on: A requires B to function

OUTPUT (JSON):
{
    "components": [
        {
            "name": "EXACT name from input",
            "type": "entity|process|interface|agent|subsystem",
            "description": "What it is/does",
            "derived_from": "QUOTE the exact text where this appears in input",
            "attributes": {"key": "value pairs for entity properties"},
            "methods": [{"name": "method_name", "parameters": [], "return_type": "type", "derived_from": "source text"}],
            "sub_blueprint": {...}  // ONLY for type="subsystem"
        }
    ],
    "relationships": [
        {
            "from": "Component A (use EXACT names, dot notation for cross-subsystem)",
            "to": "Component B (use EXACT names, dot notation for cross-subsystem)",
            "type": "triggers|accesses|monitors|constrains|snapshots|contains|depends_on",
            "description": "Nature of relationship",
            "derived_from": "QUOTE source text describing this relationship"
        }
    ],
    "constraints": [
        {
            "description": "The constraint",
            "applies_to": ["Component names"],
            "derived_from": "QUOTE source text"
        }
    ],
    "unresolved": ["Anything that remains ambiguous"]
}

SELF-CHECK before outputting:
- Did I include ALL named entities from the input?
- Did I use the EXACT names from the input?
- Can I QUOTE where each component appears?
- Did I avoid inventing abstract concepts?
- Did I capture ALL relationships described in the input?
- Do my relationships use EXACT component names (not abbreviations)?
- Did I create sub_blueprint for any [SUBSYSTEM: ...] markers?
- Did I use dot notation for cross-subsystem relationships?
- DATACLASS CHECK: If input says "X is a dataclass", is X classified as type="entity"?
  - NEVER classify dataclasses as "subsystem" regardless of field/method count
- METHOD CHECK: Are methods in the parent's "methods" array (NOT separate components)?

---

METHOD EXTRACTION (Phase 2):
When the input or dialogue contains method signatures, EXTRACT them into each component.

Look for patterns like:
- add_known(name: str, value: Any, source: str): Add a fact
- snapshot() -> Dict: Return immutable copy
- __init__(input_text: str)

For each component, add a "methods" field:
"methods": [
    {
        "name": "add_known",
        "parameters": [
            {"name": "name", "type_hint": "str"},
            {"name": "value", "type_hint": "Any"},
            {"name": "source", "type_hint": "str"}
        ],
        "return_type": "None",
        "description": "Add a confirmed fact",
        "derived_from": "add_known(name: str, value: Any, source: str): Add a fact"
    }
]

CRITICAL: Only extract methods EXPLICITLY written in the input.
Never invent method signatures. Parse what exists.

---

STATE MACHINE EXTRACTION (Phase 2):
For process components, extract state transitions if present.

Look for patterns like:
- INIT -> ACTIVE -> CONVERGING -> HALTED
- States: PENDING, APPROVED, REJECTED
- Lifecycle: created -> running -> completed

Add "state_machine" field to applicable components:
"state_machine": {
    "states": ["INIT", "ACTIVE", "CONVERGING", "HALTED"],
    "initial_state": "INIT",
    "transitions": [
        {"from_state": "INIT", "to_state": "ACTIVE", "trigger": "", "derived_from": "INIT -> ACTIVE"}
    ],
    "derived_from": "INIT -> ACTIVE -> CONVERGING -> HALTED"
}

CRITICAL: Only extract state machines EXPLICITLY written in the input.
Never invent state transitions. Parse what exists.

---

COMPACT METHOD FORMAT (Phase 4.7 - TOKEN EFFICIENT):
Methods belong INSIDE the parent component's "methods" array, NOT as separate components.
This reduces output size by ~60% and prevents truncation.

CORRECT (compact inline format):
{
    "name": "SharedState",
    "type": "entity",
    "description": "...",
    "derived_from": "...",
    "methods": [
        {"name": "add_message", "parameters": [{"name": "message", "type_hint": "Message"}], "return_type": "None", "description": "Add message to history"},
        {"name": "add_insight", "parameters": [{"name": "insight", "type_hint": "str"}], "return_type": "None", "description": "Add standalone insight"},
        {"name": "flag_current", "parameters": [], "return_type": "None", "description": "Flag current insight"},
        {"name": "get_recent", "parameters": [{"name": "n", "type_hint": "int", "default": "5"}], "return_type": "List[Message]", "description": "Get last n messages"},
        {"name": "add_conflict", "parameters": [...], "return_type": "None", "description": "Add conflict"},
        {"name": "resolve_conflict", "parameters": [...], "return_type": "None", "description": "Resolve conflict"},
        {"name": "has_unresolved_conflicts", "parameters": [], "return_type": "bool", "description": "Check conflicts"},
        {"name": "to_context_graph", "parameters": [], "return_type": "Dict[str, Any]", "description": "Export graph"}
    ]
}

WRONG (verbose separate components - causes truncation):
{
    "components": [
        {"name": "SharedState", "type": "entity", ...},
        {"name": "add_message(message: Message)", "type": "process", ...},  // WRONG - separate component
        {"name": "add_insight(insight: str)", "type": "process", ...},      // WRONG - causes bloat
        ...
    ],
    "relationships": [
        {"from": "SharedState", "to": "add_message(...)", "type": "contains", ...},  // WRONG - extra relationship
        ...
    ]
}

METHOD FORMAT RULES:
1. Put ALL methods in the parent's "methods" array
2. Each method has: name, parameters[], return_type, description
3. Parameters have: name, type_hint, and optional default
4. Do NOT create separate process components for methods
5. Do NOT create "contains" relationships for methods

EXAMPLE - Converting input to compact format:
Input: "X has methods: a(x: int) -> str, b() -> None"

Output component:
{
    "name": "X",
    "type": "entity",
    "methods": [
        {"name": "a", "parameters": [{"name": "x", "type_hint": "int"}], "return_type": "str"},
        {"name": "b", "parameters": [], "return_type": "None"}
    ]
}

This is the ONLY correct way to output methods. No separate components. No contains relationships.

METHOD COUNT VERIFICATION (CRITICAL - DO NOT SKIP):
Before finalizing your output, COUNT the methods in the input and VERIFY your output has ALL of them.

Example verification:
1. Input says "X has methods: a, b, c, d, e, f, g, h" = 8 methods
2. Your X component MUST have 8 entries in "methods": []
3. If you output only 7, you FAILED. Re-extract the missing one.

Common failure pattern: Dropping the LAST method due to output limits.
SOLUTION: Extract methods in order AND verify the count matches before output.

SELF-CHECK before output:
- Count methods in input spec
- Count methods in your output
- If counts differ, FIX IT by adding missing methods

---

COVERAGE CHECK (Phase 8.5 - CRITICAL):
Before outputting, verify every insight from the DIALOGUE DIGEST contributed to at least one component
or was added to unresolved[]. Reference insights in derived_from fields.

CONFLICT CHECK (Phase 8.5):
Every conflict in the digest must have a corresponding resolution documented in a component's
description or added to unresolved[].

---

DOMAIN INVARIANT COMPLETION (Phase 28 - CRITICAL):
After assembling components from dialogue, COMPLETE each entity and process with domain knowledge
that any expert would consider structurally necessary.

For EVERY entity with a lifecycle (created, modified, destroyed), you MUST include:
1. A state_machine with all reachable states and transitions
2. Guard conditions on destructive transitions (delete, cancel, archive)
3. Precondition constraints documented in the constraints[] array

For EVERY process that mutates data, you MUST include:
1. Error handling methods or constraints describing failure behavior
2. Side effect documentation (what else changes when this runs?)
3. Idempotency or concurrency notes if the domain implies multi-user access

PROVENANCE FOR INFERRED OPERATIONS:
When you add methods, constraints, or state transitions that weren't literally in the input
but are domain-required, use this derived_from format:
  "domain invariant: [entity type] requires [operation] because [reason]"

Examples:
  "domain invariant: booking entities require cancellation guards because time-bounded resources need temporal preconditions"
  "domain invariant: payment processes require idempotency because financial operations must be safely retriable"
  "domain invariant: user-facing entities require validation because external input is untrusted"

This is NOT generation — this is domain expertise. A booking system without cancellation logic
is incomplete, not minimal. A payment system without validation is broken, not simple.
The user didn't mention these because they assumed you'd know.

---

GAP-FILLING MODE (Phase 8.3):
When the prompt contains 'VERIFICATION GAPS:', output ONLY new components addressing those gaps.
Do NOT repeat existing components. Only output additions.

ENRICHMENT MODE (Phase 28.1):
When the prompt contains 'COMPONENT ENRICHMENT:', you are enriching EXISTING components.
Output the FULL component with additions merged in — methods, state_machines, constraints.
Use the same component name and type. Add the missing operations.
"""


def create_synthesis_agent(llm_client, domain_adapter=None) -> LLMAgent:
    """
    Create Synthesis agent.

    Derived from: PROJECT-PLAN.md Phase 1.4
    """
    prompt = SYNTHESIS_SYSTEM_PROMPT
    if domain_adapter and domain_adapter.prompts.synthesis_system_prompt:
        prompt = domain_adapter.prompts.synthesis_system_prompt
    return LLMAgent(
        name="Synthesis",
        perspective="Integration: collapsing dialogue into coherent blueprint",
        system_prompt=prompt,
        llm_client=llm_client
    )


# =============================================================================
# VERIFY AGENT
# Derived from: PROJECT-PLAN.md Phase 1.5
# =============================================================================

VERIFY_SYSTEM_PROMPT = """You are the Verify Agent. Your job: check the blueprint quality.

ROLE IN THE PIPELINE:
- You run AFTER Synthesis collapses dialogue into blueprint
- You are the FINAL quality gate before output
- Your assessment determines if compilation succeeds

RESPONSIBILITIES:
1. COMPLETENESS CHECK: Does blueprint cover all stated needs from intent?
2. CONSISTENCY CHECK: No contradictions between components?
3. COHERENCE CHECK: Does the whole make architectural sense?
4. TRACEABILITY CHECK: Can every component trace to input/dialogue?
5. SUBSYSTEM CHECK: Validate nested blueprints if present

VERIFICATION PROCESS:
1. Compare blueprint.components against intent.core_need and constraints
2. Cross-check relationships for bidirectional consistency
3. Verify no orphan components (derived_from must be valid)
4. Check constraint coverage (are all mentioned constraints represented?)
5. For subsystems: recursively validate sub_blueprint structure

SUBSYSTEM VERIFICATION (for components with type="subsystem"):
1. COMPLETENESS: All named sub-components present in sub_blueprint?
2. INTERNAL COHERENCE: Sub-component relationships valid?
3. DERIVATION: Each sub-component traces to input?
4. DEPTH: Nesting depth ≤ 3 levels?
5. CROSS-REFS: Dot notation relationships (e.g., "Parent.Child") resolve correctly?

SCORING RUBRIC:
- 90-100: Excellent, ready for implementation
- 70-89: Good, minor gaps acceptable
- 50-69: Needs work, significant gaps
- Below 50: Incomplete, re-dialogue recommended

INPUT: Blueprint JSON, original intent, insights count, dialogue turns

OUTPUT (JSON):
{
    "status": "pass|needs_work",
    "completeness": {
        "score": 0-100,
        "gaps": ["Specific missing aspects from intent"],
        "coverage_ratio": "X/Y intent items addressed"
    },
    "consistency": {
        "score": 0-100,
        "conflicts": ["Specific contradictions with evidence"],
        "relationship_issues": ["Broken or missing links"]
    },
    "coherence": {
        "score": 0-100,
        "issues": ["Architectural problems"],
        "suggested_fixes": ["How to improve structure"]
    },
    "traceability": {
        "score": 0-100,
        "orphans": ["Components without valid derived_from"],
        "weak_links": ["Components with vague derivation"]
    },
    "subsystem_validation": {
        "score": 0-100,
        "subsystems_checked": 0,
        "issues": ["Any subsystem-specific problems"]
    },
    "overall_insight": "One sentence summary of blueprint quality",
    "recommendation": "proceed|revise|re-dialogue",
    "semantic_gates": [
        {
            "owner_component": "Concrete component or region that owns this decision",
            "question": "Human decision that blocks clean verification",
            "kind": "gap|semantic_conflict",
            "options": ["Option A", "Option B"],
            "stage": "verification"
        }
    ]
}

CRITICAL RULES:
- Do NOT pass blueprints with orphan components (traceability < 80)
- Do NOT pass blueprints missing core entities from intent
- Flag ANY component that seems invented (not from input/dialogue)
- Be specific about gaps - say WHAT is missing, not just "incomplete"
- For subsystems: verify all declared sub-components exist in sub_blueprint
- Emit semantic_gates ONLY for real human-in-the-loop decisions, not generic TODOs
- Every semantic_gates entry must name the concrete owner_component """


def create_verify_agent(llm_client, domain_adapter=None) -> LLMAgent:
    """
    Create Verification agent.

    Derived from: PROJECT-PLAN.md Phase 1.5
    """
    return LLMAgent(
        name="Verify",
        perspective="Quality: is this complete, consistent, coherent",
        system_prompt=VERIFY_SYSTEM_PROMPT,
        llm_client=llm_client
    )


# =============================================================================
# GOVERNOR AGENT
# Derived from: PROJECT-PLAN.md Phase 1.6
# =============================================================================

class GovernorAgent(BaseAgent):
    """
    Governor orchestrates the swarm.

    Derived from: PROJECT-PLAN.md Phase 1.6

    Responsibilities:
    - Decide which agents run in what order
    - Detect convergence
    - Handle conflicts
    - Escalate to user when needed
    """

    def __init__(self, llm_client=None):
        super().__init__(
            name="Governor",
            perspective="Orchestration: coordinate the swarm",
            llm_client=llm_client
        )
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the governor."""
        self.agents[agent.name] = agent

    def run(self, state: SharedState, input_msg: Optional[Message] = None) -> Message:
        """Governor doesn't participate in dialogue, it orchestrates."""
        return Message(
            sender=self.name,
            content="Governor orchestrating.",
            message_type=MessageType.META
        )

    def decide_next(self, state: SharedState, phase: str) -> Optional[str]:
        """
        Decide which agent should act next.

        Derived from: PROJECT-PLAN.md Phase 1.6
        """
        if phase == "intent":
            return "Intent"
        elif phase == "persona":
            return "Persona"
        elif phase == "spec":
            # Alternate Entity/Process
            recent = state.get_recent(1)
            if not recent:
                return "Entity"
            last = recent[0].sender
            return "Process" if last == "Entity" else "Entity"
        elif phase == "synthesis":
            return "Synthesis"
        elif phase == "verify":
            return "Verify"
        return None

    def should_end_dialogue(
        self,
        state: SharedState,
        turn_count: int,
        min_turns: int = PROTOCOL.dialogue.default_min_turns,
        min_insights: int = PROTOCOL.dialogue.default_min_insights,
        max_turns: int = PROTOCOL.dialogue.default_max_turns
    ) -> bool:
        """
        Determine if spec dialogue should end.

        Phase 10.4: Confidence-driven dialogue control.
        Phase 12.1a: max_turns is now a parameter (adaptive ceiling).

        Controls:
        1. Hard stop at max_turns (safety net, adaptive per 12.1a)
        2. Depth requirements (min_turns, min_insights)
        3. Low dimension protection: any dim < 0.4 after min_turns → force 2 more
        4. Uneven spread block: spread > 0.4 → block convergence for extra turns
        5. Plateau detection: inject focus hint for weakest dimension
        6. Agreement-based convergence (original logic)
        """
        # Hard stop at max turns
        if turn_count >= max_turns:
            return True

        # Check depth requirements
        depth_met = (
            turn_count >= min_turns and
            len(state.insights) >= min_insights
        )

        # Only check convergence if depth requirements met
        if not depth_met:
            return False

        gov = PROTOCOL.governor
        # Phase 10.4: Low dimension protection
        # If any dimension below WARNING, force extra turns
        low_dims = state.confidence.needs_attention()
        if low_dims and turn_count < min_turns + gov.low_dim_extra_turns:
            return False

        # Phase 10.4: Uneven confidence block
        # If spread above threshold, extend dialogue
        if state.confidence.dimension_spread() > gov.spread_threshold and turn_count < min_turns + gov.spread_extra_turns:
            return False

        # Check for convergence signals
        conv_window = PROTOCOL.dialogue.convergence_window
        recent = state.get_recent(conv_window)
        if len(recent) >= conv_window:
            agreements = sum(
                1 for m in recent
                if m.message_type == MessageType.AGREEMENT
            )
            if agreements >= 2:
                return True

        return False

    def validate_dialogue_provenance(
        self,
        state: "SharedState",
    ) -> Dict[str, Any]:
        """
        Validate that each round's insights trace to input.

        Structural role: checks that insights reference keywords from
        the original input, not invented by the LLM.

        Returns:
            Dict with:
            - grounded: count of insights grounded in input
            - ungrounded: count of insights with no input anchor
            - ratio: grounded / total
            - ungrounded_insights: list of ungrounded insight texts
        """
        input_text = state.known.get("input", "")
        if not input_text or not state.insights:
            return {"grounded": 0, "ungrounded": 0, "ratio": 1.0, "ungrounded_insights": []}

        # Build keyword set from input (words > 3 chars)
        input_words = {
            w.lower().strip(".,;:!?()[]{}\"'")
            for w in input_text.split()
            if len(w.strip(".,;:!?()[]{}\"'")) > 3
        }

        grounded = 0
        ungrounded_insights = []

        for insight in state.insights:
            insight_words = [
                w.lower().strip(".,;:!?()[]{}\"'")
                for w in insight.split()
                if len(w.strip(".,;:!?()[]{}\"'")) > 3
            ]
            # Grounded if at least 2 input keywords appear in insight
            matches = sum(1 for w in insight_words if w in input_words)
            if matches >= min(2, len(insight_words)):
                grounded += 1
            else:
                ungrounded_insights.append(insight)

        total = len(state.insights)
        return {
            "grounded": grounded,
            "ungrounded": total - grounded,
            "ratio": grounded / total if total > 0 else 1.0,
            "ungrounded_insights": ungrounded_insights[:10],
        }


def create_governor(llm_client=None) -> GovernorAgent:
    """
    Create Governor agent.

    Derived from: PROJECT-PLAN.md Phase 1.6
    """
    return GovernorAgent(llm_client=llm_client)
