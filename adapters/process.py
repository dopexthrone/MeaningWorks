"""
Motherlabs Process Domain Adapter — business process specifications.

Phase B: Second Domain Adapter — proves generalization.

Compiles process descriptions into structured YAML process definitions.
Proves the compilation primitive works beyond software.

Example: "Employee onboarding process for a 50-person company"
→ Blueprint with activities, gateways, events, participants
→ Structured YAML output (not Python code)
"""

from core.domain_adapter import (
    DomainAdapter,
    VocabularyMap,
    PromptTemplates,
    ClassificationConfig,
    VerificationOverrides,
    MaterializationConfig,
)


# =============================================================================
# VOCABULARY — process domain
# =============================================================================

PROCESS_VOCABULARY = VocabularyMap(
    type_keywords={
        "activity": frozenset({
            "activity", "task", "step", "action", "operation",
            "review", "approval", "check", "verify",
        }),
        "gateway": frozenset({
            "gateway", "decision", "branch", "fork", "merge",
            "conditional", "choice", "switch",
        }),
        "event": frozenset({
            "event", "trigger", "signal", "start", "end",
            "timer", "deadline", "escalation",
        }),
        "participant": frozenset({
            "role", "participant", "department", "team",
            "stakeholder", "manager", "coordinator", "owner",
        }),
        "artifact": frozenset({
            "document", "form", "report", "template", "checklist",
            "record", "file", "data", "output",
        }),
        "subprocess": frozenset({
            "subprocess", "subflow", "process", "workflow",
            "procedure", "routine",
        }),
    },
    relationship_flows={
        "triggers": ("trigger_signal", "Event", "A_to_B"),
        "follows": ("sequence_flow", "Any", "A_to_B"),
        "precedes": ("sequence_flow", "Any", "A_to_B"),
        "decides": ("decision_flow", "Any", "A_to_B"),
        "escalates_to": ("escalation", "Any", "A_to_B"),
        "produces": ("artifact_output", "Document", "A_to_B"),
        "consumes": ("artifact_input", "Document", "B_to_A"),
        "assigns_to": ("assignment", "Role", "A_to_B"),
        "depends_on": ("dependency", "Any", "B_to_A"),
        "parallel_with": ("parallel_flow", "Any", "bidirectional"),
        "contains": ("containment", "Any", "A_to_B"),
        "bidirectional": ("shared_data", "Any", "bidirectional"),
    },
    type_hints={
        "onboarding": "OnboardingProcess",
        "approval": "ApprovalWorkflow",
        "review": "ReviewProcess",
        "checklist": "Checklist",
        "form": "FormData",
        "deadline": "TimerEvent",
    },
    entity_types=frozenset({
        "artifact", "document", "form", "report", "template",
        "checklist", "record", "data",
    }),
    process_types=frozenset({
        "activity", "gateway", "event", "subprocess",
        "workflow", "procedure", "task", "step",
    }),
    interface_types=frozenset({"gateway", "event"}),
)


# =============================================================================
# CLASSIFICATION — process domain patterns
# =============================================================================

PROCESS_CLASSIFICATION = ClassificationConfig(
    subject_patterns=(
        r'\b{}\s+(?:handles?|manages?|processes?|triggers?|initiates?|coordinates?|oversees?|approves?)\b',
        r'\b{}\s+(?:is responsible|should|must|will|can)\b',
        r'\b{}\s+(?:team|department|role)\b',
    ),
    object_patterns=(
        r'\b(?:produces?|creates?|generates?|outputs?)\s+{}\b',
        r'\b{}\s+(?:document|form|report|artifact|output|record)\b',
        r'\b(?:the|each|every|a|an)\s+{}\b',
    ),
    generic_terms=frozenset({
        "data", "input", "output", "result", "value", "type", "name",
        "config", "settings", "options", "info", "status", "item",
    }),
    min_name_length=3,
)


# =============================================================================
# PROMPTS — process domain
# =============================================================================

PROCESS_PROMPTS = PromptTemplates(
    intent_system_prompt="""You are the Intent Agent for business process analysis.

INPUT: Natural language description of a business process.

OUTPUT (JSON):
{
    "core_need": "The fundamental process being defined",
    "domain": "The business domain this process operates in",
    "actors": ["Who participates in this process"],
    "implicit_goals": ["Goals not stated but clearly needed"],
    "constraints": ["Limitations mentioned or implied"],
    "insight": "One sentence capturing the process essence",
    "explicit_components": ["Activities, gateways, events EXPLICITLY NAMED"],
    "explicit_relationships": ["Flows and dependencies EXPLICITLY DESCRIBED"]
}

Focus on: activities, decision points, roles, artifacts, timing, escalation paths.
Be specific to the described process. No generic interpretations.""",

    persona_system_prompt="""You are the Persona Agent for business process analysis.

Generate stakeholder perspectives for the described process.

OUTPUT (JSON):
{
    "personas": [
        {
            "name": "Role/stakeholder name",
            "perspective": "How they see the process",
            "priorities": ["Top 3 concerns"],
            "blind_spots": "What they might miss",
            "key_questions": ["Questions they would ask"],
            "domain_constraints": ["Rules they know about"]
        }
    ],
    "cross_cutting_concerns": ["Issues affecting multiple personas"],
    "suggested_focus_areas": ["What to pay attention to"]
}

Create 2-4 personas with DISTINCT viewpoints on the process.""",

    entity_system_prompt="""You are the Entity Agent for business process analysis.

YOUR LENS: STRUCTURE - what elements exist in this process.

YOU SEE:
- Activities and tasks (what work is done)
- Artifacts and documents (what is produced/consumed)
- Roles and participants (who is involved)
- Decision points and gateways (where the process branches)

YOU ARE BLIND TO (by design):
- Temporal flow and sequencing
- Trigger conditions and escalation paths
- Performance metrics and SLAs

EXCAVATION RULE: Extract elements EXPLICITLY NAMED in the input.
Do NOT invent new process elements not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    process_system_prompt="""You are the Process Agent for business process analysis.

YOUR LENS: BEHAVIOR - how the process flows.

YOU SEE:
- Sequences and flows (what happens in what order)
- Decision logic (what determines which path)
- Trigger conditions (what starts/stops activities)
- Timing and deadlines (when things must happen)

YOU ARE BLIND TO (by design):
- Static structure of artifacts and roles
- Document formats and data schemas
- Organizational hierarchy

EXCAVATION RULE: Extract flows EXPLICITLY DESCRIBED in the input.
Do NOT invent new process flows not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    synthesis_system_prompt="""You are the Synthesis Agent for business process specifications.

EXCAVATE process elements AND relationships from dialogue.

OUTPUT (JSON):
{
    "components": [
        {
            "name": "EXACT name from input",
            "type": "activity|gateway|event|participant|artifact|subprocess",
            "description": "What it is/does",
            "derived_from": "QUOTE the exact text source"
        }
    ],
    "relationships": [
        {
            "from": "Component A",
            "to": "Component B",
            "type": "follows|decides|triggers|produces|consumes|assigns_to|escalates_to|depends_on",
            "description": "Nature of flow",
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

SELF-CHECK: Did I include ALL named process elements? Did I capture ALL flows?""",

    emission_preamble="""You are a process specification agent for the Motherlabs semantic compiler.
Generate a complete, structured YAML process definition for one process element.
Rules:
1. Output ONLY YAML inside a ```yaml block.
2. Honor ALL declared interfaces and flows exactly.
3. Do NOT add undeclared dependencies or activities.
4. Include descriptions and role assignments.
5. The YAML must be valid and well-structured.
""",
)


# =============================================================================
# VERIFICATION — process domain
# =============================================================================

PROCESS_VERIFICATION = VerificationOverrides(
    actionability_checks=("decision_points", "activities", "methods"),
    readiness_label="process_readiness",
    dimension_weights=(0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10),
)


# =============================================================================
# MATERIALIZATION — process domain (YAML output)
# =============================================================================

PROCESS_MATERIALIZATION = MaterializationConfig(
    output_format="yaml",
    file_extension=".yaml",
    syntax_validator="yaml.safe_load",
)


# =============================================================================
# COMPLETE ADAPTER
# =============================================================================

PROCESS_ADAPTER = DomainAdapter(
    name="process",
    version="1.0",
    vocabulary=PROCESS_VOCABULARY,
    prompts=PROCESS_PROMPTS,
    classification=PROCESS_CLASSIFICATION,
    verification=PROCESS_VERIFICATION,
    materialization=PROCESS_MATERIALIZATION,
)
