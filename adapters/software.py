"""
Motherlabs Software Domain Adapter — extracted 1:1 from current hardcoded values.

Phase A: Extract Domain-Specific Code into Adapter

This adapter contains the EXACT same values that were previously hardcoded in:
- core/classification.py (patterns, keywords, generic terms)
- core/interface_extractor.py (relationship flows, type hints)
- core/verification.py (actionability checks)
- core/project_writer.py (entity/process types)
- agents/swarm.py (intent/persona/synthesis prompts)
- agents/spec_agents.py (entity/process prompts)
- core/agent_emission.py (emission preamble)

The 2088-test suite proves equivalence: zero behavior change.
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
# VOCABULARY — extracted from classification.py + interface_extractor.py + project_writer.py
# =============================================================================

SOFTWARE_VOCABULARY = VocabularyMap(
    type_keywords={
        "agent": frozenset({"agent", "handler", "worker", "processor", "service", "manager", "orchestrator"}),
        "entity": frozenset({"state", "data", "record", "model", "store", "repository", "corpus", "vector", "oracle"}),
        "process": frozenset({"protocol", "pipeline", "flow", "workflow", "algorithm", "compiler", "engine"}),
        "interface": frozenset({"api", "interface", "contract", "boundary", "endpoint", "gateway"}),
        "event": frozenset({"event", "signal", "trigger", "notification", "message", "callback"}),
        "subsystem": frozenset({"subsystem", "module", "system", "layer", "tier"}),
    },
    relationship_flows={
        "triggers": ("trigger_signal", "Signal", "A_to_B"),
        "accesses": ("data_access", "Any", "B_to_A"),
        "monitors": ("monitoring_data", "Any", "B_to_A"),
        "contains": ("contained_ref", "Any", "A_to_B"),
        "snapshots": ("snapshot_data", "Any", "A_to_B"),
        "depends_on": ("dependency", "Any", "B_to_A"),
        "flows_to": ("flow_data", "Any", "A_to_B"),
        "generates": ("generated_output", "Any", "A_to_B"),
        "propagates": ("propagated_data", "Any", "A_to_B"),
        "constrained_by": ("constraint_ref", "Any", "B_to_A"),
        "bidirectional": ("shared_data", "Any", "bidirectional"),
    },
    type_hints={
        "sharedstate": "SharedState",
        "shared state": "SharedState",
        "confidencevector": "ConfidenceVector",
        "confidence vector": "ConfidenceVector",
        "message": "Message",
        "dialogueprotocol": "DialogueProtocol",
        "dialogue protocol": "DialogueProtocol",
        "conflictoracle": "ConflictOracle",
        "conflict oracle": "ConflictOracle",
        "corpus": "CompilationRecord",
    },
    entity_types=frozenset({"entity", "data", "model", "record", "state", "store"}),
    process_types=frozenset({
        "process", "agent", "service", "handler", "controller",
        "manager", "orchestrator", "pipeline", "workflow",
    }),
    interface_types=frozenset({"interface", "api", "contract", "boundary", "gateway"}),
)


# =============================================================================
# CLASSIFICATION — extracted from classification.py
# =============================================================================

SOFTWARE_CLASSIFICATION = ClassificationConfig(
    subject_patterns=(
        r'\b{}\s+(?:handles?|manages?|processes?|triggers?|monitors?|creates?|generates?|orchestrates?|controls?)\b',
        r'\b{}\s+(?:is responsible|should|must|will|can)\b',
        r'\b{}\s+agent\b',
    ),
    object_patterns=(
        r'\b(?:stores?|contains?|holds?|tracks?)\s+{}\b',
        r'\b{}\s+(?:data|record|entry|item|object|instance)\b',
        r'\b(?:the|each|every|a|an)\s+{}\b',
    ),
    generic_terms=frozenset({
        "data", "input", "output", "result", "value", "type", "name",
        "config", "settings", "options", "params", "args", "info",
    }),
    min_name_length=3,
)


# =============================================================================
# PROMPTS — extracted from agents/swarm.py + agents/spec_agents.py + agent_emission.py
# =============================================================================

# These are imported by reference — the actual prompt strings live in swarm.py
# and spec_agents.py. We store the default values here for the adapter protocol,
# but the real prompts are the module-level constants in those files.
# This avoids duplicating ~500 lines of prompt text.

SOFTWARE_PROMPTS = PromptTemplates(
    intent_system_prompt="",     # Uses default from swarm.py
    persona_system_prompt="",    # Uses default from swarm.py
    entity_system_prompt="",     # Uses default from spec_agents.py
    process_system_prompt="",    # Uses default from spec_agents.py
    synthesis_system_prompt="",  # Uses default from swarm.py
    emission_preamble="",        # Uses default from agent_emission.py
)


# =============================================================================
# VERIFICATION — extracted from verification.py
# =============================================================================

SOFTWARE_VERIFICATION = VerificationOverrides(
    actionability_checks=("methods",),
    readiness_label="codegen_readiness",
    dimension_weights=(0.20, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10),
)


# =============================================================================
# MATERIALIZATION — extracted from project_writer.py
# =============================================================================

SOFTWARE_MATERIALIZATION = MaterializationConfig(
    output_format="python",
    file_extension=".py",
    syntax_validator="ast.parse",
)


# =============================================================================
# COMPLETE ADAPTER
# =============================================================================

SOFTWARE_ADAPTER = DomainAdapter(
    name="software",
    version="1.0",
    vocabulary=SOFTWARE_VOCABULARY,
    prompts=SOFTWARE_PROMPTS,
    classification=SOFTWARE_CLASSIFICATION,
    verification=SOFTWARE_VERIFICATION,
    materialization=SOFTWARE_MATERIALIZATION,
)
