"""
Motherlabs Domain Adapter — pluggable frozen configs for domain-specific behavior.

Phase A: Extract Domain-Specific Code into Adapter

LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
All frozen dataclasses. All pure functions.

A DomainAdapter is the minimal set of domain knowledge needed to make the
universal compiler work for a specific output domain. The compilation core
never hardcodes domain knowledge — it reads it from the adapter.

Extraction map (what moved here from hardcoded values):
- classification.py: _SUBJECT_PATTERNS, _OBJECT_PATTERNS, _TYPE_KEYWORDS, _GENERIC
- interface_extractor.py: _RELATIONSHIP_TO_FLOW, _TYPE_HINTS_BY_COMPONENT
- verification.py: actionability checks (has_methods)
- project_writer.py: _ENTITY_TYPES, _PROCESS_TYPES
- agents/swarm.py: INTENT/PERSONA/SYNTHESIS system prompts
- agents/spec_agents.py: ENTITY/PROCESS system prompts
- agent_emission.py: EMISSION_PREAMBLE
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple


# =============================================================================
# SUB-DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class VocabularyMap:
    """Domain-specific vocabulary for classification and interface extraction.

    Extracted from:
    - classification.py: _TYPE_KEYWORDS
    - interface_extractor.py: _RELATIONSHIP_TO_FLOW, _TYPE_HINTS_BY_COMPONENT
    - project_writer.py: _ENTITY_TYPES, _PROCESS_TYPES
    """
    # Component type → keywords that indicate that type
    type_keywords: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    # Relationship type → (flow_name, default_type, direction)
    relationship_flows: Dict[str, Tuple[str, str, str]] = field(default_factory=dict)

    # Component name pattern → type hint for interfaces
    type_hints: Dict[str, str] = field(default_factory=dict)

    # Types considered "entity" for project grouping
    entity_types: FrozenSet[str] = field(default_factory=frozenset)

    # Types considered "process" for project grouping
    process_types: FrozenSet[str] = field(default_factory=frozenset)

    # Types considered "interface" for layered emission
    interface_types: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PromptTemplates:
    """Domain-specific LLM prompt templates.

    Extracted from:
    - agents/swarm.py: INTENT/PERSONA/SYNTHESIS system prompts
    - agents/spec_agents.py: ENTITY/PROCESS system prompts
    - agent_emission.py: EMISSION_PREAMBLE
    """
    intent_system_prompt: str = ""
    persona_system_prompt: str = ""
    entity_system_prompt: str = ""
    process_system_prompt: str = ""
    synthesis_system_prompt: str = ""
    emission_preamble: str = ""


@dataclass(frozen=True)
class ClassificationConfig:
    """Domain-specific classification patterns.

    Extracted from:
    - classification.py: _SUBJECT_PATTERNS, _OBJECT_PATTERNS, _GENERIC
    """
    # Regex pattern strings for subject (actor) detection
    subject_patterns: Tuple[str, ...] = ()

    # Regex pattern strings for object (data) detection
    object_patterns: Tuple[str, ...] = ()

    # Single-word generic terms that are NOT components
    generic_terms: FrozenSet[str] = field(default_factory=frozenset)

    # Minimum name length to be considered a component
    min_name_length: int = 3


@dataclass(frozen=True)
class VerificationOverrides:
    """Domain-specific verification behavior.

    Extracted from:
    - verification.py: actionability checks (has_methods)
    """
    # What constitutes "actionable" in this domain
    # Each check is a component dict key to look for
    actionability_checks: Tuple[str, ...] = ("methods",)

    # Label for the readiness dimension (software="codegen_readiness")
    readiness_label: str = "codegen_readiness"

    # 7 weights for composite readiness score:
    # (completeness, consistency, traceability, actionability, specificity, coherence, codegen)
    dimension_weights: Tuple[float, ...] = (0.20, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10)


@dataclass(frozen=True)
class MaterializationConfig:
    """Domain-specific materialization behavior.

    Extracted from:
    - project_writer.py: output format assumptions
    - agent_emission.py: EMISSION_PREAMBLE code rules
    """
    # Output file format
    output_format: str = "python"

    # File extension for generated output
    file_extension: str = ".py"

    # Syntax validator function name (for reference)
    syntax_validator: str = "ast.parse"


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Declares what runtime infrastructure the generated system needs.

    When a domain adapter specifies runtime capabilities, the project writer
    emits deterministic template-generated infrastructure files alongside
    the LLM-generated component code. This preserves the trust guarantee:
    runtime plumbing traces to adapter config, not hallucination.
    """
    has_event_loop: bool = False          # async message processing
    has_llm_client: bool = False          # LLM API integration
    has_persistent_state: bool = False    # SQLite state store
    has_self_recompile: bool = False      # recompilation via API
    has_tool_execution: bool = False      # sandboxed tool dispatch
    detect_gaps: bool = False             # auto-detect gaps and trigger recompile
    event_loop_type: str = "none"         # "none", "asyncio", "websocket"
    state_backend: str = "none"           # "none", "sqlite", "json"
    entry_point: str = "main.py"          # generated entry point filename
    default_port: int = 8080              # for server-based systems
    tool_allowlist: Tuple[str, ...] = ()  # allowed tool names
    can_compile: bool = False             # can compile new tools via MotherlabsEngine
    can_share_tools: bool = False         # can export/import .mtool files
    corpus_path: str = ""                 # path to compilation corpus (default: ~/motherlabs/corpus.db)


# =============================================================================
# MAIN ADAPTER DATACLASS
# =============================================================================

@dataclass(frozen=True)
class DomainAdapter:
    """Pluggable frozen config that injects all domain-specific behavior.

    The compilation core never hardcodes domain knowledge. It reads
    everything from this adapter. Each adapter is a complete, frozen
    specification of how the universal compiler behaves for one domain.

    Built-in adapters: "software", "process", "api", "agent_system"
    """
    name: str
    version: str
    vocabulary: VocabularyMap = field(default_factory=VocabularyMap)
    prompts: PromptTemplates = field(default_factory=PromptTemplates)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    verification: VerificationOverrides = field(default_factory=VerificationOverrides)
    materialization: MaterializationConfig = field(default_factory=MaterializationConfig)
    runtime: Optional[RuntimeCapabilities] = None
