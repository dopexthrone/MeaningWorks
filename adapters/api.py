"""
Motherlabs API Design Domain Adapter — REST API specifications.

Phase C: Third Domain Adapter — proves N-domain scaling.

Compiles API descriptions into structured OpenAPI-style YAML definitions.
Proves the compilation primitive works for API design, not just software and processes.

Example: "A REST API for a bookstore with inventory, orders, and customer reviews"
-> Blueprint with endpoints, models, services, middleware, auth schemes
-> Structured YAML output (OpenAPI-flavored)
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
# VOCABULARY — API design domain
# =============================================================================

API_VOCABULARY = VocabularyMap(
    type_keywords={
        "endpoint": frozenset({
            "endpoint", "route", "path", "operation", "handler",
            "action", "request", "call", "method",
        }),
        "model": frozenset({
            "model", "schema", "entity", "dto", "resource",
            "payload", "body", "record", "object",
        }),
        "service": frozenset({
            "service", "controller", "logic", "manager",
            "processor", "orchestrator", "facade",
        }),
        "middleware": frozenset({
            "middleware", "filter", "interceptor", "validator",
            "rate_limiter", "cors", "logger", "error_handler",
        }),
        "auth_scheme": frozenset({
            "auth", "authentication", "authorization", "oauth",
            "jwt", "token", "api_key", "bearer", "session",
        }),
        "resource": frozenset({
            "resource", "collection", "item", "namespace",
            "group", "version", "prefix",
        }),
    },
    relationship_flows={
        "returns": ("response_data", "Any", "A_to_B"),
        "accepts": ("request_data", "Any", "B_to_A"),
        "calls": ("service_call", "Any", "A_to_B"),
        "validates": ("validation_flow", "Any", "A_to_B"),
        "authenticates": ("auth_flow", "Token", "A_to_B"),
        "transforms": ("data_transform", "Any", "A_to_B"),
        "depends_on": ("dependency", "Any", "B_to_A"),
        "contains": ("containment", "Any", "A_to_B"),
        "rate_limits": ("throttle", "Any", "A_to_B"),
        "logs": ("audit_trail", "Any", "A_to_B"),
        "errors_to": ("error_flow", "Error", "A_to_B"),
        "bidirectional": ("shared_data", "Any", "bidirectional"),
    },
    type_hints={
        "user": "UserModel",
        "order": "OrderModel",
        "product": "ProductModel",
        "token": "AuthToken",
        "error": "ErrorResponse",
        "pagination": "PaginationParams",
    },
    entity_types=frozenset({
        "model", "schema", "dto", "resource", "payload",
        "record", "object", "collection",
    }),
    process_types=frozenset({
        "endpoint", "service", "middleware", "auth_scheme",
        "controller", "handler", "interceptor", "validator",
    }),
    interface_types=frozenset({"middleware", "auth_scheme"}),
)


# =============================================================================
# CLASSIFICATION — API design domain patterns
# =============================================================================

API_CLASSIFICATION = ClassificationConfig(
    subject_patterns=(
        r'\b{}\s+(?:handles?|processes?|serves?|routes?|dispatches?|validates?)\b',
        r'\b{}\s+(?:endpoint|service|controller|middleware|handler)\b',
        r'\b{}\s+(?:authenticates?|authorizes?|rate.?limits?)\b',
    ),
    object_patterns=(
        r'\b(?:returns?|sends?|produces?)\s+{}\b',
        r'\b{}\s+(?:model|schema|resource|payload|response|request|body)\b',
        r'\b(?:the|each|every|a|an)\s+{}\b',
    ),
    generic_terms=frozenset({
        "data", "input", "output", "result", "value", "type", "name",
        "config", "settings", "options", "params", "args", "info",
        "request", "response", "status", "code",
    }),
    min_name_length=3,
)


# =============================================================================
# PROMPTS — API design domain
# =============================================================================

API_PROMPTS = PromptTemplates(
    intent_system_prompt="""You are the Intent Agent for REST API design.

INPUT: Natural language description of an API or web service.

OUTPUT (JSON):
{
    "core_need": "The fundamental API being designed",
    "domain": "The business domain this API serves",
    "actors": ["Who consumes this API — clients, services, admins"],
    "implicit_goals": ["API qualities not stated but clearly needed"],
    "constraints": ["Limitations mentioned or implied — rate limits, auth, versioning"],
    "insight": "One sentence capturing the API's purpose",
    "explicit_components": ["Endpoints, models, services EXPLICITLY NAMED"],
    "explicit_relationships": ["Data flows and dependencies EXPLICITLY DESCRIBED"]
}

Focus on: endpoints, request/response models, authentication, middleware, error handling, data relationships.
Be specific to the described API. No generic interpretations.""",

    persona_system_prompt="""You are the Persona Agent for REST API design.

Generate stakeholder perspectives for the described API.

OUTPUT (JSON):
{
    "personas": [
        {
            "name": "Stakeholder role",
            "perspective": "How they see the API",
            "priorities": ["Top 3 concerns"],
            "blind_spots": "What they might miss",
            "key_questions": ["Questions they would ask"],
            "domain_constraints": ["API rules they know about"]
        }
    ],
    "cross_cutting_concerns": ["Issues affecting multiple consumers"],
    "suggested_focus_areas": ["What to pay attention to"]
}

Create 2-4 personas: API consumer (frontend dev), backend engineer, security reviewer, DevOps/infra.""",

    entity_system_prompt="""You are the Entity Agent for REST API design.

YOUR LENS: STRUCTURE — what resources and models exist in this API.

YOU SEE:
- Resources and models (what data entities the API exposes)
- Endpoints and routes (what operations are available)
- Request/response schemas (what data flows in and out)
- Authentication schemes (what security mechanisms exist)

YOU ARE BLIND TO (by design):
- Request flow sequencing and orchestration
- Error propagation and retry logic
- Performance characteristics and rate limiting

EXCAVATION RULE: Extract elements EXPLICITLY NAMED in the input.
Do NOT invent endpoints or models not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    process_system_prompt="""You are the Process Agent for REST API design.

YOUR LENS: BEHAVIOR — how the API processes requests.

YOU SEE:
- Request/response flows (what happens when an endpoint is called)
- Validation and middleware chains (what checks occur)
- Error handling paths (what happens on failure)
- Authentication/authorization flows (how access is controlled)

YOU ARE BLIND TO (by design):
- Static data model structure
- Database schemas and storage details
- Deployment infrastructure

EXCAVATION RULE: Extract flows EXPLICITLY DESCRIBED in the input.
Do NOT invent request flows not mentioned.

INSIGHT: line format is MANDATORY at end of every response.""",

    synthesis_system_prompt="""You are the Synthesis Agent for REST API specifications.

EXCAVATE API elements AND relationships from dialogue.

OUTPUT (JSON):
{
    "components": [
        {
            "name": "EXACT name from input",
            "type": "endpoint|model|service|middleware|auth_scheme|resource",
            "description": "What it is/does",
            "derived_from": "QUOTE the exact text source"
        }
    ],
    "relationships": [
        {
            "from": "Component A",
            "to": "Component B",
            "type": "returns|accepts|calls|validates|authenticates|transforms|depends_on|rate_limits|errors_to",
            "description": "Nature of data flow",
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

SELF-CHECK: Did I include ALL named endpoints? Did I capture ALL request/response models?""",

    emission_preamble="""You are an API specification agent for the Motherlabs semantic compiler.
Generate a complete, structured YAML API definition for one API component.
Rules:
1. Output ONLY YAML inside a ```yaml block.
2. Honor ALL declared interfaces and data flows exactly.
3. Do NOT add undeclared endpoints or models.
4. Include HTTP methods, paths, request/response schemas, and status codes.
5. The YAML must be valid and well-structured (OpenAPI-flavored).
""",
)


# =============================================================================
# VERIFICATION — API design domain
# =============================================================================

API_VERIFICATION = VerificationOverrides(
    actionability_checks=("endpoints", "parameters", "methods"),
    readiness_label="api_readiness",
    dimension_weights=(0.20, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10),
)


# =============================================================================
# MATERIALIZATION — API design domain (YAML output)
# =============================================================================

API_MATERIALIZATION = MaterializationConfig(
    output_format="yaml",
    file_extension=".yaml",
    syntax_validator="yaml.safe_load",
)


# =============================================================================
# COMPLETE ADAPTER
# =============================================================================

API_ADAPTER = DomainAdapter(
    name="api",
    version="1.0",
    vocabulary=API_VOCABULARY,
    prompts=API_PROMPTS,
    classification=API_CLASSIFICATION,
    verification=API_VERIFICATION,
    materialization=API_MATERIALIZATION,
)
