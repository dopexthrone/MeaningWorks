"""
Canonical Motherlabs blueprint protocol.

This module encodes the 2026-03-07 Blueprints SSOT as transportable models and
postcode helpers. It is additive on purpose: the legacy blueprint schema still
exists, but any new blueprint-native work should depend on this protocol layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BLUEPRINTS_SSOT_VERSION = "1.0.0"
BLUEPRINTS_SSOT_DATE = "2026-03-07"

LAYER_CODES = (
    "INT",
    "SEM",
    "ORG",
    "COG",
    "AGN",
    "STR",
    "STA",
    "IDN",
    "TME",
    "EXC",
    "DAT",
    "SFX",
    "NET",
    "RES",
    "OBS",
    "SEC",
    "CTR",
    "EMG",
    "MET",
)

CONCERN_CODES = (
    "SEM",
    "ENT",
    "REL",
    "SCH",
    "ENM",
    "COL",
    "BHV",
    "FNC",
    "TRG",
    "STP",
    "LGC",
    "STA",
    "TRN",
    "DAT",
    "TRF",
    "FLW",
    "ACT",
    "PRM",
    "AUT",
    "SCO",
    "MEM",
    "PLN",
    "DLG",
    "NGT",
    "CNS",
    "HND",
    "ORC",
    "OBS",
    "CTR",
    "CFG",
    "LMT",
    "PLY",
    "LOG",
    "MET",
    "TRC",
    "ALT",
    "CND",
    "PRV",
    "VRS",
    "SIM",
    "RPT",
    "WRT",
    "EMT",
    "RED",
)

SCOPE_CODES = (
    "ECO",
    "APP",
    "DOM",
    "FET",
    "CMP",
    "FNC",
    "STP",
    "OPR",
    "EXP",
    "VAL",
)

DIMENSION_CODES = (
    "WHY",
    "WHO",
    "WHAT",
    "WHEN",
    "WHERE",
    "HOW",
    "HOW_MUCH",
    "IF",
)

DOMAIN_CODES = (
    "SFT",
    "ORG",
    "COG",
    "ECN",
    "PHY",
    "SOC",
    "NET",
    "EDU",
    "CRE",
    "LGL",
)

FILL_STATE_CODES = ("F", "P", "E", "B", "Q", "C")
NODE_STATUS_CODES = ("authored", "candidate", "quarantined", "promoted")
ANTI_GOAL_SEVERITIES = ("critical", "high", "medium")
RUNTIME_CHECK_MODES = ("startup", "periodic", "continuous")
SILENCE_TYPES = ("intentional", "deferred", "out_of_scope")
SILENCE_DECIDERS = ("agent", "human", "intent_contract")
RULE_TYPES = ("constraint", "policy", "trigger", "condition")
TEST_TYPES = ("rule", "anti_goal", "state_transition", "invariant")

CHALLENGE_TYPES = (
    "MISSING",
    "CONTRADICTION",
    "ASSUMPTION",
    "DEPTH",
    "META",
)

PIPELINE_AGENTS = (
    "Intent",
    "Persona",
    "Entity",
    "Process",
    "Synthesis",
    "Verify",
    "Governor",
)

PIPELINE_STATES = (
    "idle",
    "intent_phase",
    "persona_phase",
    "dialogue_phase",
    "synthesis_phase",
    "verification_phase",
    "governor_phase",
    "complete",
    "rejected",
    "halted",
    "recompiling",
)

EVENT_TYPES = (
    "AGENT_STARTED",
    "AGENT_COMPLETED",
    "GATE_PASSED",
    "GATE_FAILED",
    "NODE_WRITTEN",
    "CHALLENGE_ISSUED",
    "CONVERGENCE",
    "ESCALATION",
    "RECOMPILE_TRIGGERED",
    "BLUEPRINT_EMITTED",
)

NODE_REF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

LEGACY_COMPONENT_COORDINATES = {
    "entity": ("STR", "ENT", "APP", "WHAT", "SFT"),
    "process": ("EXC", "FNC", "APP", "HOW", "SFT"),
    "interface": ("NET", "SCH", "APP", "WHERE", "SFT"),
    "event": ("SFX", "EMT", "APP", "WHEN", "SFT"),
    "constraint": ("CTR", "PLY", "APP", "HOW", "SFT"),
    "subsystem": ("STR", "ENT", "DOM", "WHAT", "SFT"),
}

SCOPE_DEPTH = {scope: index for index, scope in enumerate(SCOPE_CODES)}
_LAYER_SET = frozenset(LAYER_CODES)
_CONCERN_SET = frozenset(CONCERN_CODES)
_SCOPE_SET = frozenset(SCOPE_CODES)
_DIMENSION_SET = frozenset(DIMENSION_CODES)
_DOMAIN_SET = frozenset(DOMAIN_CODES)
_FILL_STATE_SET = frozenset(FILL_STATE_CODES)
_NODE_STATUS_SET = frozenset(NODE_STATUS_CODES)
_ANTI_GOAL_SEVERITY_SET = frozenset(ANTI_GOAL_SEVERITIES)
_RUNTIME_CHECK_MODE_SET = frozenset(RUNTIME_CHECK_MODES)
_SILENCE_TYPE_SET = frozenset(SILENCE_TYPES)
_SILENCE_DECIDER_SET = frozenset(SILENCE_DECIDERS)
_RULE_TYPE_SET = frozenset(RULE_TYPES)
_TEST_TYPE_SET = frozenset(TEST_TYPES)


class BlueprintProtocolModel(BaseModel):
    """Strict base model for the blueprint protocol."""

    model_config = ConfigDict(extra="forbid")


def _validate_membership(value: str, allowed: frozenset[str], axis: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unknown {axis} code {value!r}")
    return value


class Postcode(BlueprintProtocolModel):
    """Parsed 5-axis semantic coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: str
    concern: str
    scope: str
    dimension: str
    domain: str

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, value: str) -> str:
        return _validate_membership(value, _LAYER_SET, "layer")

    @field_validator("concern")
    @classmethod
    def validate_concern(cls, value: str) -> str:
        return _validate_membership(value, _CONCERN_SET, "concern")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        return _validate_membership(value, _SCOPE_SET, "scope")

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: str) -> str:
        return _validate_membership(value, _DIMENSION_SET, "dimension")

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        return _validate_membership(value, _DOMAIN_SET, "domain")

    @property
    def key(self) -> str:
        return ".".join((self.layer, self.concern, self.scope, self.dimension, self.domain))

    @property
    def depth(self) -> int:
        return SCOPE_DEPTH[self.scope]

    def parent_scope(self) -> Optional[str]:
        index = self.depth
        if index == 0:
            return None
        return SCOPE_CODES[index - 1]

    def child_scope(self) -> Optional[str]:
        index = self.depth
        if index == len(SCOPE_CODES) - 1:
            return None
        return SCOPE_CODES[index + 1]

    def __str__(self) -> str:
        return self.key


class NodeRef(BlueprintProtocolModel):
    """Reference to a specific primitive within a postcode cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    postcode: str
    name: str

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not NODE_REF_NAME_RE.match(value):
            raise ValueError(
                "Node ref name must use letters, numbers, '.', '_' or '-'"
            )
        return value

    @property
    def key(self) -> str:
        return f"{self.postcode}/{self.name}"

    def __str__(self) -> str:
        return self.key


def parse_postcode(raw: str) -> Postcode:
    """Parse a strict postcode from its string form."""

    parts = raw.strip().split(".")
    if len(parts) != 5:
        raise ValueError(
            f"Postcode must have exactly 5 axes (got {len(parts)}): {raw!r}"
        )
    layer, concern, scope, dimension, domain = parts
    return Postcode(
        layer=layer,
        concern=concern,
        scope=scope,
        dimension=dimension,
        domain=domain,
    )


def is_postcode(raw: str) -> bool:
    try:
        parse_postcode(raw)
    except ValueError:
        return False
    return True


def parse_node_ref(raw: str) -> NodeRef:
    """Parse POSTCODE/name references used by the workbench."""

    postcode, sep, name = raw.strip().partition("/")
    if not sep or not name:
        raise ValueError(
            f"Node ref must use POSTCODE/name format, got {raw!r}"
        )
    return NodeRef(postcode=postcode, name=name)


def is_node_ref(raw: str) -> bool:
    try:
        parse_node_ref(raw)
    except ValueError:
        return False
    return True


def parse_semantic_ref(raw: str) -> str:
    """Accept node refs first, then bare postcodes for compatibility."""

    if is_node_ref(raw):
        return raw
    parse_postcode(raw)
    return raw


def normalize_node_name(raw: str) -> str:
    """Normalize primitive names into POSTCODE/name node-ref segments."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "node"


def make_node_ref(postcode: str, primitive: str) -> str:
    """Build a deterministic POSTCODE/name reference from a primitive name."""

    return str(NodeRef(postcode=postcode, name=normalize_node_name(primitive)))


class ConstraintSpec(BlueprintProtocolModel):
    """Minimal constraint shape shared by semantic nodes."""

    description: str
    expression: str = ""


class FreshnessPolicy(BlueprintProtocolModel):
    decay_rate: float = Field(0.001, ge=0.0)
    floor: float = Field(0.60, ge=0.0, le=1.0)
    stale_after: int = Field(90, ge=1)


class NodeReferences(BlueprintProtocolModel):
    read_before: List[str] = Field(default_factory=list)
    read_after: List[str] = Field(default_factory=list)
    see_also: List[str] = Field(default_factory=list)
    deep_dive: List[str] = Field(default_factory=list)
    warns: List[str] = Field(default_factory=list)

    @field_validator("read_before", "read_after", "see_also", "warns")
    @classmethod
    def validate_postcode_lists(cls, value: List[str]) -> List[str]:
        for item in value:
            parse_semantic_ref(item)
        return value


class NodeProvenance(BlueprintProtocolModel):
    source_ref: List[str] = Field(default_factory=list)
    agent_id: str
    run_id: str
    timestamp: str
    human_input: bool


class ProvenanceGate(BlueprintProtocolModel):
    source_agent: str
    source_nodes: List[str] = Field(default_factory=list)
    transformation: str
    hash: str

    @field_validator("source_nodes")
    @classmethod
    def validate_source_nodes(cls, value: List[str]) -> List[str]:
        for item in value:
            parse_postcode(item)
        return value


class BlueprintNode(BlueprintProtocolModel):
    id: str
    postcode: str
    primitive: str
    description: str
    notes: List[str] = Field(default_factory=list)

    fill_state: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str

    version: int = Field(ge=1)
    created_at: str
    updated_at: str

    last_verified: str
    freshness: FreshnessPolicy = Field(default_factory=FreshnessPolicy)

    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    connections: List[str] = Field(default_factory=list)
    depth: Optional[int] = Field(default=None, ge=0, le=9)

    references: NodeReferences = Field(default_factory=NodeReferences)

    provenance: NodeProvenance

    token_cost: int = Field(ge=0)

    constraints: List[ConstraintSpec] = Field(default_factory=list)
    constraint_source: List[str] = Field(default_factory=list)

    layer: Optional[str] = None
    concern: Optional[str] = None
    scope: Optional[str] = None
    dimension: Optional[str] = None
    domain: Optional[str] = None

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value

    @field_validator("fill_state")
    @classmethod
    def validate_fill_state(cls, value: str) -> str:
        return _validate_membership(value, _FILL_STATE_SET, "fill_state")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _validate_membership(value, _NODE_STATUS_SET, "status")

    @field_validator("connections", "constraint_source")
    @classmethod
    def validate_connections(cls, value: List[str]) -> List[str]:
        for item in value:
            parse_semantic_ref(item)
        return value

    @model_validator(mode="after")
    def derive_axes(self) -> "BlueprintNode":
        postcode = parse_postcode(self.postcode)
        for field_name in ("layer", "concern", "scope", "dimension", "domain"):
            expected = getattr(postcode, field_name)
            current = getattr(self, field_name)
            if current is not None and current != expected:
                raise ValueError(
                    f"{field_name}={current!r} conflicts with postcode {self.postcode!r}"
                )
            setattr(self, field_name, expected)

        if self.depth is not None and self.depth != postcode.depth:
            raise ValueError(
                f"depth={self.depth!r} conflicts with postcode depth {postcode.depth}"
            )
        self.depth = postcode.depth
        return self

    def effective_confidence(self, days_since_verified: float = 0.0) -> float:
        return max(
            self.confidence - (days_since_verified * self.freshness.decay_rate),
            self.freshness.floor,
        )


def _legacy_rel_from(relationship: Dict[str, Any]) -> str:
    return str(
        relationship.get("from_component")
        or relationship.get("from")
        or ""
    )


def _legacy_rel_to(relationship: Dict[str, Any]) -> str:
    return str(
        relationship.get("to_component")
        or relationship.get("to")
        or ""
    )


def _legacy_coordinate(component: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    component_type = str(component.get("type") or "entity")
    haystack = f'{component.get("name", "")} {component.get("description", "")}'.lower()

    if component_type == "constraint":
        if re.search(r"(auth|permission|token|secure|validation|sanitize|encryption|guard)", haystack):
            return ("SEC", "PLY", "APP", "HOW", "SFT")
        if re.search(r"(limit|bound|threshold|rate)", haystack):
            return ("CTR", "LMT", "APP", "IF", "SFT")

    if component_type == "process" and component.get("state_machine"):
        return ("STA", "STA", "APP", "WHEN", "SFT")

    return LEGACY_COMPONENT_COORDINATES.get(component_type, ("SEM", "SEM", "APP", "WHAT", "SFT"))


def _legacy_node_confidence(
    component: Dict[str, Any],
    related_gaps: List[str],
    trust_score: float,
) -> float:
    score = 0.56
    if component.get("description"):
        score += 0.12
    if component.get("derived_from"):
        score += 0.12
    if component.get("attributes"):
        score += 0.04
    if component.get("methods"):
        score += 0.07
    if component.get("validation_rules"):
        score += 0.05
    if component.get("state_machine"):
        score += 0.08
    score += ((trust_score / 100.0) - 0.5) * 0.08
    score -= min(0.18, len(related_gaps) * 0.08)
    return round(min(0.98, max(0.35, score)), 2)


def _legacy_fill_state(
    component: Dict[str, Any],
    related_gaps: List[str],
    confidence: float,
) -> str:
    if any(re.search(r"(blocked|needs|halt|missing dependency)", gap, re.IGNORECASE) for gap in related_gaps):
        return "B"
    if not component.get("derived_from") or related_gaps or confidence < 0.82:
        return "P"
    return "F"


def _legacy_node_status(fill_state: str) -> str:
    if fill_state == "F":
        return "promoted"
    if fill_state == "Q":
        return "quarantined"
    if fill_state == "C":
        return "candidate"
    return "authored"


def _legacy_node_notes(component: Dict[str, Any], related_gaps: List[str]) -> List[str]:
    notes: List[str] = []
    methods = component.get("methods") or []
    validation_rules = component.get("validation_rules") or []
    state_machine = component.get("state_machine") or {}

    if methods:
        notes.append(
            f"Carries {len(methods)} implementation hint{'s' if len(methods) != 1 else ''}."
        )
    if state_machine.get("states"):
        state_count = len(state_machine.get("states") or [])
        notes.append(
            f"Includes {state_count} state{'s' if state_count != 1 else ''}."
        )
    if validation_rules:
        notes.append(
            f"Protected by {len(validation_rules)} validation rule{'s' if len(validation_rules) != 1 else ''}."
        )
    if not component.get("derived_from"):
        notes.append("Direct provenance is still thin for this node.")
    if related_gaps:
        notes.append(
            f"{len(related_gaps)} open gap{'s' if len(related_gaps) != 1 else ''} still touch this node."
        )
    return notes


def _coerce_projection_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _coerce_projection_node(value: Any) -> Optional["BlueprintNode"]:
    if isinstance(value, BlueprintNode):
        return value
    if isinstance(value, dict):
        try:
            return BlueprintNode.model_validate(value)
        except Exception:
            return None
    if hasattr(value, "model_dump"):
        try:
            return BlueprintNode.model_validate(value.model_dump())
        except Exception:
            return None
    return None


def _match_projection_node(nodes: List["BlueprintNode"], text: str) -> Optional["BlueprintNode"]:
    needle = str(text or "").strip().lower()
    if not needle:
        return None

    for node in nodes:
        node_ref = make_node_ref(node.postcode, node.primitive).lower()
        if (
            needle == node.primitive.lower()
            or node_ref in needle
            or node.postcode.lower() in needle
            or node.primitive.lower() in needle
        ):
            return node
    return None


def _match_legacy_component(blueprint: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    needle = str(text or "").strip().lower()
    if not needle:
        return None

    components = list(blueprint.get("components") or [])
    topic_prefix = needle.split(":", 1)[0].strip() if ":" in needle else needle

    for component in components:
        name = str(component.get("name") or "").strip()
        if name and name.lower() == topic_prefix:
            return component

    for component in components:
        name = str(component.get("name") or "").strip()
        if name and name.lower() in needle:
            return component

    return None


def project_legacy_blueprint_nodes(
    blueprint: Dict[str, Any],
    *,
    seed_text: str = "",
    trust: Any = None,
    verification: Any = None,
    run_id: str = "legacy_projection",
    agent_id: str = "Synthesis",
    timestamp: Optional[str] = None,
) -> List[BlueprintNode]:
    """Project the legacy component blueprint into postcode-native semantic nodes."""

    trust_payload = _coerce_projection_dict(trust)
    verification_payload = _coerce_projection_dict(verification)
    blueprint = blueprint or {}
    components = list(blueprint.get("components") or [])
    relationships = list(blueprint.get("relationships") or [])
    constraints = list(blueprint.get("constraints") or [])
    gap_texts = [
        *(str(gap) for gap in trust_payload.get("gap_report", []) or []),
        *(f"Unresolved: {gap}" for gap in blueprint.get("unresolved", []) or []),
    ]
    trust_score = float(trust_payload.get("overall_score") or 0.0)
    created_at = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    base_rows: List[Dict[str, Any]] = []
    for index, component in enumerate(components):
        layer, concern, scope, dimension, domain = _legacy_coordinate(component)
        postcode = ".".join((layer, concern, scope, dimension, domain))
        primitive = str(component.get("name") or f"component_{index + 1}")
        node_ref = make_node_ref(postcode, primitive)
        related_gaps = [
            gap for gap in gap_texts
            if primitive.lower() in gap.lower() or node_ref.lower() in gap.lower() or postcode.lower() in gap.lower()
        ]
        confidence = _legacy_node_confidence(component, related_gaps, trust_score)
        fill_state = _legacy_fill_state(component, related_gaps, confidence)
        base_rows.append({
            "index": index,
            "component": component,
            "primitive": primitive,
            "postcode": postcode,
            "node_ref": node_ref,
            "related_gaps": related_gaps,
            "confidence": confidence,
            "fill_state": fill_state,
        })

    by_name = {row["primitive"]: row for row in base_rows}
    nodes: List[BlueprintNode] = []

    if seed_text or blueprint.get("core_need"):
        purpose_text = str(blueprint.get("core_need") or seed_text).strip()
        if purpose_text:
            nodes.append(BlueprintNode(
                id="node-purpose",
                postcode="INT.SEM.APP.WHY.SFT",
                primitive="purpose",
                description=purpose_text,
                notes=[],
                fill_state="F",
                confidence=0.98,
                status="promoted",
                version=1,
                created_at=created_at,
                updated_at=created_at,
                last_verified=created_at,
                parent=None,
                children=[],
                connections=[],
                references=NodeReferences(),
                provenance=NodeProvenance(
                    source_ref=[purpose_text],
                    agent_id="Intent",
                    run_id=run_id,
                    timestamp=created_at,
                    human_input=True,
                ),
                token_cost=0,
                constraints=[],
                constraint_source=[],
            ))

    intent_ref = make_node_ref("INT.SEM.APP.WHY.SFT", "purpose") if nodes else None

    for row in base_rows:
        component = row["component"]
        primitive = row["primitive"]
        inbound_refs = []
        outbound_refs = []

        for relationship in relationships:
            if _legacy_rel_to(relationship) == primitive:
                source = by_name.get(_legacy_rel_from(relationship))
                if source:
                    inbound_refs.append(source["node_ref"])
            if _legacy_rel_from(relationship) == primitive:
                target = by_name.get(_legacy_rel_to(relationship))
                if target:
                    outbound_refs.append(target["node_ref"])

        if not inbound_refs and intent_ref:
            inbound_refs.append(intent_ref)

        same_layer_refs = [
            candidate["node_ref"]
            for candidate in base_rows
            if candidate["node_ref"] != row["node_ref"]
            and candidate["postcode"].split(".", 1)[0] == row["postcode"].split(".", 1)[0]
        ][:4]

        related_constraints = [
            str(constraint.get("description") or "").strip()
            for constraint in constraints
            if primitive in (constraint.get("applies_to") or [])
            and str(constraint.get("description") or "").strip()
        ]
        validation_rules = [str(rule).strip() for rule in component.get("validation_rules") or [] if str(rule).strip()]
        constraint_items = []
        seen_constraint_texts = set()
        for description in [*validation_rules, *related_constraints]:
            if description in seen_constraint_texts:
                continue
            seen_constraint_texts.add(description)
            constraint_items.append(ConstraintSpec(description=description))

        source_refs = []
        if component.get("derived_from"):
            source_refs.append(str(component["derived_from"]))
        elif seed_text:
            source_refs.append(seed_text)
        if not source_refs and verification_payload:
            source_refs.append("verification-backed projection")

        nodes.append(BlueprintNode(
            id=f"node-{row['index'] + 1}-{normalize_node_name(primitive)}",
            postcode=row["postcode"],
            primitive=primitive,
            description=str(component.get("description") or "No semantic description yet."),
            notes=_legacy_node_notes(component, row["related_gaps"]),
            fill_state=row["fill_state"],
            confidence=row["confidence"],
            status=_legacy_node_status(row["fill_state"]),
            version=1,
            created_at=created_at,
            updated_at=created_at,
            last_verified=created_at,
            parent=inbound_refs[0] if inbound_refs else None,
            children=list(dict.fromkeys(outbound_refs)),
            connections=list(dict.fromkeys([*inbound_refs, *outbound_refs])),
            references=NodeReferences(
                read_before=list(dict.fromkeys(inbound_refs[:3])),
                read_after=list(dict.fromkeys(outbound_refs[:3])),
                see_also=list(dict.fromkeys(same_layer_refs)),
                warns=list(dict.fromkeys(outbound_refs[:3])),
            ),
            provenance=NodeProvenance(
                source_ref=source_refs or ["projection:legacy_blueprint"],
                agent_id=agent_id,
                run_id=run_id,
                timestamp=created_at,
                human_input=False,
            ),
            token_cost=0,
            constraints=constraint_items,
            constraint_source=[],
        ))

    return sorted(nodes, key=lambda node: (node.postcode, make_node_ref(node.postcode, node.primitive)))


def build_blueprint_semantic_nodes(
    blueprint: Dict[str, Any],
    *,
    seed_text: str = "",
    trust: Any = None,
    verification: Any = None,
    run_id: str = "legacy_projection",
    agent_id: str = "Synthesis",
    timestamp: Optional[str] = None,
) -> List[BlueprintNode]:
    """Prefer native semantic_nodes sidecars, then fall back to legacy projection."""

    blueprint = blueprint or {}
    native_nodes = list(blueprint.get("semantic_nodes") or [])
    created_at = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def normalize_ref_list(values: Any) -> List[str]:
        refs: List[str] = []
        for raw in values or []:
            item = str(raw or "").strip()
            if not item:
                continue
            try:
                parse_semantic_ref(item)
            except ValueError:
                continue
            refs.append(item)
        return refs

    def normalize_constraints(values: Any) -> List[ConstraintSpec]:
        constraints: List[ConstraintSpec] = []
        for raw in values or []:
            if isinstance(raw, ConstraintSpec):
                constraints.append(raw)
                continue
            if isinstance(raw, dict):
                try:
                    constraints.append(ConstraintSpec.model_validate(raw))
                    continue
                except Exception:
                    description = str(raw.get("description") or "").strip()
                    if description:
                        constraints.append(ConstraintSpec(description=description))
                    continue
            description = str(raw or "").strip()
            if description:
                constraints.append(ConstraintSpec(description=description))
        return constraints

    nodes: List[BlueprintNode] = []
    for index, raw_node in enumerate(native_nodes):
        if not isinstance(raw_node, dict):
            continue

        validated = _coerce_projection_node(raw_node)
        if validated is not None:
            nodes.append(validated)
            continue

        primitive = str(raw_node.get("primitive") or raw_node.get("name") or "").strip()
        postcode = str(raw_node.get("postcode") or "").strip()
        if not primitive or not postcode:
            continue

        fill_state = str(raw_node.get("fill_state") or ("F" if raw_node.get("confidence", 0.85) >= 0.85 else "P"))
        references_payload = raw_node.get("references") if isinstance(raw_node.get("references"), dict) else {}
        provenance_payload = raw_node.get("provenance") if isinstance(raw_node.get("provenance"), dict) else {}
        source_ref = provenance_payload.get("source_ref") or raw_node.get("source_ref") or [seed_text or primitive]
        if isinstance(source_ref, str):
            source_ref = [source_ref]

        try:
            nodes.append(BlueprintNode(
                id=str(raw_node.get("id") or f"node-native-{index + 1}-{normalize_node_name(primitive)}"),
                postcode=postcode,
                primitive=primitive,
                description=str(raw_node.get("description") or "No semantic description yet."),
                notes=[str(note).strip() for note in raw_node.get("notes", []) or [] if str(note).strip()],
                fill_state=fill_state,
                confidence=float(raw_node.get("confidence") if raw_node.get("confidence") is not None else 0.85),
                status=str(raw_node.get("status") or _legacy_node_status(fill_state)),
                version=int(raw_node.get("version") or 1),
                created_at=str(raw_node.get("created_at") or created_at),
                updated_at=str(raw_node.get("updated_at") or raw_node.get("created_at") or created_at),
                last_verified=str(raw_node.get("last_verified") or raw_node.get("updated_at") or created_at),
                freshness=FreshnessPolicy.model_validate(raw_node.get("freshness") or {}),
                parent=str(raw_node.get("parent") or "") or None,
                children=normalize_ref_list(raw_node.get("children") or []),
                connections=normalize_ref_list(raw_node.get("connections") or []),
                depth=raw_node.get("depth"),
                references=NodeReferences.model_validate({
                    "read_before": normalize_ref_list(references_payload.get("read_before") or []),
                    "read_after": normalize_ref_list(references_payload.get("read_after") or []),
                    "see_also": normalize_ref_list(references_payload.get("see_also") or []),
                    "deep_dive": [str(item).strip() for item in references_payload.get("deep_dive", []) or [] if str(item).strip()],
                    "warns": normalize_ref_list(references_payload.get("warns") or []),
                }),
                provenance=NodeProvenance(
                    source_ref=[str(item).strip() for item in source_ref if str(item).strip()],
                    agent_id=str(provenance_payload.get("agent_id") or raw_node.get("agent_id") or agent_id),
                    run_id=str(provenance_payload.get("run_id") or raw_node.get("run_id") or run_id),
                    timestamp=str(provenance_payload.get("timestamp") or raw_node.get("timestamp") or created_at),
                    human_input=bool(provenance_payload.get("human_input") or raw_node.get("human_input", False)),
                ),
                token_cost=int(raw_node.get("token_cost") or 0),
                constraints=normalize_constraints(raw_node.get("constraints") or []),
                constraint_source=normalize_ref_list(raw_node.get("constraint_source") or []),
            ))
        except Exception:
            continue

    if nodes:
        if seed_text and not any(
            node.postcode == "INT.SEM.APP.WHY.SFT" and node.primitive == "purpose"
            for node in nodes
        ):
            nodes.append(BlueprintNode(
                id="node-purpose",
                postcode="INT.SEM.APP.WHY.SFT",
                primitive="purpose",
                description=seed_text,
                notes=[],
                fill_state="F",
                confidence=0.98,
                status="promoted",
                version=1,
                created_at=created_at,
                updated_at=created_at,
                last_verified=created_at,
                parent=None,
                children=[],
                connections=[],
                references=NodeReferences(),
                provenance=NodeProvenance(
                    source_ref=[seed_text],
                    agent_id="Intent",
                    run_id=run_id,
                    timestamp=created_at,
                    human_input=True,
                ),
                token_cost=0,
                constraints=[],
                constraint_source=[],
            ))

        return sorted(nodes, key=lambda node: (node.postcode, make_node_ref(node.postcode, node.primitive)))

    return project_legacy_blueprint_nodes(
        blueprint,
        seed_text=seed_text,
        trust=trust,
        verification=verification,
        run_id=run_id,
        agent_id=agent_id,
        timestamp=timestamp,
    )


def build_blueprint_semantic_gates(
    blueprint: Dict[str, Any],
    *,
    trust: Any = None,
    verification: Any = None,
    context_graph: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Attach semantic gate ownership directly to the legacy blueprint."""
    blueprint = blueprint or {}
    trust_payload = _coerce_projection_dict(trust)
    verification_payload = _coerce_projection_dict(verification)
    context_graph = context_graph or {}

    gates: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def append_gate(
        *,
        question: str,
        options: Optional[List[str]] = None,
        kind: str,
        stage: str,
        owner_text: str,
        explicit_postcode: Optional[str] = None,
        explicit_node_ref: Optional[str] = None,
    ) -> None:
        component = _match_legacy_component(blueprint, owner_text)
        if explicit_postcode:
            postcode = explicit_postcode
            if explicit_node_ref:
                node_ref = explicit_node_ref
            elif component:
                primitive = str(component.get("name") or "region")
                node_ref = make_node_ref(postcode, primitive)
            else:
                node_ref = None
        elif component:
            postcode = ".".join(_legacy_coordinate(component))
            primitive = str(component.get("name") or "region")
            node_ref = make_node_ref(postcode, primitive)
        else:
            postcode = "INT.SEM.APP.IF.SFT"
            node_ref = explicit_node_ref

        item = (postcode, question)
        if item in seen:
            return
        seen.add(item)
        gates.append({
            "postcode": postcode,
            "question": question,
            "options": list(options or []),
            "node_ref": node_ref,
            "kind": kind,
            "stage": stage,
        })

    for gate_source in (
        blueprint.get("semantic_gates", []) or [],
        verification_payload.get("semantic_gates", []) or [],
    ):
        for raw_gate in gate_source:
            if not isinstance(raw_gate, dict):
                continue

            question = str(raw_gate.get("question") or "").strip()
            if not question:
                continue

            owner_text = str(
                raw_gate.get("owner_component")
                or raw_gate.get("owner")
                or raw_gate.get("component")
                or raw_gate.get("node_ref")
                or raw_gate.get("postcode")
                or question
            ).strip()
            explicit_postcode = str(raw_gate.get("postcode") or "").strip() or None
            explicit_node_ref = str(raw_gate.get("node_ref") or "").strip() or None
            append_gate(
                question=question,
                options=[str(option) for option in raw_gate.get("options", []) or [] if str(option).strip()],
                kind=str(raw_gate.get("kind") or "semantic_gate"),
                stage=str(raw_gate.get("stage") or "verification"),
                owner_text=owner_text,
                explicit_postcode=explicit_postcode,
                explicit_node_ref=explicit_node_ref,
            )

    conflict_summary = context_graph.get("conflict_summary") or {}
    for conflict in conflict_summary.get("unresolved", []) or []:
        if not isinstance(conflict, dict):
            continue
        category = str(conflict.get("category") or "TRADEOFF").upper()
        if category not in {"MISSING_INFO", "PRIORITY"}:
            continue

        topic = str(conflict.get("topic") or "").strip()
        if not topic:
            continue

        positions = conflict.get("positions") or {}
        options: List[str] = []
        if isinstance(positions, dict):
            for value in positions.values():
                option = str(value or "").strip()
                if option and option not in options:
                    options.append(option)

        if topic.endswith("?"):
            question = topic
        elif category == "PRIORITY":
            question = f"Which priority should Motherlabs lock for {topic}?"
        else:
            question = f"Which direction should Motherlabs lock for {topic}?"

        append_gate(
            question=question,
            options=options,
            kind="semantic_conflict",
            stage="verification",
            owner_text=topic,
        )

    gap_texts = [
        *(str(gap) for gap in trust_payload.get("gap_report", []) or []),
        *(str(gap) for gap in blueprint.get("unresolved", []) or []),
    ]
    for gap in gap_texts:
        question = re.sub(r"^unresolved:\s*", "", gap, flags=re.IGNORECASE).strip()
        if not question:
            continue
        append_gate(
            question=question,
            options=[],
            kind="gap",
            stage="verification",
            owner_text=gap,
        )

    return gates


def build_semantic_gate_escalations(
    semantic_nodes: List[Any],
    *,
    blueprint: Optional[Dict[str, Any]] = None,
    trust: Any = None,
    context_graph: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build human-in-the-loop pauses from semantic conflicts and blocked nodes."""
    nodes = [
        node
        for node in (
            _coerce_projection_node(raw_node)
            for raw_node in (semantic_nodes or [])
        )
        if node is not None
    ]
    blueprint = blueprint or {}
    trust_payload = _coerce_projection_dict(trust)
    context_graph = context_graph or {}

    native_gates = list(blueprint.get("semantic_gates") or [])
    if native_gates:
        escalations: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for gate in native_gates:
            if not isinstance(gate, dict):
                continue
            postcode = str(gate.get("postcode") or "INT.SEM.APP.IF.SFT")
            question = str(gate.get("question") or "").strip()
            if not question:
                continue
            item = (postcode, question)
            if item in seen:
                continue
            seen.add(item)
            node_ref = gate.get("node_ref")
            if not node_ref:
                node = _match_projection_node(nodes, gate.get("question") or "")
                if node:
                    node_ref = make_node_ref(node.postcode, node.primitive)
            escalations.append({
                "postcode": postcode,
                "question": question,
                "options": [str(option) for option in gate.get("options", []) or [] if str(option).strip()],
                "node_ref": str(node_ref) if node_ref else None,
                "kind": str(gate.get("kind") or "semantic_gate"),
                "stage": str(gate.get("stage") or "verification"),
            })
        return escalations

    gap_texts = [
        *(str(gap) for gap in trust_payload.get("gap_report", []) or []),
        *(str(gap) for gap in blueprint.get("unresolved", []) or []),
    ]
    escalations: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    conflict_summary = context_graph.get("conflict_summary") or {}
    unresolved_conflicts = conflict_summary.get("unresolved") or []
    for conflict in unresolved_conflicts:
        if not isinstance(conflict, dict):
            continue

        category = str(conflict.get("category") or "TRADEOFF").upper()
        if category not in {"MISSING_INFO", "PRIORITY"}:
            continue

        topic = str(conflict.get("topic") or "").strip()
        if not topic:
            continue

        positions = conflict.get("positions") or {}
        options: List[str] = []
        seen_options: set[str] = set()
        if isinstance(positions, dict):
            for value in positions.values():
                option = str(value or "").strip()
                if not option:
                    continue
                key = option.lower()
                if key in seen_options:
                    continue
                seen_options.add(key)
                options.append(option)

        if topic.endswith("?"):
            question = topic
        elif category == "PRIORITY":
            question = f"Which priority should Motherlabs lock for {topic}?"
        else:
            question = f"Which direction should Motherlabs lock for {topic}?"

        component_name = topic.split(":", 1)[0].strip() if ":" in topic else topic
        node = _match_projection_node(nodes, component_name) or _match_projection_node(nodes, topic)
        postcode = node.postcode if node else "INT.SEM.APP.IF.SFT"

        item = (postcode, question)
        if item in seen:
            continue
        seen.add(item)
        escalations.append({
            "postcode": postcode,
            "question": question,
            "options": options,
            "node_ref": make_node_ref(node.postcode, node.primitive) if node else None,
            "kind": "semantic_conflict",
            "stage": "verification",
        })

    for node in nodes:
        if node.fill_state != "B":
            continue

        primitive = str(node.primitive or node.postcode or "this region")
        node_ref = make_node_ref(node.postcode, primitive).lower()
        question = ""
        for gap in gap_texts:
            gap_lower = gap.lower()
            if (
                primitive.lower() in gap_lower
                or node.postcode.lower() in gap_lower
                or node_ref in gap_lower
            ):
                question = re.sub(r"^unresolved:\s*", "", gap, flags=re.IGNORECASE).strip()
                break

        if not question:
            question = f"What should {primitive} contain?"

        item = (node.postcode, question)
        if item in seen:
            continue
        seen.add(item)
        escalations.append({
            "postcode": node.postcode,
            "question": question,
            "options": [],
            "node_ref": make_node_ref(node.postcode, node.primitive),
            "kind": "blocked_node",
            "stage": "verification",
        })

    return escalations


class AntiGoal(BlueprintProtocolModel):
    description: str
    derived_from: str
    severity: str
    detection: str

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        return _validate_membership(value, _ANTI_GOAL_SEVERITY_SET, "anti_goal severity")


class RuntimeAnchor(BlueprintProtocolModel):
    enabled: bool
    invariants: List[str] = Field(default_factory=list)
    check_mode: str

    @field_validator("check_mode")
    @classmethod
    def validate_check_mode(cls, value: str) -> str:
        return _validate_membership(value, _RUNTIME_CHECK_MODE_SET, "runtime check_mode")


class ContextBudget(BlueprintProtocolModel):
    total: int = Field(ge=0)
    reserved: int = Field(ge=0)
    available: int = Field(ge=0)
    per_agent: int = Field(ge=0)
    compression_trigger: float = Field(ge=0.0, le=1.0)


class IntentContract(BlueprintProtocolModel):
    seed_text: str
    goals: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    layers_in_scope: List[str] = Field(default_factory=list)
    domains_in_scope: List[str] = Field(default_factory=list)
    known_unknowns: List[str] = Field(default_factory=list)
    budget_limit: float = Field(ge=0.0)
    anti_goals: List[AntiGoal] = Field(default_factory=list)
    runtime_anchor: RuntimeAnchor
    context_budget: ContextBudget
    seed_hash: str
    contract_hash: str

    @field_validator("layers_in_scope")
    @classmethod
    def validate_layers_in_scope(cls, value: List[str]) -> List[str]:
        for item in value:
            _validate_membership(item, _LAYER_SET, "layer")
        return value

    @field_validator("domains_in_scope")
    @classmethod
    def validate_domains_in_scope(cls, value: List[str]) -> List[str]:
        for item in value:
            _validate_membership(item, _DOMAIN_SET, "domain")
        return value


class Silence(BlueprintProtocolModel):
    layer: str
    reason: str
    type: str
    decided_by: str

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, value: str) -> str:
        return _validate_membership(value, _LAYER_SET, "layer")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        return _validate_membership(value, _SILENCE_TYPE_SET, "silence type")

    @field_validator("decided_by")
    @classmethod
    def validate_decided_by(cls, value: str) -> str:
        return _validate_membership(value, _SILENCE_DECIDER_SET, "silence decider")


class DepthReport(BlueprintProtocolModel):
    """User-facing compilation depth signal.

    The SSOT defines the metric and labels, but not every field name.
    This shape captures the stable core and leaves room for extension.
    """

    model_config = ConfigDict(extra="allow")

    label: Optional[str] = None
    average_scope_depth: Optional[float] = None
    filled_ratio: Optional[float] = None
    partial_ratio: Optional[float] = None
    empty_ratio: Optional[float] = None
    activated_layer_ratio: Optional[float] = None
    anti_goals_identified: Optional[int] = None
    dialogue_rounds_completed: Optional[int] = None
    gaps_remaining: Optional[int] = None


class CostReport(BlueprintProtocolModel):
    """Cost surface carried into governance."""

    model_config = ConfigDict(extra="allow")

    estimated_usd: Optional[float] = None
    actual_usd: Optional[float] = None
    by_agent: Dict[str, float] = Field(default_factory=dict)
    budget_limit_usd: Optional[float] = None
    halted: Optional[bool] = None


class EscalatedDecision(BlueprintProtocolModel):
    postcode: str
    question: str
    answer: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    node_ref: Optional[str] = None
    kind: Optional[str] = None
    stage: Optional[str] = None

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value

    @field_validator("node_ref")
    @classmethod
    def validate_node_ref(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        parse_node_ref(value)
        return value


class QuarantinedNode(BlueprintProtocolModel):
    postcode: str
    reason: str

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class AxiomViolation(BlueprintProtocolModel):
    axiom: str
    node: str
    detail: str

    @field_validator("node")
    @classmethod
    def validate_node(cls, value: str) -> str:
        parse_postcode(value)
        return value


class HumanDecision(BlueprintProtocolModel):
    postcode: str
    question: str
    answer: str
    timestamp: str

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class GovernanceReport(BlueprintProtocolModel):
    total_nodes: int = Field(ge=0)
    promoted: int = Field(ge=0)
    quarantined: List[QuarantinedNode] = Field(default_factory=list)
    escalated: List[EscalatedDecision] = Field(default_factory=list)
    axiom_violations: List[AxiomViolation] = Field(default_factory=list)
    human_decisions: List[HumanDecision] = Field(default_factory=list)
    coverage: float = Field(ge=0.0, le=100.0)
    anti_goals_checked: int = Field(ge=0)
    compilation_depth: DepthReport = Field(default_factory=DepthReport)
    cost_report: CostReport = Field(default_factory=CostReport)


class BlueprintMetadata(BlueprintProtocolModel):
    id: str
    seed: str
    seed_hash: str
    blueprint_hash: str
    created_at: str
    version: str
    compilation_depth: DepthReport = Field(default_factory=DepthReport)


class LayerCoverage(BlueprintProtocolModel):
    layer: str
    nodeCount: int = Field(ge=0)
    coverage: List[str] = Field(default_factory=list)

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, value: str) -> str:
        return _validate_membership(value, _LAYER_SET, "layer")

    @field_validator("coverage")
    @classmethod
    def validate_coverage(cls, value: List[str]) -> List[str]:
        for item in value:
            parse_postcode(item)
        return value


class CompiledEntity(BlueprintProtocolModel):
    name: str
    postcode: str
    description: str
    attributes: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class CompiledFunction(BlueprintProtocolModel):
    name: str
    postcode: str
    description: str
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class CompiledRule(BlueprintProtocolModel):
    name: str
    postcode: str
    description: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        return _validate_membership(value, _RULE_TYPE_SET, "rule type")


class CompiledRelationship(BlueprintProtocolModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_postcode: str = Field(alias="from")
    to_postcode: str = Field(alias="to")
    relation: str
    postcode: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("from_postcode", "to_postcode", "postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class StateTransition(BlueprintProtocolModel):
    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")
    trigger: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CompiledStateMachine(BlueprintProtocolModel):
    name: str
    postcode: str
    states: List[str] = Field(default_factory=list)
    transitions: List[StateTransition] = Field(default_factory=list)

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class Gap(BlueprintProtocolModel):
    layer: str
    concern: str
    scope: str
    reason: str
    priority: str

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, value: str) -> str:
        return _validate_membership(value, _LAYER_SET, "layer")

    @field_validator("concern")
    @classmethod
    def validate_concern(cls, value: str) -> str:
        return _validate_membership(value, _CONCERN_SET, "concern")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        return _validate_membership(value, _SCOPE_SET, "scope")


class RejectedPath(BlueprintProtocolModel):
    postcode: str
    reason: str

    @field_validator("postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value


class FailedPathConflict(BlueprintProtocolModel):
    nodes: List[str] = Field(default_factory=list)
    resolution: str

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, value: List[str]) -> List[str]:
        for item in value:
            parse_postcode(item)
        return value


class FailedPaths(BlueprintProtocolModel):
    rejected: List[RejectedPath] = Field(default_factory=list)
    deferred: List[RejectedPath] = Field(default_factory=list)
    conflicts: List[FailedPathConflict] = Field(default_factory=list)


class CompiledTest(BlueprintProtocolModel):
    source_postcode: str
    test_type: str
    description: str
    assertion: str

    @field_validator("source_postcode")
    @classmethod
    def validate_postcode(cls, value: str) -> str:
        parse_postcode(value)
        return value

    @field_validator("test_type")
    @classmethod
    def validate_test_type(cls, value: str) -> str:
        return _validate_membership(value, _TEST_TYPE_SET, "test type")


class CompiledBlueprint(BlueprintProtocolModel):
    metadata: BlueprintMetadata
    intent_contract: IntentContract
    layers: List[LayerCoverage] = Field(default_factory=list)
    entities: List[CompiledEntity] = Field(default_factory=list)
    functions: List[CompiledFunction] = Field(default_factory=list)
    rules: List[CompiledRule] = Field(default_factory=list)
    relationships: List[CompiledRelationship] = Field(default_factory=list)
    state_machines: List[CompiledStateMachine] = Field(default_factory=list)
    silences: List[Silence] = Field(default_factory=list)
    gaps: List[Gap] = Field(default_factory=list)
    failed_paths: FailedPaths = Field(default_factory=FailedPaths)
    governance_report: GovernanceReport
    tests: List[CompiledTest] = Field(default_factory=list)


__all__ = [
    "ANTI_GOAL_SEVERITIES",
    "BLUEPRINTS_SSOT_DATE",
    "BLUEPRINTS_SSOT_VERSION",
    "CHALLENGE_TYPES",
    "CompiledBlueprint",
    "CompiledEntity",
    "CompiledFunction",
    "CompiledRelationship",
    "CompiledRule",
    "CompiledStateMachine",
    "CompiledTest",
    "CONCERN_CODES",
    "ConstraintSpec",
    "ContextBudget",
    "CostReport",
    "DIMENSION_CODES",
    "DOMAIN_CODES",
    "EVENT_TYPES",
    "FILL_STATE_CODES",
    "FailedPaths",
    "FreshnessPolicy",
    "Gap",
    "GovernanceReport",
    "IntentContract",
    "LAYER_CODES",
    "LayerCoverage",
    "NODE_STATUS_CODES",
    "NodeProvenance",
    "NodeReferences",
    "NodeRef",
    "PIPELINE_AGENTS",
    "PIPELINE_STATES",
    "Postcode",
    "RULE_TYPES",
    "RUNTIME_CHECK_MODES",
    "SCOPE_CODES",
    "SCOPE_DEPTH",
    "SILENCE_DECIDERS",
    "SILENCE_TYPES",
    "TEST_TYPES",
    "BlueprintMetadata",
    "BlueprintNode",
    "DepthReport",
    "HumanDecision",
    "is_node_ref",
    "ProvenanceGate",
    "QuarantinedNode",
    "RuntimeAnchor",
    "Silence",
    "build_blueprint_semantic_gates",
    "build_blueprint_semantic_nodes",
    "build_semantic_gate_escalations",
    "make_node_ref",
    "normalize_node_name",
    "parse_node_ref",
    "is_postcode",
    "parse_postcode",
    "parse_semantic_ref",
    "project_legacy_blueprint_nodes",
]
