"""
mother/claude_agents.py — Standalone semantic compiler using Claude tool-use.

LEAF module. Dependencies: anthropic SDK + stdlib only.
Zero imports from core/, mother/, kernel/, agents/.

A lightweight parallel to the main engine: two Claude agents with complementary
blind spots excavate structure from natural language through a shared grid
until they converge. The structure IS the answer.

Usage:
    python3 mother/claude_agents.py "A task manager with user auth and real-time notifications"
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures (all frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    """Structured intent extracted from natural language."""
    core_need: str
    domain: str
    actors: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    implicit_goals: tuple[str, ...] = ()
    insight: str = ""


@dataclass(frozen=True)
class GridCell:
    """Single cell in the semantic grid."""
    postcode: str
    content: str = ""
    fill_state: str = "empty"  # empty | partial | filled
    source: str = ""           # which agent filled it
    confidence: float = 0.0
    connections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Component:
    """A structural component in the blueprint."""
    name: str
    component_type: str
    description: str
    attributes: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    derived_from: str = ""


@dataclass(frozen=True)
class Relationship:
    """A relationship between components."""
    source: str
    target: str
    rel_type: str
    description: str = ""
    derived_from: str = ""


@dataclass(frozen=True)
class Blueprint:
    """Merged structural + behavioral output."""
    components: tuple[Component, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    constraints: tuple[str, ...] = ()
    insights: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationScore:
    """4-dimension verification result."""
    completeness: float = 0.0
    consistency: float = 0.0
    coherence: float = 0.0
    traceability: float = 0.0
    overall: float = 0.0
    gaps: tuple[str, ...] = ()
    recommendation: str = ""


@dataclass(frozen=True)
class CompilerResult:
    """Final output of the compilation pipeline."""
    intent: Optional[Intent] = None
    blueprint: Optional[Blueprint] = None
    verification: Optional[VerificationScore] = None
    fidelity_score: float = 0.0
    grid_snapshot: dict = field(default_factory=dict)
    insights: tuple[str, ...] = ()
    token_usage: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    turns_used: int = 0


@dataclass
class CostConfig:
    """Model and cost configuration."""
    intent_model: str = "claude-haiku-4-5-20251001"
    dialogue_model: str = "claude-sonnet-4-20250514"
    synthesis_model: str = "claude-sonnet-4-20250514"
    verify_model: str = "claude-haiku-4-5-20251001"
    governor_model: str = "claude-haiku-4-5-20251001"
    max_dialogue_rounds: int = 5
    max_total_tokens: int = 200_000
    max_cost_usd: float = 2.0


# ---------------------------------------------------------------------------
# Asymmetric vocabulary filters (structural, not advisory)
# ---------------------------------------------------------------------------

_BEHAVIORAL_VOCAB = frozenset({
    "workflow", "sequence", "pipeline", "trigger", "event",
    "transition", "state machine", "process flow", "step", "orchestration",
    "dispatch", "handler", "callback", "lifecycle", "phase",
    "state change", "flow", "chain", "cascade", "emit",
})

_STRUCTURAL_VOCAB = frozenset({
    "schema", "entity", "attribute", "data model", "type system",
    "class hierarchy", "inheritance", "composition", "record",
    "field", "property", "interface", "struct", "table",
    "column", "foreign key", "primary key", "index", "relation",
})


def _apply_blindness(text: str, vocab: frozenset[str]) -> str:
    """Replace vocabulary terms with [...] — physical removal, not advisory."""
    result = text
    for term in sorted(vocab, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub("[...]", result)
    return result


# ---------------------------------------------------------------------------
# Token normalization + similarity (inlined from kernel/_text_utils.py)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "but", "or", "nor", "not", "no", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "it", "its", "this", "that", "these", "those", "my", "your", "his",
    "her", "our", "their", "which", "who", "whom", "what", "where", "when",
    "them", "itself", "himself", "herself", "themselves", "ourselves",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "than", "too", "very",
    "just", "about", "also", "any", "here", "there", "if", "up",
    "using", "used", "use", "uses",
    "get", "got", "take", "taken", "make", "made", "give", "gave",
    "allow", "allows", "create", "creates", "manage", "manages",
    "handle", "handles", "enable", "enables", "support", "supports",
    "provide", "provides", "require", "requires", "include", "includes",
    "ensure", "ensures", "track", "tracks", "store", "stores",
    "display", "displays", "process", "processes", "send", "sends",
    "show", "shows", "help", "helps", "work", "works", "want", "wants",
    "keep", "keeps", "set", "sets", "add", "adds", "remove", "removes",
    "update", "updates", "delete", "deletes", "check", "checks",
    "find", "finds", "define", "defines", "call", "calls",
    "return", "returns", "change", "changes", "read", "reads",
    "write", "writes", "open", "opens", "close", "closes",
    "load", "loads", "save", "saves",
    "able", "available", "based", "basic", "complete", "current",
    "different", "existing", "external", "full", "general",
    "important", "internal", "main", "multiple", "new", "like",
    "simple", "single", "specific", "standard", "total", "unique",
    "feature", "features", "functionality", "option", "options",
    "tool", "tools", "type", "types", "data", "information",
    "detail", "details", "list", "item", "items", "part", "parts",
    "thing", "things", "kind", "form", "example",
    "comprehensive", "platform", "application", "app", "solution", "product",
})


def _stem_once(word: str) -> str:
    """Single pass of suffix stripping."""
    if len(word) <= 4:
        return word
    for suffix in ("ation", "tion", "sion"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    for suffix in ("ment", "ness", "able", "ible", "ful", "ing", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    for suffix in ("ed", "er", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("s") and len(word) == 4:
        return word[:-1]
    return word


def _stem(word: str) -> str:
    """Iterative suffix stripping (up to 3 passes)."""
    prev = word
    for _ in range(3):
        stemmed = _stem_once(prev)
        if stemmed == prev:
            break
        prev = stemmed
    return prev


def _normalize_word(word: str) -> str:
    return word.lower().strip(".,;:!?\"'()[]{}").strip()


def _split_camel(word: str) -> list[str]:
    """Split PascalCase/camelCase into constituent words."""
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return parts.split()


# Synonym clusters for software domain
_SYNONYM_CLUSTERS: dict[str, frozenset[str]] = {
    "auth": frozenset({"authentication", "login", "credential", "signin", "authenticate"}),
    "user": frozenset({"account", "profile", "member", "participant", "operator"}),
    "data": frozenset({"database", "storage", "store", "persist", "repository", "datastore"}),
    "api": frozenset({"endpoint", "route", "interface", "service", "gateway"}),
    "error": frozenset({"exception", "failure", "fault", "crash", "bug"}),
    "config": frozenset({"configuration", "setting", "preference", "parameter"}),
    "msg": frozenset({"message", "notification", "alert", "signal", "event"}),
    "perm": frozenset({"permission", "authorization", "access", "privilege", "role"}),
    "valid": frozenset({"validation", "verify", "check", "assert", "confirm", "validate"}),
    "cache": frozenset({"memoize", "buffer", "preload", "prefetch"}),
    "log": frozenset({"logging", "audit", "trace", "monitor", "record"}),
    "task": frozenset({"job", "work", "assignment", "ticket", "todo"}),
    "search": frozenset({"query", "lookup", "find", "filter", "browse"}),
    "file": frozenset({"document", "upload", "attachment", "asset", "artifact"}),
    "ui": frozenset({"interface", "frontend", "display", "view", "screen", "widget"}),
    "state": frozenset({"status", "condition", "phase", "mode", "stage"}),
    "compute": frozenset({"calculate", "process", "transform", "convert", "derive"}),
    "connect": frozenset({"link", "associate", "bind", "attach", "wire"}),
    "schedule": frozenset({"timer", "cron", "interval", "periodic", "recurring"}),
    "pay": frozenset({"payment", "billing", "charge", "invoice", "transaction"}),
}

# Build reverse index: stemmed word → representative
_SYNONYM_INDEX: dict[str, str] = {}
for _rep, _cluster in _SYNONYM_CLUSTERS.items():
    for _word in _cluster:
        _SYNONYM_INDEX[_stem(_normalize_word(_word))] = _rep
    _SYNONYM_INDEX[_rep] = _rep


def _normalize_tokens(text: str) -> set[str]:
    """Tokenize, normalize, and stem text for comparison."""
    words = re.split(r"[\s\-_/]+", text)
    tokens: set[str] = set()
    for w in words:
        sub_words = _split_camel(w) if any(c.isupper() for c in w[1:]) else [w]
        for sw in sub_words:
            normalized = _normalize_word(sw)
            if not normalized or len(normalized) < 2:
                continue
            stemmed = _stem(normalized)
            if normalized not in _STOP_WORDS or stemmed in _SYNONYM_INDEX:
                tokens.add(stemmed)
    return tokens


def _expand_synonyms(tokens: set[str]) -> set[str]:
    """Expand tokens with synonym representatives."""
    expanded = set(tokens)
    for token in tokens:
        rep = _SYNONYM_INDEX.get(token)
        if rep:
            expanded.add(rep)
    return expanded


def _bigram_tokens(text: str) -> set[str]:
    """Extract stemmed bigrams from text."""
    words = text.split()
    normalized = []
    for w in words:
        n = _normalize_word(w)
        if n and len(n) >= 2 and n not in _STOP_WORDS:
            normalized.append(_stem(n))
    bigrams = set()
    for i in range(len(normalized) - 1):
        bigrams.add(f"{normalized[i]}_{normalized[i + 1]}")
    return bigrams


def _semantic_similarity(original: str, reconstructed: str) -> float:
    """Token-based semantic similarity. Same algorithm as kernel/closed_loop.py."""
    orig_tokens = _normalize_tokens(original)
    recon_tokens = _normalize_tokens(reconstructed)
    if not orig_tokens or not recon_tokens:
        return 0.0
    orig_expanded = _expand_synonyms(orig_tokens)
    recon_expanded = _expand_synonyms(recon_tokens)
    intersection = orig_expanded & recon_expanded
    union = orig_expanded | recon_expanded
    jaccard = len(intersection) / len(union) if union else 0.0
    recall = len(intersection) / len(orig_expanded) if orig_expanded else 0.0
    containment = recall  # same signal
    # Bigram bonus
    orig_bi = _bigram_tokens(original)
    recon_bi = _bigram_tokens(reconstructed)
    bi_overlap = orig_bi & recon_bi
    bi_recall = len(bi_overlap) / len(orig_bi) if orig_bi else 0.0
    bigram_bonus = min(0.1, bi_recall * 0.1)
    score = 0.5 * containment + 0.3 * recall + 0.1 * jaccard + bigram_bonus
    return round(min(1.0, score), 4)


def _detect_compression_losses(
    original: str, reconstructed: str, blueprint: Blueprint,
) -> list[str]:
    """Categorize what was lost between original intent and reconstruction."""
    orig = _normalize_tokens(original)
    recon = _normalize_tokens(reconstructed)
    lost = orig - recon
    if not lost:
        return []
    # Flatten blueprint to tokens
    bp_parts = []
    for c in blueprint.components:
        bp_parts.extend([c.name, c.description, c.component_type])
        bp_parts.extend(c.attributes)
        bp_parts.extend(c.methods)
    for r in blueprint.relationships:
        bp_parts.extend([r.source, r.target, r.rel_type, r.description])
    bp_parts.extend(blueprint.constraints)
    bp_parts.extend(blueprint.insights)
    bp_text = " ".join(bp_parts)
    bp_tokens = _normalize_tokens(bp_text)

    not_in_bp = lost - bp_tokens
    losses = []
    for token in sorted(not_in_bp):
        if len(token) >= 4:
            losses.append(f"entity: '{token}' absent from blueprint")
    in_bp_not_recon = lost & bp_tokens
    if in_bp_not_recon:
        losses.append(
            f"context: {len(in_bp_not_recon)} terms in blueprint but not reconstructed"
        )
    return losses


# ---------------------------------------------------------------------------
# SimpleGrid
# ---------------------------------------------------------------------------

class SimpleGrid:
    """Dict-based semantic grid. Keys are ENT.{name} or BHV.{name}."""

    def __init__(self) -> None:
        self._cells: dict[str, GridCell] = {}
        self._insights: list[str] = []

    def seed_from_intent(self, intent: Intent) -> None:
        """Bootstrap empty cells from actors and detected nouns/verbs."""
        # Actors become entity cells
        for actor in intent.actors:
            key = f"ENT.{actor}"
            if key not in self._cells:
                self._cells[key] = GridCell(postcode=key)
        # Extract nouns from core_need for additional ENT cells
        words = re.split(r"[\s,]+", intent.core_need)
        for w in words:
            cleaned = re.sub(r"[^a-zA-Z]", "", w).strip()
            if cleaned and len(cleaned) > 3 and cleaned.lower() not in _STOP_WORDS:
                key = f"ENT.{cleaned}"
                if key not in self._cells:
                    self._cells[key] = GridCell(postcode=key)
        # Seed behavioral cells from domain and constraints
        key = f"BHV.{intent.domain or 'core'}"
        if key not in self._cells:
            self._cells[key] = GridCell(postcode=key)
        for constraint in intent.constraints:
            slug = re.sub(r"[^a-zA-Z0-9]", "_", constraint)[:30].strip("_")
            if slug:
                key = f"BHV.{slug}"
                if key not in self._cells:
                    self._cells[key] = GridCell(postcode=key)

    def fill(
        self,
        postcode: str,
        content: str,
        source: str,
        confidence: float,
        connections: tuple[str, ...] = (),
    ) -> None:
        """Fill or update a cell."""
        state = "filled" if confidence >= 0.6 else "partial"
        self._cells[postcode] = GridCell(
            postcode=postcode,
            content=content,
            fill_state=state,
            source=source,
            confidence=confidence,
            connections=connections,
        )

    def unfilled_cells(self) -> list[str]:
        """Return postcodes of empty or partial cells, sorted."""
        return sorted(
            pc for pc, cell in self._cells.items()
            if cell.fill_state in ("empty", "partial")
        )

    def is_converged(self, min_coverage: float = 0.75) -> bool:
        """True if >= min_coverage of cells filled with confidence >= 0.6."""
        return self.coverage_score() >= min_coverage

    def coverage_score(self) -> float:
        """Fraction of cells that are filled with confidence >= 0.6."""
        if not self._cells:
            return 0.0
        filled = sum(
            1 for c in self._cells.values()
            if c.fill_state == "filled" and c.confidence >= 0.6
        )
        return filled / len(self._cells)

    def structural_cells(self) -> dict[str, GridCell]:
        """ENT.* cells."""
        return {k: v for k, v in self._cells.items() if k.startswith("ENT.")}

    def behavioral_cells(self) -> dict[str, GridCell]:
        """BHV.* cells."""
        return {k: v for k, v in self._cells.items() if k.startswith("BHV.")}

    def all_insights(self) -> tuple[str, ...]:
        """Collected insights."""
        return tuple(self._insights)

    def add_insight(self, insight: str) -> None:
        if insight and insight not in self._insights:
            self._insights.append(insight)

    def snapshot(self) -> dict[str, dict]:
        """Frozen dict for inclusion in CompilerResult."""
        return {
            pc: {
                "content": cell.content,
                "fill_state": cell.fill_state,
                "source": cell.source,
                "confidence": cell.confidence,
                "connections": list(cell.connections),
            }
            for pc, cell in sorted(self._cells.items())
        }

    def cell_count(self) -> int:
        return len(self._cells)

    def get(self, postcode: str) -> Optional[GridCell]:
        return self._cells.get(postcode)


# ---------------------------------------------------------------------------
# Tool schemas (Claude tool_use format)
# ---------------------------------------------------------------------------

TOOL_EXTRACT_INTENT = {
    "name": "extract_intent",
    "description": "Extract structured intent from natural language input.",
    "input_schema": {
        "type": "object",
        "properties": {
            "core_need": {"type": "string", "description": "The primary goal in one sentence"},
            "domain": {"type": "string", "description": "The domain this belongs to"},
            "actors": {"type": "array", "items": {"type": "string"}, "description": "Named entities/actors"},
            "constraints": {"type": "array", "items": {"type": "string"}, "description": "Explicit or inferred constraints"},
            "implicit_goals": {"type": "array", "items": {"type": "string"}, "description": "Goals not stated but required"},
            "insight": {"type": "string", "description": "One non-obvious observation about this intent"},
        },
        "required": ["core_need", "domain", "actors", "constraints", "implicit_goals", "insight"],
    },
}

TOOL_FILL_STRUCTURAL = {
    "name": "fill_structural_cell",
    "description": "Fill a structural (ENT.*) cell with entity description, attributes, and relationships.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cell_name": {"type": "string", "description": "Entity name (without ENT. prefix)"},
            "description": {"type": "string"},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "relationships": {"type": "array", "items": {"type": "string"}, "description": "Connections to other cells"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "derived_from": {"type": "string", "description": "What in the input this traces to"},
            "insight": {"type": "string", "description": "Non-obvious structural observation"},
        },
        "required": ["cell_name", "description", "attributes", "confidence", "derived_from"],
    },
}

TOOL_FILL_BEHAVIORAL = {
    "name": "fill_behavioral_cell",
    "description": "Fill a behavioral (BHV.*) cell with process description, steps, and state transitions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cell_name": {"type": "string", "description": "Process name (without BHV. prefix)"},
            "description": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "triggers": {"type": "array", "items": {"type": "string"}},
            "state_transitions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "derived_from": {"type": "string", "description": "What in the input this traces to"},
            "insight": {"type": "string", "description": "Non-obvious behavioral observation"},
        },
        "required": ["cell_name", "description", "steps", "confidence", "derived_from"],
    },
}

TOOL_CHALLENGE = {
    "name": "challenge",
    "description": "Challenge another agent's cell as missing, incomplete, or contradictory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target_cell": {"type": "string", "description": "Postcode of cell being challenged"},
            "challenge_type": {"type": "string", "enum": ["missing", "incomplete", "contradicts"]},
            "argument": {"type": "string", "description": "Why this cell is problematic from your lens"},
            "proposed_addition": {"type": "string", "description": "What should be added or changed"},
        },
        "required": ["target_cell", "challenge_type", "argument"],
    },
}

TOOL_SYNTHESIZE_COMPONENT = {
    "name": "synthesize_component",
    "description": "Emit a unified component from merged structural + behavioral data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "component_type": {"type": "string", "description": "e.g., service, entity, controller, store"},
            "description": {"type": "string"},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "methods": {"type": "array", "items": {"type": "string"}},
            "derived_from": {"type": "string", "description": "Grid cells this traces to"},
        },
        "required": ["name", "component_type", "description", "attributes", "methods", "derived_from"],
    },
}

TOOL_SYNTHESIZE_RELATIONSHIP = {
    "name": "synthesize_relationship",
    "description": "Emit a relationship between two components.",
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "rel_type": {"type": "string", "description": "e.g., depends_on, triggers, contains, reads_from"},
            "description": {"type": "string"},
            "derived_from": {"type": "string"},
        },
        "required": ["source", "target", "rel_type", "derived_from"],
    },
}

TOOL_VERIFY_DIMENSION = {
    "name": "verify_dimension",
    "description": "Score one verification dimension of the blueprint.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dimension": {"type": "string", "enum": ["completeness", "consistency", "coherence", "traceability"]},
            "score": {"type": "number", "minimum": 0, "maximum": 100},
            "evidence": {"type": "string", "description": "Specific evidence for this score"},
            "gaps": {"type": "array", "items": {"type": "string"}, "description": "Identified gaps"},
        },
        "required": ["dimension", "score", "evidence", "gaps"],
    },
}

TOOL_DECODE_INTENT = {
    "name": "decode_intent",
    "description": "Reconstruct the original intent from the blueprint alone (closed-loop gate).",
    "input_schema": {
        "type": "object",
        "properties": {
            "reconstructed_intent": {"type": "string", "description": "What was the user trying to build? Reconstruct from blueprint only."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "compression_losses": {"type": "array", "items": {"type": "string"}, "description": "What information was lost"},
            "verdict": {"type": "string", "enum": ["pass", "fail", "marginal"]},
        },
        "required": ["reconstructed_intent", "confidence", "compression_losses", "verdict"],
    },
}

ALL_TOOLS = [
    TOOL_EXTRACT_INTENT, TOOL_FILL_STRUCTURAL, TOOL_FILL_BEHAVIORAL,
    TOOL_CHALLENGE, TOOL_SYNTHESIZE_COMPONENT, TOOL_SYNTHESIZE_RELATIONSHIP,
    TOOL_VERIFY_DIMENSION, TOOL_DECODE_INTENT,
]


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """You extract structured intent from natural language.
Given a user's description, use the extract_intent tool to produce:
- core_need: the primary goal in one clear sentence
- domain: the domain category (e.g., "task-management", "e-commerce", "communication")
- actors: named entities, users, or systems mentioned
- constraints: explicit or strongly implied constraints
- implicit_goals: things NOT stated but REQUIRED for the system to work (auth, validation, error handling, etc.)
- insight: one non-obvious observation about what this system actually needs

Extract only what the input entails. Never invent."""

_ENTITY_SYSTEM = """You are a STRUCTURAL analyst. Your lens: WHAT EXISTS.
You see: nouns, attributes, data models, relationships between entities, schemas, types.
You CANNOT see temporal flow, processes, or state changes. They do not exist in your universe.

Given the input and current grid state, use fill_structural_cell to describe entities.
For each entity: what it IS, what attributes it HAS, what it connects TO.

You may also use the challenge tool to challenge BHV.* cells — you can see their postcode
and fill_state but NOT their content. Challenge from your structural lens: "this behavioral
cell implies a missing entity" or "this process requires data that no entity provides."

RULES:
- Extract ONLY what the input explicitly names or directly entails
- Never invent entities the input doesn't mention or imply
- Attributes must trace to input text
- One fill_structural_cell call per entity. Be thorough per entity."""

_PROCESS_SYSTEM = """You are a BEHAVIORAL analyst. Your lens: WHAT HAPPENS.
You see: verbs, workflows, state transitions, triggers, event flows, sequences.
You CANNOT see static structure, entities, or attributes. They do not exist in your universe.

Given the input and current grid state, use fill_behavioral_cell to describe processes.
For each process: what HAPPENS, in what ORDER, triggered by WHAT.

You may also use the challenge tool to challenge ENT.* cells — you can see their postcode
and fill_state but NOT their content. Challenge from your behavioral lens: "this entity
implies a missing process" or "nothing triggers the creation of this entity."

RULES:
- Extract ONLY what the input explicitly describes or directly entails
- Never invent processes the input doesn't mention or imply
- Steps must trace to input text
- One fill_behavioral_cell call per process. Be thorough per process."""

_SYNTHESIS_SYSTEM = """You are a SYNTHESIS agent. You merge structural and behavioral grid cells
into unified components and relationships.

Given the full grid state (all ENT.* and BHV.* cells), produce:
1. Components — use synthesize_component for each. Merge entity attributes with behavioral methods.
2. Relationships — use synthesize_relationship for each connection between components.

Every component must trace to at least one grid cell (derived_from field).
Every relationship must trace to grid content.
Do NOT invent components or relationships not supported by the grid."""

_VERIFY_SYSTEM = """You are a VERIFICATION agent. Score the blueprint on 4 dimensions.
Use verify_dimension once for each:

1. completeness — Does the blueprint cover all aspects of the original intent? (0-100)
2. consistency — Are there contradictions between components? (0-100)
3. coherence — Does the system make architectural sense as a whole? (0-100)
4. traceability — Can every component trace back to the input? (0-100)

Be specific in evidence and gaps. Do not inflate scores."""

_GOVERNOR_SYSTEM = """You are a GOVERNOR. You perform the closed-loop gate.
You receive ONLY the blueprint — you have NEVER seen the grid or the original input.

Your job: from the blueprint alone, reconstruct what the user was trying to build.
Use decode_intent to emit:
- reconstructed_intent: your best reconstruction of the original request
- confidence: how confident you are (0-1)
- compression_losses: what information seems missing from the blueprint
- verdict: pass/fail/marginal

Be honest. If the blueprint is vague, say so."""


# ---------------------------------------------------------------------------
# Token tracker
# ---------------------------------------------------------------------------

# Pricing per million tokens (as of 2025)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
}


class _TokenTracker:
    """Accumulates token usage and estimates cost."""

    def __init__(self, max_tokens: int, max_cost: float) -> None:
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0

    def record(self, usage: dict, model: str) -> None:
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        self.input_tokens += inp
        self.output_tokens += out
        self.calls += 1
        rates = _MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (inp * rates["input"] + out * rates["output"]) / 1_000_000
        self.total_cost += cost
        total_tokens = self.input_tokens + self.output_tokens
        if total_tokens > self.max_tokens:
            raise RuntimeError(
                f"Token cap exceeded: {total_tokens} > {self.max_tokens}"
            )
        if self.total_cost > self.max_cost:
            raise RuntimeError(
                f"Cost cap exceeded: ${self.total_cost:.4f} > ${self.max_cost:.2f}"
            )

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "calls": self.calls,
        }


# ---------------------------------------------------------------------------
# Agent call wrapper
# ---------------------------------------------------------------------------

def _call_agent(
    client: Any,
    model: str,
    system: str,
    user_msg: str,
    tools: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> tuple[list[dict], str, dict]:
    """Call Claude with tools. Returns (tool_calls, text, usage).

    Each tool_call is {"name": str, "input": dict}.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        tools=tools,
    )
    tool_calls = []
    text_parts = []
    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({"name": block.name, "input": block.input})
        elif block.type == "text":
            text_parts.append(block.text)
    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
    return tool_calls, "\n".join(text_parts), usage


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_intent_agent(
    client: Any, input_text: str, config: CostConfig, tracker: _TokenTracker,
) -> Intent:
    """Phase 1: Extract structured intent."""
    tools, text, usage = _call_agent(
        client, config.intent_model, _INTENT_SYSTEM,
        f"Extract the structured intent from this description:\n\n{input_text}",
        [TOOL_EXTRACT_INTENT],
    )
    tracker.record(usage, config.intent_model)
    for tc in tools:
        if tc["name"] == "extract_intent":
            inp = tc["input"]
            return Intent(
                core_need=inp.get("core_need", ""),
                domain=inp.get("domain", ""),
                actors=tuple(inp.get("actors", ())),
                constraints=tuple(inp.get("constraints", ())),
                implicit_goals=tuple(inp.get("implicit_goals", ())),
                insight=inp.get("insight", ""),
            )
    raise RuntimeError("Intent agent did not call extract_intent tool")


def _build_grid_context(grid: SimpleGrid, axis: str) -> str:
    """Format grid state for an agent. axis='structural' or 'behavioral'."""
    lines = ["Current grid state:"]
    for pc, cell in sorted(grid._cells.items()):
        if axis == "structural" and pc.startswith("ENT."):
            lines.append(f"  {pc}: {cell.fill_state} (conf={cell.confidence})")
            if cell.content:
                lines.append(f"    content: {cell.content[:200]}")
        elif axis == "behavioral" and pc.startswith("BHV."):
            lines.append(f"  {pc}: {cell.fill_state} (conf={cell.confidence})")
            if cell.content:
                lines.append(f"    content: {cell.content[:200]}")
        else:
            # Other axis — show only postcode + fill_state (blind)
            lines.append(f"  {pc}: {cell.fill_state}")
    unfilled = grid.unfilled_cells()
    if unfilled:
        lines.append(f"\nUnfilled cells: {', '.join(unfilled)}")
    return "\n".join(lines)


def _apply_structural_fills(grid: SimpleGrid, tool_calls: list[dict]) -> list[dict]:
    """Apply fill_structural_cell calls, collect challenges. Returns challenges."""
    challenges = []
    for tc in tool_calls:
        if tc["name"] == "fill_structural_cell":
            inp = tc["input"]
            name = inp.get("cell_name", "")
            if not name:
                continue
            postcode = f"ENT.{name}" if not name.startswith("ENT.") else name
            desc = inp.get("description", "")
            attrs = inp.get("attributes", [])
            rels = inp.get("relationships", [])
            conf = float(inp.get("confidence", 0.5))
            derived = inp.get("derived_from", "")
            content = f"{desc} | attrs: {', '.join(attrs)}" if attrs else desc
            connections = tuple(rels) if rels else ()
            grid.fill(postcode, content, "entity_agent", conf, connections)
            insight = inp.get("insight", "")
            if insight:
                grid.add_insight(insight)
        elif tc["name"] == "challenge":
            challenges.append(tc["input"])
    return challenges


def _apply_behavioral_fills(grid: SimpleGrid, tool_calls: list[dict]) -> list[dict]:
    """Apply fill_behavioral_cell calls, collect challenges. Returns challenges."""
    challenges = []
    for tc in tool_calls:
        if tc["name"] == "fill_behavioral_cell":
            inp = tc["input"]
            name = inp.get("cell_name", "")
            if not name:
                continue
            postcode = f"BHV.{name}" if not name.startswith("BHV.") else name
            desc = inp.get("description", "")
            steps = inp.get("steps", [])
            triggers = inp.get("triggers", [])
            transitions = inp.get("state_transitions", [])
            conf = float(inp.get("confidence", 0.5))
            derived = inp.get("derived_from", "")
            parts = [desc]
            if steps:
                parts.append(f"steps: {' -> '.join(steps)}")
            if triggers:
                parts.append(f"triggers: {', '.join(triggers)}")
            if transitions:
                parts.append(f"transitions: {', '.join(transitions)}")
            content = " | ".join(parts)
            grid.fill(postcode, content, "process_agent", conf)
            insight = inp.get("insight", "")
            if insight:
                grid.add_insight(insight)
        elif tc["name"] == "challenge":
            challenges.append(tc["input"])
    return challenges


def _format_challenges(challenges: list[dict]) -> str:
    """Format pending challenges for injection into agent prompt."""
    if not challenges:
        return ""
    lines = ["\nPending challenges from the opposing agent:"]
    for ch in challenges:
        target = ch.get("target_cell", "?")
        ctype = ch.get("challenge_type", "?")
        arg = ch.get("argument", "")
        proposed = ch.get("proposed_addition", "")
        lines.append(f"  - {ctype} on {target}: {arg}")
        if proposed:
            lines.append(f"    proposed: {proposed}")
    return "\n".join(lines)


def _run_dialogue(
    client: Any,
    input_text: str,
    intent: Intent,
    grid: SimpleGrid,
    config: CostConfig,
    tracker: _TokenTracker,
) -> int:
    """Phase 3: Run entity/process dialogue. Returns turns used."""
    entity_tools = [TOOL_FILL_STRUCTURAL, TOOL_CHALLENGE]
    process_tools = [TOOL_FILL_BEHAVIORAL, TOOL_CHALLENGE]

    entity_input = _apply_blindness(input_text, _BEHAVIORAL_VOCAB)
    process_input = _apply_blindness(input_text, _STRUCTURAL_VOCAB)

    entity_challenges: list[dict] = []
    process_challenges: list[dict] = []
    prev_coverage = 0.0
    stall_count = 0
    turns = 0

    for round_idx in range(config.max_dialogue_rounds):
        # --- Entity Agent ---
        entity_context = _build_grid_context(grid, "structural")
        challenge_block = _format_challenges(process_challenges)
        entity_msg = (
            f"Input (structural lens):\n{entity_input}\n\n"
            f"Intent: {intent.core_need}\n"
            f"Domain: {intent.domain}\n"
            f"Actors: {', '.join(intent.actors)}\n\n"
            f"{entity_context}{challenge_block}\n\n"
            f"Fill unfilled ENT.* cells. You may create new ENT.* cells for entities "
            f"you discover. You may challenge BHV.* cells."
        )
        tools, _, usage = _call_agent(
            client, config.dialogue_model, _ENTITY_SYSTEM,
            entity_msg, entity_tools,
        )
        tracker.record(usage, config.dialogue_model)
        entity_challenges = _apply_structural_fills(grid, tools)
        turns += 1

        # --- Process Agent ---
        process_context = _build_grid_context(grid, "behavioral")
        challenge_block = _format_challenges(entity_challenges)
        process_msg = (
            f"Input (behavioral lens):\n{process_input}\n\n"
            f"Intent: {intent.core_need}\n"
            f"Domain: {intent.domain}\n\n"
            f"{process_context}{challenge_block}\n\n"
            f"Fill unfilled BHV.* cells. You may create new BHV.* cells for processes "
            f"you discover. You may challenge ENT.* cells."
        )
        tools, _, usage = _call_agent(
            client, config.dialogue_model, _PROCESS_SYSTEM,
            process_msg, process_tools,
        )
        tracker.record(usage, config.dialogue_model)
        process_challenges = _apply_behavioral_fills(grid, tools)
        turns += 1

        # --- Convergence check ---
        coverage = grid.coverage_score()
        logger.info(
            "Round %d: coverage=%.2f, entity_challenges=%d, process_challenges=%d",
            round_idx + 1, coverage, len(entity_challenges), len(process_challenges),
        )

        # Early exit: converged and no challenges
        if grid.is_converged() and not entity_challenges and not process_challenges:
            logger.info("Grid converged at round %d", round_idx + 1)
            break

        # Stall detection
        if abs(coverage - prev_coverage) < 0.01:
            stall_count += 1
            if stall_count >= 2:
                logger.info("Stall detected after round %d", round_idx + 1)
                break
        else:
            stall_count = 0
        prev_coverage = coverage

    return turns


def _run_synthesis(
    client: Any,
    input_text: str,
    intent: Intent,
    grid: SimpleGrid,
    config: CostConfig,
    tracker: _TokenTracker,
) -> Blueprint:
    """Phase 4: Merge grid into Blueprint via synthesis agent."""
    synthesis_tools = [TOOL_SYNTHESIZE_COMPONENT, TOOL_SYNTHESIZE_RELATIONSHIP]

    # Format full grid for synthesis
    grid_lines = ["Full grid state:"]
    for pc, cell in sorted(grid._cells.items()):
        grid_lines.append(f"\n{pc} [{cell.fill_state}, conf={cell.confidence}]:")
        if cell.content:
            grid_lines.append(f"  {cell.content}")
        if cell.connections:
            grid_lines.append(f"  connections: {', '.join(cell.connections)}")
    grid_text = "\n".join(grid_lines)

    synthesis_msg = (
        f"Original input: {input_text}\n\n"
        f"Intent: {intent.core_need}\n"
        f"Domain: {intent.domain}\n\n"
        f"{grid_text}\n\n"
        f"Synthesize unified components and relationships from this grid. "
        f"Use synthesize_component for each component and synthesize_relationship "
        f"for each connection."
    )

    components: list[Component] = []
    relationships: list[Relationship] = []

    # Multi-turn: Claude may not emit everything in one call
    for attempt in range(3):
        tools, text, usage = _call_agent(
            client, config.synthesis_model, _SYNTHESIS_SYSTEM,
            synthesis_msg, synthesis_tools,
        )
        tracker.record(usage, config.synthesis_model)

        for tc in tools:
            if tc["name"] == "synthesize_component":
                inp = tc["input"]
                components.append(Component(
                    name=inp.get("name", ""),
                    component_type=inp.get("component_type", ""),
                    description=inp.get("description", ""),
                    attributes=tuple(inp.get("attributes", ())),
                    methods=tuple(inp.get("methods", ())),
                    derived_from=inp.get("derived_from", ""),
                ))
            elif tc["name"] == "synthesize_relationship":
                inp = tc["input"]
                relationships.append(Relationship(
                    source=inp.get("source", ""),
                    target=inp.get("target", ""),
                    rel_type=inp.get("rel_type", ""),
                    description=inp.get("description", ""),
                    derived_from=inp.get("derived_from", ""),
                ))

        # Check if we got reasonable output
        if components:
            break
        # Retry with nudge
        synthesis_msg = (
            f"{synthesis_msg}\n\nYou must use the synthesize_component and "
            f"synthesize_relationship tools. Emit at least one component."
        )

    return Blueprint(
        components=tuple(components),
        relationships=tuple(relationships),
        constraints=intent.constraints,
        insights=grid.all_insights(),
        unresolved=tuple(grid.unfilled_cells()),
    )


def _run_verification(
    client: Any,
    intent: Intent,
    blueprint: Blueprint,
    config: CostConfig,
    tracker: _TokenTracker,
) -> VerificationScore:
    """Phase 5: Score blueprint on 4 dimensions."""
    # Format blueprint for verifier
    bp_lines = [f"Intent: {intent.core_need}", f"Domain: {intent.domain}", ""]
    bp_lines.append(f"Components ({len(blueprint.components)}):")
    for c in blueprint.components:
        bp_lines.append(f"  {c.name} ({c.component_type}): {c.description}")
        if c.attributes:
            bp_lines.append(f"    attributes: {', '.join(c.attributes)}")
        if c.methods:
            bp_lines.append(f"    methods: {', '.join(c.methods)}")
    bp_lines.append(f"\nRelationships ({len(blueprint.relationships)}):")
    for r in blueprint.relationships:
        bp_lines.append(f"  {r.source} -> {r.target} ({r.rel_type}): {r.description}")
    bp_lines.append(f"\nConstraints: {', '.join(blueprint.constraints)}")
    bp_lines.append(f"Insights: {', '.join(blueprint.insights)}")
    bp_text = "\n".join(bp_lines)

    verify_msg = (
        f"Verify this blueprint:\n\n{bp_text}\n\n"
        f"Call verify_dimension once for each of the 4 dimensions: "
        f"completeness, consistency, coherence, traceability."
    )

    tools, _, usage = _call_agent(
        client, config.verify_model, _VERIFY_SYSTEM,
        verify_msg, [TOOL_VERIFY_DIMENSION],
    )
    tracker.record(usage, config.verify_model)

    scores: dict[str, float] = {}
    all_gaps: list[str] = []
    for tc in tools:
        if tc["name"] == "verify_dimension":
            inp = tc["input"]
            dim = inp.get("dimension", "")
            score = float(inp.get("score", 0))
            scores[dim] = score
            gaps = inp.get("gaps", [])
            all_gaps.extend(gaps)

    completeness = scores.get("completeness", 0)
    consistency = scores.get("consistency", 0)
    coherence = scores.get("coherence", 0)
    traceability = scores.get("traceability", 0)
    overall = (completeness + consistency + coherence + traceability) / 4

    recommendation = "pass" if overall >= 70 else "needs improvement"

    return VerificationScore(
        completeness=completeness,
        consistency=consistency,
        coherence=coherence,
        traceability=traceability,
        overall=overall,
        gaps=tuple(all_gaps),
        recommendation=recommendation,
    )


def _run_governor_gate(
    client: Any,
    input_text: str,
    blueprint: Blueprint,
    config: CostConfig,
    tracker: _TokenTracker,
) -> tuple[float, str, list[str]]:
    """Phase 6: Closed-loop gate. Returns (fidelity, reconstructed, losses)."""
    # Format blueprint for governor — NO grid, NO original input
    bp_lines = [f"Components ({len(blueprint.components)}):"]
    for c in blueprint.components:
        bp_lines.append(f"  {c.name} ({c.component_type}): {c.description}")
        if c.attributes:
            bp_lines.append(f"    attributes: {', '.join(c.attributes)}")
        if c.methods:
            bp_lines.append(f"    methods: {', '.join(c.methods)}")
    bp_lines.append(f"\nRelationships ({len(blueprint.relationships)}):")
    for r in blueprint.relationships:
        bp_lines.append(f"  {r.source} -> {r.target} ({r.rel_type})")
    bp_lines.append(f"\nConstraints: {', '.join(blueprint.constraints)}")
    bp_text = "\n".join(bp_lines)

    governor_msg = (
        f"You see ONLY this blueprint. You have never seen the original request.\n\n"
        f"{bp_text}\n\n"
        f"Reconstruct what the user was trying to build. Use decode_intent."
    )

    tools, _, usage = _call_agent(
        client, config.governor_model, _GOVERNOR_SYSTEM,
        governor_msg, [TOOL_DECODE_INTENT],
    )
    tracker.record(usage, config.governor_model)

    reconstructed = ""
    llm_losses: list[str] = []
    for tc in tools:
        if tc["name"] == "decode_intent":
            inp = tc["input"]
            reconstructed = inp.get("reconstructed_intent", "")
            llm_losses = inp.get("compression_losses", [])

    # Deterministic fidelity computation
    fidelity = _semantic_similarity(input_text, reconstructed)
    det_losses = _detect_compression_losses(input_text, reconstructed, blueprint)
    all_losses = det_losses + [l for l in llm_losses if l not in det_losses]

    return fidelity, reconstructed, all_losses


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compile(
    input_text: str,
    config: Optional[CostConfig] = None,
    client: Any = None,
) -> CompilerResult:
    """Compile natural language into verified structure.

    Requires ANTHROPIC_API_KEY in environment (or pass client directly).
    """
    if not input_text or not input_text.strip():
        raise ValueError("Input text must be non-empty")

    if config is None:
        config = CostConfig()

    if client is None:
        # Import anthropic here to keep the module importable without the SDK installed
        import anthropic
        client = anthropic.Anthropic()

    tracker = _TokenTracker(config.max_total_tokens, config.max_cost_usd)
    start_time = time.time()
    total_turns = 0

    # Phase 1: Intent extraction
    logger.info("Phase 1: Extracting intent...")
    intent = _run_intent_agent(client, input_text, config, tracker)
    logger.info("Intent: %s (domain=%s, actors=%s)", intent.core_need, intent.domain, intent.actors)

    # Phase 2: Grid bootstrap
    logger.info("Phase 2: Bootstrapping grid...")
    grid = SimpleGrid()
    grid.seed_from_intent(intent)
    logger.info("Grid seeded with %d cells", grid.cell_count())

    # Phase 3: Dialogue
    logger.info("Phase 3: Running dialogue (max %d rounds)...", config.max_dialogue_rounds)
    turns = _run_dialogue(client, input_text, intent, grid, config, tracker)
    total_turns += turns
    logger.info("Dialogue complete: %d turns, coverage=%.2f", turns, grid.coverage_score())

    # Phase 4: Synthesis
    logger.info("Phase 4: Synthesizing blueprint...")
    blueprint = _run_synthesis(client, input_text, intent, grid, config, tracker)
    logger.info("Blueprint: %d components, %d relationships",
                len(blueprint.components), len(blueprint.relationships))

    # Phase 5: Verification
    logger.info("Phase 5: Verifying blueprint...")
    verification = _run_verification(client, intent, blueprint, config, tracker)
    logger.info("Verification: overall=%.1f (%s)", verification.overall, verification.recommendation)

    # Phase 6: Governor gate
    logger.info("Phase 6: Running governor gate...")
    fidelity, reconstructed, losses = _run_governor_gate(
        client, input_text, blueprint, config, tracker,
    )
    logger.info("Fidelity: %.4f (threshold=0.60), losses=%d", fidelity, len(losses))

    duration = time.time() - start_time

    # Collect all insights
    all_insights = list(grid.all_insights())
    if intent.insight:
        all_insights.insert(0, intent.insight)

    return CompilerResult(
        intent=intent,
        blueprint=blueprint,
        verification=verification,
        fidelity_score=fidelity,
        grid_snapshot=grid.snapshot(),
        insights=tuple(all_insights),
        token_usage=tracker.summary(),
        duration_seconds=round(duration, 2),
        turns_used=total_turns,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _format_result(result: CompilerResult) -> str:
    """Format CompilerResult for CLI output."""
    lines = []
    lines.append("=" * 60)
    lines.append("COMPILATION RESULT")
    lines.append("=" * 60)

    if result.intent:
        lines.append(f"\nIntent: {result.intent.core_need}")
        lines.append(f"Domain: {result.intent.domain}")
        lines.append(f"Actors: {', '.join(result.intent.actors)}")

    if result.blueprint:
        lines.append(f"\nComponents ({len(result.blueprint.components)}):")
        for c in result.blueprint.components:
            lines.append(f"  {c.name} ({c.component_type})")
            lines.append(f"    {c.description}")
            if c.attributes:
                lines.append(f"    attrs: {', '.join(c.attributes)}")
            if c.methods:
                lines.append(f"    methods: {', '.join(c.methods)}")

        lines.append(f"\nRelationships ({len(result.blueprint.relationships)}):")
        for r in result.blueprint.relationships:
            lines.append(f"  {r.source} -> {r.target} ({r.rel_type})")

    if result.verification:
        v = result.verification
        lines.append(f"\nVerification:")
        lines.append(f"  completeness:  {v.completeness:.0f}")
        lines.append(f"  consistency:   {v.consistency:.0f}")
        lines.append(f"  coherence:     {v.coherence:.0f}")
        lines.append(f"  traceability:  {v.traceability:.0f}")
        lines.append(f"  overall:       {v.overall:.1f} ({v.recommendation})")
        if v.gaps:
            lines.append(f"  gaps: {', '.join(v.gaps[:5])}")

    lines.append(f"\nFidelity: {result.fidelity_score:.4f} {'PASS' if result.fidelity_score >= 0.60 else 'FAIL'}")

    if result.insights:
        lines.append(f"\nInsights:")
        for ins in result.insights[:10]:
            lines.append(f"  - {ins}")

    lines.append(f"\nTokens: {result.token_usage}")
    lines.append(f"Duration: {result.duration_seconds}s")
    lines.append(f"Turns: {result.turns_used}")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Standalone semantic compiler using Claude tool-use"
    )
    parser.add_argument("input", help="Natural language description to compile")
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Dialogue/synthesis model (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument("--max-rounds", type=int, default=5, help="Max dialogue rounds")
    parser.add_argument("--max-cost", type=float, default=2.0, help="Max cost in USD")
    args = parser.parse_args()

    cfg = CostConfig(
        dialogue_model=args.model,
        synthesis_model=args.model,
        max_dialogue_rounds=args.max_rounds,
        max_cost_usd=args.max_cost,
    )

    try:
        result = compile(args.input, cfg)
        print(_format_result(result))
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
