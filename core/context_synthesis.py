"""
Context synthesis — extract a structured context map from grid state.

LEAF module. Stdlib only. No LLM calls.

Used by CONTEXT compilation mode. Given filled grid cells and dialogue
messages, extracts concepts, relationships, assumptions, unknowns,
vocabulary, and memory connections into a frozen ContextMap.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass(frozen=True)
class Concept:
    """A concept extracted from the grid."""
    name: str
    description: str
    layer: str          # kernel layer (ORG, SEM, etc.)
    concern: str        # kernel concern axis (ENT, BHV, etc.)
    confidence: float   # from cell confidence
    source_postcode: str


@dataclass(frozen=True)
class Relationship:
    """A relationship between concepts."""
    source: str         # concept name
    target: str         # concept name
    relation_type: str  # "connected", "parent-child", "co-layer"
    strength: float     # 0.0 - 1.0


@dataclass(frozen=True)
class Assumption:
    """An assumption detected in the input or dialogue."""
    text: str
    category: str       # "structural", "behavioral", "domain", "constraint"
    confidence: float   # how confident we are this IS an assumption


@dataclass(frozen=True)
class Unknown:
    """Something not yet understood — gap in the context."""
    question: str
    category: str       # "missing_entity", "unclear_behavior", "unspecified_constraint"
    priority: float     # 0.0 - 1.0


@dataclass(frozen=True)
class ContextMap:
    """Structured context understanding from a CONTEXT compilation."""
    original_intent: str
    concepts: Tuple[Concept, ...]
    relationships: Tuple[Relationship, ...]
    assumptions: Tuple[Assumption, ...]
    unknowns: Tuple[Unknown, ...]
    vocabulary: Tuple[str, ...]         # domain-specific terms extracted
    memory_connections: Tuple[str, ...]  # links to prior knowledge
    confidence: float                    # overall context confidence


def synthesize_context(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
    original_intent: str,
    memory_results: Optional[List[Dict[str, Any]]] = None,
) -> ContextMap:
    """Synthesize a ContextMap from grid state and dialogue.

    Heuristic extraction — no LLM calls. Extracts structure from
    cell metadata and dialogue text patterns.

    Args:
        grid_cells: List of cell dicts with keys: postcode, primitive,
            content, fill, confidence, connections, layer, concern.
        dialogue_messages: List of message dicts with keys: sender, content.
        original_intent: The user's original input text.
        memory_results: Optional prior memory matches.

    Returns:
        Frozen ContextMap with extracted context.
    """
    concepts = _extract_concepts(grid_cells)
    relationships = _extract_relationships(grid_cells, concepts)
    assumptions = _extract_assumptions(dialogue_messages, original_intent)
    unknowns = _extract_unknowns(grid_cells, dialogue_messages)
    vocabulary = _extract_vocabulary(grid_cells, dialogue_messages)
    memory_connections = _extract_memory_connections(memory_results)

    # Overall confidence: weighted average of concept confidences
    if concepts:
        overall_conf = sum(c.confidence for c in concepts) / len(concepts)
    else:
        overall_conf = 0.0

    return ContextMap(
        original_intent=original_intent,
        concepts=tuple(concepts),
        relationships=tuple(relationships),
        assumptions=tuple(assumptions),
        unknowns=tuple(unknowns),
        vocabulary=tuple(vocabulary),
        memory_connections=tuple(memory_connections),
        confidence=round(overall_conf, 3),
    )


def context_map_to_dict(ctx: ContextMap) -> Dict[str, Any]:
    """Serialize a ContextMap to a plain dict (JSON-safe)."""
    return {
        "original_intent": ctx.original_intent,
        "concepts": [
            {
                "name": c.name,
                "description": c.description,
                "layer": c.layer,
                "concern": c.concern,
                "confidence": c.confidence,
                "source_postcode": c.source_postcode,
            }
            for c in ctx.concepts
        ],
        "relationships": [
            {
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "strength": r.strength,
            }
            for r in ctx.relationships
        ],
        "assumptions": [
            {
                "text": a.text,
                "category": a.category,
                "confidence": a.confidence,
            }
            for a in ctx.assumptions
        ],
        "unknowns": [
            {
                "question": u.question,
                "category": u.category,
                "priority": u.priority,
            }
            for u in ctx.unknowns
        ],
        "vocabulary": list(ctx.vocabulary),
        "memory_connections": list(ctx.memory_connections),
        "confidence": ctx.confidence,
    }


def format_context_summary(ctx: ContextMap) -> str:
    """Format a human-readable summary of a ContextMap."""
    lines = []
    lines.append(f"Context Map: {ctx.original_intent[:80]}")
    lines.append(f"Confidence: {ctx.confidence:.0%}")
    lines.append("")

    if ctx.concepts:
        lines.append(f"Concepts ({len(ctx.concepts)}):")
        for c in ctx.concepts[:10]:
            lines.append(f"  - {c.name} [{c.layer}.{c.concern}] ({c.confidence:.0%})")
        if len(ctx.concepts) > 10:
            lines.append(f"  ... and {len(ctx.concepts) - 10} more")
        lines.append("")

    if ctx.relationships:
        lines.append(f"Relationships ({len(ctx.relationships)}):")
        for r in ctx.relationships[:8]:
            lines.append(f"  - {r.source} → {r.target} ({r.relation_type})")
        if len(ctx.relationships) > 8:
            lines.append(f"  ... and {len(ctx.relationships) - 8} more")
        lines.append("")

    if ctx.assumptions:
        lines.append(f"Assumptions ({len(ctx.assumptions)}):")
        for a in ctx.assumptions[:5]:
            lines.append(f"  - [{a.category}] {a.text}")
        lines.append("")

    if ctx.unknowns:
        lines.append(f"Unknowns ({len(ctx.unknowns)}):")
        for u in ctx.unknowns[:5]:
            lines.append(f"  - [{u.category}] {u.question}")
        lines.append("")

    if ctx.vocabulary:
        lines.append(f"Vocabulary: {', '.join(ctx.vocabulary[:15])}")
        lines.append("")

    if ctx.memory_connections:
        lines.append(f"Memory connections: {len(ctx.memory_connections)}")

    return "\n".join(lines)


# ============================================================
# Internal extraction functions
# ============================================================

def _extract_concepts(grid_cells: List[Dict[str, Any]]) -> List[Concept]:
    """Extract concepts from filled grid cells."""
    concepts = []
    seen_names = set()
    for cell in grid_cells:
        fill = cell.get("fill", "E")
        # Only extract from filled or partial cells
        if fill not in ("F", "P", "filled", "partial"):
            continue

        name = cell.get("primitive", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        postcode = cell.get("postcode", "")
        # Parse layer and concern from postcode (format: LAYER.CONCERN.SCOPE.DIM.DOMAIN)
        parts = postcode.split(".") if postcode else []
        layer = parts[0] if len(parts) > 0 else ""
        concern = parts[1] if len(parts) > 1 else ""

        concepts.append(Concept(
            name=name,
            description=cell.get("content", ""),
            layer=layer,
            concern=concern,
            confidence=cell.get("confidence", 0.0),
            source_postcode=postcode,
        ))

    # Sort by confidence descending
    concepts.sort(key=lambda c: c.confidence, reverse=True)
    return concepts


def _extract_relationships(
    grid_cells: List[Dict[str, Any]],
    concepts: List[Concept],
) -> List[Relationship]:
    """Extract relationships from cell connections and parent refs."""
    relationships = []
    seen = set()

    # Index: postcode → concept name
    pc_to_name = {}
    for c in concepts:
        pc_to_name[c.source_postcode] = c.name

    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill not in ("F", "P", "filled", "partial"):
            continue

        src_pc = cell.get("postcode", "")
        src_name = pc_to_name.get(src_pc)
        if not src_name:
            continue

        # Connection-based relationships
        connections = cell.get("connections", ())
        for tgt_pc in connections:
            tgt_name = pc_to_name.get(tgt_pc)
            if tgt_name and tgt_name != src_name:
                key = tuple(sorted([src_name, tgt_name]))
                if key not in seen:
                    seen.add(key)
                    relationships.append(Relationship(
                        source=src_name,
                        target=tgt_name,
                        relation_type="connected",
                        strength=0.7,
                    ))

        # Parent-child relationships
        parent_pc = cell.get("parent")
        if parent_pc:
            parent_name = pc_to_name.get(parent_pc)
            if parent_name and parent_name != src_name:
                key = (parent_name, src_name, "parent-child")
                if key not in seen:
                    seen.add(key)
                    relationships.append(Relationship(
                        source=parent_name,
                        target=src_name,
                        relation_type="parent-child",
                        strength=0.9,
                    ))

    return relationships


# Patterns that indicate assumptions in text
_ASSUMPTION_PATTERNS = [
    (re.compile(r"(?:assum|presuppos|tak(?:e|ing) for granted|given that|provided that)", re.I), "structural"),
    (re.compile(r"(?:should|must|always|never|every|all users|all clients)", re.I), "constraint"),
    (re.compile(r"(?:typically|usually|normally|standard|conventional|common practice)", re.I), "domain"),
    (re.compile(r"(?:will need to|requires|depends on|relies on)", re.I), "behavioral"),
]


def _extract_assumptions(
    dialogue_messages: List[Dict[str, Any]],
    original_intent: str,
) -> List[Assumption]:
    """Extract assumptions from dialogue text and original intent."""
    assumptions = []
    seen_texts = set()

    all_texts = [original_intent] + [
        m.get("content", "") for m in dialogue_messages
    ]

    for text in all_texts:
        if not text:
            continue
        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 200:
                continue
            for pattern, category in _ASSUMPTION_PATTERNS:
                if pattern.search(sentence):
                    # Deduplicate by normalized text
                    norm = sentence.lower().strip()
                    if norm not in seen_texts:
                        seen_texts.add(norm)
                        assumptions.append(Assumption(
                            text=sentence,
                            category=category,
                            confidence=0.6,
                        ))
                    break  # one match per sentence

    # Cap at 20 assumptions
    return assumptions[:20]


def _extract_unknowns(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
) -> List[Unknown]:
    """Extract unknowns from empty/questioned cells and dialogue questions."""
    unknowns = []
    seen = set()

    # Empty cells with connections = structural unknowns
    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill in ("E", "empty", "Q", "questioned"):
            name = cell.get("primitive", "")
            connections = cell.get("connections", ())
            if name and connections:
                q = f"What is {name}? (referenced but not explored)"
                if q not in seen:
                    seen.add(q)
                    category = "unclear_behavior" if cell.get("concern") == "BHV" else "missing_entity"
                    unknowns.append(Unknown(
                        question=q,
                        category=category,
                        priority=0.7,
                    ))
            elif fill in ("Q", "questioned"):
                q = f"Unresolved: {name or cell.get('postcode', 'unknown')}"
                if q not in seen:
                    seen.add(q)
                    unknowns.append(Unknown(
                        question=q,
                        category="unspecified_constraint",
                        priority=0.8,
                    ))

    # Questions from dialogue
    for msg in dialogue_messages:
        content = msg.get("content", "")
        for sentence in _split_sentences(content):
            sentence = sentence.strip()
            if sentence.endswith("?") and 10 < len(sentence) < 200:
                if sentence not in seen:
                    seen.add(sentence)
                    unknowns.append(Unknown(
                        question=sentence,
                        category="unspecified_constraint",
                        priority=0.5,
                    ))

    # Sort by priority descending, cap at 15
    unknowns.sort(key=lambda u: u.priority, reverse=True)
    return unknowns[:15]


def _extract_vocabulary(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
) -> List[str]:
    """Extract domain-specific vocabulary from cells and dialogue."""
    terms = set()

    # Cell primitives are domain vocabulary
    for cell in grid_cells:
        name = cell.get("primitive", "").strip()
        if name and len(name) > 1:
            terms.add(name)

    # CamelCase/PascalCase words from dialogue are likely domain terms
    camel_re = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
    for msg in dialogue_messages:
        content = msg.get("content", "")
        for match in camel_re.finditer(content):
            terms.add(match.group())

    # Sort alphabetically for stability
    return sorted(terms)


def _extract_memory_connections(
    memory_results: Optional[List[Dict[str, Any]]],
) -> List[str]:
    """Extract connection references from prior memory matches."""
    if not memory_results:
        return []
    connections = []
    for mem in memory_results:
        source = mem.get("source", "")
        text = mem.get("text", mem.get("content", ""))
        if source and text:
            connections.append(f"[{source}] {text[:100]}")
        elif text:
            connections.append(text[:100])
    return connections[:10]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    # Split on period/question-mark/exclamation followed by space or end
    return re.split(r"(?<=[.!?])\s+", text)
