"""
Exploration synthesis — extract divergent insights from grid state.

LEAF module. Stdlib only. No LLM calls.

Used by EXPLORE compilation mode. Given filled grid cells and dialogue
messages, surfaces insights the user hasn't considered: frontier questions,
adjacent domains, alternative framings, and depth chains.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


@dataclass(frozen=True)
class Insight:
    """A non-obvious insight extracted from exploration."""
    text: str
    category: str       # "pattern", "gap", "contradiction", "opportunity"
    confidence: float
    source: str          # what produced this insight


@dataclass(frozen=True)
class FrontierQuestion:
    """A question at the edge of explored territory."""
    question: str
    domain: str          # which domain/layer this question probes
    priority: float      # 0.0 - 1.0


@dataclass(frozen=True)
class AlternativeFraming:
    """A different way to frame the original intent."""
    framing: str
    perspective: str     # e.g. "user", "system", "economic", "temporal"
    divergence: float    # how different from original (0.0 = same, 1.0 = opposite)


@dataclass(frozen=True)
class ExplorationMap:
    """Structured divergent exploration from an EXPLORE compilation."""
    original_intent: str
    insights: Tuple[Insight, ...]
    frontier_questions: Tuple[FrontierQuestion, ...]
    adjacent_domains: Tuple[str, ...]
    alternative_framings: Tuple[AlternativeFraming, ...]
    depth_chains: Tuple[Dict[str, Any], ...]  # from endpoint extractor


def synthesize_exploration(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
    original_intent: str,
    endpoint_chains: Optional[List[Dict[str, Any]]] = None,
) -> ExplorationMap:
    """Synthesize an ExplorationMap from grid state and dialogue.

    Heuristic extraction — no LLM calls. Finds patterns, gaps,
    contradictions, and frontier questions from cell state and dialogue.

    Args:
        grid_cells: List of cell dicts from grid_to_structured().
        dialogue_messages: List of message dicts with keys: sender, content.
        original_intent: The user's original input text.
        endpoint_chains: Optional depth chain dicts from endpoint extractor.

    Returns:
        Frozen ExplorationMap with extracted exploration.
    """
    insights = _extract_insights(grid_cells, dialogue_messages)
    frontier_questions = _extract_frontier_questions(grid_cells, original_intent)
    adjacent_domains = _extract_adjacent_domains(grid_cells, dialogue_messages)
    alternative_framings = _extract_alternative_framings(original_intent, dialogue_messages)
    chains = tuple(endpoint_chains) if endpoint_chains else ()

    return ExplorationMap(
        original_intent=original_intent,
        insights=tuple(insights),
        frontier_questions=tuple(frontier_questions),
        adjacent_domains=tuple(adjacent_domains),
        alternative_framings=tuple(alternative_framings),
        depth_chains=chains,
    )


def exploration_map_to_dict(exp: ExplorationMap) -> Dict[str, Any]:
    """Serialize an ExplorationMap to a plain dict (JSON-safe)."""
    return {
        "original_intent": exp.original_intent,
        "insights": [
            {
                "text": i.text,
                "category": i.category,
                "confidence": i.confidence,
                "source": i.source,
            }
            for i in exp.insights
        ],
        "frontier_questions": [
            {
                "question": q.question,
                "domain": q.domain,
                "priority": q.priority,
            }
            for q in exp.frontier_questions
        ],
        "adjacent_domains": list(exp.adjacent_domains),
        "alternative_framings": [
            {
                "framing": f.framing,
                "perspective": f.perspective,
                "divergence": f.divergence,
            }
            for f in exp.alternative_framings
        ],
        "depth_chains": [dict(c) for c in exp.depth_chains],
    }


def format_exploration_summary(exp: ExplorationMap) -> str:
    """Format a human-readable summary of an ExplorationMap."""
    lines = []
    lines.append(f"Exploration: {exp.original_intent[:80]}")
    lines.append("")

    if exp.insights:
        lines.append(f"Insights ({len(exp.insights)}):")
        for i in exp.insights[:8]:
            lines.append(f"  [{i.category}] {i.text}")
        if len(exp.insights) > 8:
            lines.append(f"  ... and {len(exp.insights) - 8} more")
        lines.append("")

    if exp.frontier_questions:
        lines.append(f"Frontier Questions ({len(exp.frontier_questions)}):")
        for q in exp.frontier_questions[:6]:
            lines.append(f"  ? {q.question} ({q.domain})")
        lines.append("")

    if exp.adjacent_domains:
        lines.append(f"Adjacent Domains: {', '.join(exp.adjacent_domains)}")
        lines.append("")

    if exp.alternative_framings:
        lines.append(f"Alternative Framings ({len(exp.alternative_framings)}):")
        for f in exp.alternative_framings[:4]:
            lines.append(f"  [{f.perspective}] {f.framing}")
        lines.append("")

    if exp.depth_chains:
        lines.append(f"Depth Chains: {len(exp.depth_chains)} unexplored endpoint(s)")

    return "\n".join(lines)


# ============================================================
# Internal extraction functions
# ============================================================

def _extract_insights(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
) -> List[Insight]:
    """Extract non-obvious insights from grid patterns and dialogue."""
    insights = []

    # Pattern: layers with many filled cells indicate strong conceptual areas
    layer_counts: Dict[str, int] = {}
    layer_confidence: Dict[str, List[float]] = {}
    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill in ("F", "P", "filled", "partial"):
            parts = cell.get("postcode", "").split(".")
            layer = parts[0] if parts else ""
            if layer:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                conf = cell.get("confidence", 0.0)
                layer_confidence.setdefault(layer, []).append(conf)

    if layer_counts:
        # Dominant layer insight
        top_layer = max(layer_counts, key=lambda k: layer_counts[k])
        top_count = layer_counts[top_layer]
        if top_count >= 3:
            insights.append(Insight(
                text=f"Dominant conceptual layer: {top_layer} ({top_count} concepts)",
                category="pattern",
                confidence=0.8,
                source="grid_layer_analysis",
            ))

        # Low-confidence cluster insight
        for layer, confs in layer_confidence.items():
            avg_conf = sum(confs) / len(confs)
            if avg_conf < 0.5 and len(confs) >= 2:
                insights.append(Insight(
                    text=f"Layer {layer} has low confidence (avg {avg_conf:.0%}) — needs deeper exploration",
                    category="gap",
                    confidence=0.7,
                    source="grid_confidence_analysis",
                ))

    # Contradictions: cells in same layer with very different confidence
    for layer, confs in layer_confidence.items():
        if len(confs) >= 2:
            spread = max(confs) - min(confs)
            if spread > 0.5:
                insights.append(Insight(
                    text=f"Layer {layer} has high confidence spread ({spread:.0%}) — possible inconsistency",
                    category="contradiction",
                    confidence=0.6,
                    source="grid_spread_analysis",
                ))

    # Dialogue insight extraction — look for INSIGHT: markers
    for msg in dialogue_messages:
        content = msg.get("content", "")
        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("INSIGHT:"):
                text = line[len("INSIGHT:"):].strip()
                if text and len(text) > 5:
                    insights.append(Insight(
                        text=text,
                        category="pattern",
                        confidence=0.7,
                        source=f"dialogue:{msg.get('sender', 'unknown')}",
                    ))

    # Opportunity: empty cells connected to high-confidence filled cells
    filled_pcs = set()
    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill in ("F", "filled") and cell.get("confidence", 0) > 0.7:
            filled_pcs.add(cell.get("postcode", ""))

    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill in ("E", "empty"):
            connections = cell.get("connections", ())
            connected_to_strong = any(c in filled_pcs for c in connections)
            if connected_to_strong:
                name = cell.get("primitive", cell.get("postcode", ""))
                insights.append(Insight(
                    text=f"Unexplored concept '{name}' is connected to well-understood areas",
                    category="opportunity",
                    confidence=0.65,
                    source="grid_frontier_analysis",
                ))

    # Cap at 20, sorted by confidence
    insights.sort(key=lambda i: i.confidence, reverse=True)
    return insights[:20]


def _extract_frontier_questions(
    grid_cells: List[Dict[str, Any]],
    original_intent: str,
) -> List[FrontierQuestion]:
    """Generate frontier questions from grid boundaries."""
    questions = []

    # Layers with cells but low fill rate
    layer_total: Dict[str, int] = {}
    layer_filled: Dict[str, int] = {}
    for cell in grid_cells:
        parts = cell.get("postcode", "").split(".")
        layer = parts[0] if parts else ""
        if layer:
            layer_total[layer] = layer_total.get(layer, 0) + 1
            fill = cell.get("fill", "E")
            if fill in ("F", "P", "filled", "partial"):
                layer_filled[layer] = layer_filled.get(layer, 0) + 1

    for layer, total in layer_total.items():
        filled = layer_filled.get(layer, 0)
        if total > 0 and filled / total < 0.5:
            questions.append(FrontierQuestion(
                question=f"What aspects of {layer} layer remain unexplored?",
                domain=layer,
                priority=0.7,
            ))

    # Cross-layer gaps: layers referenced in connections but not activated
    activated_layers = set(layer_total.keys())
    referenced_layers = set()
    for cell in grid_cells:
        for conn in cell.get("connections", ()):
            parts = conn.split(".")
            if parts:
                referenced_layers.add(parts[0])

    for layer in referenced_layers - activated_layers:
        if layer:
            questions.append(FrontierQuestion(
                question=f"Layer {layer} is referenced but not explored — what lives there?",
                domain=layer,
                priority=0.8,
            ))

    # Generic frontier questions based on intent keywords
    intent_lower = original_intent.lower()
    if "user" in intent_lower and not any("USR" in q.domain for q in questions):
        questions.append(FrontierQuestion(
            question="What user behaviors haven't been considered?",
            domain="USR",
            priority=0.6,
        ))
    if any(w in intent_lower for w in ("scale", "grow", "large")):
        questions.append(FrontierQuestion(
            question="What happens at 10x the expected scale?",
            domain="RES",
            priority=0.6,
        ))
    if any(w in intent_lower for w in ("team", "collaborat", "multi")):
        questions.append(FrontierQuestion(
            question="What coordination problems emerge with multiple actors?",
            domain="AGN",
            priority=0.6,
        ))

    # Sort by priority, cap at 10
    questions.sort(key=lambda q: q.priority, reverse=True)
    return questions[:10]


# Domain adjacency map — what domains tend to neighbor each other
_ADJACENT_DOMAINS = {
    "SFT": ["security", "operations", "data science", "UX design"],
    "ORG": ["change management", "knowledge management", "training"],
    "DOM": ["regulatory compliance", "market analysis", "competitive intelligence"],
    "APP": ["platform engineering", "API design", "developer experience"],
    "ECO": ["sustainability", "supply chain", "partnership management"],
}


def _extract_adjacent_domains(
    grid_cells: List[Dict[str, Any]],
    dialogue_messages: List[Dict[str, Any]],
) -> List[str]:
    """Identify adjacent domains worth exploring."""
    # Determine which layers are active
    active_layers = set()
    for cell in grid_cells:
        fill = cell.get("fill", "E")
        if fill in ("F", "P", "filled", "partial"):
            parts = cell.get("postcode", "").split(".")
            if parts:
                active_layers.add(parts[0])

    # Collect adjacent domains from active layers
    adjacents = set()
    for layer in active_layers:
        for domain in _ADJACENT_DOMAINS.get(layer, []):
            adjacents.add(domain)

    # Check dialogue for domain mentions that could expand
    all_text = " ".join(m.get("content", "") for m in dialogue_messages)
    domain_hints = [
        "compliance", "security", "privacy", "accessibility",
        "performance", "cost", "reliability", "observability",
    ]
    for hint in domain_hints:
        if hint in all_text.lower():
            adjacents.add(hint)

    return sorted(adjacents)[:8]


# Perspective templates for alternative framings
_PERSPECTIVES = [
    ("user", "From the end user's perspective: {intent}"),
    ("economic", "As an economic system: {intent}"),
    ("temporal", "Over a 5-year timeline: {intent}"),
    ("failure", "If this fails completely: {intent}"),
    ("inverse", "The opposite approach to: {intent}"),
]


def _extract_alternative_framings(
    original_intent: str,
    dialogue_messages: List[Dict[str, Any]],
) -> List[AlternativeFraming]:
    """Generate alternative framings of the original intent."""
    if not original_intent:
        return []

    framings = []
    # Truncate intent for template insertion
    intent_short = original_intent[:120]

    for perspective, template in _PERSPECTIVES:
        framing_text = template.format(intent=intent_short)
        divergence = 0.3 if perspective in ("user", "temporal") else 0.6
        if perspective == "inverse":
            divergence = 0.9
        framings.append(AlternativeFraming(
            framing=framing_text,
            perspective=perspective,
            divergence=divergence,
        ))

    # Dialogue-derived reframings — if agents proposed alternative names
    reframe_re = re.compile(r"(?:alternatively|another way|could also|instead of|reframe)", re.I)
    for msg in dialogue_messages:
        content = msg.get("content", "")
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if reframe_re.search(sentence) and 15 < len(sentence) < 200:
                framings.append(AlternativeFraming(
                    framing=sentence.strip(),
                    perspective="dialogue",
                    divergence=0.5,
                ))
                if len(framings) >= 8:
                    break
        if len(framings) >= 8:
            break

    return framings[:8]
