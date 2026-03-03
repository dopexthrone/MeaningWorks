"""Convergence detection for semantic compilation — LEAF MODULE (stdlib only).

Measures blueprint stability across dialogue turns. When the blueprint stops
changing meaningfully, the dialogue has converged — stop excavating, move to
synthesis.

The key invariant: F(F) ≅ F. When another turn of dialogue doesn't add or
modify components, you've reached the fixed point.

Two mechanisms:
1. blueprint_delta() — compares two blueprint snapshots, returns 0-1 score
2. ConvergenceTracker — accumulates deltas, detects plateau, signals stop
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


def _extract_component_names(messages: list) -> Set[str]:
    """Extract capitalized component names from dialogue messages.

    Scans for capitalized multi-char words that look like component names,
    filtering out common noise words (agent names, discourse markers).
    """
    _NOISE = {
        "The", "This", "That", "These", "Those", "What", "When", "Where",
        "Which", "Why", "How", "You", "Your", "Agent", "Entity", "Process",
        "Here", "Let", "Yes", "Now", "Our", "Both", "They", "Its", "Not",
        "But", "And", "For", "With", "From", "Into", "Each", "Also", "All",
        "Has", "Have", "Will", "Would", "Could", "Should", "Must", "May",
        "Can", "Are", "Was", "Were", "Been", "Being", "Does", "Did",
        "INSIGHT", "SECTION", "COMPONENT", "RELATIONSHIP", "CONSTRAINT",
        "INPUT", "OUTPUT", "TYPE", "NOTE", "TODO", "REQUIRED", "CRITICAL",
        "Governor", "Synthesis", "Verify", "System", "None", "True", "False",
    }
    names: Set[str] = set()
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        words = set(re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', content))
        names |= (words - _NOISE)
    return names


def _extract_relationships(messages: list) -> Set[Tuple[str, str]]:
    """Extract relationship pairs from dialogue messages.

    Looks for patterns like:
    - "A connects to B", "A depends on B", "A uses B"
    - "A → B", "A -> B", "A->B"
    """
    pairs: Set[Tuple[str, str]] = set()
    arrow_pattern = re.compile(r'([A-Z][a-zA-Z]+)\s*(?:→|->|--|connects?\s+to|depends?\s+on|uses?|contains?)\s*([A-Z][a-zA-Z]+)')
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        for m in arrow_pattern.finditer(content):
            pairs.add((m.group(1), m.group(2)))
    return pairs


def _extract_insights(messages: list) -> Set[str]:
    """Extract insight strings from messages."""
    insights: Set[str] = set()
    for msg in messages:
        insight = msg.get("insight", "") if isinstance(msg, dict) else getattr(msg, "insight", "")
        if insight:
            insights.add(insight.strip()[:100])  # Normalize: first 100 chars
    return insights


def _count_domains(text: str) -> int:
    """Estimate number of distinct domains in input text.

    Looks for semantic domain markers: distinct noun clusters, explicit
    domain keywords, section headers, topic shifts.
    """
    # Explicit domain markers
    domain_signals = [
        r'\btattoo\b', r'\bstudio\b', r'\btrading\b', r'\btrader\b',
        r'\bhome\b', r'\bfridge\b', r'\bsecurity\b', r'\bCCTV\b',
        r'\bmarketing\b', r'\bSEO\b', r'\baccounting\b', r'\bfinancial\b',
        r'\bhealth\b', r'\bfitness\b', r'\bshopping\b', r'\bcrypto\b',
        r'\bnetwork\b', r'\bmarketplace\b', r'\bdevice\b', r'\bIoT\b',
        r'\bvoice\b', r'\bcustomer\s*service\b', r'\bemail\b',
        r'\bcalendar\b', r'\bsocial\b',
    ]
    text_lower = text.lower()
    count = sum(1 for p in domain_signals if re.search(p, text_lower))
    # Also count section headers (## in markdown)
    sections = len(re.findall(r'^##\s+', text, re.MULTILINE))
    # Rough estimate: each 2 domain signals ≈ 1 domain, each section ≈ 0.5 domain
    domain_estimate = max(1, (count + 1) // 2 + sections // 2)
    return min(domain_estimate, 15)  # Cap at 15


def blueprint_delta(prev_snapshot: dict, curr_snapshot: dict) -> float:
    """Compare two blueprint-like snapshots and return normalized delta.

    Snapshots are dicts with keys:
    - components: set of component names
    - relationships: set of (source, target) tuples
    - insights: set of insight strings
    - insight_count: int

    Returns float 0.0 (identical) to 1.0 (completely different).
    """
    prev_comps = prev_snapshot.get("components", set())
    curr_comps = curr_snapshot.get("components", set())
    prev_rels = prev_snapshot.get("relationships", set())
    curr_rels = curr_snapshot.get("relationships", set())
    prev_insights = prev_snapshot.get("insights", set())
    curr_insights = curr_snapshot.get("insights", set())

    # Component delta: Jaccard distance
    comp_union = prev_comps | curr_comps
    if comp_union:
        comp_delta = 1.0 - len(prev_comps & curr_comps) / len(comp_union)
    else:
        comp_delta = 0.0

    # Relationship delta: Jaccard distance
    rel_union = prev_rels | curr_rels
    if rel_union:
        rel_delta = 1.0 - len(prev_rels & curr_rels) / len(rel_union)
    else:
        rel_delta = 0.0

    # Insight delta: new insights as fraction
    insight_union = prev_insights | curr_insights
    if insight_union:
        insight_delta = 1.0 - len(prev_insights & curr_insights) / len(insight_union)
    else:
        insight_delta = 0.0

    # Weighted: components matter most, then relationships, then insights
    delta = 0.5 * comp_delta + 0.3 * rel_delta + 0.2 * insight_delta
    return min(max(delta, 0.0), 1.0)


def estimate_turn_budget(
    input_text: str,
    corpus_avg_turns: Optional[float] = None,
    corpus_sample_size: int = 0,
) -> Tuple[int, int, int]:
    """Estimate dialogue turn budget from input complexity and corpus history.

    Returns (min_turns, recommended_turns, max_turns).

    Factors:
    - Input length (longer = more to excavate)
    - Domain count (more domains = more dialogue needed)
    - Corpus history (if similar inputs took N turns, budget accordingly)
    """
    # Base from input length
    char_count = len(input_text)
    if char_count < 500:
        base = 6
    elif char_count < 2000:
        base = 9
    elif char_count < 5000:
        base = 12
    else:
        base = 15

    # Domain count bonus
    domains = _count_domains(input_text)
    domain_bonus = min(domains * 2, 12)  # 2 extra turns per domain, max 12

    # Corpus adjustment: if we have data, weight toward historical average
    if corpus_avg_turns and corpus_sample_size >= 3:
        corpus_weight = min(corpus_sample_size / 10, 0.6)  # Max 60% corpus influence
        estimated = base + domain_bonus
        adjusted = corpus_weight * corpus_avg_turns + (1 - corpus_weight) * estimated
        base_budget = int(adjusted)
    else:
        base_budget = base + domain_bonus

    min_turns = max(6, base_budget - 3)
    recommended = base_budget
    max_turns = min(base_budget + 9, 60)  # Hard ceiling at 60 turns

    return (min_turns, recommended, max_turns)


def take_snapshot(messages: list) -> dict:
    """Build a convergence snapshot from current dialogue state.

    Extracts components, relationships, and insights from all messages
    in the dialogue history.
    """
    return {
        "components": _extract_component_names(messages),
        "relationships": _extract_relationships(messages),
        "insights": _extract_insights(messages),
        "insight_count": len(_extract_insights(messages)),
    }


@dataclass
class ConvergenceTracker:
    """Track blueprint stability across dialogue turns.

    After each dialogue turn (or round), call update() with current messages.
    Check has_converged() to see if we've reached the fixed point.
    """
    plateau_window: int = 2          # Consecutive low-delta turns to declare convergence
    delta_threshold: float = 0.05    # Below this = "no meaningful change"
    min_turns_before_convergence: int = 6  # Don't converge before this many turns
    recommended_turns: Optional[int] = None  # When set, relax plateau_window past this

    # Internal state
    _snapshots: List[dict] = field(default_factory=list)
    _deltas: List[float] = field(default_factory=list)
    _turn_count: int = 0

    def update(self, messages: list, total_turns: int = 0) -> float:
        """Take a snapshot of current state and compute delta from previous.

        Args:
            messages: Full dialogue message history
            total_turns: Actual dialogue turn count (for convergence gating)

        Returns:
            Delta from previous snapshot (0.0 if first snapshot)
        """
        snapshot = take_snapshot(messages)
        self._turn_count = total_turns or (self._turn_count + 1)

        if not self._snapshots:
            self._snapshots.append(snapshot)
            self._deltas.append(1.0)  # First turn always has maximum delta
            return 1.0

        prev = self._snapshots[-1]
        delta = blueprint_delta(prev, snapshot)
        self._snapshots.append(snapshot)
        self._deltas.append(delta)
        return delta

    def has_converged(self) -> bool:
        """Check if the blueprint has stabilized.

        Returns True if:
        1. We've had at least min_turns_before_convergence turns
        2. The last plateau_window deltas are all below delta_threshold
           (relaxed to window=1 past recommended_turns)
        """
        if self._turn_count < self.min_turns_before_convergence:
            return False

        # Past recommended budget: single low delta is sufficient
        effective_window = self.plateau_window
        if self.recommended_turns and self._turn_count >= self.recommended_turns:
            effective_window = 1

        if len(self._deltas) < effective_window:
            return False

        recent = self._deltas[-effective_window:]
        return all(d < self.delta_threshold for d in recent)

    def should_continue(self, current_turn: int, max_turns: int) -> bool:
        """Decide whether dialogue should continue.

        Returns True if:
        - Haven't hit max_turns AND
        - Haven't converged (or haven't had enough turns to check)
        """
        if current_turn >= max_turns:
            return False
        return not self.has_converged()

    @property
    def last_delta(self) -> float:
        """Most recent delta value."""
        return self._deltas[-1] if self._deltas else 1.0

    @property
    def component_count(self) -> int:
        """Number of components discovered so far."""
        if self._snapshots:
            return len(self._snapshots[-1].get("components", set()))
        return 0

    @property
    def convergence_summary(self) -> dict:
        """Summary of convergence state for telemetry."""
        return {
            "turns": self._turn_count,
            "deltas": list(self._deltas),
            "converged": self.has_converged(),
            "component_count": self.component_count,
            "final_delta": self.last_delta,
        }


def grid_convergence_summary(grid) -> dict:
    """Build convergence summary from grid state (structural, not text-proxy).

    This replaces text-based convergence snapshots with direct grid introspection.
    The grid IS convergence — fill states are the ground truth, not regex extraction.

    Args:
        grid: A kernel Grid instance.

    Returns:
        Dict with fill_rate, total_cells, filled, empty, unfilled_connections,
        and converged flag.
    """
    from kernel.navigator import is_converged

    filled = grid.filled_cells()
    empty = grid.empty_cells()
    unfilled_conns = grid.unfilled_connections()

    return {
        "fill_rate": grid.fill_rate,
        "total_cells": grid.total_cells,
        "filled": len(filled),
        "empty": len(empty),
        "unfilled_connections": len(unfilled_conns),
        "converged": is_converged(grid),
    }
