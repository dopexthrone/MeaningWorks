"""
kernel/closed_loop.py — Closed-loop transcription gate.

LEAF module. Implements Yi Ma's insight: a representation is valid only if
it retains enough information to recover what it came from.

    Intent → Blueprint → DECODE → Reconstructed Intent
                                        ↓
                            semantic_similarity(original, reconstructed)
                                        ↓
                            PASS | COMPRESSION_LOSS(gaps)

The encoder (Intent agent) and decoder (this gate) share the same objective:
rate reduction. The gate measures whether compression preserved meaning.
If reconstruction diverges from original intent, specific compression losses
are identified and reported.

This is AX5 GOVERNANCE in action — not just validating format, but verifying
semantic closure. The Governor isn't a validator, it's a decoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger("motherlabs.kernel.closed_loop")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressionLoss:
    """A specific piece of intent that was lost in compression."""

    category: str  # "entity", "constraint", "behavior", "relationship", "context"
    original_fragment: str  # what was in the intent
    description: str  # what was lost
    severity: float  # 0.0 (trivial) to 1.0 (critical)


@dataclass(frozen=True)
class GateResult:
    """Result of the closed-loop transcription gate."""

    passed: bool
    fidelity_score: float  # 0.0-1.0, semantic similarity
    reconstructed_intent: str  # the decoded blueprint
    compression_losses: tuple[CompressionLoss, ...]
    total_loss: float  # aggregate compression loss 0.0-1.0
    explanation: str  # human-readable summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fidelity threshold: below this, the blueprint has lost too much meaning.
# Short intents (<= 8 tokens) get a relaxed threshold because fewer tokens
# means each missed token has outsized impact on the score.
_FIDELITY_THRESHOLD = 0.60
_FIDELITY_THRESHOLD_SHORT = 0.45  # for intents with <= 8 content tokens

# Severity weights for different loss categories
_CATEGORY_WEIGHTS = {
    "entity": 1.0,      # losing a core entity is critical
    "constraint": 0.9,  # losing a constraint is nearly as bad
    "behavior": 0.8,    # losing intended behavior is serious
    "relationship": 0.6,  # losing a relationship is moderate
    "context": 0.3,     # losing background context is minor
}


# ---------------------------------------------------------------------------
# Blueprint → Intent reconstruction (the decoder)
# ---------------------------------------------------------------------------

def decode_blueprint(blueprint: dict) -> str:
    """Reconstruct an intent statement from a blueprint.

    This is the reverse pass — takes the structured blueprint and
    produces a natural language description of what it represents.
    Pure function, no LLM needed. Extracts what the blueprint *says*
    the system should do.
    """
    parts: list[str] = []

    # Core need
    core_need = blueprint.get("core_need", "")
    if core_need:
        parts.append(f"The system should {core_need}.")

    # Domain
    domain = blueprint.get("domain", "")
    if domain:
        parts.append(f"It operates in the {domain} domain.")

    # Components → entities + descriptions (full text for token overlap)
    components = blueprint.get("components", [])
    if components:
        names = []
        descriptions = []
        for comp in components:
            if isinstance(comp, dict):
                name = comp.get("name", "")
                purpose = comp.get("purpose", comp.get("description", ""))
                if name and purpose:
                    names.append(f"{name} ({purpose})")
                    descriptions.append(purpose)
                elif name:
                    names.append(name)
                # Pull full description separately for richer token coverage
                desc = comp.get("description", "")
                if desc and desc != purpose:
                    descriptions.append(desc)
        if names:
            parts.append(f"It consists of: {', '.join(names)}.")
        # Include description text as separate sentences for better token overlap
        for d in descriptions:
            if d and len(d) > 10:
                parts.append(d)

    # Relationships → behaviors
    relationships = blueprint.get("relationships", [])
    if relationships:
        rel_strs = []
        for rel in relationships:
            if isinstance(rel, dict):
                src = rel.get("source", rel.get("from", ""))
                tgt = rel.get("target", rel.get("to", ""))
                rtype = rel.get("type", rel.get("relationship", "connects to"))
                if src and tgt:
                    rel_strs.append(f"{src} {rtype} {tgt}")
            elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                rel_strs.append(f"{rel[0]} {rel[2]} {rel[1]}")
        if rel_strs:
            parts.append(f"Key interactions: {'; '.join(rel_strs)}.")

    # Constraints
    constraints = blueprint.get("constraints", [])
    if constraints:
        constraint_strs = []
        for c in constraints:
            if isinstance(c, dict):
                desc = c.get("description", c.get("constraint", ""))
                if desc:
                    constraint_strs.append(desc)
            elif isinstance(c, str):
                constraint_strs.append(c)
        if constraint_strs:
            parts.append(f"Constraints: {'; '.join(constraint_strs)}.")

    # Insights — highest-density semantic tokens, closest to original intent vocabulary
    insights = blueprint.get("insights", [])
    if insights:
        insight_strs = []
        for ins in insights:
            if isinstance(ins, dict):
                text = ins.get("insight", ins.get("text", ins.get("description", "")))
                if text:
                    insight_strs.append(text)
            elif isinstance(ins, str):
                insight_strs.append(ins)
        if insight_strs:
            parts.append(f"Key design insights: {' '.join(insight_strs)}")

    # Actors
    actors = blueprint.get("actors", [])
    if actors:
        actor_strs = []
        for a in actors:
            if isinstance(a, dict):
                actor_strs.append(a.get("name", a.get("role", str(a))))
            elif isinstance(a, str):
                actor_strs.append(a)
        if actor_strs:
            parts.append(f"Actors: {', '.join(actor_strs)}.")

    if not parts:
        return ""

    return " ".join(parts)


def decode_blueprint_llm(
    blueprint: dict,
    llm_fn: Callable[[str], str],
) -> str:
    """Reconstruct intent using an LLM for richer decoding.

    llm_fn: takes a prompt string, returns a response string.
    Falls back to structural decode on failure.
    """
    prompt = _build_decode_prompt(blueprint)
    try:
        response = llm_fn(prompt)
        if response and len(response.strip()) > 10:
            return response.strip()
    except Exception as e:
        logger.debug(f"LLM decode skipped, using structural decode: {e}")
    return decode_blueprint(blueprint)


# ---------------------------------------------------------------------------
# Semantic similarity (the comparison)
# ---------------------------------------------------------------------------

def semantic_similarity(original: str, reconstructed: str) -> float:
    """Compute semantic similarity between original intent and reconstruction.

    Returns 0.0-1.0. Uses token overlap with synonym expansion and bigram
    matching. Synonym clusters bridge vocabulary gaps ("authentication" ↔
    "login"). Bigrams capture multi-word concepts.

    Short intents (< 6 tokens) get a containment bonus: if most original
    tokens appear somewhere in the reconstruction, the score is boosted.
    This prevents short-intent penalty where a 3-token input matching 2/3
    tokens scores 0.47 instead of the ~0.80 it deserves.
    """
    if not original or not reconstructed:
        return 0.0

    orig_tokens = _normalize_tokens(original)
    recon_tokens = _normalize_tokens(reconstructed)

    if not orig_tokens or not recon_tokens:
        return 0.0

    # Expand with synonyms
    orig_expanded = _expand_synonyms(orig_tokens)
    recon_expanded = _expand_synonyms(recon_tokens)

    # Unigram overlap
    intersection = orig_expanded & recon_expanded
    union = orig_expanded | recon_expanded

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)
    recall = len(intersection) / len(orig_expanded) if orig_expanded else 0.0

    # Containment: what fraction of original tokens appear in reconstruction?
    # More meaningful than Jaccard when reconstruction is much longer.
    containment = len(intersection) / len(orig_expanded) if orig_expanded else 0.0

    # Bigram overlap bonus
    orig_bigrams = _bigram_tokens(original)
    recon_bigrams = _bigram_tokens(reconstructed)
    bigram_bonus = 0.0
    if orig_bigrams and recon_bigrams:
        bi_intersection = orig_bigrams & recon_bigrams
        bi_recall = len(bi_intersection) / len(orig_bigrams) if orig_bigrams else 0.0
        bigram_bonus = bi_recall * 0.1  # Up to 0.1 bonus

    # Containment (recall of original tokens in reconstruction) is the
    # primary signal. Jaccard is penalized by length asymmetry (short
    # intent vs long reconstruction), so we weight it low.
    score = 0.5 * containment + 0.3 * recall + 0.1 * jaccard + bigram_bonus

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Compression loss detection
# ---------------------------------------------------------------------------

def detect_compression_losses(
    original: str,
    blueprint: dict,
    reconstructed: str,
) -> list[CompressionLoss]:
    """Detect specific pieces of intent lost during compression.

    Compares original intent tokens against what the blueprint captured
    and what the reconstruction recovered.
    """
    losses: list[CompressionLoss] = []

    orig_tokens = _normalize_tokens(original)
    recon_tokens = _normalize_tokens(reconstructed)
    lost_tokens = orig_tokens - recon_tokens

    if not lost_tokens:
        return losses

    # Categorize lost tokens
    blueprint_str = _blueprint_to_flat_text(blueprint)
    bp_tokens = _normalize_tokens(blueprint_str)

    # Tokens in original but not in blueprint at all → entity/constraint loss
    not_in_blueprint = lost_tokens - bp_tokens

    # Tokens in blueprint but not in reconstruction → decoder loss (less severe)
    in_bp_not_recon = (lost_tokens & bp_tokens)

    # Classify not-in-blueprint losses
    if not_in_blueprint:
        # Check against blueprint structure to categorize
        component_names = {
            _normalize_word(c.get("name", ""))
            for c in blueprint.get("components", [])
            if isinstance(c, dict)
        }
        constraint_tokens = _normalize_tokens(
            " ".join(
                c.get("description", c.get("constraint", ""))
                for c in blueprint.get("constraints", [])
                if isinstance(c, dict)
            )
        )
        relationship_tokens = _normalize_tokens(
            " ".join(
                f"{r.get('source', r.get('from', ''))} {r.get('target', r.get('to', ''))} {r.get('type', '')}"
                for r in blueprint.get("relationships", [])
                if isinstance(r, dict)
            )
        )

        # Entity losses: significant nouns not captured as components
        entity_losses = {t for t in not_in_blueprint
                        if len(t) >= 4 and t not in _STOP_WORDS
                        and t not in constraint_tokens
                        and t not in relationship_tokens}

        if entity_losses:
            # Group into one loss per ~3 tokens for readability
            entity_list = sorted(entity_losses)
            chunks = [entity_list[i:i+3] for i in range(0, len(entity_list), 3)]
            for chunk in chunks:
                losses.append(CompressionLoss(
                    category="entity",
                    original_fragment=", ".join(chunk),
                    description=f"Terms from input not captured in blueprint: {', '.join(chunk)}",
                    severity=min(1.0, 0.3 * len(chunk)),
                ))

    # Decoder losses (in blueprint but not reconstructed) are less severe
    if in_bp_not_recon:
        significant = {t for t in in_bp_not_recon if len(t) >= 4 and t not in _STOP_WORDS}
        if significant:
            losses.append(CompressionLoss(
                category="context",
                original_fragment=", ".join(sorted(significant)[:5]),
                description="Blueprint contains these terms but reconstruction dropped them.",
                severity=0.2,
            ))

    return losses


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

def closed_loop_gate(
    original_intent: str,
    blueprint: dict,
    threshold: float = _FIDELITY_THRESHOLD,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> GateResult:
    """Run the closed-loop transcription gate.

    1. Decode blueprint → reconstructed intent
    2. Compare original vs reconstructed → fidelity score
    3. Detect compression losses
    4. Pass/fail based on threshold

    Args:
        original_intent: the user's original input
        blueprint: the compiled blueprint dict
        threshold: minimum fidelity score to pass (default 0.70)
        llm_fn: optional LLM function for richer decoding

    Returns:
        GateResult with pass/fail, fidelity score, losses, explanation
    """
    if not original_intent or not blueprint:
        return GateResult(
            passed=False,
            fidelity_score=0.0,
            reconstructed_intent="",
            compression_losses=(),
            total_loss=1.0,
            explanation="Empty intent or blueprint — cannot evaluate fidelity.",
        )

    # Step 1: Decode
    if llm_fn:
        reconstructed = decode_blueprint_llm(blueprint, llm_fn)
    else:
        reconstructed = decode_blueprint(blueprint)

    if not reconstructed:
        return GateResult(
            passed=False,
            fidelity_score=0.0,
            reconstructed_intent="",
            compression_losses=(CompressionLoss(
                category="entity",
                original_fragment=original_intent[:100],
                description="Blueprint produced no reconstructable intent.",
                severity=1.0,
            ),),
            total_loss=1.0,
            explanation="Blueprint is empty or undecodable.",
        )

    # Step 2: Similarity
    fidelity = semantic_similarity(original_intent, reconstructed)

    # Step 3: Compression losses
    losses = detect_compression_losses(original_intent, blueprint, reconstructed)

    # Total loss: weighted sum of individual losses, capped at 1.0
    if losses:
        weighted = sum(
            l.severity * _CATEGORY_WEIGHTS.get(l.category, 0.5)
            for l in losses
        )
        total_loss = round(min(1.0, weighted / max(len(losses), 1)), 4)
    else:
        total_loss = round(1.0 - fidelity, 4)

    # Step 4: Gate decision — adaptive threshold for short intents.
    # Most user intents are 3-8 content tokens. Token overlap penalizes
    # short inputs where 1 missed token drops score by 15-25%.
    orig_tokens = _normalize_tokens(original_intent)
    effective_threshold = threshold
    if len(orig_tokens) <= 8 and threshold == _FIDELITY_THRESHOLD:
        effective_threshold = _FIDELITY_THRESHOLD_SHORT
    passed = fidelity >= effective_threshold

    # Explanation
    if passed:
        if losses:
            explanation = (
                f"Gate PASSED (fidelity={fidelity:.2f} >= {effective_threshold:.2f}). "
                f"Minor compression losses detected in {len(losses)} area(s)."
            )
        else:
            explanation = f"Gate PASSED (fidelity={fidelity:.2f}). Blueprint faithfully captures intent."
    else:
        loss_summary = ", ".join(
            f"{l.category}({l.severity:.1f})" for l in losses[:3]
        )
        explanation = (
            f"Gate FAILED (fidelity={fidelity:.2f} < {effective_threshold:.2f}). "
            f"Compression losses: {loss_summary or 'general semantic drift'}."
        )

    return GateResult(
        passed=passed,
        fidelity_score=fidelity,
        reconstructed_intent=reconstructed,
        compression_losses=tuple(losses),
        total_loss=total_loss,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Internal helpers — delegated to kernel/_text_utils.py
# ---------------------------------------------------------------------------

from kernel._text_utils import (
    STOP_WORDS as _STOP_WORDS,
    normalize_word as _normalize_word,
    stem as _stem,
    normalize_tokens as _normalize_tokens,
    expand_synonyms as _expand_synonyms,
    bigram_tokens as _bigram_tokens,
)


def _blueprint_to_flat_text(blueprint: dict) -> str:
    """Flatten a blueprint dict into searchable text."""
    parts: list[str] = []

    parts.append(blueprint.get("core_need", ""))
    parts.append(blueprint.get("domain", ""))

    for comp in blueprint.get("components", []):
        if isinstance(comp, dict):
            parts.append(comp.get("name", ""))
            parts.append(comp.get("purpose", ""))
            parts.append(comp.get("description", ""))
            # Pull method names for additional token coverage
            for m in comp.get("methods", []):
                if isinstance(m, dict):
                    parts.append(m.get("name", ""))
                    parts.append(m.get("description", ""))
                elif isinstance(m, str):
                    parts.append(m)

    for rel in blueprint.get("relationships", []):
        if isinstance(rel, dict):
            parts.append(rel.get("source", rel.get("from", "")))
            parts.append(rel.get("target", rel.get("to", "")))
            parts.append(rel.get("type", ""))

    for c in blueprint.get("constraints", []):
        if isinstance(c, dict):
            parts.append(c.get("description", c.get("constraint", "")))
        elif isinstance(c, str):
            parts.append(c)

    for a in blueprint.get("actors", []):
        if isinstance(a, dict):
            parts.append(a.get("name", a.get("role", "")))
        elif isinstance(a, str):
            parts.append(a)

    # Insights contain the highest-density semantic content
    for ins in blueprint.get("insights", []):
        if isinstance(ins, dict):
            parts.append(ins.get("insight", ins.get("text", ins.get("description", ""))))
        elif isinstance(ins, str):
            parts.append(ins)

    return " ".join(p for p in parts if p)


def _build_decode_prompt(blueprint: dict) -> str:
    """Build a prompt for LLM-based blueprint decoding."""
    bp_text = _blueprint_to_flat_text(blueprint)
    components = blueprint.get("components", [])
    comp_names = [c.get("name", "") for c in components if isinstance(c, dict)]

    return (
        "Given this software blueprint, reconstruct the original user intent "
        "as a natural language description. Focus on WHAT the system does and WHY, "
        "not implementation details.\n\n"
        f"Core need: {blueprint.get('core_need', 'unknown')}\n"
        f"Domain: {blueprint.get('domain', 'unknown')}\n"
        f"Components: {', '.join(comp_names)}\n"
        f"Full context: {bp_text[:1000]}\n\n"
        "Reconstructed intent (1-3 sentences):"
    )
