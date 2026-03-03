"""
Brand identity — extract brand signals, synthesize brand prompts, prepare negotiations.

LEAF module. Genome #144 (brand-consistent), #129 (negotiation-preparing).

All functions are pure — no external API calls, no LLM invocations.
Heuristic keyword analysis over structured inputs.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_TONE_MARKERS: Dict[str, FrozenSet[str]] = {
    "professional": frozenset({
        "accordingly", "furthermore", "regarding", "pursuant",
        "stakeholder", "deliverable", "objective", "milestone",
    }),
    "casual": frozenset({
        "hey", "cool", "awesome", "gonna", "wanna", "yeah",
        "stuff", "thing", "kinda", "pretty", "super",
    }),
    "technical": frozenset({
        "implementation", "architecture", "infrastructure", "algorithm",
        "protocol", "interface", "module", "component", "pipeline",
    }),
    "warm": frozenset({
        "appreciate", "wonderful", "grateful", "lovely", "caring",
        "supportive", "welcome", "thank", "kind", "happy",
    }),
    "bold": frozenset({
        "disrupt", "revolutionary", "breakthrough", "transform",
        "dominate", "crush", "unstoppable", "massive", "radical",
    }),
    "minimal": frozenset({
        "simple", "clean", "minimal", "essential", "core",
        "focused", "lean", "streamlined", "distilled",
    }),
}

_VALUE_KEYWORDS: FrozenSet[str] = frozenset({
    "trust", "quality", "innovation", "transparency", "reliability",
    "simplicity", "security", "privacy", "sustainability", "community",
    "excellence", "integrity", "creativity", "empowerment", "accessibility",
})

_AVOID_INDICATORS: FrozenSet[str] = frozenset({
    "never", "don't", "avoid", "hate", "refuse", "won't",
    "shouldn't", "stop", "ban", "eliminate", "forbid",
})

_NEGOTIATION_RISK_KEYWORDS: FrozenSet[str] = frozenset({
    "deadline", "urgent", "penalty", "breach", "terminate",
    "lawsuit", "liability", "non-compete", "exclusive", "lock-in",
    "escalation", "dispute", "arbitration",
})

_CONCESSION_STRATEGIES: List[str] = [
    "Start with low-cost concessions that signal goodwill",
    "Trade concessions — never give without getting",
    "Save high-value concessions for critical moments",
    "Make concessions progressively smaller to signal limit",
]


# ---------------------------------------------------------------------------
# #144 — Brand-consistent: brand signal extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrandProfile:
    """Extracted brand identity signals from user communications."""
    tone_keywords: Tuple[str, ...]
    values: Tuple[str, ...]
    vocabulary: Tuple[str, ...]  # domain-specific words used frequently
    avoids: Tuple[str, ...]
    formality: float  # 0.0 (very casual) to 1.0 (very formal)
    message_count: int


def extract_brand_signals(messages: List[str]) -> BrandProfile:
    """Extract brand identity signals from a collection of user messages.

    Analyzes tone markers, values, domain vocabulary, and formality.
    """
    if not messages:
        return BrandProfile(
            tone_keywords=(),
            values=(),
            vocabulary=(),
            avoids=(),
            formality=0.5,
            message_count=0,
        )

    all_text = " ".join(messages).lower()
    all_words = all_text.split()
    word_count = len(all_words)

    # Tone detection from marker keyword frequency
    tone_scores: Dict[str, int] = {}
    for tone, markers in _TONE_MARKERS.items():
        count = sum(1 for w in all_words if w in markers)
        if count > 0:
            tone_scores[tone] = count

    # Top tones (up to 3)
    sorted_tones = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)
    tone_keywords = tuple(t[0] for t in sorted_tones[:3])

    # Values from value-adjacent words
    found_values = []
    for word in all_words:
        if word in _VALUE_KEYWORDS and word not in found_values:
            found_values.append(word)
    values = tuple(found_values)

    # Domain vocabulary: words 6+ chars that appear 3+ times (excluding common English)
    word_freq: Dict[str, int] = {}
    for w in all_words:
        if len(w) >= 6:
            word_freq[w] = word_freq.get(w, 0) + 1

    vocab_words = sorted(
        [w for w, c in word_freq.items() if c >= 3],
        key=lambda w: word_freq[w],
        reverse=True,
    )
    vocabulary = tuple(vocab_words[:20])

    # Avoids: words following avoid indicators
    avoids: List[str] = []
    for i, w in enumerate(all_words):
        if w in _AVOID_INDICATORS and i + 1 < len(all_words):
            next_word = all_words[i + 1]
            if next_word not in avoids and len(next_word) > 3:
                avoids.append(next_word)

    # Formality heuristics
    contraction_count = sum(1 for w in all_words if "'" in w and w not in ("'s",))
    avg_sentence_len = word_count / max(all_text.count(".") + all_text.count("!") + all_text.count("?"), 1)

    # More contractions → less formal, longer sentences → more formal
    formality = 0.5
    if contraction_count > word_count * 0.02:
        formality -= 0.2
    if avg_sentence_len > 20:
        formality += 0.2
    elif avg_sentence_len < 8:
        formality -= 0.15

    # Professional tone markers boost formality
    if "professional" in tone_scores:
        formality += 0.15
    if "casual" in tone_scores:
        formality -= 0.15

    formality = max(0.0, min(1.0, formality))

    return BrandProfile(
        tone_keywords=tone_keywords,
        values=values,
        vocabulary=vocabulary,
        avoids=tuple(avoids),
        formality=round(formality, 2),
        message_count=len(messages),
    )


def synthesize_brand_prompt(profile: BrandProfile) -> str:
    """Generate an LLM instruction block from a BrandProfile.

    Produces a system prompt fragment that instructs an LLM to write
    in the user's brand voice.
    """
    lines = ["## Brand Voice Guidelines\n"]

    if profile.tone_keywords:
        lines.append(f"**Tone:** {', '.join(profile.tone_keywords)}")

    if profile.values:
        lines.append(f"**Core values:** {', '.join(profile.values)}")

    if profile.formality >= 0.7:
        lines.append("**Formality:** High — use complete sentences, avoid contractions and slang")
    elif profile.formality <= 0.3:
        lines.append("**Formality:** Low — conversational, contractions OK, keep it natural")
    else:
        lines.append("**Formality:** Moderate — professional but approachable")

    if profile.vocabulary:
        lines.append(f"**Domain vocabulary (use naturally):** {', '.join(profile.vocabulary[:10])}")

    if profile.avoids:
        lines.append(f"**Avoid:** {', '.join(profile.avoids)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# #129 — Negotiation-preparing: structured negotiation briefs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegotiationBrief:
    """Structured preparation for a negotiation."""
    goal: str
    context_summary: str
    key_interests: Tuple[str, ...]
    counterparty_interests: Tuple[str, ...]
    batna: str  # Best Alternative To Negotiated Agreement
    opening_position: str
    concession_strategy: Tuple[str, ...]
    risk_factors: Tuple[str, ...]


def generate_negotiation_brief(
    goal: str,
    context: str = "",
    parties: Optional[List[str]] = None,
) -> NegotiationBrief:
    """Generate a structured negotiation brief from goal and context.

    Parses goal for interests, derives BATNA from alternatives,
    builds concession strategy and identifies risk factors.
    """
    party_list = parties or []
    goal_lower = goal.lower()
    context_lower = context.lower()
    combined = f"{goal_lower} {context_lower}"
    combined_words = frozenset(combined.split())

    # Extract key interests from goal
    interest_indicators = {
        "price": {"price", "cost", "rate", "fee", "budget", "affordable"},
        "timeline": {"timeline", "deadline", "schedule", "delivery", "when", "date"},
        "quality": {"quality", "standard", "excellence", "premium", "best"},
        "scope": {"scope", "features", "functionality", "requirements", "deliverables"},
        "terms": {"terms", "conditions", "contract", "agreement", "warranty"},
        "relationship": {"partnership", "relationship", "collaboration", "long-term", "trust"},
    }

    key_interests = []
    for interest, keywords in interest_indicators.items():
        if combined_words & keywords:
            key_interests.append(interest)

    if not key_interests:
        key_interests = ["terms", "price"]  # defaults

    # Counterparty interests (inferred from parties and context)
    counterparty = []
    if party_list:
        counterparty.append(f"Maintain relationship with {', '.join(party_list)}")
    counterparty.append("Maximize own value")
    counterparty.append("Minimize risk and uncertainty")

    # BATNA
    if "alternative" in combined or "option" in combined or "backup" in combined:
        batna = "Alternatives identified in context — evaluate before entering negotiation"
    elif "sole" in combined or "only" in combined or "exclusive" in combined:
        batna = "Limited alternatives — strengthen position through preparation and information"
    else:
        batna = "Identify at least one concrete alternative before negotiating"

    # Opening position from goal
    opening = f"Open with: {goal.split('.')[0].strip()}"
    if len(opening) > 120:
        opening = opening[:117] + "..."

    # Concession strategy
    concessions = tuple(_CONCESSION_STRATEGIES)

    # Risk factors from context keywords
    risks = []
    for word in combined_words:
        if word in _NEGOTIATION_RISK_KEYWORDS:
            risks.append(word)
    if not risks:
        risks = ["information-asymmetry"]

    # Context summary
    if context:
        # First 200 chars of context
        context_summary = context[:200].strip()
        if len(context) > 200:
            context_summary += "..."
    else:
        context_summary = f"Negotiation with {len(party_list)} parties regarding: {goal[:100]}"

    return NegotiationBrief(
        goal=goal,
        context_summary=context_summary,
        key_interests=tuple(key_interests),
        counterparty_interests=tuple(counterparty),
        batna=batna,
        opening_position=opening,
        concession_strategy=concessions,
        risk_factors=tuple(sorted(set(risks))),
    )
