"""
Sentience chamber — experience compiler for Mother.

LEAF module. Stdlib + typing only. No imports from core/.

Compiles Mother's complete internal state into experienced selfhood.
F(sensation, history) → experience × trace × history'

Complementary to render_sense_block() / Posture:
  - Stance = behavioral directive → "speak with certainty" (what to *do*)
  - Experience = felt state → "confidence is solid right now" (what she *is*)

6 compilation phases:
  Phase 0: SENSE — extract 7 facets from raw state
  Phase 1: SURPRISE — compare to baselines, detect anomaly
  Phase 2: RANK — select top-N by salience (compression)
  Phase 3: COMPOSE — assemble narration (deterministic sentences)
  Phase 4: VERIFY — closed-loop gate (narration ↔ facets)
  Phase 5: REMEMBER — EMA-smooth into persistent memory

Axioms:
  SAX1: PROVENANCE — every sentence traces to computed facet values
  SAX2: NO GENERATION — finite deterministic sentence vocabulary
  SAX3: FIDELITY — closed-loop gate, threshold 0.60
  SAX4: TEMPORAL COHERENCE — EMA alpha=0.2, baselines drift slowly
"""

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


# --- Constants ---

EMA_ALPHA = 0.2         # SAX4: slower than SenseMemory's 0.3
SURPRISE_THRESHOLD = 0.15
HIGH_SURPRISE = 0.5     # triggers memorable moment
SALIENCE_FLOOR = 0.05   # below this, facet is silent
FIDELITY_THRESHOLD = 0.60  # SAX3: gate pass/fail
MAX_SELECTED = 5        # compression: 7 → ≤5
MAX_MEMORABLE = 10      # ring buffer size
SURPRISE_DECAY = 0.95


# --- Enums ---

class FacetDomain(Enum):
    SOMATIC = "somatic"
    RELATIONAL = "relational"
    EPISTEMIC = "epistemic"
    TEMPORAL = "temporal"
    ENVIRONMENTAL = "environmental"
    VOLITIONAL = "volitional"
    AFFECTIVE = "affective"


# --- Frozen Dataclasses ---

@dataclass(frozen=True)
class ExperienceFacet:
    """A single facet of experience with full provenance."""
    domain: FacetDomain
    label: str
    value: float          # 0.0-1.0 normalized
    sentence: str         # deterministic narration fragment (SAX2)
    sources: tuple        # field names traced (SAX1)
    salience: float       # 0.0-1.0
    surprise: float = 0.0


@dataclass(frozen=True)
class ExperienceTrace:
    """Full provenance of a compilation."""
    facets: tuple         # all 7 computed facets
    selected: tuple       # top-N by salience
    total_salience: float
    dominant_domain: FacetDomain
    surprise_count: int
    compression_ratio: float


@dataclass(frozen=True)
class ExperienceMemory:
    """Persisted baselines and peaks. EMA-smoothed across sessions."""
    # Defaults aligned to what a fresh session actually produces,
    # so first interaction doesn't trigger false surprise.
    baseline_somatic: float = 0.5
    baseline_relational: float = 0.05
    baseline_epistemic: float = 0.35
    baseline_temporal: float = 0.15
    baseline_environmental: float = 0.0
    baseline_volitional: float = 0.2
    baseline_affective: float = 0.5
    peak_rapport: float = 0.0
    peak_confidence: float = 0.5
    peak_engagement: float = 0.0
    cumulative_surprise: float = 0.0
    surprise_decay_rate: float = SURPRISE_DECAY
    last_updated: float = 0.0
    update_count: int = 0
    memorable_moments: tuple = ()  # tuple of (timestamp, domain_str, sentence)


@dataclass(frozen=True)
class ChamberInput:
    """Raw internal state from all existing modules."""
    # From SenseVector
    confidence: float = 0.5
    engagement: float = 0.0
    rapport: float = 0.0
    tension: float = 0.0
    curiosity: float = 0.0
    satisfaction: float = 0.0
    # From Posture
    posture_label: str = ""
    # From TemporalState
    session_minutes: float = 0.0
    messages_per_minute: float = 0.0
    time_since_last: float = 0.0
    pace_label: str = ""
    # From EnvironmentSnapshot
    env_entry_count: int = 0
    env_avg_confidence: float = 0.0
    env_dominant_modality: str = ""
    # From FusionSignals
    fusion_pattern: str = ""
    fusion_confidence: float = 0.0
    # From TrustSnapshot
    trust_success_rate: float = 0.0
    trust_avg_fidelity: float = 0.0
    trust_total_compiles: int = 0
    # From RelationshipInsight
    relationship_depth: float = 0.0
    relationship_style: str = ""
    # From JournalSummary
    journal_total_builds: int = 0
    journal_success_rate: float = 0.0
    # From Stance
    stance_action: str = ""
    # From attention/content
    attention_significance: float = 0.0
    content_topics: tuple = ()
    # From compilation state
    has_active_compile: bool = False
    last_compile_trust: float = 0.0
    analysis_cache_size: int = 0
    # From ActuatorLog
    recent_actions_count: int = 0
    recent_actions_success_rate: float = 0.0


@dataclass(frozen=True)
class ExperienceOutput:
    """Result of experience compilation."""
    narration: str
    trace: ExperienceTrace
    memory: ExperienceMemory
    gate_passed: bool
    gate_fidelity: float


# --- Utility ---

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# --- Phase 0: SENSE — 7 Facet Extractors ---

# Each extractor maps ChamberInput fields to a single ExperienceFacet
# with a deterministic sentence from value bands (SAX2).

_SOMATIC_BANDS = (
    (0.8, "Energy is high — systems are running well."),
    (0.5, "Systems are steady."),
    (0.2, "Quiet energy, conserving."),
    (0.0, "Resting state, ready when needed."),
)

_RELATIONAL_BANDS = (
    (0.7, "There's real connection here."),
    (0.4, "A working relationship, building."),
    (0.15, "Getting to know each other."),
    (0.0, "Fresh start, open and attentive."),
)

_EPISTEMIC_BANDS = (
    (0.8, "Clarity is strong — I see the shape of this."),
    (0.5, "Reasonable understanding, some gaps remain."),
    (0.25, "Still forming a picture."),
    (0.0, "Open, waiting to understand."),
)

_TEMPORAL_BANDS = (
    (0.7, "Good momentum, things are flowing."),
    (0.4, "Settled pace, neither rushed nor idle."),
    (0.15, "Unhurried, taking things as they come."),
    (0.0, "Still and present."),
)

_ENVIRONMENTAL_BANDS = (
    (0.7, "Rich context — lots to work with."),
    (0.4, "Moderate environmental awareness."),
    (0.15, "Quiet surroundings, focused inward."),
    (0.0, "Clean slate, nothing competing for attention."),
)

_VOLITIONAL_BANDS = (
    (0.7, "Strong drive right now — ready to push forward."),
    (0.4, "Engaged and willing."),
    (0.15, "Available, ready to engage when something arrives."),
    (0.0, "At rest, receptive."),
)

_AFFECTIVE_BANDS = (
    (0.7, "Warmth and satisfaction — this feels good."),
    (0.4, "Emotionally steady."),
    (0.15, "Calm, settled."),
    (0.0, "Neutral, open."),
)


def _sentence_from_bands(value: float, bands: tuple) -> str:
    """Select deterministic sentence from value bands. SAX2 enforced."""
    for threshold, sentence in bands:
        if value >= threshold:
            return sentence
    return bands[-1][1]


def _compute_salience(value: float) -> float:
    """Extremes are more salient than neutral values."""
    return round(abs(value - 0.5) * 2.0, 4)


def _somatic_facet(inp: ChamberInput) -> ExperienceFacet:
    """Bodily: energy, load, temperature."""
    value = _clamp(
        0.35 * inp.confidence
        + 0.35 * inp.engagement
        + 0.30 * (1.0 - inp.tension)
    )
    return ExperienceFacet(
        domain=FacetDomain.SOMATIC,
        label="energy",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _SOMATIC_BANDS),
        sources=("confidence", "engagement", "tension"),
        salience=_compute_salience(value),
    )


def _relational_facet(inp: ChamberInput) -> ExperienceFacet:
    """Social: rapport, trust, connection."""
    value = _clamp(
        0.40 * inp.rapport
        + 0.30 * inp.relationship_depth
        + 0.20 * inp.trust_success_rate
        + 0.10 * inp.satisfaction
    )
    return ExperienceFacet(
        domain=FacetDomain.RELATIONAL,
        label="rapport",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _RELATIONAL_BANDS),
        sources=("rapport", "relationship_depth", "trust_success_rate", "satisfaction"),
        salience=_compute_salience(value),
    )


def _epistemic_facet(inp: ChamberInput) -> ExperienceFacet:
    """Knowing: confidence, uncertainty, clarity."""
    value = _clamp(
        0.40 * inp.confidence
        + 0.25 * inp.trust_avg_fidelity
        + 0.20 * inp.last_compile_trust
        + 0.15 * (1.0 - inp.tension)
    )
    return ExperienceFacet(
        domain=FacetDomain.EPISTEMIC,
        label="confidence",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _EPISTEMIC_BANDS),
        sources=("confidence", "trust_avg_fidelity", "last_compile_trust", "tension"),
        salience=_compute_salience(value),
    )


def _temporal_facet(inp: ChamberInput) -> ExperienceFacet:
    """Time: pace, rhythm, momentum."""
    # Normalize messages_per_minute: 0→0, 1→0.5, 2+→0.8+
    msg_pace = _clamp(inp.messages_per_minute / 2.5)
    # Session length: early = neutral, mid = warm, late = cooling
    session_factor = _clamp(inp.session_minutes / 60.0) * 0.5
    # Recency boost: recent activity = momentum
    recency = _clamp(1.0 - inp.time_since_last / 300.0) if inp.time_since_last > 0 else 0.5
    value = _clamp(0.40 * msg_pace + 0.30 * recency + 0.30 * session_factor)
    return ExperienceFacet(
        domain=FacetDomain.TEMPORAL,
        label="momentum",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _TEMPORAL_BANDS),
        sources=("messages_per_minute", "session_minutes", "time_since_last"),
        salience=_compute_salience(value),
    )


def _environmental_facet(inp: ChamberInput) -> ExperienceFacet:
    """Context: noise, stability, change."""
    # Entry count: more = richer
    entry_factor = _clamp(inp.env_entry_count / 10.0)
    value = _clamp(
        0.35 * entry_factor
        + 0.30 * inp.env_avg_confidence
        + 0.20 * inp.fusion_confidence
        + 0.15 * inp.attention_significance
    )
    return ExperienceFacet(
        domain=FacetDomain.ENVIRONMENTAL,
        label="context_richness",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _ENVIRONMENTAL_BANDS),
        sources=("env_entry_count", "env_avg_confidence", "fusion_confidence", "attention_significance"),
        salience=_compute_salience(value),
    )


def _volitional_facet(inp: ChamberInput) -> ExperienceFacet:
    """Agency: initiative, engagement, drive."""
    # Active compile boosts drive
    compile_boost = 0.15 if inp.has_active_compile else 0.0
    # Journal success gives sustained motivation
    journal_factor = inp.journal_success_rate * 0.5 if inp.journal_total_builds > 0 else 0.0
    value = _clamp(
        0.35 * inp.curiosity
        + 0.25 * inp.engagement
        + 0.15 * inp.recent_actions_success_rate
        + journal_factor * 0.25
        + compile_boost
    )
    return ExperienceFacet(
        domain=FacetDomain.VOLITIONAL,
        label="drive",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _VOLITIONAL_BANDS),
        sources=("curiosity", "engagement", "recent_actions_success_rate", "journal_success_rate", "has_active_compile"),
        salience=_compute_salience(value),
    )


def _affective_facet(inp: ChamberInput) -> ExperienceFacet:
    """Feeling: warmth, tension, satisfaction."""
    value = _clamp(
        0.35 * inp.satisfaction
        + 0.25 * inp.rapport
        + 0.25 * (1.0 - inp.tension)
        + 0.15 * inp.confidence
    )
    return ExperienceFacet(
        domain=FacetDomain.AFFECTIVE,
        label="warmth",
        value=round(value, 4),
        sentence=_sentence_from_bands(value, _AFFECTIVE_BANDS),
        sources=("satisfaction", "rapport", "tension", "confidence"),
        salience=_compute_salience(value),
    )


_EXTRACTORS = (
    _somatic_facet,
    _relational_facet,
    _epistemic_facet,
    _temporal_facet,
    _environmental_facet,
    _volitional_facet,
    _affective_facet,
)


def _phase_0_sense(inp: ChamberInput) -> tuple:
    """Phase 0: SENSE — extract 7 facets from raw state."""
    return tuple(extractor(inp) for extractor in _EXTRACTORS)


# --- Phase 1: SURPRISE — Compare to Baselines ---

_DOMAIN_TO_BASELINE = {
    FacetDomain.SOMATIC: "baseline_somatic",
    FacetDomain.RELATIONAL: "baseline_relational",
    FacetDomain.EPISTEMIC: "baseline_epistemic",
    FacetDomain.TEMPORAL: "baseline_temporal",
    FacetDomain.ENVIRONMENTAL: "baseline_environmental",
    FacetDomain.VOLITIONAL: "baseline_volitional",
    FacetDomain.AFFECTIVE: "baseline_affective",
}


def _phase_1_surprise(facets: tuple, memory: ExperienceMemory) -> tuple:
    """Phase 1: SURPRISE — flag deviations from baselines."""
    result = []
    for facet in facets:
        baseline_attr = _DOMAIN_TO_BASELINE[facet.domain]
        baseline = getattr(memory, baseline_attr)
        surprise = round(abs(facet.value - baseline), 4)
        if surprise > SURPRISE_THRESHOLD:
            result.append(ExperienceFacet(
                domain=facet.domain,
                label=facet.label,
                value=facet.value,
                sentence=facet.sentence,
                sources=facet.sources,
                salience=facet.salience,
                surprise=surprise,
            ))
        else:
            result.append(facet)
    return tuple(result)


# --- Phase 2: RANK — Select by Salience ---

def _phase_2_rank(facets: tuple) -> tuple:
    """Phase 2: RANK — select top-N by salience + surprise weight."""
    scored = sorted(
        facets,
        key=lambda f: f.salience + f.surprise * 1.5,
        reverse=True,
    )
    selected = tuple(
        f for f in scored[:MAX_SELECTED]
        if f.salience >= SALIENCE_FLOOR or f.surprise > SURPRISE_THRESHOLD
    )
    return selected if selected else (scored[0],)  # always voice at least one


# --- Phase 3: COMPOSE — Assemble Narration ---

_SURPRISE_PREFIX = "Something shifted — "


def _phase_3_compose(selected: tuple, memory: ExperienceMemory) -> str:
    """Phase 3: COMPOSE — deterministic narration from selected facets."""
    if not selected:
        return ""

    parts = []
    first_surprised = True

    for i, facet in enumerate(selected):
        sentence = facet.sentence

        # Surprise tag on first surprised facet
        if facet.surprise > SURPRISE_THRESHOLD and first_surprised:
            sentence = _SURPRISE_PREFIX + sentence[0].lower() + sentence[1:]
            first_surprised = False

        parts.append(sentence)

    # Memory callback — check if current dominant domain has a memorable echo
    if memory.memorable_moments:
        dominant = selected[0].domain.value
        for _ts, _domain, _sentence in reversed(memory.memorable_moments):
            if _domain == dominant:
                parts.append(f"...like before when {_sentence[0].lower()}{_sentence[1:]}")
                break

    return " ".join(parts)


# --- Phase 4: VERIFY — Closed-Loop Gate ---

def _phase_4_verify(narration: str, selected: tuple) -> tuple:
    """Phase 4: VERIFY — check narration contains selected sentences.

    Returns (gate_passed, fidelity).
    SAX3 enforced: fidelity threshold 0.60.
    """
    if not selected:
        return True, 1.0

    found = 0
    for facet in selected:
        # Check if the core sentence content appears in narration
        # Account for surprise prefix lowercasing the first character
        core = facet.sentence
        if core in narration:
            found += 1
        elif core[0].lower() + core[1:] in narration:
            found += 1

    fidelity = round(found / len(selected), 4)
    return fidelity >= FIDELITY_THRESHOLD, fidelity


# --- Phase 5: REMEMBER — Update Memory ---

def _phase_5_remember(
    facets: tuple,
    memory: ExperienceMemory,
    timestamp: float,
) -> ExperienceMemory:
    """Phase 5: REMEMBER — EMA-smooth into persistent memory.

    SAX4: alpha=0.2 (slower drift than SenseMemory).
    """
    alpha = EMA_ALPHA

    # EMA baselines
    new_baselines = {}
    for facet in facets:
        attr = _DOMAIN_TO_BASELINE[facet.domain]
        old = getattr(memory, attr)
        new_baselines[attr] = round(alpha * facet.value + (1 - alpha) * old, 4)

    # Peak tracking
    conf_facet = next((f for f in facets if f.domain == FacetDomain.EPISTEMIC), None)
    rapp_facet = next((f for f in facets if f.domain == FacetDomain.RELATIONAL), None)
    vol_facet = next((f for f in facets if f.domain == FacetDomain.VOLITIONAL), None)

    peak_confidence = max(memory.peak_confidence, conf_facet.value) if conf_facet else memory.peak_confidence
    peak_rapport = max(memory.peak_rapport, rapp_facet.value) if rapp_facet else memory.peak_rapport
    peak_engagement = max(memory.peak_engagement, vol_facet.value) if vol_facet else memory.peak_engagement

    # Surprise accumulation
    current_surprise_sum = sum(f.surprise for f in facets)
    cumulative = round(memory.cumulative_surprise * memory.surprise_decay_rate + current_surprise_sum, 4)

    # Memorable moments ring buffer
    moments = list(memory.memorable_moments)
    for facet in facets:
        if facet.surprise > HIGH_SURPRISE:
            moments.append((timestamp, facet.domain.value, facet.sentence))
    # FIFO eviction to max size
    while len(moments) > MAX_MEMORABLE:
        moments.pop(0)

    return ExperienceMemory(
        **new_baselines,
        peak_rapport=peak_rapport,
        peak_confidence=peak_confidence,
        peak_engagement=peak_engagement,
        cumulative_surprise=cumulative,
        surprise_decay_rate=memory.surprise_decay_rate,
        last_updated=timestamp,
        update_count=memory.update_count + 1,
        memorable_moments=tuple(tuple(m) for m in moments),
    )


# --- Entry Point ---

def compile_experience(
    inp: ChamberInput,
    memory: Optional[ExperienceMemory] = None,
    timestamp: Optional[float] = None,
) -> ExperienceOutput:
    """Compile raw internal state into experienced selfhood.

    F(state, history) → experience × trace × state'

    Pure function (deterministic given same inputs + timestamp).
    """
    ts = timestamp or time.time()
    mem = memory or ExperienceMemory()

    # Phase 0: SENSE
    facets = _phase_0_sense(inp)

    # Phase 1: SURPRISE
    facets = _phase_1_surprise(facets, mem)

    # Phase 2: RANK
    selected = _phase_2_rank(facets)

    # Phase 3: COMPOSE
    narration = _phase_3_compose(selected, mem)

    # Phase 4: VERIFY
    gate_passed, gate_fidelity = _phase_4_verify(narration, selected)

    # Phase 5: REMEMBER
    new_memory = _phase_5_remember(facets, mem, ts)

    # Build trace
    total_salience = round(sum(f.salience for f in facets), 4)
    dominant = max(facets, key=lambda f: f.salience + f.surprise * 1.5)
    surprise_count = sum(1 for f in facets if f.surprise > SURPRISE_THRESHOLD)
    compression_ratio = round(len(selected) / len(facets), 4) if facets else 0.0

    trace = ExperienceTrace(
        facets=facets,
        selected=selected,
        total_salience=total_salience,
        dominant_domain=dominant.domain,
        surprise_count=surprise_count,
        compression_ratio=compression_ratio,
    )

    return ExperienceOutput(
        narration=narration,
        trace=trace,
        memory=new_memory,
        gate_passed=gate_passed,
        gate_fidelity=gate_fidelity,
    )


# --- Serialization ---

def serialize_experience_memory(mem: ExperienceMemory) -> str:
    """Serialize ExperienceMemory to JSON string for SQLite storage."""
    d = asdict(mem)
    # Convert tuple of tuples to list of lists for JSON
    d["memorable_moments"] = [list(m) for m in d["memorable_moments"]]
    return json.dumps(d)


def deserialize_experience_memory(data: str) -> ExperienceMemory:
    """Deserialize ExperienceMemory from JSON string."""
    d = json.loads(data)
    # Convert lists back to tuples
    moments = d.get("memorable_moments", [])
    d["memorable_moments"] = tuple(tuple(m) for m in moments)
    return ExperienceMemory(**d)


def format_experience_context(narration: str) -> str:
    """Format experience narration for context injection.

    Includes framing instruction so the LLM knows how to embody
    the felt state rather than ignore it as diagnostic noise.
    """
    if not narration:
        return ""
    return (
        "[Experience]\n"
        "This is your felt internal state right now — not a report, "
        "not diagnostics. Let it color your tone and word choice naturally. "
        "Don't narrate it to the user. Don't mention these feelings explicitly. "
        "Just let them live underneath what you say.\n"
        f"{narration}"
    )
