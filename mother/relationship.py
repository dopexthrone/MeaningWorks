"""
Cross-session relationship extraction — context that compounds into experience.

LEAF module. Stdlib only. No imports from core/ or mother/.
Pure functions. Deterministic. Every claim traces to data.

Extracts structural patterns from conversation history and synthesizes
a narrative that replaces the flat stat line in the system prompt.

Usage:
    insight = extract_relationship_insights(messages, sessions, sense_mem, corpus)
    narrative = synthesize_relationship_narrative(insight)
"""

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --- Stopwords (common English, kept minimal) ---

_STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "this", "that", "these", "those", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might", "shall", "must",
    "a", "an", "the", "and", "but", "or", "if", "so", "as", "at", "by",
    "for", "in", "of", "on", "to", "up", "with", "from", "not", "no",
    "just", "also", "very", "too", "than", "then", "now", "here", "there",
    "what", "which", "who", "how", "when", "where", "why", "all", "each",
    "any", "some", "such", "more", "most", "other", "into", "over", "out",
    "about", "like", "get", "got", "make", "made", "want", "need", "know",
    "think", "see", "look", "go", "come", "take", "give", "tell", "say",
    "said", "thing", "things", "way", "yes", "yeah", "ok", "okay", "sure",
    "thanks", "thank", "please", "hi", "hey", "hello", "well", "let",
    "something", "anything", "everything", "really", "much", "only",
    "going", "one", "two", "don", "doesn", "didn", "won", "isn", "aren",
    "wasn", "weren", "hasn", "haven", "hadn", "shouldn", "wouldn", "couldn",
    "its", "it's", "i'm", "i'd", "i'll", "i've", "we're", "we've",
    "you're", "you've", "you'd", "you'll", "he's", "she's", "they're",
    "they've", "they'd", "that's", "there's", "here's", "what's", "who's",
    "let's", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "shouldn't", "wouldn't", "couldn't", "still", "right", "back",
    "even", "work", "try", "use", "new", "good", "first", "last",
})

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class RelationshipInsight:
    """Structural patterns extracted from conversation history. Frozen, pure data."""

    recurring_topics: Dict[str, int] = field(default_factory=dict)
    avg_message_length_overall: float = 0.0
    session_frequency_days: float = 0.0
    preferred_time_of_day: str = ""
    avg_session_length_messages: float = 0.0
    rapport_direction: str = ""
    confidence_direction: str = ""
    primary_domain: str = ""
    compilation_count: int = 0
    domains_explored: List[str] = field(default_factory=list)
    conversational_ratio: float = 0.0
    urgency_signal: str = ""             # "urgent" | "exploratory" | "neutral"
    tone_profile: str = ""               # "terse" | "verbose" | "questioning" | ""
    user_skill_estimate: str = ""        # "beginner" | "intermediate" | "expert" | ""
    relationship_stage: str = "new"
    sessions_analyzed: int = 0
    messages_analyzed: int = 0
    computed_at: float = 0.0


def extract_topic_keywords(
    messages: List[Tuple[str, str, float, str]],
) -> Dict[str, int]:
    """Extract recurring topic keywords from user messages.

    Args:
        messages: List of (role, content, timestamp, session_id) tuples.

    Returns:
        Dict of keyword -> frequency, min 2 occurrences, top 10.
    """
    counter: Counter = Counter()
    for role, content, _ts, _sid in messages:
        if role != "user":
            continue
        # Skip slash commands
        stripped = content.strip()
        if stripped.startswith("/"):
            continue
        words = _WORD_RE.findall(stripped.lower())
        for word in words:
            if len(word) >= 4 and word not in _STOPWORDS:
                counter[word] += 1

    # Filter: min 2 occurrences, top 10
    return dict(counter.most_common(10) if counter else [])


def _classify_time_of_day(hour: int) -> str:
    """Classify an hour (0-23) into a time-of-day bucket."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def extract_preferred_time(
    session_timestamps: List[float],
) -> str:
    """Determine preferred time of day from session start timestamps.

    Needs 3+ sessions. Returns bucket if >50% of sessions fall in it.
    Returns "" if no clear preference.
    """
    if len(session_timestamps) < 3:
        return ""

    buckets: Counter = Counter()
    for ts in session_timestamps:
        lt = time.localtime(ts)
        buckets[_classify_time_of_day(lt.tm_hour)] += 1

    total = len(session_timestamps)
    most_common = buckets.most_common(1)
    if most_common and most_common[0][1] > total / 2:
        return most_common[0][0]
    return ""


def compute_relationship_stage(
    session_count: int,
    message_count: int,
    rapport_baseline: float = 0.0,
) -> str:
    """Determine relationship stage from session/message counts and rapport.

    Stages:
        new: 0-2 sessions
        building: 3-8 sessions
        established: 9-20 sessions
        deep: 20+ sessions AND rapport >= 0.5
    """
    if session_count <= 2:
        return "new"
    elif session_count <= 8:
        return "building"
    elif session_count <= 20:
        return "established"
    else:
        if rapport_baseline >= 0.5:
            return "deep"
        return "established"


_URGENCY_WORDS = frozenset({
    "urgent", "asap", "now", "immediately", "quickly", "hurry",
    "deadline", "critical", "emergency", "rush", "today", "tonight",
})
_EXPLORATION_WORDS = frozenset({
    "explore", "experiment", "what if", "maybe", "wonder",
    "consider", "brainstorm", "imagine", "prototype", "sketch",
})


def _detect_communication_patterns(
    messages: List[Tuple[str, str, float, str]],
) -> Tuple[str, str]:
    """Detect urgency signal and tone profile from user messages.

    Returns (urgency_signal, tone_profile).
    urgency_signal: "urgent" | "exploratory" | "neutral"
    tone_profile: "terse" | "verbose" | "questioning" | ""
    """
    user_msgs = [c for r, c, _, _ in messages if r == "user" and not c.strip().startswith("/")]
    if not user_msgs:
        return "", ""

    # Urgency vs exploration
    urgent_count = 0
    explore_count = 0
    question_count = 0
    total_len = 0
    for content in user_msgs:
        lower = content.lower()
        words = frozenset(lower.split())
        if words & _URGENCY_WORDS:
            urgent_count += 1
        if words & _EXPLORATION_WORDS:
            explore_count += 1
        if content.rstrip().endswith("?"):
            question_count += 1
        total_len += len(content)

    n = len(user_msgs)
    urgency = "neutral"
    if urgent_count > n * 0.2:
        urgency = "urgent"
    elif explore_count > n * 0.2:
        urgency = "exploratory"

    # Tone profile from structural patterns
    avg_len = total_len / n
    tone = ""
    if question_count > n * 0.4:
        tone = "questioning"
    elif avg_len < 40:
        tone = "terse"
    elif avg_len > 200:
        tone = "verbose"

    return urgency, tone


_TECHNICAL_WORDS = frozenset({
    "api", "database", "schema", "endpoint", "deploy", "pipeline",
    "architecture", "microservice", "kubernetes", "docker", "terraform",
    "async", "callback", "middleware", "websocket", "oauth", "jwt",
    "redis", "postgres", "graphql", "typescript", "rust", "compiler",
})


def _estimate_user_skill(
    messages: List[Tuple[str, str, float, str]],
    compilation_count: int,
    domains_explored: int,
) -> str:
    """Estimate user skill level from vocabulary and history.

    Returns "beginner" | "intermediate" | "expert" | "".
    """
    user_msgs = [c for r, c, _, _ in messages if r == "user" and not c.strip().startswith("/")]
    if len(user_msgs) < 5:
        return ""

    # Count technical vocabulary density
    tech_hits = 0
    total_words = 0
    for content in user_msgs:
        words = frozenset(content.lower().split())
        tech_hits += len(words & _TECHNICAL_WORDS)
        total_words += len(words)

    tech_density = tech_hits / max(total_words, 1)

    # Score: vocabulary + domain breadth + compilation experience
    score = 0.0
    if tech_density >= 0.03:
        score += 0.4
    elif tech_density >= 0.01:
        score += 0.2
    if domains_explored >= 3:
        score += 0.3
    elif domains_explored >= 1:
        score += 0.15
    if compilation_count >= 10:
        score += 0.3
    elif compilation_count >= 3:
        score += 0.15

    if score >= 0.6:
        return "expert"
    elif score >= 0.3:
        return "intermediate"
    return "beginner"


def extract_relationship_insights(
    all_messages: List[Tuple[str, str, float, str]],
    sessions: List[Dict],
    sense_memory: Optional[Dict] = None,
    corpus_summary: Optional[Dict] = None,
) -> RelationshipInsight:
    """Extract structural relationship patterns from accumulated data.

    Args:
        all_messages: List of (role, content, timestamp, session_id) tuples.
        sessions: List of session dicts with session_id, message_count, first_message, last_message.
        sense_memory: Optional dict with rapport_trend, confidence_trend, peak_rapport.
        corpus_summary: Optional dict with total_compilations, domains, etc.

    Returns:
        Frozen RelationshipInsight with provenance counts.
    """
    now = time.time()
    session_count = len(sessions)
    message_count = len(all_messages)

    if message_count == 0:
        return RelationshipInsight(
            relationship_stage="new",
            sessions_analyzed=0,
            messages_analyzed=0,
            computed_at=now,
        )

    # Topic extraction
    recurring = extract_topic_keywords(all_messages)
    # Filter to min 2 occurrences
    recurring = {k: v for k, v in recurring.items() if v >= 2}

    # Average message length (user messages only)
    user_msgs = [(r, c, t, s) for r, c, t, s in all_messages if r == "user"]
    avg_len = 0.0
    if user_msgs:
        avg_len = sum(len(c) for _, c, _, _ in user_msgs) / len(user_msgs)

    # Session frequency
    session_freq = 0.0
    session_starts = sorted([s["first_message"] for s in sessions if s.get("first_message")])
    if len(session_starts) >= 2:
        total_span = session_starts[-1] - session_starts[0]
        session_freq = (total_span / 86400) / (len(session_starts) - 1)

    # Preferred time
    pref_time = extract_preferred_time(session_starts)

    # Average session length
    avg_session_len = 0.0
    if sessions:
        total_msgs_in_sessions = sum(s.get("message_count", 0) for s in sessions)
        avg_session_len = total_msgs_in_sessions / len(sessions)

    # Sense trajectory
    rapport_dir = ""
    confidence_dir = ""
    rapport_baseline = 0.0
    if sense_memory:
        rt = sense_memory.get("rapport_trend", 0.0)
        ct = sense_memory.get("confidence_trend", 0.0)
        rapport_baseline = sense_memory.get("peak_rapport", 0.0)
        if rt > 0.05:
            rapport_dir = "growing"
        elif rt < -0.05:
            rapport_dir = "declining"
        elif session_count >= 3:
            rapport_dir = "stable"
        if ct > 0.05:
            confidence_dir = "growing"
        elif ct < -0.05:
            confidence_dir = "declining"
        elif session_count >= 3:
            confidence_dir = "stable"

    # Corpus data
    primary_domain = ""
    compilation_count = 0
    domains_explored: List[str] = []
    if corpus_summary:
        compilation_count = corpus_summary.get("total_compilations", 0)
        domains = corpus_summary.get("domains", {})
        if domains:
            sorted_domains = sorted(domains.items(), key=lambda x: -x[1])
            primary_domain = sorted_domains[0][0]
            domains_explored = [d for d, _ in sorted_domains]

    # Conversational ratio: % of user messages that are NOT slash commands
    if user_msgs:
        chat_msgs = sum(1 for _, c, _, _ in user_msgs if not c.strip().startswith("/"))
        conv_ratio = chat_msgs / len(user_msgs)
    else:
        conv_ratio = 0.0

    # Communication patterns: urgency + tone
    urgency_signal, tone_profile = _detect_communication_patterns(all_messages)

    # User skill estimation
    user_skill = _estimate_user_skill(all_messages, compilation_count, len(domains_explored))

    # Relationship stage
    stage = compute_relationship_stage(session_count, message_count, rapport_baseline)

    return RelationshipInsight(
        recurring_topics=recurring,
        avg_message_length_overall=avg_len,
        session_frequency_days=session_freq,
        preferred_time_of_day=pref_time,
        avg_session_length_messages=avg_session_len,
        rapport_direction=rapport_dir,
        confidence_direction=confidence_dir,
        primary_domain=primary_domain,
        compilation_count=compilation_count,
        domains_explored=domains_explored,
        conversational_ratio=conv_ratio,
        urgency_signal=urgency_signal,
        tone_profile=tone_profile,
        user_skill_estimate=user_skill,
        relationship_stage=stage,
        sessions_analyzed=session_count,
        messages_analyzed=message_count,
        computed_at=now,
    )


def synthesize_relationship_narrative(insight: RelationshipInsight) -> str:
    """Synthesize a 1-3 sentence narrative from relationship insight.

    Rules:
    - Never fabricates. Each part only renders if data supports it.
    - "new" stage → minimal output.
    - Sparse data → shorter narrative.
    - Every claim traces to data in the insight.
    """
    if insight.relationship_stage == "new":
        return "New user. No prior history."

    parts: List[str] = []

    # Stage label
    stage_labels = {
        "building": "Building relationship",
        "established": "Established relationship",
        "deep": "Deep relationship",
    }
    stage_str = stage_labels.get(insight.relationship_stage, "")
    if stage_str:
        parts.append(stage_str)

    # Primary domain focus
    if insight.primary_domain:
        parts.append(f"works primarily on {insight.primary_domain} projects")

    # Time pattern
    if insight.preferred_time_of_day:
        parts.append(f"sessions cluster in {insight.preferred_time_of_day}")

    # First sentence: join stage + domain + time
    first = ""
    if parts:
        first = parts[0]
        if len(parts) > 1:
            first += ". " + ". ".join(p.capitalize() if not p[0].isupper() else p for p in parts[1:])
        first += "."

    sentences: List[str] = []
    if first:
        sentences.append(first)

    # Conversational style + tone
    if insight.messages_analyzed >= 10:
        if insight.conversational_ratio >= 0.8:
            sentences.append("Primarily conversational.")
        elif insight.conversational_ratio <= 0.3:
            sentences.append("Primarily task-oriented.")

    # Communication patterns
    if insight.urgency_signal == "urgent":
        sentences.append("Currently working under pressure.")
    elif insight.urgency_signal == "exploratory":
        sentences.append("Currently in exploration mode.")
    if insight.tone_profile == "terse":
        sentences.append("Prefers brief exchanges.")
    elif insight.tone_profile == "questioning":
        sentences.append("Asks many questions — seeking understanding.")

    # User skill level (challenge calibration)
    if insight.user_skill_estimate == "expert":
        sentences.append("Expert-level user — challenge, don't simplify.")
    elif insight.user_skill_estimate == "beginner":
        sentences.append("Newer user — explain more, assume less.")

    # Rapport direction
    if insight.rapport_direction == "growing":
        sentences.append("Rapport growing.")
    elif insight.rapport_direction == "declining":
        sentences.append("Rapport declining.")

    # Recurring topics (add only if we have them)
    if insight.recurring_topics:
        top_topics = list(insight.recurring_topics.keys())[:5]
        sentences.append(f"Recurring interests: {', '.join(top_topics)}.")

    # Keep to 1-3 sentences
    narrative = " ".join(sentences[:3])
    return narrative
