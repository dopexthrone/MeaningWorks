"""
Input Quality Analyzer — Phase 9.1

Prevents wasting LLM tokens on vague input. Pure heuristics, no LLM calls.

Scoring dimensions:
- Length: raw character/word count
- Specificity: domain terms, proper nouns, technical vocabulary
- Noun/verb density: ratio of meaningful words to filler
- Actionability: presence of actors, actions, goals

Thresholds:
- REJECT_THRESHOLD (0.15): Below this, reject outright
- WARN_THRESHOLD (0.35): Below this, warn but proceed
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QualityScore:
    """Result of input quality analysis."""

    overall: float  # 0.0-1.0 composite score
    length_score: float = 0.0  # word count contribution
    specificity_score: float = 0.0  # domain terms, proper nouns
    density_score: float = 0.0  # noun/verb ratio
    actionability_score: float = 0.0  # actors + actions + goals
    details: List[str] = field(default_factory=list)  # human-readable breakdown
    suggestion: str = ""  # what to improve

    @property
    def is_acceptable(self) -> bool:
        """True if score is above reject threshold."""
        return self.overall >= InputQualityAnalyzer.REJECT_THRESHOLD

    @property
    def is_hollow(self) -> bool:
        """True if score is in the hollow zone (above reject, below hollow threshold)."""
        return (
            self.overall >= InputQualityAnalyzer.REJECT_THRESHOLD
            and self.overall < InputQualityAnalyzer.HOLLOW_THRESHOLD
        )

    @property
    def has_warnings(self) -> bool:
        """True if score is below warn threshold but above reject."""
        return self.is_acceptable and self.overall < InputQualityAnalyzer.WARN_THRESHOLD


# Common filler/stop words that don't add specificity
_FILLER_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that",
    "this", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "these", "those", "am", "about", "up",
    "make", "like", "want", "need", "something", "thing", "stuff",
    "get", "got", "also", "really", "please", "help", "build",
})

# Action verbs that indicate the user has thought about what the system does
_ACTION_VERBS = frozenset({
    # Business operations
    "manage", "track", "schedule", "book", "create", "send", "receive",
    "process", "validate", "authenticate", "authorize", "notify", "search",
    "filter", "sort", "export", "import", "generate", "analyze", "monitor",
    "report", "assign", "approve", "reject", "review", "submit", "publish",
    "archive", "delete", "update", "edit", "configure", "register", "login",
    "logout", "purchase", "checkout", "ship", "deliver", "invoice", "pay",
    "refund", "subscribe", "unsubscribe", "upload", "download", "sync",
    "calculate", "convert", "transform", "aggregate", "display", "render",
    "route", "dispatch", "queue", "retry", "log", "audit", "encrypt",
    "decrypt", "compress", "decompress", "cache", "index", "migrate",
    "deploy", "rollback", "scale", "balance", "throttle", "rate-limit",
    # Technical / capability actions
    "browse", "fetch", "parse", "crawl", "scrape", "extract", "request",
    "connect", "stream", "record", "capture", "listen", "transcribe",
    "recognize", "detect", "classify", "embed", "vectorize", "query",
    "infer", "compile", "execute", "invoke", "poll", "observe", "scan",
    "resolve", "negotiate", "handshake", "serialize", "deserialize",
    "encode", "decode", "hash", "sign", "verify", "wrap", "unwrap",
    "proxy", "forward", "relay", "bridge", "adapt", "translate",
    "orchestrate", "coordinate", "schedule", "spawn", "terminate",
})

# Domain indicator words — if these appear, the input has specificity
_DOMAIN_INDICATORS = frozenset({
    # People / actors
    "user", "admin", "customer", "client", "patient", "student", "teacher",
    "employee", "manager", "owner", "vendor", "supplier", "driver",
    "artist", "author", "reader", "viewer", "subscriber", "member",
    # Business objects
    "account", "profile", "session", "order", "invoice", "payment",
    "product", "item", "catalog", "inventory", "cart", "wishlist",
    "appointment", "booking", "reservation", "schedule", "calendar",
    "message", "notification", "email", "chat", "comment", "review",
    "rating", "feedback", "report", "dashboard", "analytics",
    "database", "api", "endpoint", "webhook", "queue", "pipeline",
    "workflow", "template", "document", "file", "image", "video",
    "permission", "role", "group", "team", "organization", "workspace",
    "project", "task", "ticket", "issue", "milestone", "sprint",
    "recipe", "ingredient", "menu", "restaurant", "delivery",
    "course", "lesson", "quiz", "grade", "certificate",
    "property", "listing", "tenant", "landlord", "lease",
    "flight", "hotel", "trip", "itinerary", "passenger",
    # Technical / capability objects
    "url", "html", "page", "browser", "capability", "agent", "model",
    "sensor", "microphone", "camera", "screen", "websocket", "http",
    "request", "response", "token", "embedding", "vector", "plugin",
    "extension", "hook", "bridge", "adapter", "service", "worker",
    "process", "thread", "stream", "feed", "channel", "socket",
    "server", "client", "proxy", "gateway", "router", "controller",
    "handler", "middleware", "protocol", "schema", "config", "registry",
    "cache", "store", "index", "log", "metric", "event", "signal",
    "command", "query", "mutation", "subscription", "cursor", "batch",
})


class InputQualityAnalyzer:
    """
    Analyzes input text quality using heuristics.

    No LLM calls — pure string analysis. Fast enough to run on every request.
    """

    REJECT_THRESHOLD = 0.15
    HOLLOW_THRESHOLD = 0.25
    WARN_THRESHOLD = 0.35

    def analyze(self, text: str) -> QualityScore:
        """
        Analyze input text and return a quality score.

        Args:
            text: The user's natural language description

        Returns:
            QualityScore with composite score and breakdown
        """
        if not text or not text.strip():
            return QualityScore(
                overall=0.0,
                details=["Input is empty"],
                suggestion="Describe what you want to build, including the domain, key actors, and main actions.",
            )

        text = text.strip()
        # Tokenize: split on whitespace, strip punctuation from edges
        raw_words = text.lower().split()
        words = [re.sub(r'^[^\w]+|[^\w]+$', '', w) for w in raw_words]
        words = [w for w in words if w]  # Remove empty after stripping
        word_count = len(words)

        # 1. Length score (0-1)
        length_score = self._score_length(word_count)

        # 2. Specificity score (0-1)
        specificity_score, specificity_details = self._score_specificity(words)

        # 3. Density score (0-1) — ratio of meaningful words
        density_score = self._score_density(words)

        # 4. Actionability score (0-1) — actors + verbs + goals
        actionability_score, action_details = self._score_actionability(text, words)

        # Composite: weighted average
        overall = (
            length_score * 0.15
            + specificity_score * 0.30
            + density_score * 0.20
            + actionability_score * 0.35
        )

        # Build details
        details = []
        if length_score < 0.3:
            details.append(f"Very short input ({word_count} words)")
        elif length_score < 0.6:
            details.append(f"Short input ({word_count} words)")

        details.extend(specificity_details)
        details.extend(action_details)

        if density_score < 0.3:
            details.append("Most words are filler with no domain meaning")

        # Build suggestion
        suggestion = self._build_suggestion(
            length_score, specificity_score, density_score, actionability_score,
            details=details,
        )

        return QualityScore(
            overall=round(overall, 3),
            length_score=round(length_score, 3),
            specificity_score=round(specificity_score, 3),
            density_score=round(density_score, 3),
            actionability_score=round(actionability_score, 3),
            details=details,
            suggestion=suggestion,
        )

    def _score_length(self, word_count: int) -> float:
        """Score based on word count. Sweet spot is 15-100 words."""
        if word_count == 0:
            return 0.0
        if word_count < 3:
            return 0.05
        if word_count < 6:
            return 0.15
        if word_count < 10:
            return 0.35
        if word_count < 15:
            return 0.55
        if word_count <= 100:
            return min(1.0, 0.6 + (word_count - 15) * 0.005)
        # Very long — still fine but no extra bonus
        return 1.0

    def _score_specificity(self, words: list) -> tuple:
        """Score based on domain-specific vocabulary."""
        details = []
        if not words:
            return 0.0, details

        # Match words including simple plural forms (strip trailing 's'/'es')
        unique_words = set(words)
        stemmed = set()
        for w in unique_words:
            stemmed.add(w)
            if w.endswith("es") and len(w) > 3:
                stemmed.add(w[:-2])
            if w.endswith("s") and len(w) > 2:
                stemmed.add(w[:-1])
            # Also add -ing -> base
            if w.endswith("ing") and len(w) > 5:
                stemmed.add(w[:-3])

        domain_hits = stemmed & _DOMAIN_INDICATORS
        action_hits = stemmed & _ACTION_VERBS

        meaningful_terms = len(domain_hits) + len(action_hits)

        if meaningful_terms == 0:
            details.append("No domain-specific terms found")
            return 0.0, details

        if len(domain_hits) == 0:
            details.append("No domain actors/entities detected")

        # Score: 1 term = 0.2, 3 terms = 0.5, 6+ terms = 1.0
        score = min(1.0, meaningful_terms * 0.15 + 0.05)
        return score, details

    def _score_density(self, words: list) -> float:
        """Score based on ratio of meaningful words to filler."""
        if not words:
            return 0.0

        meaningful = [w for w in words if w not in _FILLER_WORDS and len(w) > 2]
        ratio = len(meaningful) / len(words)

        # Map ratio to score: 0.2 ratio → 0.3 score, 0.5+ → 1.0
        if ratio < 0.1:
            return 0.1
        if ratio < 0.2:
            return 0.25
        return min(1.0, ratio * 1.8)

    def _stem_words(self, words: list) -> set:
        """Simple stemming: strip trailing s/es/ing for matching."""
        stemmed = set()
        for w in set(words):
            stemmed.add(w)
            if w.endswith("es") and len(w) > 3:
                stemmed.add(w[:-2])
            if w.endswith("s") and len(w) > 2:
                stemmed.add(w[:-1])
            if w.endswith("ing") and len(w) > 5:
                stemmed.add(w[:-3])
        return stemmed

    def _score_actionability(self, text: str, words: list) -> tuple:
        """Score based on presence of actors, actions, and goals."""
        details = []
        if not words:
            return 0.0, details

        score = 0.0
        stemmed = self._stem_words(words)

        # Check for actors (domain indicators that are person-like)
        actor_words = stemmed & {
            "user", "admin", "customer", "client", "patient", "student",
            "teacher", "employee", "manager", "owner", "vendor", "driver",
            "artist", "author", "reader", "viewer", "subscriber", "member",
        }
        if actor_words:
            score += 0.3
        else:
            details.append("No actors/users mentioned (who uses this?)")

        # Check for action verbs
        action_words = stemmed & _ACTION_VERBS
        if action_words:
            score += 0.35
        else:
            details.append("No specific actions described (what does it do?)")

        # Check for system/domain nouns
        domain_words = stemmed & _DOMAIN_INDICATORS
        if domain_words:
            score += 0.35
        else:
            details.append("No domain objects mentioned (what does it manage?)")

        return min(1.0, score), details

    def _build_suggestion(
        self,
        length: float,
        specificity: float,
        density: float,
        actionability: float,
        details: list | None = None,
    ) -> str:
        """Build a suggestion from what's actually missing in this input.

        No hardcoded domain examples. The details list already says what's
        absent — we just synthesize that into an actionable prompt.
        """
        missing = []

        if length < 0.5:
            missing.append("more detail about what you're building")
        if specificity < 0.5:
            missing.append("the domain and specific entities it manages")
        if actionability < 0.5:
            parts = []
            if details:
                if any("actor" in d.lower() for d in details):
                    parts.append("who uses it")
                if any("action" in d.lower() for d in details):
                    parts.append("what actions they perform")
                if any("domain object" in d.lower() for d in details):
                    parts.append("what objects the system manages")
            if parts:
                missing.append(", ".join(parts))
            else:
                missing.append("the actors and what they do")
        if density < 0.3 and not missing:
            missing.append("concrete terms instead of vague language")

        if not missing:
            return ""

        return "Be more specific. Add " + "; ".join(missing) + "."
