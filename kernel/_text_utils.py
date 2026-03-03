"""
kernel/_text_utils.py — Shared text processing utilities for the kernel.

LEAF module. Provides stemming, normalization, stop words, and synonym
expansion used by closed_loop.py, content_validator.py, and navigator.py.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

STOP_WORDS = frozenset({
    # Determiners, articles, copulas
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    # Auxiliaries
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    # Conjunctions, negation
    "and", "but", "or", "nor", "not", "no", "so", "yet", "for",
    # Prepositions
    "in", "on", "at", "to", "of", "by", "with", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    # Pronouns
    "it", "its", "this", "that", "these", "those", "my", "your", "his",
    "her", "our", "their", "which", "who", "whom", "what", "where", "when",
    "them", "itself", "himself", "herself", "themselves", "ourselves",
    # Quantifiers, degree words
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "than", "too", "very",
    "just", "about", "also", "any", "here", "there", "if", "up",
    "everything", "something", "anything", "nothing",
    "better", "best", "worse", "worst",
    "new", "like", "well", "back", "still", "even", "way",
    # Common verbs (never entities — these are actions, not nouns)
    "should", "using", "used", "use", "uses",
    "get", "got", "take", "taken", "make", "made", "give", "gave",
    "allow", "allows", "allowed", "allowing",
    "create", "creates", "created", "creating",
    "manage", "manages", "managed", "managing",
    "handle", "handles", "handled", "handling",
    "enable", "enables", "enabled", "enabling",
    "support", "supports", "supported", "supporting",
    "provide", "provides", "provided", "providing",
    "require", "requires", "required", "requiring",
    "include", "includes", "included", "including",
    "ensure", "ensures", "ensured", "ensuring",
    "track", "tracks", "tracked", "tracking",
    "store", "stores", "stored", "storing",
    "display", "displays", "displayed", "displaying",
    "process", "processes", "processed", "processing",
    "send", "sends", "sent", "sending",
    "receive", "receives", "received", "receiving",
    "show", "shows", "showed", "showing",
    "help", "helps", "helped", "helping",
    "work", "works", "worked", "working",
    "want", "wants", "wanted", "wanting",
    "keep", "keeps", "kept", "keeping",
    "start", "starts", "started", "starting",
    "stop", "stops", "stopped", "stopping",
    "run", "runs", "running",
    "set", "sets", "setting",
    "add", "adds", "added", "adding",
    "remove", "removes", "removed", "removing",
    "update", "updates", "updated", "updating",
    "delete", "deletes", "deleted", "deleting",
    "check", "checks", "checked", "checking",
    "find", "finds", "found", "finding",
    "look", "looks", "looked", "looking",
    "define", "defines", "defined", "defining",
    "call", "calls", "called", "calling",
    "return", "returns", "returned", "returning",
    "move", "moves", "moved", "moving",
    "change", "changes", "changed", "changing",
    "connect", "connects", "connected", "connecting",
    "follow", "follows", "followed", "following",
    "read", "reads", "reading",
    "write", "writes", "written", "writing",
    "open", "opens", "opened", "opening",
    "close", "closes", "closed", "closing",
    "load", "loads", "loaded", "loading",
    "save", "saves", "saved", "saving",
    # Common qualifiers (never entities)
    "able", "available", "based", "basic", "certain", "clear",
    "complete", "current", "custom", "daily", "default", "different",
    "easy", "entire", "existing", "external", "final", "first",
    "full", "general", "given", "good", "great", "high", "important",
    "internal", "last", "late", "local", "long", "main", "many",
    "multiple", "next", "online", "open", "original", "overall",
    "particular", "personal", "possible", "potential", "previous",
    "primary", "proper", "public", "real", "recent", "related",
    "relevant", "right", "simple", "single", "small", "social",
    "special", "specific", "standard", "sure", "total", "true",
    "unique", "various", "whole", "within", "without",
    # Meta terms (describe the request, not the domain)
    "feature", "features", "functionality", "option", "options",
    "tool", "tools", "type", "types", "data", "info", "information",
    "detail", "details", "list", "item", "items", "part", "parts",
    "thing", "things", "stuff", "kind", "form", "example",
    "people", "person",
    # Request framing (describe the ask, not the system)
    "comprehensive", "platform", "application", "app", "program",
    "solution", "product",
})


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------

def stem(word: str) -> str:
    """Lightweight suffix-stripping stem. No external deps.

    Handles: -ing, -tion, -sion, -ment, -ness, -able, -ible, -ful,
    -ly, -ed, -er, -es, -s (if word > 4 chars).
    Applied iteratively until stable.
    """
    prev = word
    for _ in range(3):
        stemmed = _stem_once(prev)
        if stemmed == prev:
            break
        prev = stemmed
    return prev


def _stem_once(word: str) -> str:
    """Single pass of suffix stripping."""
    if len(word) <= 4:
        return word
    # Long suffixes require residual >= 4 to avoid over-stemming
    # (e.g. "creation" → "cre" is too aggressive)
    for suffix in ("ation", "tion", "sion"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    for suffix in ("ment", "ness", "able", "ible", "ful", "ing", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    for suffix in ("ed", "er", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("s") and len(word) == 4:
        return word[:-1]
    return word


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_word(word: str) -> str:
    """Normalize a single word for comparison."""
    return word.lower().strip(".,;:!?\"'()[]{}").strip()


def _split_camel(word: str) -> list[str]:
    """Split PascalCase/camelCase into constituent words.

    'AuthService' → ['Auth', 'Service']
    'tokenManager' → ['token', 'Manager']
    'HTMLParser' → ['HTML', 'Parser']
    """
    import re
    # Insert boundary before uppercase-followed-by-lowercase (camelCase)
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
    # Insert boundary between consecutive uppercase and uppercase+lowercase (acronyms)
    parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts)
    return parts.split()


def normalize_tokens(text: str) -> set[str]:
    """Tokenize, normalize, and stem text for comparison.

    Splits on whitespace, hyphens, underscores AND camelCase/PascalCase
    boundaries so 'AuthService' → {"auth", "servic"} and
    'TokenManager' → {"token", "manag"}.

    Tokens in synonym clusters survive stop-word filtering — "process",
    "store", "handle" etc. are domain-meaningful even though they're
    common verbs.
    """
    import re
    words = re.split(r'[\s\-_/]+', text)
    tokens = set()
    for w in words:
        # Split camelCase/PascalCase before normalizing
        sub_words = _split_camel(w) if any(c.isupper() for c in w[1:]) else [w]
        for sw in sub_words:
            normalized = normalize_word(sw)
            if not normalized or len(normalized) < 2:
                continue
            stemmed = stem(normalized)
            # Keep token if: not a stop word, OR maps to a synonym cluster
            if normalized not in STOP_WORDS or stemmed in _SYNONYM_INDEX:
                tokens.add(stemmed)
    return tokens


# ---------------------------------------------------------------------------
# Synonym clusters — hand-curated for software domain
# ---------------------------------------------------------------------------

SYNONYM_CLUSTERS: dict[str, frozenset[str]] = {
    "auth": frozenset({"authentication", "login", "credential", "signin", "signon", "authenticate"}),
    "user": frozenset({"account", "profile", "member", "participant", "operator"}),
    "data": frozenset({"database", "storage", "store", "persist", "repository", "datastore"}),
    "api": frozenset({"endpoint", "route", "interface", "service", "gateway"}),
    "error": frozenset({"exception", "failure", "fault", "crash", "bug"}),
    "config": frozenset({"configuration", "setting", "preference", "option", "parameter"}),
    "msg": frozenset({"message", "notification", "alert", "signal", "event"}),
    "perm": frozenset({"permission", "authorization", "access", "privilege", "role"}),
    "valid": frozenset({"validation", "verify", "check", "assert", "confirm", "validate"}),
    "cache": frozenset({"memoize", "buffer", "preload", "prefetch"}),
    "log": frozenset({"logging", "audit", "trace", "monitor", "record"}),
    "queue": frozenset({"buffer", "backlog", "pipeline", "stream"}),
    "test": frozenset({"testing", "spec", "assertion", "fixture", "mock"}),
    "deploy": frozenset({"deployment", "release", "publish", "ship", "rollout"}),
    "task": frozenset({"job", "work", "assignment", "ticket", "todo"}),
    "search": frozenset({"query", "lookup", "find", "filter", "browse"}),
    "file": frozenset({"document", "upload", "attachment", "asset", "artifact"}),
    "ui": frozenset({"interface", "frontend", "display", "view", "screen", "widget"}),
    "state": frozenset({"status", "condition", "phase", "mode", "stage"}),
    "compute": frozenset({"calculate", "process", "transform", "convert", "derive"}),
    "connect": frozenset({"link", "associate", "bind", "attach", "wire"}),
    "delete": frozenset({"remove", "destroy", "purge", "drop", "discard"}),
    "create": frozenset({"generate", "produce", "build", "construct", "initialize"}),
    "update": frozenset({"modify", "change", "edit", "patch", "mutate"}),
    "read": frozenset({"fetch", "retrieve", "load", "get", "access"}),
    "send": frozenset({"transmit", "emit", "dispatch", "publish", "broadcast"}),
    "receive": frozenset({"consume", "accept", "handle", "ingest", "subscribe"}),
    "schedule": frozenset({"timer", "cron", "interval", "periodic", "recurring"}),
    "encrypt": frozenset({"cipher", "hash", "secure", "protect", "obfuscate"}),
    "pay": frozenset({"payment", "billing", "charge", "invoice", "transaction"}),
}

# Build reverse index: word → representative
_SYNONYM_INDEX: dict[str, str] = {}
for _rep, _cluster in SYNONYM_CLUSTERS.items():
    for _word in _cluster:
        _SYNONYM_INDEX[stem(normalize_word(_word))] = _rep
    _SYNONYM_INDEX[_rep] = _rep


def expand_synonyms(tokens: set[str]) -> set[str]:
    """Expand tokens with synonym representatives.

    For each token, if it matches a synonym cluster member, add the
    cluster representative. Original tokens are preserved.
    """
    expanded = set(tokens)
    for token in tokens:
        rep = _SYNONYM_INDEX.get(token)
        if rep:
            expanded.add(rep)
    return expanded


# ---------------------------------------------------------------------------
# Bigram support
# ---------------------------------------------------------------------------

def bigram_tokens(text: str) -> set[str]:
    """Extract stemmed bigrams from text.

    Bigrams capture multi-word concepts that unigrams miss:
    "login handler" → {"login_handl"} (stemmed bigram).
    """
    words = text.split()
    normalized = []
    for w in words:
        n = normalize_word(w)
        if n and len(n) >= 2 and n not in STOP_WORDS:
            normalized.append(stem(n))

    bigrams = set()
    for i in range(len(normalized) - 1):
        bigrams.add(f"{normalized[i]}_{normalized[i+1]}")
    return bigrams


# ---------------------------------------------------------------------------
# Semantic grouping for emergence detection
# ---------------------------------------------------------------------------

def semantic_jaccard(a: str, b: str) -> float:
    """Stemmed-token Jaccard similarity between two strings."""
    tokens_a = normalize_tokens(a)
    tokens_b = normalize_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0
