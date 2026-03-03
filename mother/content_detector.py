"""
Content-generation detection — pure-function module.

Detects when user input calls for long-form content generation
(blog posts, documentation, letters, tutorials, etc.) and returns
a prompt directive that overrides conversational brevity constraints.

LEAF module — no imports from core/, no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ContentSignal", "detect_content_request"]


@dataclass(frozen=True)
class ContentSignal:
    """Result of content-generation detection."""
    detected: bool = False
    content_type: str = ""
    directive: str = ""


# --- Detection vocabulary ---

_WRITING_VERBS = frozenset({
    "write", "draft", "compose", "create", "generate", "produce",
    "author", "prepare", "craft", "make", "develop", "formulate",
    "design", "outline", "sketch", "put", "lay",
})

# Ordered longest-first to prevent substring collisions
# ("newsletter" before "letter", "blog posts" before "blog post")
_CONTENT_NOUNS: tuple[tuple[str, str], ...] = (
    # --- Multi-word phrases first ---
    ("blog posts", "blog_post"),
    ("blog post", "blog_post"),
    ("press releases", "press_release"),
    ("press release", "press_release"),
    ("product descriptions", "product_description"),
    ("product description", "product_description"),
    ("white paper", "white_paper"),
    ("whitepaper", "white_paper"),
    ("case studies", "case_study"),
    ("case study", "case_study"),
    ("landing pages", "landing_page"),
    ("landing page", "landing_page"),
    ("business plan", "business_plan"),
    ("business plans", "business_plan"),
    ("business case", "business_case"),
    ("business cases", "business_case"),
    ("pitch deck", "pitch_deck"),
    ("pitch decks", "pitch_deck"),
    ("cover letter", "cover_letter"),
    ("cover letters", "cover_letter"),
    ("job descriptions", "job_description"),
    ("job description", "job_description"),
    ("terms of service", "terms_of_service"),
    ("privacy policy", "privacy_policy"),
    ("privacy policies", "privacy_policy"),
    ("social media posts", "social_media_post"),
    ("social media post", "social_media_post"),
    ("social posts", "social_media_post"),
    ("linkedin posts", "social_media_post"),
    ("linkedin post", "social_media_post"),
    ("ad copy", "ad_copy"),
    ("knowledge base", "knowledge_base"),
    ("user stories", "user_story"),
    ("user story", "user_story"),
    ("release notes", "release_notes"),
    ("talking points", "talking_points"),
    ("investor update", "investor_update"),
    ("investor updates", "investor_update"),
    ("board deck", "board_deck"),
    ("strategy doc", "strategy_document"),
    ("strategy document", "strategy_document"),
    ("mission statement", "mission_statement"),
    ("vision statement", "vision_statement"),
    ("creative brief", "creative_brief"),
    ("creative briefs", "creative_brief"),
    ("onboarding doc", "onboarding_document"),
    ("onboarding docs", "onboarding_document"),
    ("onboarding guide", "onboarding_document"),
    ("deep dive", "deep_dive"),
    # --- Longer single words before substrings ---
    ("newsletters", "newsletter"),
    ("newsletter", "newsletter"),
    ("documentation", "documentation"),
    ("specification", "specification"),
    ("announcements", "announcement"),
    ("announcement", "announcement"),
    ("descriptions", "description"),
    ("description", "description"),
    ("testimonials", "testimonial"),
    ("testimonial", "testimonial"),
    # --- Regular nouns (plural before singular) ---
    ("articles", "article"),
    ("article", "article"),
    ("tutorials", "tutorial"),
    ("tutorial", "tutorial"),
    ("guides", "guide"),
    ("guide", "guide"),
    ("essays", "essay"),
    ("essay", "essay"),
    ("emails", "email"),
    ("email", "email"),
    ("tweets", "tweet"),
    ("tweet", "tweet"),
    ("threads", "thread"),
    ("thread", "thread"),
    ("captions", "caption"),
    ("caption", "caption"),
    ("taglines", "tagline"),
    ("tagline", "tagline"),
    ("slogans", "slogan"),
    ("slogan", "slogan"),
    ("docs", "documentation"),
    ("readme", "documentation"),
    ("proposals", "proposal"),
    ("proposal", "proposal"),
    ("contracts", "contract"),
    ("contract", "contract"),
    ("letters", "letter"),
    ("letter", "letter"),
    ("reports", "report"),
    ("report", "report"),
    ("summaries", "summary"),
    ("summary", "summary"),
    ("outlines", "outline"),
    ("outline", "outline"),
    ("runbooks", "runbook"),
    ("runbook", "runbook"),
    ("playbooks", "playbook"),
    ("playbook", "playbook"),
    ("briefs", "brief"),
    ("brief", "brief"),
    ("changelogs", "changelog"),
    ("changelog", "changelog"),
    ("resumes", "resume"),
    ("resume", "resume"),
    ("spec", "specification"),
    ("faqs", "faq"),
    ("faq", "faq"),
    ("sops", "sop"),
    ("sop", "sop"),
    ("okrs", "okr"),
    ("rfps", "rfp"),
    ("rfp", "rfp"),
    ("copy", "copy"),
    ("script", "script"),
    ("memo", "memo"),
    ("memos", "memo"),
    ("bio", "bio"),
    ("bios", "bio"),
    ("speech", "speech"),
    ("speeches", "speech"),
    ("manifesto", "manifesto"),
    ("review", "review"),
    ("reviews", "review"),
    ("cv", "resume"),
)

# Checked with word-boundary regex (\b...\b) to prevent substring
# collisions ("some" in "someone", "long" in "along", etc.)
_QUANTITY_LENGTH_SIGNALS: tuple[str, ...] = (
    "a few", "a couple", "a batch", "a series", "a set of", "a list of",
    "some", "several", "multiple", "many",
    "detailed", "comprehensive", "in-depth", "in full",
    "thorough", "extensive", "complete",
    "long", "full",
    "fleshed out", "flesh out", "expanded", "deep dive",
)

# Pre-compiled word-boundary patterns for quantity/length signals
_QUANTITY_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(r"\b" + re.escape(s) + r"\b") for s in _QUANTITY_LENGTH_SIGNALS
)

# Phrases that indicate code, not content — exclusions (word-level check)
_CODE_NOUNS = frozenset({
    "code", "function", "class", "method", "module", "script",
    "test", "tests", "endpoint", "api", "component", "hook",
    "bug", "fix", "feature", "app", "application", "system",
    "server", "database", "schema", "migration", "query",
    "program", "algorithm", "library", "package",
    "website", "site", "page", "frontend", "backend",
    "service", "microservice", "pipeline", "workflow",
    "dashboard", "interface", "widget", "plugin",
    "bot", "chatbot", "tool", "cli", "sdk",
})

# Content nouns that have code-domain meanings. When one of these matches
# AND the message also contains code-context indicators, reject detection.
_AMBIGUOUS_NOUNS: dict[str, frozenset[str]] = {
    "script": frozenset({
        "python", "javascript", "bash", "shell", "js", "ts",
        "typescript", "node", "ruby", "perl", "php",
    }),
    "thread": frozenset({
        "pool", "safe", "concurrent", "mutex", "lock", "async",
        "threading", "parallel", "executor", "worker", "daemon",
    }),
    "review": frozenset({
        "code", "pr", "merge", "diff", "commit", "branch",
        "pull",
    }),
}


def _tokenize(text: str) -> list[str]:
    """Lowercase split for word-level matching."""
    return text.lower().split()


def _has_quantity_signal(lower: str) -> bool:
    """Check for quantity/length signals using word-boundary matching."""
    return any(p.search(lower) for p in _QUANTITY_PATTERNS)


def detect_content_request(message: str) -> ContentSignal:
    """Detect content-generation intent from a user message.

    Returns ContentSignal with detected=True when the message
    requests written content (blog posts, docs, letters, etc.).

    Detection requires a writing verb (or quantity/length signal)
    AND a content noun. Code-related requests are excluded.
    """
    if not message or not message.strip():
        return ContentSignal()

    lower = message.lower()
    words = _tokenize(message)

    # Check for code exclusions (word-level)
    has_code_noun = any(cn in words for cn in _CODE_NOUNS)

    # Find writing verbs (word-level)
    has_writing_verb = any(v in words for v in _WRITING_VERBS)

    # Find quantity/length signals (word-boundary regex)
    has_quantity_length = _has_quantity_signal(lower)

    # Need at least one trigger (verb or quantity/length)
    if not has_writing_verb and not has_quantity_length:
        return ContentSignal()

    # Find content nouns (match longest first — "blog post" before "blog")
    matched_type = ""
    for phrase, content_type in _CONTENT_NOUNS:
        if phrase in lower:
            matched_type = content_type
            break

    if not matched_type:
        return ContentSignal()

    # Ambiguous noun check — content nouns that have code-domain meanings.
    # Fires regardless of quantity signal presence.
    if matched_type in _AMBIGUOUS_NOUNS:
        indicators = _AMBIGUOUS_NOUNS[matched_type]
        if any(ind in words for ind in indicators):
            return ContentSignal()

    directive = (
        f"[Response Mode: Content Generation]\n"
        f"The user is requesting written content ({matched_type}). "
        f"Produce the actual content directly in your response — "
        f"do not just acknowledge the request or offer to help. "
        f"Write the full content. Aim for substantial, complete output. "
        f"The [VOICE] tag length guidance does not apply to content "
        f"generation — write as long as the content requires. "
        f"Structure with headings, paragraphs, and formatting as "
        f"appropriate for the content type."
    )

    return ContentSignal(
        detected=True,
        content_type=matched_type,
        directive=directive,
    )
