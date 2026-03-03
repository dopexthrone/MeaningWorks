"""Pre-dialogue interrogation — LEAF MODULE (stdlib + protocol_spec only).

When user input is ambiguous, sparse, or multi-domain, generate targeted
clarification questions BEFORE the expensive dialogue phase. Zero overhead
when input is clear.

Insertion point: after intent extraction, before persona generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.protocol_spec import PROTOCOL


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class InterrogationQuestion:
    id: str                    # "q1", "q2", etc.
    question: str              # User-level language
    category: str              # "domain_scope" | "missing_actors" | "missing_actions" | "missing_components" | "unknown_domain"
    options: list              # Suggested answers (empty for open-ended)
    required: bool = True


@dataclass
class InterrogationRequest:
    questions: list            # list[InterrogationQuestion]
    context: str               # Why we're asking
    original_intent: dict      # For reference
    quality_score: float
    domain_count: int = 1


@dataclass
class InterrogationResponse:
    answers: Dict[str, str]    # question_id -> answer text
    skip: bool = False


@dataclass
class InterrogationResult:
    triggered: bool
    questions_asked: int = 0
    answers_received: int = 0
    refined_description: Optional[str] = None
    domain_choice: Optional[str] = None
    should_fracture: bool = False
    skipped: bool = False


# =============================================================================
# DOMAIN LABEL EXTRACTION
# =============================================================================

_DOMAIN_PATTERNS: List[Tuple[str, str]] = [
    (r'\btattoo\b', "Tattoo studio"),
    (r'\btrading\b|\btrader\b', "Trading"),
    (r'\bsecurity\b|\bCCTV\b', "Security / CCTV"),
    (r'\bmarketing\b|\bSEO\b', "Marketing"),
    (r'\baccounting\b|\bfinancial\b|\binvoic', "Finance / Accounting"),
    (r'\bhealth\b|\bfitness\b|\bmedical\b', "Health / Fitness"),
    (r'\bshopping\b|\be-?commerce\b|\bmarketplace\b', "E-commerce / Marketplace"),
    (r'\bcrypto\b|\bblockchain\b', "Crypto / Blockchain"),
    (r'\bnetwork\b|\bmesh\b|\bIoT\b|\bdevice\b', "Network / IoT"),
    (r'\bvoice\b|\baudio\b|\bspeech\b', "Voice / Audio"),
    (r'\bcustomer\s*service\b|\bsupport\b|\bticket', "Customer Service"),
    (r'\bcalendar\b|\bscheduling\b|\bbooking\b', "Scheduling / Booking"),
    (r'\bsocial\b|\bcommunity\b', "Social / Community"),
]


def _extract_domain_labels(text: str) -> List[str]:
    """Extract domain labels from input text, plus meta-options."""
    text_lower = text.lower()
    labels = []
    for pattern, label in _DOMAIN_PATTERNS:
        if re.search(pattern, text_lower):
            labels.append(label)
    # Append meta-options
    if len(labels) >= 2:
        labels.append("All together as one system")
        labels.append("Separate compilations")
    return labels


# =============================================================================
# TRIGGER DETECTION
# =============================================================================

def should_interrogate(
    quality_score: float,
    intent: dict,
    domain_count: int,
    quality_details: List[str],
) -> Tuple[bool, List[str]]:
    """Decide whether to interrogate the user before dialogue.

    Returns (should_ask, list_of_reasons).

    Triggers on ANY of:
    1. domain_count >= multi_domain_threshold (multi-domain)
    2. quality_score < quality_interrogate_threshold (warn zone)
    3. No explicit_components AND quality_score < 0.5
    4. domain is "unknown" / "general" / empty
    """
    spec = PROTOCOL.interrogation
    reasons: List[str] = []

    if domain_count >= spec.multi_domain_threshold:
        reasons.append(f"multi_domain:{domain_count}")

    if quality_score < spec.quality_interrogate_threshold:
        reasons.append("quality_warn_zone")

    if not intent.get("explicit_components") and quality_score < 0.5:
        reasons.append("no_explicit_components")

    domain = intent.get("domain", "")
    if not domain or domain.lower() in ("unknown", "general"):
        reasons.append("unknown_domain")

    return (bool(reasons), reasons)


# =============================================================================
# QUESTION GENERATION (template-based, no LLM)
# =============================================================================

_DETAIL_MAP: Dict[str, Tuple[str, str]] = {
    # quality detail substring -> (category, question)
    "no actors": ("missing_actors", "Who will use this system? (e.g. customers, admins, devices)"),
    "no action": ("missing_actions", "What are the main actions or workflows?"),
    "no goals": ("missing_actions", "What should this system accomplish?"),
    "too short": ("missing_components", "Can you describe the main parts of what you're building?"),
    "vague": ("missing_components", "What are the key features or modules?"),
}


def _detail_to_question(detail_str: str) -> Optional[InterrogationQuestion]:
    """Convert a quality analyzer detail string into a user-facing question."""
    detail_lower = detail_str.lower()
    for substring, (category, question) in _DETAIL_MAP.items():
        if substring in detail_lower:
            return InterrogationQuestion(
                id="",  # assigned later
                question=question,
                category=category,
                options=[],
            )
    return None


def generate_questions(
    reasons: List[str],
    intent: dict,
    quality_details: List[str],
    domain_count: int,
    input_text: str,
) -> List[InterrogationQuestion]:
    """Generate targeted questions from trigger reasons.

    Template-based — zero LLM cost. Capped at max_questions.
    """
    spec = PROTOCOL.interrogation
    questions: List[InterrogationQuestion] = []
    seen_categories: set = set()

    for reason in reasons:
        if reason.startswith("multi_domain:"):
            if "domain_scope" not in seen_categories:
                labels = _extract_domain_labels(input_text)
                questions.append(InterrogationQuestion(
                    id="",
                    question=f"Your description spans {domain_count} areas. Which should I focus on?",
                    category="domain_scope",
                    options=labels if labels else [],
                ))
                seen_categories.add("domain_scope")

        elif reason == "quality_warn_zone":
            for detail in quality_details:
                q = _detail_to_question(detail)
                if q and q.category not in seen_categories:
                    questions.append(q)
                    seen_categories.add(q.category)

        elif reason == "no_explicit_components":
            if "missing_components" not in seen_categories:
                questions.append(InterrogationQuestion(
                    id="",
                    question="What are the main parts of what you're building?",
                    category="missing_components",
                    options=[],
                ))
                seen_categories.add("missing_components")

        elif reason == "unknown_domain":
            if "unknown_domain" not in seen_categories:
                questions.append(InterrogationQuestion(
                    id="",
                    question="What industry or area is this for?",
                    category="unknown_domain",
                    options=[
                        "Software / SaaS",
                        "E-commerce",
                        "Finance",
                        "Health / Medical",
                        "Education",
                        "Other",
                    ],
                ))
                seen_categories.add("unknown_domain")

    # Cap and assign IDs
    questions = questions[:spec.max_questions]
    for i, q in enumerate(questions):
        q.id = f"q{i + 1}"

    return questions


# =============================================================================
# INTENT REFINEMENT
# =============================================================================

def refine_intent_from_answers(
    description: str,
    intent: dict,
    response: InterrogationResponse,
    questions: List[InterrogationQuestion],
) -> Tuple[str, dict, bool]:
    """Apply user answers to refine description and intent.

    Returns (refined_description, refined_intent, should_fracture).
    """
    refined_desc = description
    refined_intent = dict(intent)  # shallow copy
    should_fracture = False

    # Build category -> answer lookup
    q_map = {q.id: q for q in questions}

    for qid, answer in response.answers.items():
        q = q_map.get(qid)
        if not q:
            continue

        answer_lower = answer.lower().strip()

        if q.category == "domain_scope":
            if "separate" in answer_lower:
                should_fracture = True
            elif "all together" not in answer_lower:
                refined_desc = f"{refined_desc}\n\n[FOCUS: {answer}]"
                refined_intent["domain"] = answer

        elif q.category == "unknown_domain":
            refined_intent["domain"] = answer

        elif q.category == "missing_components":
            # Parse comma/and-separated components
            parts = re.split(r'[,\n]+|\band\b', answer)
            components = [p.strip() for p in parts if p.strip()]
            if components:
                refined_intent["explicit_components"] = components

        elif q.category in ("missing_actors", "missing_actions"):
            refined_desc = f"{refined_desc}\n\n[CLARIFICATION: {answer}]"

    return (refined_desc, refined_intent, should_fracture)
