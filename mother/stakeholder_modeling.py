"""
Stakeholder modeling — role detection and concern mapping from descriptions.

LEAF module. Genome #38 (stakeholder-modeling).

All functions are pure — no external API calls, no LLM invocations.
Keyword-intersection analysis over description text.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_STAKEHOLDER_ROLES: Dict[str, FrozenSet[str]] = {
    "end-user": frozenset({"user", "customer", "consumer", "visitor", "subscriber", "client", "buyer"}),
    "developer": frozenset({"developer", "engineer", "programmer", "devops", "architect", "coder"}),
    "business": frozenset({"founder", "ceo", "manager", "executive", "stakeholder", "investor", "board"}),
    "operations": frozenset({"ops", "admin", "support", "helpdesk", "moderator", "sysadmin"}),
    "compliance": frozenset({"legal", "compliance", "auditor", "regulator", "privacy", "security"}),
    "partner": frozenset({"partner", "vendor", "supplier", "integration", "third-party", "affiliate"}),
}

_CONCERN_AREAS: Dict[str, FrozenSet[str]] = {
    "usability": frozenset({"usability", "ux", "interface", "intuitive", "accessible", "easy"}),
    "performance": frozenset({"performance", "speed", "latency", "throughput", "fast", "scalable"}),
    "security": frozenset({"security", "encryption", "authentication", "authorization", "vulnerability"}),
    "cost": frozenset({"cost", "budget", "pricing", "affordable", "expense", "savings"}),
    "reliability": frozenset({"reliability", "uptime", "availability", "resilience", "backup", "recovery"}),
    "compliance": frozenset({"compliance", "regulation", "audit", "gdpr", "hipaa", "sox"}),
}

# Concern conflict pairs — these role-concern combinations create natural tension
_CONCERN_CONFLICTS: List[Tuple[str, str]] = [
    ("usability", "security"),
    ("cost", "performance"),
    ("cost", "security"),
    ("usability", "compliance"),
]


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StakeholderMap:
    """Stakeholder analysis with roles, concerns, and conflict zones."""
    stakeholders: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (role, concerns) pairs
    primary_stakeholder: str
    concern_overlap: Tuple[str, ...]  # concerns shared by 2+ roles
    conflict_zones: Tuple[Tuple[str, str], ...]  # conflicting concern pairs
    summary: str


# ---------------------------------------------------------------------------
# #38 — Stakeholder-modeling: detection and mapping
# ---------------------------------------------------------------------------

def identify_stakeholders(description: str) -> List[str]:
    """Identify stakeholder roles from a text description.

    Uses frozenset intersection per role keyword set.
    Returns list of detected role names.
    """
    if not description:
        return []

    words = frozenset(description.lower().split())
    detected: List[str] = []

    for role, keywords in _STAKEHOLDER_ROLES.items():
        if words & keywords:
            detected.append(role)

    return detected


def build_stakeholder_map(
    description: str,
    explicit_stakeholders: Optional[List[str]] = None,
) -> StakeholderMap:
    """Build a stakeholder map from description and optional explicit roles.

    Detects roles, infers concerns per role, finds concern overlaps,
    and identifies conflict zones.

    Returns frozen StakeholderMap.
    """
    if not description:
        return StakeholderMap(
            stakeholders=(),
            primary_stakeholder="unknown",
            concern_overlap=(),
            conflict_zones=(),
            summary="No description provided.",
        )

    words = frozenset(description.lower().split())

    # Detect roles
    detected_roles = identify_stakeholders(description)
    if explicit_stakeholders:
        for role in explicit_stakeholders:
            if role not in detected_roles:
                detected_roles.append(role)

    if not detected_roles:
        return StakeholderMap(
            stakeholders=(),
            primary_stakeholder="unknown",
            concern_overlap=(),
            conflict_zones=(),
            summary="No stakeholders detected in description.",
        )

    # Infer concerns per role from keyword overlap
    role_concerns: Dict[str, List[str]] = {}
    for role in detected_roles:
        concerns: List[str] = []
        for concern_name, concern_keywords in _CONCERN_AREAS.items():
            if words & concern_keywords:
                concerns.append(concern_name)
        role_concerns[role] = concerns

    # Build stakeholder tuples
    stakeholders = tuple(
        (role, tuple(role_concerns.get(role, [])))
        for role in detected_roles
    )

    # Find primary stakeholder — most keyword matches
    role_match_counts: List[Tuple[str, int]] = []
    for role in detected_roles:
        role_keywords = _STAKEHOLDER_ROLES.get(role, frozenset())
        match_count = len(words & role_keywords)
        role_match_counts.append((role, match_count))

    role_match_counts.sort(key=lambda x: x[1], reverse=True)
    primary_stakeholder = role_match_counts[0][0] if role_match_counts else "unknown"

    # Find concern overlap (concerns shared by 2+ roles)
    concern_counts: Dict[str, int] = {}
    for concerns in role_concerns.values():
        for concern in concerns:
            concern_counts[concern] = concern_counts.get(concern, 0) + 1

    concern_overlap = tuple(
        sorted(c for c, count in concern_counts.items() if count >= 2)
    )

    # Identify conflict zones from detected concerns
    all_concerns = set()
    for concerns in role_concerns.values():
        all_concerns.update(concerns)

    conflict_zones = tuple(
        (a, b) for a, b in _CONCERN_CONFLICTS
        if a in all_concerns and b in all_concerns
    )

    # Summary
    role_list = ", ".join(detected_roles)
    summary = (
        f"Detected {len(detected_roles)} stakeholder(s): {role_list}. "
        f"Primary: {primary_stakeholder}. "
        f"{len(concern_overlap)} shared concern(s), {len(conflict_zones)} conflict zone(s)."
    )

    return StakeholderMap(
        stakeholders=stakeholders,
        primary_stakeholder=primary_stakeholder,
        concern_overlap=concern_overlap,
        conflict_zones=conflict_zones,
        summary=summary,
    )
