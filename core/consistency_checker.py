"""
core/consistency_checker.py — Cross-agent factual consistency detection.

LEAF module. Stdlib only. No imports from core/ or mother/ (operates on plain dicts/strings).

Detects contradictions across agent responses within a compilation dialogue.
Agents may disagree structurally (which is healthy friction) but should not
produce factually inconsistent claims about the same entity — e.g., one agent
says "UserService handles authentication" while another says "AuthService
handles authentication" without acknowledging the discrepancy.

Three detection strategies:
  1. Entity-attribute contradictions: same entity assigned conflicting properties
  2. Cardinality contradictions: agent A says "single database" while B says "two databases"
  3. Negation contradictions: agent A says X requires Y, agent B says X does NOT require Y

Returns ConsistencyReport with findings. Does NOT block compilation — findings
are advisory signals that feed into trust scoring and verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Contradiction:
    """A detected factual inconsistency between agents."""
    category: str           # "entity_attribute", "cardinality", "negation"
    entity: str             # the entity/topic in question
    agent_a: str            # first agent name
    claim_a: str            # first agent's claim (excerpt)
    agent_b: str            # second agent name
    claim_b: str            # second agent's claim (excerpt)
    severity: str           # "hard" (direct contradiction) or "soft" (possible inconsistency)
    explanation: str        # why this is flagged


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of cross-agent consistency analysis."""
    contradictions: tuple[Contradiction, ...]
    messages_analyzed: int
    agents_seen: tuple[str, ...]

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def hard_count(self) -> int:
        return sum(1 for c in self.contradictions if c.severity == "hard")

    @property
    def soft_count(self) -> int:
        return sum(1 for c in self.contradictions if c.severity == "soft")


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# "X handles/manages/provides/is responsible for Y"
_RESPONSIBILITY_RE = re.compile(
    r"(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+"
    r"(?:handles?|manages?|provides?|is\s+responsible\s+for|controls?|owns?)\s+"
    r"(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)

# "X requires/needs/depends on Y" and "X does not require/need Y"
# NOTE: entity capture is case-sensitive (PascalCase), rest is case-insensitive
_REQUIREMENT_POS_RE = re.compile(
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+"
    r"(?:requires?|needs?|depends?\s+on|must\s+have)\s+"
    r"(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)
_REQUIREMENT_NEG_RE = re.compile(
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+"
    r"(?:does\s+not\s+|doesn't\s+)"
    r"(?:require|need|depend\s+on|must\s+have)\s+"
    r"(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)

# Cardinality: "single/one X", "two/multiple/several X"
_SINGLE_RE = re.compile(
    r"\b(?:single|one|a\s+single|unified|centralized)\s+"
    r"([a-zA-Z]+)\b",
    re.IGNORECASE,
)
_PLURAL_RE = re.compile(
    r"\b(?:two|three|multiple|several|many|distributed|separate)\s+"
    r"([a-zA-Z]+)\b",
    re.IGNORECASE,
)

# "X is a/an Y" or "X is Y"
_TYPE_RE = re.compile(
    r"(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+"
    r"is\s+(?:a\s+|an\s+)?"
    r"([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})",
    re.IGNORECASE,
)


def _normalize(s: str) -> str:
    """Lowercase and strip for comparison."""
    return s.strip().lower()


def _stem(s: str) -> str:
    """Naive English stemming for entity comparison.

    Strips trailing 's' so 'databases' matches 'database'.
    Conservative: only strips the final 's' to avoid over-stemming.
    """
    w = s.strip().lower()
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def _extract_claims(
    messages: List[Dict],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Extract structured claims from message dicts.

    Args:
        messages: list of {"sender": str, "content": str} dicts

    Returns:
        dict of claim_type → [(agent, entity, value), ...]
    """
    claims: Dict[str, List[Tuple[str, str, str]]] = {
        "responsibility": [],
        "requirement": [],
        "negated_requirement": [],
        "cardinality_single": [],
        "cardinality_plural": [],
        "type_assertion": [],
    }

    for msg in messages:
        agent = msg.get("sender", "")
        content = msg.get("content", "")
        if not agent or not content:
            continue

        # Responsibility claims
        for m in _RESPONSIBILITY_RE.finditer(content):
            entity = _normalize(m.group(1))
            responsibility = _normalize(m.group(2))
            if len(entity) > 2 and len(responsibility) > 2:
                claims["responsibility"].append((agent, entity, responsibility))

        # Requirement claims — negative first (more specific pattern)
        for m in _REQUIREMENT_NEG_RE.finditer(content):
            entity = _normalize(m.group(1))
            requirement = _normalize(m.group(2))
            if len(entity) > 2 and len(requirement) > 2:
                claims["negated_requirement"].append((agent, entity, requirement))

        # Positive requirements
        for m in _REQUIREMENT_POS_RE.finditer(content):
            entity = _normalize(m.group(1))
            requirement = _normalize(m.group(2))
            if len(entity) > 2 and len(requirement) > 2:
                # Skip if already captured as negative
                if not any(
                    ne == entity and nr == requirement
                    for _, ne, nr in claims["negated_requirement"]
                    if _ == agent
                ):
                    claims["requirement"].append((agent, entity, requirement))

        # Cardinality claims
        for m in _SINGLE_RE.finditer(content):
            entity = _normalize(m.group(1))
            if len(entity) > 2:
                claims["cardinality_single"].append((agent, entity, "single"))

        for m in _PLURAL_RE.finditer(content):
            entity = _normalize(m.group(1))
            if len(entity) > 2:
                claims["cardinality_plural"].append((agent, entity, "plural"))

        # Type assertions
        for m in _TYPE_RE.finditer(content):
            entity = _normalize(m.group(1))
            type_claim = _normalize(m.group(2))
            if len(entity) > 2 and len(type_claim) > 2:
                claims["type_assertion"].append((agent, entity, type_claim))

    return claims


def _detect_responsibility_contradictions(
    claims: Dict[str, List[Tuple[str, str, str]]],
) -> List[Contradiction]:
    """Detect when different agents assign the same responsibility to different entities."""
    contradictions = []
    resp_claims = claims.get("responsibility", [])

    # Group by responsibility (what's being handled)
    by_responsibility: Dict[str, List[Tuple[str, str]]] = {}
    for agent, entity, responsibility in resp_claims:
        key = responsibility
        if key not in by_responsibility:
            by_responsibility[key] = []
        by_responsibility[key].append((agent, entity))

    for responsibility, agent_entities in by_responsibility.items():
        # Find cross-agent disagreements on who handles this
        entities_by_agent: Dict[str, set] = {}
        for agent, entity in agent_entities:
            if agent not in entities_by_agent:
                entities_by_agent[agent] = set()
            entities_by_agent[agent].add(entity)

        agents = list(entities_by_agent.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                # Different agents, different entities for same responsibility
                overlap = entities_by_agent[a] & entities_by_agent[b]
                diff_a = entities_by_agent[a] - entities_by_agent[b]
                diff_b = entities_by_agent[b] - entities_by_agent[a]
                if diff_a and diff_b and not overlap:
                    contradictions.append(Contradiction(
                        category="entity_attribute",
                        entity=responsibility,
                        agent_a=a,
                        claim_a=f"{', '.join(sorted(diff_a))} handles {responsibility}",
                        agent_b=b,
                        claim_b=f"{', '.join(sorted(diff_b))} handles {responsibility}",
                        severity="soft",
                        explanation=(
                            f"Agents disagree on which component handles '{responsibility}'"
                        ),
                    ))

    return contradictions


def _detect_negation_contradictions(
    claims: Dict[str, List[Tuple[str, str, str]]],
) -> List[Contradiction]:
    """Detect when one agent requires X and another negates it."""
    contradictions = []
    pos = claims.get("requirement", [])
    neg = claims.get("negated_requirement", [])

    # Index positive requirements by (entity, requirement)
    pos_by_key: Dict[Tuple[str, str], List[str]] = {}
    for agent, entity, req in pos:
        key = (entity, req)
        if key not in pos_by_key:
            pos_by_key[key] = []
        pos_by_key[key].append(agent)

    for agent, entity, req in neg:
        key = (entity, req)
        if key in pos_by_key:
            for pos_agent in pos_by_key[key]:
                if pos_agent != agent:
                    contradictions.append(Contradiction(
                        category="negation",
                        entity=entity,
                        agent_a=pos_agent,
                        claim_a=f"{entity} requires {req}",
                        agent_b=agent,
                        claim_b=f"{entity} does not require {req}",
                        severity="hard",
                        explanation=(
                            f"Direct contradiction: {pos_agent} says '{entity}' requires "
                            f"'{req}', but {agent} says it does not"
                        ),
                    ))

    return contradictions


def _detect_cardinality_contradictions(
    claims: Dict[str, List[Tuple[str, str, str]]],
) -> List[Contradiction]:
    """Detect single vs plural contradictions for the same entity."""
    contradictions = []
    singles = claims.get("cardinality_single", [])
    plurals = claims.get("cardinality_plural", [])

    # Index singles by stemmed entity → (raw_entity, agents)
    single_by_stem: Dict[str, List[Tuple[str, str]]] = {}
    for agent, entity, _ in singles:
        stem = _stem(entity)
        if stem not in single_by_stem:
            single_by_stem[stem] = []
        single_by_stem[stem].append((entity, agent))

    for agent, entity, _ in plurals:
        stem = _stem(entity)
        if stem in single_by_stem:
            for raw_single, single_agent in single_by_stem[stem]:
                if single_agent != agent:
                    contradictions.append(Contradiction(
                        category="cardinality",
                        entity=entity,
                        agent_a=single_agent,
                        claim_a=f"single {raw_single}",
                        agent_b=agent,
                        claim_b=f"multiple {entity}",
                        severity="hard",
                        explanation=(
                            f"Cardinality conflict: {single_agent} says single '{raw_single}', "
                            f"{agent} says multiple '{entity}'"
                        ),
                    ))

    return contradictions


def check_consistency(
    messages: List[Dict],
) -> ConsistencyReport:
    """Run all consistency checks across agent messages.

    Args:
        messages: list of {"sender": str, "content": str} dicts
                  (matches Message.to_dict() format from protocol.py)

    Returns:
        ConsistencyReport with detected contradictions
    """
    if not messages:
        return ConsistencyReport(
            contradictions=(),
            messages_analyzed=0,
            agents_seen=(),
        )

    agents = tuple(sorted(set(
        m.get("sender", "") for m in messages if m.get("sender")
    )))

    claims = _extract_claims(messages)

    contradictions: List[Contradiction] = []
    contradictions.extend(_detect_responsibility_contradictions(claims))
    contradictions.extend(_detect_negation_contradictions(claims))
    contradictions.extend(_detect_cardinality_contradictions(claims))

    return ConsistencyReport(
        contradictions=tuple(contradictions),
        messages_analyzed=len(messages),
        agents_seen=agents,
    )


def format_consistency_warnings(report: ConsistencyReport) -> str:
    """Format consistency report as text for insertion into verification prompts."""
    if not report.has_contradictions:
        return ""

    lines = [f"[CONSISTENCY] {len(report.contradictions)} cross-agent inconsistencies detected:"]
    for c in report.contradictions:
        severity_tag = "HARD" if c.severity == "hard" else "SOFT"
        lines.append(
            f"  [{severity_tag}] {c.category}: {c.explanation}"
        )
    return "\n".join(lines)
