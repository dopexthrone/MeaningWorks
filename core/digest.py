"""
Motherlabs Digest - Structured dialogue summary for synthesis.

Phase 8.1: Compilation Quality - Dialogue Digest

Problem: Synthesis receives only flat insights and truncated dialogue (300 chars/msg).
Solution: Build a token-efficient structured digest from full SharedState.

The digest gives synthesis visibility into:
- Full insights with source attribution
- Confidence vectors
- Conflict records with resolution status
- Unknowns
- Key exchanges (CHALLENGE/ACCOMMODATION messages - highest signal)
- Persona priorities
"""

import re
from typing import Dict, Any, List
from core.protocol import SharedState, MessageType


# Token budget: ~4 chars per token, target 2500 tokens max
MAX_DIGEST_CHARS = 10000
MAX_EXCHANGE_CHARS = 500
MAX_EXCHANGES = 8


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _truncate_to_budget(text: str, max_chars: int) -> str:
    """Truncate text to character budget, breaking at newline if possible."""
    if len(text) <= max_chars:
        return text
    # Try to break at last newline before limit
    truncated = text[:max_chars]
    last_nl = truncated.rfind('\n')
    if last_nl > max_chars * 0.7:
        return truncated[:last_nl] + "\n[...truncated]"
    return truncated + "\n[...truncated]"


def _extract_pattern_matches(insights: List[str]) -> List[Dict[str, Any]]:
    """
    Parse insights for L3/L4 pattern matches ("works like" / "like" analogies).

    Phase 11.3: Extracts structured pattern data from insights that contain
    cross-domain pattern recognition. These become PATTERN TRANSFER directives
    in the synthesis prompt.

    Args:
        insights: List of insight strings from dialogue

    Returns:
        List of dicts with keys: source, analog, hints, raw
    """
    matches = []

    for insight in insights:
        # Pattern 1: "X works like Y" (strongest signal)
        m = re.search(
            r'(.+?)\s+works\s+like\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            insight, re.IGNORECASE
        )
        if not m:
            # Pattern 2: "X is like Y" or "X like Y"
            m = re.search(
                r'(.+?)\s+(?:is\s+)?like\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                insight, re.IGNORECASE
            )

        if not m:
            continue

        # Validate analog starts with uppercase (proper noun / domain reference)
        analog_raw = m.group(2).strip()
        if not analog_raw or not analog_raw[0].isupper():
            continue

        source = m.group(1).strip()
        if not source:
            continue
        # Clean source: remove leading "INSIGHT:" or similar prefixes
        source = re.sub(r'^(?:INSIGHT:\s*)', '', source, flags=re.IGNORECASE).strip()
        if not source:
            continue

        # Strip trailing "needs" from analog if it leaked through regex
        if re.search(r'\bneeds$', analog_raw, re.IGNORECASE):
            analog_raw = re.sub(r'\s+needs$', '', analog_raw, flags=re.IGNORECASE).strip()

        # Extract hints from text after the analog in the original insight
        # Find where analog ends in the insight to get the rest
        analog_end = insight.lower().find(analog_raw.lower())
        if analog_end >= 0:
            rest = insight[analog_end + len(analog_raw):]
        else:
            rest = insight[m.end():]

        hints = []

        # Try "— hint1, hint2" or "— hint1 + hint2" delimiter
        dash_match = re.search(r'[—\-–]\s*(?:needs:\s*)?(.+)', rest)
        if dash_match:
            hint_text = dash_match.group(1).strip()
        else:
            # Try "needs: hint1, hint2" anywhere after analog
            needs_match = re.search(r'needs:\s*(.+)', rest, re.IGNORECASE)
            if needs_match:
                hint_text = needs_match.group(1).strip()
            else:
                hint_text = ""

        if hint_text:
            # Split on "+", ",", " and "
            parts = re.split(r'\s*[+,]\s*|\s+and\s+', hint_text)
            hints = [p.strip() for p in parts if p.strip()]

        matches.append({
            "source": source,
            "analog": analog_raw,
            "hints": hints,
            "raw": insight,
        })

    return matches


def _format_pattern_transfers(patterns: List[Dict[str, Any]]) -> str:
    """
    Format pattern matches into a PATTERN TRANSFERS digest section.

    Args:
        patterns: Output from _extract_pattern_matches()

    Returns:
        Formatted string section
    """
    lines = []
    for p in patterns:
        hint_str = ", ".join(p["hints"]) if p["hints"] else "(no specific hints)"
        lines.append(f"  {p['source']} \u2192 {p['analog']}: {hint_str}")
    return "PATTERN TRANSFERS:\n" + "\n".join(lines)


def _parse_method_params(params_str: str) -> List[Dict[str, str]]:
    """
    Parse a method parameter string into structured params.

    Phase 12.3a: Parses "param1: Type1, param2: Type2" into list of dicts.
    Default type_hint = "Any" if no type given.

    Args:
        params_str: Comma-separated parameter string (e.g., "source: Any, target: Any")

    Returns:
        List of {"name": ..., "type_hint": ...} dicts
    """
    params_str = params_str.strip()
    if not params_str:
        return []

    params = []
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, type_hint = part.split(":", 1)
            params.append({"name": name.strip(), "type_hint": type_hint.strip()})
        else:
            params.append({"name": part.strip(), "type_hint": "Any"})
    return params


def extract_dialogue_methods(state) -> List[Dict[str, Any]]:
    """
    Extract METHOD: lines from dialogue history.

    Phase 12.3a: Scans Entity/Process agent messages for explicit method
    declarations in the format:
        METHOD: ComponentName.method_name(param1: type, param2: type) -> return_type

    Args:
        state: SharedState with dialogue history

    Returns:
        List of method dicts with component, name, parameters, return_type, derived_from, source
    """
    methods = []
    method_re = re.compile(
        r'METHOD:\s*(\w+)\.(\w+)\(([^)]*)\)\s*(?:->\s*(\S+))?'
    )

    for msg in state.history:
        # Only parse Entity/Process agent messages
        if msg.sender not in ("Entity", "Process"):
            continue

        for match in method_re.finditer(msg.content):
            component = match.group(1)
            name = match.group(2)
            params_str = match.group(3)
            return_type = match.group(4) or "None"

            methods.append({
                "component": component,
                "name": name,
                "parameters": _parse_method_params(params_str),
                "return_type": return_type,
                "derived_from": match.group(0),
                "source": "dialogue",
            })

    return methods


def extract_dialogue_state_machines(state) -> List[Dict[str, Any]]:
    """
    Extract STATES: blocks from dialogue history.

    Phase 12.3a: Scans Process Agent messages for state machine declarations:
        STATES: ComponentName
          states: [STATE_A, STATE_B, STATE_C]
          transitions:
            - STATE_A -> STATE_B on "trigger_event"

    Args:
        state: SharedState with dialogue history

    Returns:
        List of state machine dicts with component, states, transitions, derived_from, source
    """
    machines = []
    states_header_re = re.compile(r'^STATES:\s*(\w+)', re.MULTILINE)
    states_list_re = re.compile(r'states:\s*\[([^\]]+)\]')
    transition_re = re.compile(r'(\w+)\s*->\s*(\w+)\s+on\s+"([^"]+)"')

    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue

        content = msg.content
        for header_match in states_header_re.finditer(content):
            component = header_match.group(1)
            # Collect text from after the header until next non-indented line or end
            start_pos = header_match.end()
            block_lines = []
            remaining = content[start_pos:]
            for line in remaining.split("\n"):
                # First line after header may be empty
                if not block_lines and not line.strip():
                    continue
                # Stop at next non-indented non-empty line (but not the first line)
                if block_lines and line.strip() and not line.startswith((" ", "\t", "-")):
                    break
                block_lines.append(line)

            block_text = "\n".join(block_lines)

            # Parse states list
            states = []
            states_match = states_list_re.search(block_text)
            if states_match:
                states = [s.strip() for s in states_match.group(1).split(",") if s.strip()]

            # Parse transitions
            transitions = []
            for t_match in transition_re.finditer(block_text):
                transitions.append({
                    "from": t_match.group(1),
                    "to": t_match.group(2),
                    "trigger": t_match.group(3),
                })

            # Build derived_from from the matched block
            derived_from = header_match.group(0)

            machines.append({
                "component": component,
                "states": states,
                "transitions": transitions,
                "derived_from": derived_from,
                "source": "dialogue",
            })

    return machines


def extract_dialogue_algorithms(state) -> List[Dict[str, Any]]:
    """
    Extract ALGORITHM: blocks from CONSTRAIN stage dialogue.

    Format:
        ALGORITHM: ComponentName.method_name
          1. Step description
          2. If condition then action else alternative
          3. Return result
          PRE: precondition
          POST: postcondition

    Returns:
        List of dicts: component, method_name, steps, preconditions, postconditions, derived_from, source
    """
    algorithms = []
    algo_header_re = re.compile(r'^ALGORITHM:\s*(\w+)\.(\w+)', re.MULTILINE)

    for msg in state.history:
        # Only parse Entity/Process agent messages
        if msg.sender not in ("Entity", "Process"):
            continue

        content = msg.content
        for header_match in algo_header_re.finditer(content):
            component = header_match.group(1)
            method_name = header_match.group(2)

            # Collect indented block after header
            start_pos = header_match.end()
            block_lines = []
            remaining = content[start_pos:]
            for line in remaining.split("\n"):
                if not block_lines and not line.strip():
                    continue
                if block_lines and line.strip() and not line.startswith((" ", "\t")):
                    break
                block_lines.append(line)

            # Parse numbered steps, PRE:, POST: from block
            steps = []
            preconditions = []
            postconditions = []
            for line in block_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.upper().startswith("PRE:"):
                    preconditions.append(stripped[4:].strip())
                elif stripped.upper().startswith("POST:"):
                    postconditions.append(stripped[5:].strip())
                elif re.match(r'^\d+\.', stripped):
                    steps.append(stripped)

            if steps:
                algorithms.append({
                    "component": component,
                    "method_name": method_name,
                    "steps": steps,
                    "preconditions": preconditions,
                    "postconditions": postconditions,
                    "derived_from": header_match.group(0),
                    "source": "dialogue",
                })

    return algorithms


def extract_pattern_method_stubs(insights: List[str]) -> List[Dict[str, Any]]:
    """
    Convert pattern transfer hints into method stubs.

    Phase 12.3a: Reuses _extract_pattern_matches() to find "X works like Y — a, b, c"
    patterns and creates method stubs from each hint.

    Args:
        insights: List of insight strings

    Returns:
        List of method stub dicts
    """
    stubs = []
    patterns = _extract_pattern_matches(insights)

    for pm in patterns:
        for hint in pm.get("hints", []):
            stubs.append({
                "component": pm["source"],
                "name": hint,
                "parameters": [],
                "return_type": "None",
                "derived_from": pm["raw"],
                "source": "pattern_transfer",
            })

    return stubs


def format_method_section(methods: List[Dict[str, Any]], state_machines: List[Dict[str, Any]]) -> str:
    """
    Format extracted methods and state machines for synthesis prompt.

    Phase 12.3a: Groups by component, produces readable text for LLM consumption.

    Args:
        methods: List of method dicts from extract_dialogue_methods / extract_pattern_method_stubs
        state_machines: List of state machine dicts from extract_dialogue_state_machines

    Returns:
        Formatted string for synthesis prompt, empty string if nothing to format
    """
    if not methods and not state_machines:
        return ""

    parts = []

    # Group methods by component
    if methods:
        by_component: Dict[str, List[Dict[str, Any]]] = {}
        for m in methods:
            comp = m["component"]
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(m)

        lines = ["EXTRACTED METHODS — include in component methods[] arrays:\n"]
        for comp, comp_methods in by_component.items():
            lines.append(f"{comp}:")
            for m in comp_methods:
                params = ", ".join(
                    f"{p['name']}: {p['type_hint']}" for p in m["parameters"]
                ) if m["parameters"] else ""
                sig = f"{m['name']}({params}) -> {m['return_type']}"
                lines.append(f"  - {sig}")
                lines.append(f"    derived_from: \"{m['derived_from']}\"")
            lines.append("")
        parts.append("\n".join(lines))

    # Format state machines
    if state_machines:
        lines = ["EXTRACTED STATE MACHINES — include as state_machine on component:\n"]
        for sm in state_machines:
            lines.append(f"{sm['component']}:")
            if sm["states"]:
                lines.append(f"  states: [{', '.join(sm['states'])}]")
            for t in sm["transitions"]:
                lines.append(f"  transitions: {t['from']} -> {t['to']} on \"{t['trigger']}\"")
            lines.append(f"  derived_from: \"{sm['derived_from']}\"")
            lines.append("")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _rank_insights(state: SharedState) -> List[Dict[str, Any]]:
    """
    Rank insights by signal strength.

    Phase 12.1c: Prioritized synthesis — insights from CHALLENGE/ACCOMMODATION
    exchanges and pattern matches rank higher than routine propositions.

    Scoring:
    - +3 if insight's source message was CHALLENGE or ACCOMMODATION
    - +2 if insight contains a pattern match ("works like" / "is like")
    - +1 if insight's source message was PROPOSITION
    - Tiers: HIGH (score >= 4), MEDIUM (2-3), LOW (1)

    Returns:
        List of {insight, source, score, tier} sorted by score descending.
    """
    ranked = []

    for i, insight in enumerate(state.insights):
        score = 0
        source = _find_insight_source(state, insight, i)

        # Find the source message to determine its type
        source_msg = None
        for m in state.history:
            if m.insight and m.insight == insight:
                source_msg = m
                break

        # Score by message type
        if source_msg:
            if source_msg.message_type in (MessageType.CHALLENGE, MessageType.ACCOMMODATION):
                score += 3
            elif source_msg.message_type == MessageType.PROPOSITION:
                score += 1
        else:
            score += 1  # Default for unattributed insights

        # Pattern match bonus
        insight_lower = insight.lower()
        if "works like" in insight_lower or "is like" in insight_lower:
            score += 2

        # Determine tier
        if score >= 4:
            tier = "HIGH"
        elif score >= 2:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        ranked.append({
            "insight": insight,
            "source": source,
            "score": score,
            "tier": tier,
        })

    # Sort by score descending, then by original order (stable sort)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "for", "of", "in", "on", "at", "to", "from", "by", "with", "and",
    "or", "but", "not", "no", "so", "if", "then", "than", "that", "this",
    "it", "its", "as", "into", "via", "use", "using", "based",
})


def _normalize_insight(text: str) -> str:
    """Normalize insight text for comparison: arrows, whitespace, equation order."""
    # Normalize unicode arrows to ASCII
    normalized = text.replace("→", "->").replace("←", "<-").replace("↔", "<->")
    # Normalize multiple spaces
    normalized = " ".join(normalized.split())
    return normalized


def _deduplicate_insights(ranked: list[dict]) -> list[dict]:
    """Merge semantically similar insights, keeping the highest-scored version.

    Uses overlap coefficient (intersection / min(|A|, |B|)) on content words after
    stopword removal. Threshold 0.4 catches both direct overlap ("Kanban board" vs
    "Kanban-style columns") and thesaurus loops where synonyms substitute
    ("structural_immortality" vs "architectural_immortality").

    Also detects commutative equation duplicates: "A + B = C" matches "C = A + B".
    """
    if len(ranked) <= 1:
        return ranked

    import re

    def _content_tokens(text: str) -> set:
        # Normalize arrows before tokenizing
        normalized = _normalize_insight(text)
        words = set(re.findall(r'\w+', normalized.lower()))
        content = words - _STOP_WORDS
        return content if content else words

    def _equation_parts(text: str) -> tuple[frozenset, ...] | None:
        """Extract parts from 'A + B = C' style insights for commutative matching."""
        normalized = _normalize_insight(text)
        if " = " not in normalized:
            return None
        sides = normalized.split(" = ", 1)
        if len(sides) != 2:
            return None
        # Split each side on + and normalize terms
        left_terms = frozenset(t.strip().lower() for t in sides[0].split("+") if t.strip())
        right_terms = frozenset(t.strip().lower() for t in sides[1].split("+") if t.strip())
        return (left_terms, right_terms)

    def _is_commutative_dup(text1: str, text2: str) -> bool:
        """Check if two equation insights are the same with sides swapped."""
        parts1 = _equation_parts(text1)
        parts2 = _equation_parts(text2)
        if parts1 is None or parts2 is None:
            return False
        # A + B = C matches C = A + B (swap sides)
        if parts1[0] == parts2[1] and parts1[1] == parts2[0]:
            return True
        # Also catch A + B = C matches A + B = C (same sides, different order within +)
        if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
            return True
        return False

    deduped = []
    for candidate in ranked:
        c_tokens = _content_tokens(candidate["insight"])
        is_dup = False
        for existing in deduped:
            # Check commutative equation match first (catches exact equation swaps)
            if _is_commutative_dup(candidate["insight"], existing["insight"]):
                is_dup = True
                break
            # Then check token overlap
            e_tokens = _content_tokens(existing["insight"])
            intersection = c_tokens & e_tokens
            min_size = min(len(c_tokens), len(e_tokens))
            if min_size > 0:
                overlap = len(intersection) / min_size
                if overlap > 0.4:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(candidate)

    return deduped


def build_dialogue_digest(state: SharedState) -> str:
    """
    Build a structured dialogue digest from SharedState for synthesis.

    Compresses full SharedState into a token-efficient summary (~2000-2500 tokens)
    with sections for insights, confidence, conflicts, unknowns, key exchanges,
    and persona priorities.

    Args:
        state: SharedState with dialogue history, insights, confidence, etc.

    Returns:
        Formatted string digest, deterministic (same input = same output)
    """
    sections = []

    # SECTION 1: INSIGHTS with tiered priority (Phase 12.1c)
    # Cap at 10 to prevent degenerate dialogue tail from polluting synthesis
    if state.insights:
        ranked = _deduplicate_insights(_rank_insights(state))[:10]
        tier_groups = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for r in ranked:
            tier_groups[r["tier"]].append(f"  [{r['source']}] {r['insight']}")

        insight_lines = []
        for tier_name in ("HIGH", "MEDIUM", "LOW"):
            items = tier_groups[tier_name]
            if items:
                insight_lines.append(f"  {tier_name} PRIORITY:")
                insight_lines.extend([f"  {line}" for line in items])
        sections.append("INSIGHTS:\n" + "\n".join(insight_lines))

    # SECTION 2: CONFIDENCE VECTOR
    cv = state.confidence
    if cv.overall() > 0:
        sections.append(
            f"CONFIDENCE:\n"
            f"  structural: {cv.structural:.0%}\n"
            f"  behavioral: {cv.behavioral:.0%}\n"
            f"  coverage: {cv.coverage:.0%}\n"
            f"  consistency: {cv.consistency:.0%}\n"
            f"  overall: {cv.overall():.0%}"
        )

    # SECTION 3: CONFLICTS
    if state.conflicts:
        conflict_lines = []
        for c in state.conflicts:
            status = "RESOLVED" if c.get("resolved") else "UNRESOLVED"
            agents = ", ".join(c.get("agents", []))
            topic = c.get("topic", "unknown")
            positions = c.get("positions", {})
            pos_str = "; ".join(f"{k}: {v}" for k, v in positions.items())
            resolution = c.get("resolution", "")
            line = f"  [{status}] {agents} on '{topic}': {pos_str}"
            if resolution:
                line += f" -> {resolution}"
            conflict_lines.append(line)
        sections.append("CONFLICTS:\n" + "\n".join(conflict_lines))

    # SECTION 4: UNKNOWNS
    if state.unknown:
        unknown_lines = [f"  - {u}" for u in state.unknown]
        sections.append("UNKNOWNS:\n" + "\n".join(unknown_lines))

    # SECTION 5: KEY EXCHANGES (CHALLENGE and ACCOMMODATION messages - highest signal)
    key_messages = [
        m for m in state.history
        if m.message_type in (MessageType.CHALLENGE, MessageType.ACCOMMODATION)
        and m.sender in ("Entity", "Process")
    ]
    if key_messages:
        exchange_lines = []
        for m in key_messages[:MAX_EXCHANGES]:
            content = m.content[:MAX_EXCHANGE_CHARS]
            if len(m.content) > MAX_EXCHANGE_CHARS:
                content += "..."
            mtype = m.message_type.value.upper()
            exchange_lines.append(f"  [{m.sender}/{mtype}] {content}")
        sections.append("KEY EXCHANGES:\n" + "\n".join(exchange_lines))

    # SECTION 6: PATTERN TRANSFERS (Phase 11.3)
    if state.insights:
        pattern_matches = _extract_pattern_matches(state.insights)
        if pattern_matches:
            sections.append(_format_pattern_transfers(pattern_matches))

    # SECTION 7: PERSONA PRIORITIES
    if state.personas:
        persona_lines = []
        for p in state.personas:
            name = p.get("name", "Unknown")
            priorities = p.get("priorities", [])
            top_priorities = priorities[:2] if priorities else []
            if top_priorities:
                prio_str = "; ".join(str(pr) for pr in top_priorities)
                persona_lines.append(f"  {name}: {prio_str}")
            else:
                perspective = p.get("perspective", "")
                if perspective:
                    persona_lines.append(f"  {name}: {perspective[:100]}")
        if persona_lines:
            sections.append("PERSONA PRIORITIES:\n" + "\n".join(persona_lines))

    # Combine and enforce token budget
    digest = "\n\n".join(sections)
    digest = _truncate_to_budget(digest, MAX_DIGEST_CHARS)

    return digest


def _find_insight_source(state: SharedState, insight: str, insight_index: int) -> str:
    """
    Find which agent produced a given insight.

    Searches history for messages whose insight field matches.
    Falls back to turn-based attribution.
    """
    # Direct match: find message with matching insight
    for m in state.history:
        if m.insight and m.insight == insight:
            return f"{m.sender}/T{state.history.index(m) + 1}"

    # Fallback: attribute by position (insights are added in order)
    # Entity and Process alternate, so use index parity
    agent_msgs = [m for m in state.history if m.sender in ("Entity", "Process")]
    if insight_index < len(agent_msgs):
        m = agent_msgs[insight_index]
        return f"{m.sender}/T{state.history.index(m) + 1}"

    return "unknown"
