"""
kernel/llm_bridge.py — Bridge between kernel LLMFunction protocol and real LLM clients.

Adapts core/llm.py clients (GrokClient, ClaudeClient, etc.) into the kernel's
LLMFunction protocol: (str) → list[dict].

The kernel never imports core/llm.py directly. This bridge is the only coupling point.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def make_llm_function(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> "LLMFunction":
    """Create a kernel-compatible LLM function from a provider config.

    Args:
        provider: "grok", "claude", "openai", "gemini", or "auto" (detect from env).
        model: Model name override. Defaults to provider's default.
        api_key: API key override. Defaults to env var.
        temperature: Generation temperature. 0.0 for deterministic.
        max_tokens: Max response tokens.
        system_prompt: Optional system prompt override.

    Returns:
        A callable matching LLMFunction protocol: (str) → list[dict]
    """
    from core.llm import create_client

    client = create_client(provider=provider, model=model, api_key=api_key)

    sys_prompt = system_prompt or (
        "You are a semantic compiler. You extract structured concepts from text "
        "and return them as a JSON array. Return ONLY valid JSON — no markdown "
        "fences, no explanation, just the array."
    )

    def llm_fn(prompt: str) -> list[dict]:
        raw = client.complete_with_system(
            system_prompt=sys_prompt,
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return parse_extractions(raw)

    return llm_fn


def parse_extractions(raw: str) -> list[dict]:
    """Parse LLM response into structured extractions.

    Handles:
    - Clean JSON arrays
    - JSON wrapped in markdown code fences
    - JSON with trailing commas or minor syntax issues
    - Partial/truncated responses (extracts what we can)

    Returns list of dicts with keys: postcode, primitive, content, confidence, connections.
    Invalid entries are silently dropped.
    """
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _validate_extractions(parsed)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from surrounding text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _validate_extractions(parsed)
        except json.JSONDecodeError:
            pass

    # Try fixing trailing commas
    if match:
        fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return _validate_extractions(parsed)
        except json.JSONDecodeError:
            pass

    # Last resort: try parsing individual objects
    objects = re.findall(r'\{[^{}]+\}', text)
    results = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return _validate_extractions(results) if results else []


def _validate_extractions(items: list) -> list[dict]:
    """Filter extractions to only valid entries.

    Required keys: postcode, primitive, content.
    Optional: confidence (default 0.5), connections (default []).
    """
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not item.get("postcode") or not item.get("primitive"):
            continue

        entry = {
            "postcode": str(item["postcode"]).strip(),
            "primitive": str(item["primitive"]).strip(),
            "content": str(item.get("content", "")).strip(),
            "confidence": float(item.get("confidence", 0.5)),
            "connections": list(item.get("connections", [])),
        }

        # Clamp confidence
        entry["confidence"] = max(0.0, min(1.0, entry["confidence"]))

        valid.append(entry)

    return valid


# ---------------------------------------------------------------------------
# Agent response → grid fill parsing
# ---------------------------------------------------------------------------

# Confidence markers found in Entity/Process agent responses
_HIGH_CONFIDENCE_MARKERS = frozenset({
    "clearly", "certainly", "must", "requires", "critical",
    "essential", "fundamental", "core", "primary",
})
_LOW_CONFIDENCE_MARKERS = frozenset({
    "might", "perhaps", "possibly", "could", "optional",
    "may", "potential", "consider",
})


def parse_agent_response_to_fill(
    response_content: str,
    target_postcode: str,
) -> dict:
    """Extract primitive, content, confidence from an agent response for grid fill.

    Parses Entity/Process agent messages to extract structured fill data.
    Uses INSIGHT lines for content, component/process mentions for primitives,
    and confidence markers for scoring.

    Args:
        response_content: The agent's response text.
        target_postcode: The postcode being filled (for concern-axis routing).

    Returns:
        Dict with keys: primitive, content, confidence.
    """
    lines = response_content.strip().split("\n")

    # Extract INSIGHT line as primary content
    content_parts = []
    insight_text = ""
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("INSIGHT:"):
            insight_text = stripped[len("INSIGHT:"):].strip()
            content_parts.append(insight_text)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content_parts.append(stripped[2:].strip())

    content = insight_text if insight_text else " ".join(content_parts[:3])
    if not content:
        # Fallback: first substantive line
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith(("#", "```", "---")):
                content = stripped[:200]
                break

    # Extract primitive name from component/process mentions
    primitive = _extract_primitive(response_content, target_postcode)

    # Estimate confidence from markers
    confidence = _estimate_confidence(response_content)

    return {
        "primitive": primitive,
        "content": content or "extracted from dialogue",
        "confidence": confidence,
    }


def _extract_primitive(response: str, target_postcode: str) -> str:
    """Extract a primitive name from agent response text.

    Looks for capitalized component names, process names, or falls back
    to a postcode-derived default.
    """
    # Try to find explicit component/process names
    # Pattern: "ComponentName", "ProcessName", or quoted names
    component_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'  # CamelCase
    )
    matches = component_pattern.findall(response)
    # Filter noise
    noise = {"Entity", "Process", "Governor", "System", "SharedState", "Message"}
    filtered = [m for m in matches if m not in noise]
    if filtered:
        # Use the most frequently mentioned CamelCase name
        from collections import Counter
        counts = Counter(filtered)
        return counts.most_common(1)[0][0].lower()

    # Try snake_case names in quotes
    quoted = re.findall(r'"([a-z_]+)"', response)
    if quoted:
        return quoted[0]

    # Fallback: derive from postcode concern axis
    parts = target_postcode.split(".")
    concern = parts[1] if len(parts) > 1 else "unknown"
    scope = parts[2] if len(parts) > 2 else "eco"
    return f"{concern.lower()}_{scope.lower()}"


def _estimate_confidence(response: str) -> float:
    """Estimate fill confidence from language markers in response text."""
    words = set(response.lower().split())
    high_count = len(words & _HIGH_CONFIDENCE_MARKERS)
    low_count = len(words & _LOW_CONFIDENCE_MARKERS)

    base = 0.7
    base += high_count * 0.03
    base -= low_count * 0.05
    return max(0.3, min(0.95, base))
