"""
Motherlabs Input Enrichment — expand sparse input for richer compilation.

Phase 2 of Agent Ship: Addresses BUILD-OBSERVATIONS #1:
"depth is proportional to noun density. Sparse input → shallow output."

Contains prompt templates and response parsing only.
The LLM call happens in the orchestrator.

This is a LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
"""

import re
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class EnrichmentResult:
    """Result of input enrichment."""
    original_input: str
    enriched_input: str
    expansion_ratio: float      # len(enriched) / len(original)


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """You are a software requirements analyst. Your job is to expand a sparse product description into a detailed specification that preserves the user's original intent.

CRITICAL RULES:
1. Do NOT add features the user didn't mention. Expand depth of what they said, not breadth.
2. Do NOT change the domain or purpose.
3. Do NOT add technical implementation details (databases, frameworks, etc.).
4. DO identify implicit actors, actions, and constraints.
5. DO name analogous existing products for context.
6. Output ONLY the enriched description, no preamble or explanation.

Your output should be 200-400 words covering:
- Domain identification (what space this operates in)
- 3-5 actors (who interacts with this system)
- 5-10 key actions (what the system does)
- 1-2 analogous existing products (for structural reference)
- Implicit constraints (what must be true for this to work)
- Expanded description preserving ALL original requirements"""


def build_enrichment_prompt(sparse_input: str) -> str:
    """Build the user-facing prompt for input enrichment.

    Args:
        sparse_input: The user's original sparse description

    Returns:
        Formatted prompt string to send as user message
    """
    return (
        f"Expand this product description into a detailed specification.\n\n"
        f"Original description:\n"
        f'"{sparse_input.strip()}"\n\n'
        f"Produce an enriched description (200-400 words) that adds depth "
        f"without adding breadth. Identify actors, actions, constraints, "
        f"and analogous products. Output ONLY the enriched text."
    )


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_enrichment_response(
    llm_response: str,
    original_input: str,
) -> EnrichmentResult:
    """Parse LLM enrichment response into EnrichmentResult.

    Graceful degradation: if response is empty or too short,
    falls back to original input.

    Args:
        llm_response: Raw LLM response text
        original_input: The user's original input (for fallback)

    Returns:
        EnrichmentResult with enriched text and expansion ratio
    """
    if not llm_response or not llm_response.strip():
        return EnrichmentResult(
            original_input=original_input,
            enriched_input=original_input,
            expansion_ratio=1.0,
        )

    enriched = llm_response.strip()

    # Strip common LLM preambles
    enriched = _strip_preamble(enriched)

    # If enrichment is shorter than original, something went wrong — fallback
    if len(enriched) < len(original_input) * 0.8:
        return EnrichmentResult(
            original_input=original_input,
            enriched_input=original_input,
            expansion_ratio=1.0,
        )

    # Compute expansion ratio
    orig_len = max(len(original_input), 1)
    ratio = len(enriched) / orig_len

    return EnrichmentResult(
        original_input=original_input,
        enriched_input=enriched,
        expansion_ratio=round(ratio, 2),
    )


def _strip_preamble(text: str) -> str:
    """Strip common LLM response preambles.

    Handles patterns like:
    - "Here is the enriched description:"
    - "Enriched Description:\n..."
    - "Sure! Here's..."
    """
    # Pattern: starts with common preamble phrases
    preamble_patterns = [
        r'^(?:Here\s+is|Here\'s|Sure[!,.]?\s*(?:Here\s+is|Here\'s))[^:]*:\s*\n?',
        r'^Enriched\s+(?:Description|Specification|Text)\s*:\s*\n?',
        r'^(?:Expanded|Detailed)\s+(?:Description|Specification)\s*:\s*\n?',
    ]

    for pattern in preamble_patterns:
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)

    return text.strip()
