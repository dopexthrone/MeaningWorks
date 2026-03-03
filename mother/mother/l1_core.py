import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from quarantine import Cell

def track_losses(input_text: str, blueprint: dict) -> list[tuple[str, float]]:
    \"\"\"
    Compute compression losses between input description and blueprint.
    Returns list of (category, severity 0.0-1.0).
    Categories: entity, constraint, behavior, relationship, context.
    \"\"\"
    losses = []

    # Parse input nouns (simple heuristic: capitalized words)
    input_nouns = set(re.findall(r'\b[A-Z][a-z]*\b', input_text))
    # Blueprint component names
    bp_names = set(c.get('name', '').strip() for c in blueprint.get('components', []))
    
    missing_entities = input_nouns - bp_names
    if missing_entities:
        losses.append(('entity', min(1.0, len(missing_entities) * 0.3)))

    # Constraints: keywords like must, before, validate
    constraint_keywords = ['must', 'before', 'validate', 'handle']
    has_constraint = any(kw in input_text.lower() for kw in constraint_keywords)
    if has_constraint:
        # Check if blueprint mentions auth/validation
        bp_text = ' '.join(c.get('description', '') for c in blueprint.get('components', []))
        has_handling = any(kw in bp_text.lower() for kw in ['auth', 'validate', 'handle'])
        if not has_handling:
            losses.append(('constraint', 0.5))

    # Trivial input: no losses
    if len(input_text.strip()) < 20:
        return []

    return losses

try:
    from subsystems.compression_improvement_system import CompressionMetrics, CompressionCore
except ImportError:
    CompressionMetrics = None
    CompressionCore = None

def initialize_state_tracking(input_text: Optional[str] = None, blueprint: Optional[dict] = None) -> None:
    if CompressionMetrics is None or CompressionCore is None:
        return
    metrics = CompressionMetrics()
    if input_text is not None and blueprint is not None:
        losses = track_losses(input_text, blueprint)
        for category, severity in losses:
            comp_pct = round((1.0 - severity) * 100, 1)
            metrics.update_dimension(category, comp_pct)
    else:
        metrics.create_compression_entity(1.1)
    if metrics.get_compression_entity() < 20.0:
        core = CompressionCore()
        core.drive_compression_improvements()
