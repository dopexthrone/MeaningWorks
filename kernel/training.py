"""
kernel/training.py — Training data emission for future fine-tuning.

LEAF module. After each compile, emits structured training examples
to ~/.motherlabs/training/YYYY-MM-DD.jsonl. Positive examples from
filled cells, negative from rejected/quarantined, instruction from
governor corrections.

No imports from core/ or mother/. Uses only stdlib + kernel types.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------

_DEFAULT_TRAINING_DIR = Path.home() / ".motherlabs" / "training"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingExample:
    type: str               # "positive", "negative", "instruction"
    postcode: str
    intent_context: str
    output_text: str
    confidence: float
    rejection_reason: str
    domain: str
    run_id: str


# ---------------------------------------------------------------------------
# Extraction — build training examples from grid + compile artifacts
# ---------------------------------------------------------------------------

def extract_training_examples(
    grid,
    result,
    state,
    run_id: str,
) -> list[TrainingExample]:
    """Extract training examples from a compilation's grid and result.

    Args:
        grid: kernel.grid.Grid instance (the semantic map)
        result: CompileResult dataclass
        state: SharedState from the compilation
        run_id: Unique compilation identifier

    Returns:
        List of TrainingExample ready for JSONL emission.
    """
    examples = []
    intent_context = (state.known.get("input", "") or "")[:500]
    domain = state.known.get("domain", "software")

    cells = getattr(grid, "cells", {})
    if not cells:
        return examples

    for pc_key, cell in cells.items():
        fill_name = cell.fill.name if hasattr(cell.fill, "name") else str(cell.fill)
        content = getattr(cell, "content", "") or ""
        confidence = getattr(cell, "confidence", 0.0)
        postcode = pc_key

        if fill_name == "F" and content.strip():
            # Positive example: filled cell with real content
            examples.append(TrainingExample(
                type="positive",
                postcode=postcode,
                intent_context=intent_context,
                output_text=content[:2000],
                confidence=confidence,
                rejection_reason="",
                domain=domain,
                run_id=run_id,
            ))
        elif fill_name == "Q":
            # Negative example: quarantined cell (governor rejected)
            examples.append(TrainingExample(
                type="negative",
                postcode=postcode,
                intent_context=intent_context,
                output_text=content[:2000],
                confidence=confidence,
                rejection_reason="quarantined_by_governor",
                domain=domain,
                run_id=run_id,
            ))
        elif fill_name == "P" and content.strip():
            # Instruction example: partial cell — shows what needs improvement
            examples.append(TrainingExample(
                type="instruction",
                postcode=postcode,
                intent_context=intent_context,
                output_text=content[:2000],
                confidence=confidence,
                rejection_reason="partial_fill",
                domain=domain,
                run_id=run_id,
            ))

    # Cap at 200 examples per compilation
    return examples[:200]


# ---------------------------------------------------------------------------
# Emission — write training data to JSONL
# ---------------------------------------------------------------------------

def emit_jsonl(
    examples: list[TrainingExample],
    output_dir: Optional[Path] = None,
) -> Path:
    """Write training examples to a dated JSONL file.

    Appends to existing file if it exists for the same date.
    Returns the path written to.
    """
    out = output_dir or _DEFAULT_TRAINING_DIR
    out.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = out / f"{date_str}.jsonl"

    with open(filepath, "a", encoding="utf-8") as f:
        for ex in examples:
            line = json.dumps(asdict(ex), ensure_ascii=False)
            f.write(line + "\n")

    return filepath


# ---------------------------------------------------------------------------
# Stats — aggregate training data metrics
# ---------------------------------------------------------------------------

def training_stats(output_dir: Optional[Path] = None) -> dict:
    """Aggregate statistics over all training JSONL files."""
    out = output_dir or _DEFAULT_TRAINING_DIR
    if not out.exists():
        return {"total_files": 0, "total_examples": 0}

    files = sorted(out.glob("*.jsonl"))
    total_examples = 0
    type_counts = {"positive": 0, "negative": 0, "instruction": 0}
    domain_counts: dict[str, int] = {}

    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_examples += 1
                    t = rec.get("type", "unknown")
                    if t in type_counts:
                        type_counts[t] += 1
                    d = rec.get("domain", "unknown")
                    domain_counts[d] = domain_counts.get(d, 0) + 1
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue

    return {
        "total_files": len(files),
        "total_examples": total_examples,
        "type_counts": type_counts,
        "domain_counts": domain_counts,
    }
