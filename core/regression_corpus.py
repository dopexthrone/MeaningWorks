"""
core/regression_corpus.py — Append-only regression corpus for trust validation.

LEAF module. Stdlib + json only. No imports from core/ or mother/.

Captures full compilation snapshots (input + output + trust dimensions)
as append-only JSONL records. This enables:
  1. External validation that trust scores match expected ranges for known inputs
  2. Regression detection when compiler changes degrade specific dimensions
  3. Training data for future trust calibration models

Each record is self-contained: it includes the input, the trust scores,
the verification details, and enough blueprint summary to reconstruct
what was evaluated. Records are never deleted or modified.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_DIR = Path.home() / ".motherlabs"
_DEFAULT_FILENAME = "regression_corpus.jsonl"
_MAX_INPUT_CHARS = 2000       # truncate input beyond this
_MAX_COMPONENT_DETAIL = 50    # max components to record detail for


@dataclass(frozen=True)
class RegressionRecord:
    """A single compilation snapshot for regression testing.

    Self-contained: includes enough to reproduce the trust evaluation.
    """
    compile_id: str
    timestamp: float
    input_text: str              # truncated to _MAX_INPUT_CHARS
    domain: str
    provider: str
    model: str

    # Trust dimensions (0-100 each)
    trust_score: float
    completeness: float
    consistency: float
    coherence: float
    traceability: float

    # Blueprint summary
    component_count: int
    relationship_count: int
    component_names: tuple       # names only, for lightweight matching

    # Verification metadata
    verification_mode: str
    rejected: bool
    rejection_reason: str
    fidelity_score: float        # closed-loop fidelity (0.0-1.0)

    # Compression losses (category → count)
    compression_losses: dict

    # Process metadata
    dialogue_turns: int
    total_duration_s: float
    cost_usd: float


def build_record(
    compile_id: str,
    input_text: str,
    domain: str = "software",
    provider: str = "",
    model: str = "",
    trust_score: float = 0.0,
    completeness: float = 0.0,
    consistency: float = 0.0,
    coherence: float = 0.0,
    traceability: float = 0.0,
    component_count: int = 0,
    relationship_count: int = 0,
    component_names: tuple = (),
    verification_mode: str = "unknown",
    rejected: bool = False,
    rejection_reason: str = "",
    fidelity_score: float = 0.0,
    compression_losses: Optional[dict] = None,
    dialogue_turns: int = 0,
    total_duration_s: float = 0.0,
    cost_usd: float = 0.0,
) -> RegressionRecord:
    """Build a RegressionRecord with truncation and normalization."""
    return RegressionRecord(
        compile_id=compile_id,
        timestamp=time.time(),
        input_text=input_text[:_MAX_INPUT_CHARS],
        domain=domain,
        provider=provider,
        model=model,
        trust_score=trust_score,
        completeness=completeness,
        consistency=consistency,
        coherence=coherence,
        traceability=traceability,
        component_count=component_count,
        relationship_count=relationship_count,
        component_names=tuple(list(component_names)[:_MAX_COMPONENT_DETAIL]),
        verification_mode=verification_mode,
        rejected=rejected,
        rejection_reason=rejection_reason,
        fidelity_score=fidelity_score,
        compression_losses=compression_losses or {},
        dialogue_turns=dialogue_turns,
        total_duration_s=total_duration_s,
        cost_usd=cost_usd,
    )


def append_record(record: RegressionRecord, corpus_dir: Optional[Path] = None) -> Path:
    """Append a record to the JSONL corpus file. Returns the file path."""
    d = corpus_dir or _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _DEFAULT_FILENAME

    data = asdict(record)
    # Convert tuple to list for JSON serialization
    data["component_names"] = list(data["component_names"])

    with open(path, "a") as f:
        f.write(json.dumps(data, separators=(",", ":")) + "\n")

    return path


def load_corpus(corpus_dir: Optional[Path] = None) -> List[RegressionRecord]:
    """Load all records from the corpus. Returns [] if file doesn't exist."""
    d = corpus_dir or _DEFAULT_DIR
    path = d / _DEFAULT_FILENAME
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["component_names"] = tuple(data.get("component_names", []))
            records.append(RegressionRecord(**data))
    return records


def corpus_stats(corpus_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Summary statistics over the corpus."""
    records = load_corpus(corpus_dir)
    if not records:
        return {"count": 0}

    trust_scores = [r.trust_score for r in records]
    rejected_count = sum(1 for r in records if r.rejected)
    domains = {}
    for r in records:
        domains[r.domain] = domains.get(r.domain, 0) + 1

    return {
        "count": len(records),
        "rejection_rate": rejected_count / len(records),
        "trust_mean": sum(trust_scores) / len(trust_scores),
        "trust_min": min(trust_scores),
        "trust_max": max(trust_scores),
        "domains": domains,
    }


def check_regression(
    current: RegressionRecord,
    corpus_dir: Optional[Path] = None,
    threshold_drop: float = 15.0,
) -> List[str]:
    """Check if current record shows regression vs corpus baseline.

    Returns list of warnings. Empty = no regression detected.
    threshold_drop: max acceptable drop in any dimension vs corpus mean.
    """
    records = load_corpus(corpus_dir)
    if len(records) < 3:
        return []  # not enough data

    # Filter to same domain for fair comparison
    domain_records = [r for r in records if r.domain == current.domain]
    if len(domain_records) < 3:
        domain_records = records  # fall back to all

    warnings = []
    dims = ("trust_score", "completeness", "consistency", "coherence", "traceability")

    for dim in dims:
        baseline = sum(getattr(r, dim) for r in domain_records) / len(domain_records)
        current_val = getattr(current, dim)
        drop = baseline - current_val
        if drop > threshold_drop:
            warnings.append(
                f"{dim}: {current_val:.1f} vs baseline {baseline:.1f} "
                f"(dropped {drop:.1f}, threshold {threshold_drop:.1f})"
            )

    return warnings
