"""
mother/goal_generator.py — Self-generated improvement targets.

LEAF module. Analyzes the current state of the semantic grid, compilation
history, and feedback signals to generate concrete improvement goals.
This is L3 in action: F(F) → evolution. The compiler identifies its own
gaps and generates goals to fill them.

Goals are not arbitrary — they're derived from:
1. Low-confidence cells in the semantic grid (what's uncertain)
2. Missing cells that should exist (structural gaps)
3. Weakness patterns from governor feedback (what keeps failing)
4. Observation anomalies (what's surprising)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImprovementGoal:
    """A single self-generated improvement goal."""

    goal_id: str
    priority: str  # "critical", "high", "medium", "low"
    category: str  # "confidence", "coverage", "quality", "resilience"
    description: str
    source: str  # what generated this goal (e.g., "observer:low_confidence")
    target_postcodes: tuple[str, ...] = ()  # affected grid cells
    estimated_effort: str = "unknown"  # "trivial", "small", "medium", "large"
    success_metric: str = ""  # how to know this goal is met


@dataclass(frozen=True)
class GoalSet:
    """Ordered collection of improvement goals from one analysis."""

    goals: tuple[ImprovementGoal, ...]
    analysis_summary: str
    total_goals: int
    critical_count: int
    coverage_gaps: int
    confidence_issues: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Priority weights for scoring
_PRIORITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Minimum confidence for a cell to be considered "healthy"
_HEALTHY_CONFIDENCE = 0.75

# Minimum fill rate (percentage of expected cells that exist)
_HEALTHY_FILL_RATE = 0.60

# Expected layers in a well-mapped system
_CORE_LAYERS = frozenset({"INT", "SEM", "ORG", "COG", "AGN", "STR", "STA"})

# Expected concerns per layer (minimum — not all need all)
_MIN_CONCERNS_PER_LAYER = 2


# ---------------------------------------------------------------------------
# Goal generation from grid analysis
# ---------------------------------------------------------------------------

def goals_from_grid(
    cell_data: list[tuple[str, float, str, str]],
    activated_layers: set[str],
    total_cells: int,
) -> list[ImprovementGoal]:
    """Generate goals from semantic grid state.

    cell_data: list of (postcode, confidence, fill_state, primitive)
    activated_layers: set of layer codes currently in the grid
    total_cells: total cell count in grid
    """
    goals: list[ImprovementGoal] = []
    goal_counter = 0

    # 1. Low confidence cells
    low_conf_cells = [(pk, conf, fs, prim) for pk, conf, fs, prim in cell_data
                      if conf < _HEALTHY_CONFIDENCE and fs in ("F", "P")]
    if low_conf_cells:
        # Group by severity
        critical = [c for c in low_conf_cells if c[1] < 0.30]
        warning = [c for c in low_conf_cells if 0.30 <= c[1] < 0.50]
        watch = [c for c in low_conf_cells if 0.50 <= c[1] < _HEALTHY_CONFIDENCE]

        if critical:
            top_crit = sorted(critical, key=lambda x: x[1])[:5]
            goal_counter += 1
            goals.append(ImprovementGoal(
                goal_id=f"G-{goal_counter:03d}",
                priority="critical",
                category="confidence",
                description=f"{len(top_crit)} cells below 30% confidence need immediate attention. (Total critical: {len(critical)})",
                source="grid:critical_confidence",
                target_postcodes=tuple(c[0] for c in top_crit),
                estimated_effort="medium",
                success_metric="All target cells above 30% confidence.",
            ))

        if warning:
            top_warn = sorted(warning, key=lambda x: x[1])[:10]
            goal_counter += 1
            goals.append(ImprovementGoal(
                goal_id=f"G-{goal_counter:03d}",
                priority="high",
                category="confidence",
                description=f"{len(top_warn)} cells between 30-50% confidence need reinforcement. (Total warning: {len(warning)})",
                source="grid:low_confidence",
                target_postcodes=tuple(c[0] for c in top_warn),
                estimated_effort="small",
                success_metric="All target cells above 50% confidence.",
            ))

        if watch:
            goal_counter += 1
            goals.append(ImprovementGoal(
                goal_id=f"G-{goal_counter:03d}",
                priority="medium",
                category="confidence",
                description=f"{len(watch)} cells between 50-75% confidence could improve.",
                source="grid:marginal_confidence",
                target_postcodes=tuple(c[0] for c in watch),
                estimated_effort="trivial",
                success_metric=f"All target cells above {_HEALTHY_CONFIDENCE * 100:.0f}% confidence.",
            ))

    # 2. Missing core layers
    missing_layers = _CORE_LAYERS - activated_layers
    if missing_layers:
        goal_counter += 1
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority="high",
            category="coverage",
            description=f"Core layers not yet mapped: {', '.join(sorted(missing_layers))}.",
            source="grid:missing_layers",
            estimated_effort="large",
            success_metric="All core layers have at least one cell.",
        ))

    # 3. Sparse layers (activated but with very few cells)
    layer_counts: dict[str, int] = {}
    for pk, _, _, _ in cell_data:
        parts = pk.split(".")
        if parts:
            layer = parts[0]
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    sparse_layers = [l for l, c in layer_counts.items()
                     if c < _MIN_CONCERNS_PER_LAYER and l in _CORE_LAYERS]
    if sparse_layers:
        goal_counter += 1
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority="medium",
            category="coverage",
            description=f"Sparse layers need more cells: {', '.join(sorted(sparse_layers))}.",
            source="grid:sparse_layers",
            estimated_effort="medium",
            success_metric=f"Each core layer has at least {_MIN_CONCERNS_PER_LAYER} cells.",
        ))

    # 4. Quarantined cells need investigation
    quarantined = [(pk, conf, prim) for pk, conf, fs, prim in cell_data if fs == "Q"]
    if quarantined:
        top_q = sorted(quarantined, key=lambda x: x[1])[:10]
        num_q = len(quarantined)
        goal_counter += 1
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority="high" if num_q <= 10 else "medium",
            category="resilience",
            description=f"{len(top_q)} quarantined cells need investigation or removal. (Total: {num_q})",
            source="grid:quarantined",
            target_postcodes=tuple(c[0] for c in top_q),
            estimated_effort="small",
            success_metric="Target quarantined cells rehabilitated or removed.",
        ))

    return goals


def goals_from_feedback(
    weaknesses: list[tuple[str, str, float]],
    rejection_rate: float,
    trend: str,
) -> list[ImprovementGoal]:
    """Generate goals from governor feedback signals.

    weaknesses: list of (dimension, severity, mean_score)
    rejection_rate: 0.0-1.0
    trend: "improving", "degrading", "stable", "insufficient_data"
    """
    goals: list[ImprovementGoal] = []
    goal_counter = 100  # offset to avoid collision with grid goals

    # High rejection rate
    if rejection_rate > 0.3:
        goal_counter += 1
        if rejection_rate >= 1.0:
            priority = "critical"
            description = "CRITICAL: Rejection rate at 100%. Improve compiler prompts, verification logic, agent orchestration to eliminate rejections."
            success_metric = "Zero rejections over 10 consecutive compilations."
        else:
            priority = "critical" if rejection_rate > 0.5 else "high"
            description=f"Rejection rate at {rejection_rate:.0%}. Compiler output quality needs improvement."
            success_metric = "Rejection rate below 20%."
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority=priority,
            category="quality",
            description=description,
            source="feedback:rejection_rate",
            estimated_effort="large",
            success_metric=success_metric,
        ))

    # Degrading trend
    if trend == "degrading":
        goal_counter += 1
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority="high",
            category="quality",
            description="Trust scores are trending downward. Investigate recent compilation changes.",
            source="feedback:degrading_trend",
            estimated_effort="medium",
            success_metric="Trust score trend returns to stable or improving.",
        ))

    # Per-dimension weaknesses
    for dim, severity, score in weaknesses:
        if severity in ("critical", "warning"):
            goal_counter += 1
            goals.append(ImprovementGoal(
                goal_id=f"G-{goal_counter:03d}",
                priority="high" if severity == "critical" else "medium",
                category="quality",
                description=f"{dim} dimension at {score:.1f}% needs targeted improvement.",
                source=f"feedback:weakness:{dim}",
                estimated_effort="medium",
                success_metric=f"{dim} mean score above 60%.",
            ))

    return goals


def goals_from_anomalies(
    anomaly_count: int,
    anomaly_postcodes: list[str],
) -> list[ImprovementGoal]:
    """Generate goals from observation anomalies.

    anomaly_count: total anomalies detected
    anomaly_postcodes: postcodes where anomalies occurred
    """
    goals: list[ImprovementGoal] = []

    if anomaly_count == 0:
        return goals

    priority = "critical" if anomaly_count > 5 else "high" if anomaly_count > 2 else "medium"

    goals.append(ImprovementGoal(
        goal_id="G-200",
        priority=priority,
        category="resilience",
        description=f"{anomaly_count} anomalies detected. Expected outcomes did not match actual results.",
        source="observer:anomalies",
        target_postcodes=tuple(anomaly_postcodes[:10]),  # cap at 10
        estimated_effort="medium" if anomaly_count <= 5 else "large",
        success_metric="Zero anomalies in next observation batch.",
    ))

    return goals


def goals_from_compression_losses(
    category_frequencies: dict[str, int],
    total_compilations: int,
) -> list[ImprovementGoal]:
    """Generate goals from compression loss category patterns.

    category_frequencies: {category: count} across compilations
    total_compilations: how many compilations had any compression losses
    """
    goals: list[ImprovementGoal] = []
    goal_counter = 300  # offset to avoid collision

    if total_compilations == 0:
        return goals

    for cat, count in sorted(category_frequencies.items(), key=lambda x: -x[1]):
        freq = count / total_compilations
        if freq <= 0.50:
            continue
        goal_counter += 1
        goals.append(ImprovementGoal(
            goal_id=f"G-{goal_counter:03d}",
            priority="critical" if freq > 0.75 else "high",
            category="quality",
            description=f"{cat} compression losses in {freq:.0%} of compilations. Synthesis is systematically dropping {cat} information.",
            source=f"compression:{cat}",
            estimated_effort="medium",
            success_metric=f"{cat} compression loss frequency below 30%.",
        ))

    return goals


# ---------------------------------------------------------------------------
# Goal enrichment — diagnostic → actionable
# ---------------------------------------------------------------------------

# Layer/concern descriptions for enriching diagnostic goals
_LAYER_DESCRIPTIONS = {
    "INT": "Intent", "SEM": "Semantic", "ORG": "Organization",
    "COG": "Cognitive", "AGN": "Agency", "STR": "Structure",
    "STA": "State", "IDN": "Identity", "TME": "Time",
    "EXC": "Execution", "CTR": "Control", "RES": "Resource",
    "OBS": "Observability", "NET": "Network", "EMG": "Emergence",
    "MET": "Meta", "DAT": "Data", "SFX": "Side Effects",
}

_CONCERN_DESCRIPTIONS = {
    "ENT": "Entity", "BHV": "Behavior", "FNC": "Function",
    "REL": "Relation", "PLN": "Plan", "MEM": "Memory",
    "ORC": "Orchestration", "AGT": "Agent", "ACT": "Actor",
    "STA": "State", "GTE": "Gate", "PLY": "Policy",
    "MET": "Metric", "LOG": "Log", "FLW": "Flow",
    "PRV": "Provenance", "CNS": "Constraint", "GOL": "Goal",
    "CFG": "Config", "EMT": "Emit", "RED": "Read", "TRC": "Trace",
    "SEM": "Semantic", "SCO": "Scope", "TRN": "Transition",
    "SNP": "Snapshot", "VRS": "Version", "SCH": "Schedule",
    "LMT": "Limit", "CND": "Candidate", "INT": "Integrity",
    "ENM": "Enumeration", "PRM": "Permission", "TMO": "Timeout",
    "LCK": "Lock", "RTY": "Retry", "TRF": "Transform",
    "COL": "Collection", "WRT": "Write", "ALT": "Alert",
}

_CATEGORY_TO_VERB = {
    "confidence": "Strengthen implementation and add test coverage for",
    "coverage": "Implement missing capability for",
    "quality": "Improve output quality of",
    "resilience": "Fix reliability issues in",
}


def goal_to_actionable(
    goal: ImprovementGoal,
    grid_cells: list[tuple[str, float, str, str]] | None = None,
) -> ImprovementGoal:
    """Enrich a diagnostic goal with actionable build context from postcodes.

    Transforms generic descriptions like "3 cells below 30% confidence" into
    specific build instructions like "Strengthen implementation and add test
    coverage for Semantic Entity modeling, Cognitive Behavior processing".

    Returns a new ImprovementGoal with the enriched description.
    """
    postcodes = goal.target_postcodes
    if not postcodes:
        # No postcodes to expand — return as-is
        return goal

    # Build territory description from postcodes
    layers: set[str] = set()
    concerns: set[str] = set()
    for pc_str in postcodes:
        parts = pc_str.split(".")
        if len(parts) >= 1:
            layers.add(parts[0])
        if len(parts) >= 2:
            concerns.add(parts[1])

    territory_parts = []
    for layer in sorted(layers):
        layer_name = _LAYER_DESCRIPTIONS.get(layer, layer)
        layer_concerns = []
        for concern in sorted(concerns):
            concern_name = _CONCERN_DESCRIPTIONS.get(concern, concern)
            layer_concerns.append(concern_name)
        if layer_concerns:
            territory_parts.append(f"{layer_name} {'/'.join(layer_concerns)}")
        else:
            territory_parts.append(layer_name)

    territory = ", ".join(territory_parts)

    # Build actionable description
    verb = _CATEGORY_TO_VERB.get(goal.category, "Improve")
    enriched_desc = f"{verb} {territory}."

    # Append specific cell details if available
    if grid_cells:
        pc_set = set(postcodes)
        relevant = [
            (pc, conf, fs, prim)
            for pc, conf, fs, prim in grid_cells
            if pc in pc_set
        ]
        if relevant:
            cell_lines = []
            for pc, conf, fs, prim in relevant[:5]:
                cell_lines.append(f"{pc} ({prim}): {fs} at {conf:.0%}")
            enriched_desc += " Target cells: " + "; ".join(cell_lines) + "."

    # Preserve original description as context
    enriched_desc += f" (Original: {goal.description})"

    return ImprovementGoal(
        goal_id=goal.goal_id,
        priority=goal.priority,
        category=goal.category,
        description=enriched_desc,
        source=goal.source,
        target_postcodes=goal.target_postcodes,
        estimated_effort=goal.estimated_effort,
        success_metric=goal.success_metric,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def generate_goal_set(
    grid_goals: list[ImprovementGoal],
    feedback_goals: list[ImprovementGoal],
    anomaly_goals: list[ImprovementGoal],
    compression_goals: list[ImprovementGoal] | None = None,
) -> GoalSet:
    """Merge and prioritize goals from all sources into a GoalSet."""
    all_goals = grid_goals + feedback_goals + anomaly_goals
    if compression_goals:
        all_goals.extend(compression_goals)

    # Sort by priority (critical first) then by category
    all_goals.sort(key=lambda g: (
        _PRIORITY_SCORE.get(g.priority, 0) * -1,
        g.category,
    ))

    critical = sum(1 for g in all_goals if g.priority == "critical")
    coverage = sum(1 for g in all_goals if g.category == "coverage")
    confidence = sum(1 for g in all_goals if g.category == "confidence")

    if not all_goals:
        summary = "No improvement goals generated. System is operating within normal parameters."
    elif critical > 0:
        summary = f"{critical} critical issue(s) require immediate attention."
    else:
        summary = f"{len(all_goals)} improvement goals identified. Highest priority: {all_goals[0].description[:80]}"

    return GoalSet(
        goals=tuple(all_goals),
        analysis_summary=summary,
        total_goals=len(all_goals),
        critical_count=critical,
        coverage_gaps=coverage,
        confidence_issues=confidence,
    )
