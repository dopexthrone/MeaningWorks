"""
mother/self_build_planner.py — Translate goals into rich Claude Code prompts.

LEAF module. Stdlib only. No imports from core/ or mother/.

Given a diagnostic goal (e.g., "3 cells below 30% confidence"), this module
rewrites it into an actionable build instruction, infers target files from
postcode axes, extracts relevant blueprint context, and assembles a complete
Claude Code prompt with architectural rules and boundary constraints.

No LLM calls — keyword heuristics and template expansion only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelfBuildSpec:
    """Everything needed to dispatch a self-build via Claude Code."""

    goal_description: str
    build_intent: str                # Actionable rewrite of goal
    target_postcodes: tuple[str, ...]
    target_files: tuple[str, ...]    # Inferred from postcodes
    protected_files: tuple[str, ...]
    boundary_rules: tuple[str, ...]
    test_command: str
    blueprint_context: str           # Relevant component excerpts
    prompt: str                      # The assembled Claude Code prompt


# ---------------------------------------------------------------------------
# Mapping tables — general, not per-capability
# ---------------------------------------------------------------------------

# Layer axis → likely module directories
_LAYER_TO_MODULE: dict[str, tuple[str, ...]] = {
    "INT": ("core",),
    "SEM": ("kernel",),
    "ORG": ("core", "mother"),
    "COG": ("mother", "agents"),
    "AGN": ("agents",),
    "STR": ("kernel",),
    "STA": ("kernel", "core"),
    "IDN": ("mother",),
    "TME": ("core",),
    "EXC": ("core",),
    "CTR": ("core",),
    "RES": ("core",),
    "OBS": ("kernel",),
    "NET": ("mother",),
    "EMG": ("kernel",),
    "MET": ("kernel", "core"),
    "DAT": ("core",),
    "SFX": ("core",),
}

# Concern axis → filename patterns to match
_CONCERN_TO_PATTERN: dict[str, tuple[str, ...]] = {
    "ENT": ("entity", "model", "type", "schema"),
    "BHV": ("behavior", "process", "action", "flow"),
    "FNC": ("function", "handler", "op", "util"),
    "REL": ("relation", "connect", "link", "graph"),
    "PLN": ("plan", "schedule", "sequence"),
    "MEM": ("memory", "store", "cache", "recall"),
    "ORC": ("orchestrat", "dispatch", "route"),
    "AGT": ("agent",),
    "ACT": ("actor", "executor"),
    "STA": ("state", "status", "snapshot"),
    "GTE": ("gate", "guard", "check", "verify"),
    "PLY": ("policy", "rule", "constraint"),
    "MET": ("metric", "measure", "score"),
    "LOG": ("log", "trace", "record"),
    "FLW": ("flow", "pipeline", "stream"),
    "PRV": ("provenance", "source", "trace"),
    "CNS": ("constraint", "limit", "bound"),
    "GOL": ("goal", "objective", "target"),
    "CFG": ("config", "setting", "preference"),
    "EMT": ("emit", "output", "write", "render"),
    "RED": ("read", "load", "fetch", "parse"),
    "TRC": ("trace", "provenance", "audit"),
}

# Goal category → actionable verb phrase
_CATEGORY_TO_ACTION: dict[str, str] = {
    "confidence": "strengthen implementation and add test coverage for",
    "coverage": "implement missing capability for",
    "quality": "improve output quality of",
    "resilience": "fix reliability issues in",
}

# Protected files — never modify
_PROTECTED_FILES: tuple[str, ...] = (
    "mother/context.py",
    "mother/persona.py",
    "mother/senses.py",
)

# Boundary rules — always included
_BOUNDARY_RULES: tuple[str, ...] = (
    "Avoid modifying these core files: mother/context.py, mother/persona.py, mother/senses.py",
    "mother/bridge.py is the ONLY file in mother/ that imports from core/",
    "New files must be LEAF modules (no circular imports)",
    "Use .venv/bin/python3.14 (no `python` alias)",
    "DomainAdapter is frozen — register via adapter_registry, never mutate",
    "engine.compile() returns CompileResult dataclass, NOT a dict",
)

_TEST_COMMAND = ".venv/bin/pytest tests/ -x -q"

# Layer descriptions for enriching diagnostic goals
_LAYER_DESCRIPTIONS: dict[str, str] = {
    "INT": "Intent (input processing)",
    "SEM": "Semantic (meaning extraction)",
    "ORG": "Organization (structural layout)",
    "COG": "Cognitive (reasoning and decision)",
    "AGN": "Agency (agent orchestration)",
    "STR": "Structure (grid/map architecture)",
    "STA": "State (lifecycle management)",
    "IDN": "Identity (self-model and persona)",
    "TME": "Time (scheduling and temporal)",
    "EXC": "Execution (runtime pipeline)",
    "CTR": "Control (flow control and gates)",
    "RES": "Resource (cost and allocation)",
    "OBS": "Observability (monitoring and feedback)",
    "NET": "Network (multi-instance and comms)",
    "EMG": "Emergence (pattern detection)",
    "MET": "Meta (self-referential)",
    "DAT": "Data (storage and retrieval)",
    "SFX": "Side Effects (external actions)",
}

_CONCERN_DESCRIPTIONS: dict[str, str] = {
    "ENT": "Entity modeling",
    "BHV": "Behavior/process",
    "FNC": "Function/handler",
    "REL": "Relationships",
    "PLN": "Planning",
    "MEM": "Memory/storage",
    "ORC": "Orchestration",
    "AGT": "Agent",
    "ACT": "Actor/executor",
    "STA": "State management",
    "GTE": "Gate/guard",
    "PLY": "Policy/constraint",
    "MET": "Metrics",
    "LOG": "Logging/tracing",
    "FLW": "Flow/pipeline",
    "PRV": "Provenance",
    "CNS": "Constraints",
    "GOL": "Goals",
    "CFG": "Configuration",
    "EMT": "Emission/output",
    "RED": "Read/load",
    "TRC": "Trace/audit",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def goal_to_build_intent(
    goal: dict,
    grid_cells: list[tuple[str, float, str, str]] | None = None,
    blueprint: dict | None = None,
) -> str:
    """Rewrite a diagnostic goal into an actionable build instruction.

    goal: dict with at least 'description', 'category', optionally 'target_postcodes'
    grid_cells: (postcode, confidence, fill_state, primitive) for context
    blueprint: optional compiled blueprint for additional context

    Returns a clear, actionable instruction string.
    """
    desc = goal.get("description", "")
    category = goal.get("category", "")
    postcodes = goal.get("target_postcodes", ())

    # Start with the action verb from category
    action = _CATEGORY_TO_ACTION.get(category, "improve")

    # Expand postcodes into human-readable territory descriptions
    territory = _describe_postcodes(postcodes)

    # Build the actionable intent
    parts = [action]
    if territory:
        parts.append(territory)
    else:
        parts.append(desc)

    intent = " ".join(parts)

    # Append specific cell context if available
    if grid_cells and postcodes:
        pc_set = set(postcodes)
        relevant = [
            (pc, conf, fs, prim)
            for pc, conf, fs, prim in grid_cells
            if pc in pc_set
        ]
        if relevant:
            cell_details = []
            for pc, conf, fs, prim in relevant[:5]:
                cell_details.append(f"  - {pc} ({prim}): {fs} at {conf:.0%}")
            intent += "\n\nTarget cells:\n" + "\n".join(cell_details)

    # Append blueprint component names if available
    if blueprint:
        components = blueprint.get("components", [])
        if components:
            names = [c.get("name", "") for c in components[:5] if c.get("name")]
            if names:
                intent += f"\n\nRelevant components: {', '.join(names)}"

    return intent


def blueprint_to_build_context(
    blueprint: dict,
    max_components: int = 5,
) -> str:
    """Extract relevant component info from a blueprint as structured text.

    Returns a compact string with component names, descriptions,
    constraints, and key methods.
    """
    components = blueprint.get("components", [])
    if not components:
        return ""

    lines = []
    for comp in components[:max_components]:
        name = comp.get("name", "unnamed")
        desc = comp.get("description", "")
        lines.append(f"## {name}")
        if desc:
            lines.append(f"  {desc}")

        # Constraints
        constraints = comp.get("constraints", [])
        if constraints:
            for c in constraints[:3]:
                if isinstance(c, dict):
                    lines.append(f"  - constraint: {c.get('description', c.get('name', str(c)))}")
                else:
                    lines.append(f"  - constraint: {c}")

        # Methods / properties
        methods = comp.get("methods", comp.get("properties", []))
        if methods:
            method_names = []
            for m in methods[:5]:
                if isinstance(m, dict):
                    method_names.append(m.get("name", str(m)))
                else:
                    method_names.append(str(m))
            if method_names:
                lines.append(f"  methods: {', '.join(method_names)}")

        # Relationships
        relationships = comp.get("relationships", [])
        if relationships:
            for r in relationships[:3]:
                if isinstance(r, dict):
                    target = r.get("target", r.get("to", ""))
                    rtype = r.get("type", r.get("relationship", ""))
                    if target:
                        lines.append(f"  -> {target} ({rtype})")

        lines.append("")

    # Also include top-level relationships
    top_rels = blueprint.get("relationships", [])
    if top_rels:
        lines.append("## Relationships")
        for r in top_rels[:8]:
            if isinstance(r, dict):
                src = r.get("source", r.get("from", ""))
                tgt = r.get("target", r.get("to", ""))
                rtype = r.get("type", r.get("relationship", ""))
                if src and tgt:
                    lines.append(f"  {src} -> {tgt} ({rtype})")

    return "\n".join(lines).strip()


def infer_target_files(
    postcodes: tuple[str, ...] | list[str],
    repo_dir: str | Path,
) -> tuple[str, ...]:
    """Map postcodes to likely .py files in the repo.

    Uses layer→module and concern→pattern tables to locate
    relevant source files. Returns relative paths.
    """
    if not postcodes:
        return ()

    repo = Path(repo_dir)
    modules: set[str] = set()
    patterns: set[str] = set()

    for pc_str in postcodes:
        parts = pc_str.split(".")
        if len(parts) >= 1:
            layer = parts[0]
            for mod in _LAYER_TO_MODULE.get(layer, ()):
                modules.add(mod)
        if len(parts) >= 2:
            concern = parts[1]
            for pat in _CONCERN_TO_PATTERN.get(concern, ()):
                patterns.add(pat)

    # Find matching files
    found: list[str] = []

    for mod in sorted(modules):
        mod_dir = repo / mod
        if not mod_dir.is_dir():
            continue
        for py_file in sorted(mod_dir.glob("*.py")):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            fname_lower = py_file.stem.lower()
            # If no patterns, include all non-test files in the module
            if not patterns:
                rel = str(py_file.relative_to(repo))
                if rel not in found:
                    found.append(rel)
            else:
                for pat in patterns:
                    if pat in fname_lower:
                        rel = str(py_file.relative_to(repo))
                        if rel not in found:
                            found.append(rel)
                        break

    return tuple(found[:20])  # Cap at 20 files


def assemble_self_build_prompt(
    build_intent: str,
    repo_dir: str | Path,
    target_files: tuple[str, ...] | None = None,
    target_postcodes: tuple[str, ...] = (),
    blueprint_context: str = "",
    learning_context: str = "",
) -> SelfBuildSpec:
    """Assemble a complete Claude Code prompt from components.

    Returns a SelfBuildSpec with the full prompt and metadata.
    """
    # Infer target files if not provided
    if target_files is None:
        target_files = infer_target_files(target_postcodes, repo_dir)

    # Build the prompt
    sections = []

    sections.append(f"TASK: {build_intent}")

    if target_files:
        sections.append(
            "TARGET FILES (investigate these first):\n"
            + "\n".join(f"  - {f}" for f in target_files)
        )

    if blueprint_context:
        sections.append(f"BLUEPRINT CONTEXT:\n{blueprint_context}")

    # Architectural rules
    rules = "GUIDELINES:\n" + "\n".join(f"- {r}" for r in _BOUNDARY_RULES)
    sections.append(f"{rules}")

    sections.append(f"SAFETY NOTE: Git snapshot taken before changes. After edits, run '{_TEST_COMMAND}' — must pass 100% (-x stops on first failure). If fail, diagnose and fix. Rollback automatic if tests fail.")
    sections.append(f"TEST COMMAND: {_TEST_COMMAND}")

    if learning_context:
        sections.append(f"LEARNING CONTEXT:\n{learning_context}")

    prompt = "\n\n".join(sections)

    return SelfBuildSpec(
        goal_description=build_intent,
        build_intent=build_intent,
        target_postcodes=target_postcodes,
        target_files=target_files,
        protected_files=_PROTECTED_FILES,
        boundary_rules=_BOUNDARY_RULES,
        test_command=_TEST_COMMAND,
        blueprint_context=blueprint_context,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _describe_postcodes(postcodes: tuple[str, ...] | list[str]) -> str:
    """Expand postcodes into human-readable territory descriptions."""
    if not postcodes:
        return ""

    layers: set[str] = set()
    concerns: set[str] = set()

    for pc_str in postcodes:
        parts = pc_str.split(".")
        if len(parts) >= 1:
            layers.add(parts[0])
        if len(parts) >= 2:
            concerns.add(parts[1])

    parts_out = []

    if layers:
        layer_names = [
            _LAYER_DESCRIPTIONS.get(l, l)
            for l in sorted(layers)
        ]
        parts_out.append("layers: " + ", ".join(layer_names))

    if concerns:
        concern_names = [
            _CONCERN_DESCRIPTIONS.get(c, c)
            for c in sorted(concerns)
        ]
        parts_out.append("concerns: " + ", ".join(concern_names))

    return " — ".join(parts_out) if parts_out else ""
