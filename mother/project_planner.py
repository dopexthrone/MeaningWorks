"""
mother/project_planner.py — Translate compiled blueprints into code engine prompts.

LEAF module. Stdlib only. No imports from core/ or mother/.

Given a blueprint (dict from engine.compile() → CompileResult.blueprint),
this module renders it into a structured prompt that the native code engine
or CLI coding agents can execute to produce a runnable project.

No LLM calls — keyword heuristics and template expansion only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectBuildSpec:
    """Everything needed to dispatch a project build via the code engine."""

    project_name: str           # Slugified name
    project_dir: str            # Absolute target path
    blueprint_summary: str      # Compact blueprint rendering for prompt
    component_count: int
    domain: str
    core_need: str
    language: str               # Default "python"
    framework_hints: tuple[str, ...]
    prompt: str                 # Assembled user prompt
    system_prompt: str          # Specialized system prompt for green-field projects
    trust_score: float          # From verification, for display


# ---------------------------------------------------------------------------
# Language / framework detection tables
# ---------------------------------------------------------------------------

_FRAMEWORK_SIGNALS: dict[str, tuple[str, ...]] = {
    # JS/TS frontend
    "react": ("react", "jsx", "tsx", "next", "nextjs", "next.js"),
    "vue": ("vue", "vuejs", "vue.js", "nuxt"),
    "svelte": ("svelte", "sveltekit"),
    "angular": ("angular",),
    # JS/TS backend
    "express": ("express", "expressjs"),
    "fastify": ("fastify",),
    "nestjs": ("nestjs", "nest.js", "nest"),
    # Python
    "flask": ("flask",),
    "django": ("django",),
    "fastapi": ("fastapi", "fast api"),
    "streamlit": ("streamlit",),
    # Rust
    "actix": ("actix",),
    "axum": ("axum",),
    # Go
    "gin": ("gin",),
    "fiber": ("fiber",),
}

_LANGUAGE_SIGNALS: dict[str, tuple[str, ...]] = {
    "typescript": ("typescript", "tsx", "react", "next", "nextjs",
                   "angular", "vue", "svelte", "nestjs"),
    "javascript": ("javascript", "js", "jsx", "node", "express", "fastify"),
    "python": ("python", "flask", "django", "fastapi", "streamlit", "pip",
               "pytest", "poetry"),
    "rust": ("rust", "cargo", "actix", "axum", "tokio"),
    "go": ("go", "golang", "gin", "fiber"),
    "java": ("java", "spring", "maven", "gradle"),
    "swift": ("swift", "swiftui", "ios"),
    "kotlin": ("kotlin", "android"),
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def infer_project_name(blueprint: dict) -> str:
    """Infer a slugified project name from blueprint domain/core_need.

    Falls back to 'project' if nothing useful is found.
    """
    candidates = []
    for key in ("core_need", "domain", "description", "name"):
        val = blueprint.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    if not candidates:
        return "project"

    # Use the first candidate, slugify it
    raw = candidates[0]
    # Keep only alphanumeric and spaces, then slugify
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", raw)
    words = cleaned.lower().split()
    # Cap at 4 words to keep names reasonable
    slug = "-".join(words[:4])
    return slug or "project"


def extract_language_and_framework(blueprint: dict) -> tuple[str, tuple[str, ...]]:
    """Scan blueprint for language and framework signals.

    Checks component constraints, types, descriptions, and top-level fields
    for keywords that indicate a language or framework.

    Returns (language, framework_hints). Defaults to ("python", ()).
    """
    # Collect all text from blueprint into a searchable corpus
    text_parts: list[str] = []
    for key in ("description", "core_need", "domain", "language", "framework",
                "tech_stack", "stack"):
        val = blueprint.get(key)
        if isinstance(val, str):
            text_parts.append(val.lower())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    text_parts.append(item.lower())

    components = blueprint.get("components", [])
    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue
            for field in ("name", "description", "type"):
                v = comp.get(field)
                if isinstance(v, str):
                    text_parts.append(v.lower())
            constraints = comp.get("constraints", [])
            if isinstance(constraints, list):
                for c in constraints:
                    if isinstance(c, str):
                        text_parts.append(c.lower())
                    elif isinstance(c, dict):
                        for cv in c.values():
                            if isinstance(cv, str):
                                text_parts.append(cv.lower())

    corpus = " ".join(text_parts)
    if not corpus.strip():
        return ("python", ())

    # Detect frameworks
    found_frameworks: list[str] = []
    for framework, signals in _FRAMEWORK_SIGNALS.items():
        for signal in signals:
            if signal in corpus:
                if framework not in found_frameworks:
                    found_frameworks.append(framework)
                break

    # Detect language — score by signal hits
    lang_scores: dict[str, int] = {}
    for lang, signals in _LANGUAGE_SIGNALS.items():
        score = 0
        for signal in signals:
            if signal in corpus:
                score += 1
        if score > 0:
            lang_scores[lang] = score

    if lang_scores:
        language = max(lang_scores, key=lambda k: lang_scores[k])
    else:
        language = "python"

    return (language, tuple(found_frameworks))


def blueprint_to_project_context(
    blueprint: dict,
    max_components: int = 20,
) -> str:
    """Render a full blueprint into prompt context.

    Includes components (name, description, methods, constraints),
    relationships, and top-level metadata. Capped at max_components.
    """
    lines: list[str] = []

    # Top-level metadata
    for key in ("domain", "core_need", "description"):
        val = blueprint.get(key)
        if isinstance(val, str) and val.strip():
            lines.append(f"{key}: {val.strip()}")

    # Components
    components = blueprint.get("components", [])
    if isinstance(components, list) and components:
        lines.append("")
        lines.append(f"## Components ({len(components)} total)")
        for comp in components[:max_components]:
            if not isinstance(comp, dict):
                continue
            name = comp.get("name", "unnamed")
            desc = comp.get("description", "")
            lines.append(f"\n### {name}")
            if desc:
                lines.append(f"  {desc}")

            # Type/role
            comp_type = comp.get("type", "")
            if comp_type:
                lines.append(f"  type: {comp_type}")

            # Constraints
            constraints = comp.get("constraints", [])
            if isinstance(constraints, list):
                for c in constraints[:5]:
                    if isinstance(c, dict):
                        lines.append(f"  - constraint: {c.get('description', c.get('name', str(c)))}")
                    elif isinstance(c, str):
                        lines.append(f"  - constraint: {c}")

            # Methods / properties
            methods = comp.get("methods", comp.get("properties", []))
            if isinstance(methods, list) and methods:
                method_names = []
                for m in methods[:8]:
                    if isinstance(m, dict):
                        method_names.append(m.get("name", str(m)))
                    elif isinstance(m, str):
                        method_names.append(m)
                if method_names:
                    lines.append(f"  methods: {', '.join(method_names)}")

            # Relationships on component
            rels = comp.get("relationships", [])
            if isinstance(rels, list):
                for r in rels[:3]:
                    if isinstance(r, dict):
                        target = r.get("target", r.get("to", ""))
                        rtype = r.get("type", r.get("relationship", ""))
                        if target:
                            lines.append(f"  -> {target} ({rtype})")

        if len(components) > max_components:
            lines.append(f"\n  ... and {len(components) - max_components} more components")

    # Top-level relationships
    top_rels = blueprint.get("relationships", [])
    if isinstance(top_rels, list) and top_rels:
        lines.append("")
        lines.append("## Relationships")
        for r in top_rels[:15]:
            if isinstance(r, dict):
                src = r.get("source", r.get("from", ""))
                tgt = r.get("target", r.get("to", ""))
                rtype = r.get("type", r.get("relationship", ""))
                if src and tgt:
                    lines.append(f"  {src} -> {tgt} ({rtype})")

    # Insights
    insights = blueprint.get("insights", [])
    if isinstance(insights, list) and insights:
        lines.append("")
        lines.append("## Insights")
        for ins in insights[:5]:
            if isinstance(ins, dict):
                lines.append(f"  - {ins.get('description', ins.get('insight', str(ins)))}")
            elif isinstance(ins, str):
                lines.append(f"  - {ins}")

    return "\n".join(lines).strip()


def build_project_system_prompt(
    language: str,
    framework_hints: tuple[str, ...],
    project_dir: str,
) -> str:
    """Build system prompt for green-field project creation.

    This is NOT the self-modification prompt — it instructs creation
    of a new standalone project from scratch.
    """
    framework_line = ""
    if framework_hints:
        framework_line = f"\nFrameworks: {', '.join(framework_hints)}."

    return f"""You are a senior software engineer building a new project from scratch.

Language: {language}.{framework_line}
Project directory: {project_dir}

RULES:
- Create the complete project structure: directories, source files, config, dependencies.
- Write ALL implementation code — no stubs, no placeholders, no TODOs.
- Include a dependency manifest (requirements.txt / package.json / Cargo.toml / go.mod as appropriate).
- Include an entry point that runs the application.
- Include basic tests that verify core functionality.
- After writing all files, run the entry point to verify it starts without errors.
- After that, run the test suite to verify tests pass.
- If anything fails, diagnose and fix immediately — do not leave broken code.
- Follow {language} best practices and idiomatic patterns.
- Keep the implementation practical and production-ready, not over-engineered.
- Write clear, self-documenting code. Comments only where logic is non-obvious."""


def assemble_project_build_spec(
    blueprint: dict,
    verification: dict,
    output_dir: str,
    project_name: str = "",
    language: str = "",
) -> ProjectBuildSpec:
    """Main entry point. Blueprint → frozen ProjectBuildSpec ready for code engine.

    blueprint: dict from CompileResult.blueprint
    verification: dict from CompileResult.verification
    output_dir: base directory for projects (e.g., ~/motherlabs/projects)
    project_name: override slug (auto-inferred if empty)
    language: override language (auto-detected if empty)
    """
    # Infer project name
    name = project_name.strip() if project_name else infer_project_name(blueprint)

    # Detect language and framework
    detected_lang, framework_hints = extract_language_and_framework(blueprint)
    lang = language.strip() if language else detected_lang

    # Build project directory path
    project_dir = os.path.join(output_dir, name)

    # Render blueprint context
    blueprint_summary = blueprint_to_project_context(blueprint)

    # Count components
    components = blueprint.get("components", [])
    component_count = len(components) if isinstance(components, list) else 0

    # Extract domain and core_need
    domain = blueprint.get("domain", "")
    if isinstance(domain, str):
        domain = domain.strip()
    else:
        domain = ""
    core_need = blueprint.get("core_need", "")
    if isinstance(core_need, str):
        core_need = core_need.strip()
    else:
        core_need = ""

    # Trust score from verification
    trust_score = 0.0
    if isinstance(verification, dict):
        trust_score = float(verification.get("overall_score", 0.0))

    # System prompt
    system_prompt = build_project_system_prompt(lang, framework_hints, project_dir)

    # Assemble the user prompt
    prompt_sections: list[str] = []

    prompt_sections.append(
        f"Build a complete {lang} project in {project_dir}."
    )

    if core_need:
        prompt_sections.append(f"Core requirement: {core_need}")

    if blueprint_summary:
        prompt_sections.append(f"BLUEPRINT:\n{blueprint_summary}")

    prompt_sections.append(
        "Create the full project: directory structure, all source files, "
        "dependency manifest, entry point, and tests. "
        "Run the entry point and tests to verify everything works."
    )

    prompt = "\n\n".join(prompt_sections)

    return ProjectBuildSpec(
        project_name=name,
        project_dir=project_dir,
        blueprint_summary=blueprint_summary,
        component_count=component_count,
        domain=domain,
        core_need=core_need,
        language=lang,
        framework_hints=framework_hints,
        prompt=prompt,
        system_prompt=system_prompt,
        trust_score=trust_score,
    )
