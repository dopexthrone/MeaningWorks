"""
Meeting preparation — agendas, briefings, and notes templates.

LEAF module (stdlib only). Generates meeting documents from goals,
compile context, and risk data.

Genome #134: meeting-preparing — prepares agendas, briefings, and notes.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MeetingAgenda:
    """A structured meeting agenda."""

    title: str
    objective: str
    date: str
    items: List[Dict[str, str]]          # [{topic, description, time_estimate}]
    success_criteria: List[str]
    attendee_notes: str = ""
    preparation: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Briefing:
    """A pre-meeting briefing document."""

    title: str
    executive_summary: str
    key_components: List[str]
    risks: List[str]
    open_questions: List[str]
    recent_progress: List[str]
    recommendations: List[str]


@dataclass(frozen=True)
class NotesTemplate:
    """A structured meeting notes template."""

    title: str
    date: str
    decisions: List[str]                  # pre-filled decision slots
    action_items: List[Dict[str, str]]    # [{task, owner, deadline}]
    open_questions: List[str]
    follow_up_date: str = ""


def generate_agenda(
    goal_description: str,
    domain: str = "software",
    steps: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    duration_minutes: int = 60,
) -> MeetingAgenda:
    """Generate a meeting agenda from a goal and its context."""
    title = _extract_meeting_title(goal_description)
    objective = f"Align on approach and next steps for: {goal_description[:100]}"

    # Build agenda items
    items: List[Dict[str, str]] = []

    # Opening
    items.append({
        "topic": "Context & Objectives",
        "description": "Review current state and meeting goals",
        "time_estimate": "5 min",
    })

    # Main items from steps
    if steps:
        per_step_time = max(5, (duration_minutes - 20) // len(steps))
        for i, step in enumerate(steps[:5]):
            items.append({
                "topic": step[:60],
                "description": f"Discuss approach and requirements for step {i + 1}",
                "time_estimate": f"{per_step_time} min",
            })
    else:
        # Default structure
        items.append({
            "topic": "Requirements Review",
            "description": f"Walk through {domain} requirements",
            "time_estimate": f"{(duration_minutes - 20) // 2} min",
        })
        items.append({
            "topic": "Approach & Trade-offs",
            "description": "Evaluate implementation options",
            "time_estimate": f"{(duration_minutes - 20) // 2} min",
        })

    # Risks if any
    if risks:
        items.append({
            "topic": "Risks & Mitigations",
            "description": f"{len(risks)} identified risk(s) to review",
            "time_estimate": "5 min",
        })

    # Closing
    items.append({
        "topic": "Action Items & Next Steps",
        "description": "Assign owners and deadlines",
        "time_estimate": "5 min",
    })

    # Success criteria
    success = [
        "Clear next steps with owners and deadlines",
        "All open questions addressed or escalated",
    ]
    if risks:
        success.append("Risk mitigations agreed upon")

    # Preparation
    prep: List[str] = []
    if steps:
        prep.append(f"Review the {len(steps)} planned steps")
    prep.append(f"Familiarize with the {domain} domain context")

    return MeetingAgenda(
        title=title,
        objective=objective,
        date=time.strftime("%Y-%m-%d"),
        items=items,
        success_criteria=success,
        preparation=prep,
    )


def generate_briefing(
    goal_description: str,
    components: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    recent_builds: Optional[List[str]] = None,
    open_questions: Optional[List[str]] = None,
) -> Briefing:
    """Generate a pre-meeting briefing document."""
    title = _extract_meeting_title(goal_description)

    summary = (
        f"This briefing covers the current state of: {goal_description[:150]}. "
        f"{'There are ' + str(len(risks)) + ' identified risks.' if risks else ''} "
        f"{'Recent progress includes ' + str(len(recent_builds or [])) + ' build(s).' if recent_builds else ''}"
    ).strip()

    recommendations: List[str] = []
    if risks and len(risks) > 2:
        recommendations.append("Address high-priority risks before proceeding")
    if not components:
        recommendations.append("Define system components before implementation")
    else:
        recommendations.append("Validate component boundaries with stakeholders")

    return Briefing(
        title=title,
        executive_summary=summary,
        key_components=components or [],
        risks=risks or [],
        open_questions=open_questions or [],
        recent_progress=recent_builds or [],
        recommendations=recommendations,
    )


def generate_notes_template(
    goal_description: str,
    steps: Optional[List[str]] = None,
) -> NotesTemplate:
    """Generate a structured meeting notes template."""
    title = _extract_meeting_title(goal_description)

    # Pre-fill decision slots
    decisions = [""] * 3  # 3 empty decision slots

    # Action items from steps
    action_items: List[Dict[str, str]] = []
    if steps:
        for step in steps[:5]:
            action_items.append({
                "task": step[:80],
                "owner": "",
                "deadline": "",
            })
    else:
        # Default empty slots
        for _ in range(3):
            action_items.append({
                "task": "",
                "owner": "",
                "deadline": "",
            })

    return NotesTemplate(
        title=title,
        date=time.strftime("%Y-%m-%d"),
        decisions=decisions,
        action_items=action_items,
        open_questions=[""],
    )


def format_agenda_markdown(agenda: MeetingAgenda) -> str:
    """Format a MeetingAgenda as Markdown."""
    lines = [f"# Meeting: {agenda.title}\n"]
    lines.append(f"**Date:** {agenda.date}")
    lines.append(f"**Objective:** {agenda.objective}\n")

    if agenda.preparation:
        lines.append("## Preparation\n")
        for p in agenda.preparation:
            lines.append(f"- {p}")
        lines.append("")

    lines.append("## Agenda\n")
    lines.append("| # | Topic | Time |")
    lines.append("|---|-------|------|")
    for i, item in enumerate(agenda.items, 1):
        topic = item.get("topic", "")
        time_est = item.get("time_estimate", "")
        lines.append(f"| {i} | {topic} | {time_est} |")
    lines.append("")

    if agenda.success_criteria:
        lines.append("## Success Criteria\n")
        for c in agenda.success_criteria:
            lines.append(f"- [ ] {c}")
        lines.append("")

    return "\n".join(lines)


def format_briefing_markdown(briefing: Briefing) -> str:
    """Format a Briefing as Markdown."""
    lines = [f"# Briefing: {briefing.title}\n"]

    lines.append("## Summary\n")
    lines.append(briefing.executive_summary + "\n")

    if briefing.key_components:
        lines.append("## Key Components\n")
        for comp in briefing.key_components:
            lines.append(f"- {comp}")
        lines.append("")

    if briefing.recent_progress:
        lines.append("## Recent Progress\n")
        for p in briefing.recent_progress:
            lines.append(f"- {p}")
        lines.append("")

    if briefing.risks:
        lines.append("## Risks\n")
        for r in briefing.risks:
            lines.append(f"- {r}")
        lines.append("")

    if briefing.open_questions:
        lines.append("## Open Questions\n")
        for q in briefing.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    if briefing.recommendations:
        lines.append("## Recommendations\n")
        for r in briefing.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def format_notes_markdown(notes: NotesTemplate) -> str:
    """Format a NotesTemplate as Markdown."""
    lines = [f"# Meeting Notes: {notes.title}\n"]
    lines.append(f"**Date:** {notes.date}\n")

    lines.append("## Decisions\n")
    for i, d in enumerate(notes.decisions, 1):
        lines.append(f"{i}. {d or '_________________'}")
    lines.append("")

    lines.append("## Action Items\n")
    lines.append("| Task | Owner | Deadline |")
    lines.append("|------|-------|----------|")
    for item in notes.action_items:
        task = item.get("task", "") or "_________________"
        owner = item.get("owner", "") or "___"
        deadline = item.get("deadline", "") or "___"
        lines.append(f"| {task} | {owner} | {deadline} |")
    lines.append("")

    lines.append("## Open Questions\n")
    for q in notes.open_questions:
        lines.append(f"- {q or '_________________'}")
    lines.append("")

    if notes.follow_up_date:
        lines.append(f"**Next meeting:** {notes.follow_up_date}\n")

    return "\n".join(lines)


# --- Internal helpers ---

def _extract_meeting_title(description: str) -> str:
    """Extract a meeting title from a goal description."""
    first_sentence = description.split(".")[0].strip()
    if len(first_sentence) <= 60:
        return first_sentence
    return first_sentence[:57] + "..."
