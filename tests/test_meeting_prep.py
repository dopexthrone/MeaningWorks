"""Tests for mother/meeting_prep.py.

Covers MeetingAgenda, Briefing, NotesTemplate, generate_* and format_*_markdown.
"""

import pytest
from mother.meeting_prep import (
    MeetingAgenda,
    Briefing,
    NotesTemplate,
    generate_agenda,
    generate_briefing,
    generate_notes_template,
    format_agenda_markdown,
    format_briefing_markdown,
    format_notes_markdown,
    _extract_meeting_title,
)


# === Dataclass tests ===

class TestMeetingAgenda:
    def test_frozen(self):
        agenda = MeetingAgenda(
            title="Test", objective="Obj", date="2026-01-01",
            items=[], success_criteria=[],
        )
        with pytest.raises(AttributeError):
            agenda.title = "Changed"

    def test_default_fields(self):
        agenda = MeetingAgenda(
            title="T", objective="O", date="D",
            items=[], success_criteria=[],
        )
        assert agenda.attendee_notes == ""
        assert agenda.preparation == []


class TestBriefing:
    def test_frozen(self):
        briefing = Briefing(
            title="T", executive_summary="S",
            key_components=[], risks=[],
            open_questions=[], recent_progress=[],
            recommendations=[],
        )
        with pytest.raises(AttributeError):
            briefing.title = "Changed"


class TestNotesTemplate:
    def test_frozen(self):
        notes = NotesTemplate(
            title="T", date="D",
            decisions=[], action_items=[],
            open_questions=[],
        )
        with pytest.raises(AttributeError):
            notes.title = "Changed"

    def test_default_follow_up(self):
        notes = NotesTemplate(
            title="T", date="D",
            decisions=[], action_items=[],
            open_questions=[],
        )
        assert notes.follow_up_date == ""


# === generate_agenda ===

class TestGenerateAgenda:
    def test_basic_generation(self):
        agenda = generate_agenda("Review auth implementation")
        assert isinstance(agenda, MeetingAgenda)
        assert agenda.title
        assert agenda.objective
        assert agenda.date
        assert len(agenda.items) >= 2  # at least opening + closing

    def test_with_steps(self):
        steps = ["Design API", "Write tests", "Deploy"]
        agenda = generate_agenda("Build feature X", steps=steps)
        # Should have opening + 3 step items + closing = 5
        assert len(agenda.items) >= 5

    def test_without_steps_default_structure(self):
        agenda = generate_agenda("Build feature X")
        topics = [item.get("topic", "") for item in agenda.items]
        topic_text = " ".join(topics)
        assert "Requirements" in topic_text or "Context" in topic_text

    def test_with_risks(self):
        agenda = generate_agenda("Build it", risks=["Data loss", "Downtime"])
        topics = [item.get("topic", "") for item in agenda.items]
        assert any("Risk" in t for t in topics)

    def test_without_risks_no_risk_section(self):
        agenda = generate_agenda("Build it")
        topics = [item.get("topic", "") for item in agenda.items]
        assert not any("Risk" in t for t in topics)

    def test_success_criteria(self):
        agenda = generate_agenda("Build it")
        assert "Clear next steps" in agenda.success_criteria[0]

    def test_success_criteria_includes_risk_mitigation(self):
        agenda = generate_agenda("Build it", risks=["r1"])
        criteria_text = " ".join(agenda.success_criteria)
        assert "mitigation" in criteria_text.lower() or "risk" in criteria_text.lower()

    def test_preparation(self):
        agenda = generate_agenda("Build it", steps=["a", "b"])
        assert any("2" in p for p in agenda.preparation)

    def test_domain_in_preparation(self):
        agenda = generate_agenda("Build it", domain="api")
        prep_text = " ".join(agenda.preparation)
        assert "api" in prep_text

    def test_custom_duration(self):
        agenda = generate_agenda("Build it", duration_minutes=30)
        # All time estimates should be shorter
        for item in agenda.items:
            time_str = item.get("time_estimate", "0 min")
            minutes = int(time_str.split()[0])
            assert minutes <= 30

    def test_steps_capped_at_5(self):
        steps = [f"Step {i}" for i in range(10)]
        agenda = generate_agenda("Build it", steps=steps)
        step_items = [i for i in agenda.items
                      if "step" in i.get("description", "").lower()]
        assert len(step_items) <= 5


# === generate_briefing ===

class TestGenerateBriefing:
    def test_basic_generation(self):
        briefing = generate_briefing("Review auth implementation")
        assert isinstance(briefing, Briefing)
        assert briefing.title
        assert briefing.executive_summary

    def test_with_components(self):
        briefing = generate_briefing("Build it", components=["Auth", "DB"])
        assert briefing.key_components == ["Auth", "DB"]

    def test_without_components(self):
        briefing = generate_briefing("Build it")
        assert briefing.key_components == []
        rec_text = " ".join(briefing.recommendations)
        assert "define" in rec_text.lower() or "component" in rec_text.lower()

    def test_with_risks(self):
        briefing = generate_briefing("Build it", risks=["r1", "r2", "r3"])
        assert briefing.risks == ["r1", "r2", "r3"]
        rec_text = " ".join(briefing.recommendations)
        assert "risk" in rec_text.lower()

    def test_with_recent_builds(self):
        briefing = generate_briefing("Build it", recent_builds=["v1.0", "v1.1"])
        assert briefing.recent_progress == ["v1.0", "v1.1"]
        assert "2 build" in briefing.executive_summary.lower()

    def test_with_open_questions(self):
        briefing = generate_briefing("Build it", open_questions=["What API?"])
        assert briefing.open_questions == ["What API?"]

    def test_recommendations_always_present(self):
        briefing = generate_briefing("Build it")
        assert len(briefing.recommendations) > 0


# === generate_notes_template ===

class TestGenerateNotesTemplate:
    def test_basic_generation(self):
        notes = generate_notes_template("Review auth")
        assert isinstance(notes, NotesTemplate)
        assert notes.title
        assert notes.date

    def test_decision_slots(self):
        notes = generate_notes_template("Review auth")
        assert len(notes.decisions) == 3
        assert all(d == "" for d in notes.decisions)

    def test_with_steps(self):
        notes = generate_notes_template("Build it", steps=["a", "b", "c"])
        assert len(notes.action_items) == 3
        assert notes.action_items[0]["task"] == "a"

    def test_without_steps_empty_slots(self):
        notes = generate_notes_template("Build it")
        assert len(notes.action_items) == 3
        assert all(item["task"] == "" for item in notes.action_items)

    def test_action_items_have_owner_deadline(self):
        notes = generate_notes_template("Build it", steps=["do X"])
        item = notes.action_items[0]
        assert "owner" in item
        assert "deadline" in item

    def test_open_questions_slot(self):
        notes = generate_notes_template("Build it")
        assert len(notes.open_questions) >= 1

    def test_steps_capped_at_5(self):
        steps = [f"Step {i}" for i in range(10)]
        notes = generate_notes_template("Build it", steps=steps)
        assert len(notes.action_items) <= 5


# === format_agenda_markdown ===

class TestFormatAgendaMarkdown:
    def test_contains_title(self):
        agenda = generate_agenda("Review auth")
        md = format_agenda_markdown(agenda)
        assert "# Meeting:" in md

    def test_contains_date(self):
        agenda = generate_agenda("Review auth")
        md = format_agenda_markdown(agenda)
        assert agenda.date in md

    def test_contains_table(self):
        agenda = generate_agenda("Review auth")
        md = format_agenda_markdown(agenda)
        assert "| # | Topic | Time |" in md

    def test_contains_success_criteria(self):
        agenda = generate_agenda("Review auth")
        md = format_agenda_markdown(agenda)
        assert "Success Criteria" in md
        assert "- [ ]" in md

    def test_contains_preparation(self):
        agenda = generate_agenda("Review auth", steps=["a"])
        md = format_agenda_markdown(agenda)
        assert "Preparation" in md


# === format_briefing_markdown ===

class TestFormatBriefingMarkdown:
    def test_contains_title(self):
        briefing = generate_briefing("Review auth")
        md = format_briefing_markdown(briefing)
        assert "# Briefing:" in md

    def test_contains_summary(self):
        briefing = generate_briefing("Review auth")
        md = format_briefing_markdown(briefing)
        assert "Summary" in md

    def test_contains_risks_section(self):
        briefing = generate_briefing("Build it", risks=["r1"])
        md = format_briefing_markdown(briefing)
        assert "Risks" in md
        assert "r1" in md

    def test_contains_recommendations(self):
        briefing = generate_briefing("Build it")
        md = format_briefing_markdown(briefing)
        assert "Recommendations" in md

    def test_components_section(self):
        briefing = generate_briefing("Build it", components=["Auth"])
        md = format_briefing_markdown(briefing)
        assert "Key Components" in md
        assert "Auth" in md

    def test_no_empty_sections(self):
        briefing = generate_briefing("Build it")
        md = format_briefing_markdown(briefing)
        # No risks → no Risks section
        assert "Risks" not in md


# === format_notes_markdown ===

class TestFormatNotesMarkdown:
    def test_contains_title(self):
        notes = generate_notes_template("Review auth")
        md = format_notes_markdown(notes)
        assert "# Meeting Notes:" in md

    def test_contains_decisions(self):
        notes = generate_notes_template("Review auth")
        md = format_notes_markdown(notes)
        assert "Decisions" in md

    def test_contains_action_items_table(self):
        notes = generate_notes_template("Review auth")
        md = format_notes_markdown(notes)
        assert "| Task | Owner | Deadline |" in md

    def test_contains_open_questions(self):
        notes = generate_notes_template("Review auth")
        md = format_notes_markdown(notes)
        assert "Open Questions" in md

    def test_follow_up_date_shown(self):
        notes = NotesTemplate(
            title="T", date="D",
            decisions=[], action_items=[],
            open_questions=[],
            follow_up_date="2026-03-01",
        )
        md = format_notes_markdown(notes)
        assert "2026-03-01" in md
        assert "Next meeting" in md

    def test_no_follow_up_hidden(self):
        notes = generate_notes_template("Review auth")
        md = format_notes_markdown(notes)
        assert "Next meeting" not in md


# === _extract_meeting_title ===

class TestExtractMeetingTitle:
    def test_short_description(self):
        title = _extract_meeting_title("Review the auth system")
        assert title == "Review the auth system"

    def test_long_description_truncates(self):
        desc = "A" * 100
        title = _extract_meeting_title(desc)
        assert len(title) <= 60
        assert title.endswith("...")

    def test_multi_sentence_uses_first(self):
        title = _extract_meeting_title("Fix the bug. Then ship it.")
        assert title == "Fix the bug"
