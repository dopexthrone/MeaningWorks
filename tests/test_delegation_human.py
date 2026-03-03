"""Tests for human delegation in mother/delegation.py.

Covers HumanDelegationTask, generate_delegation_task(),
format_delegation_markdown(), format_delegation_json().
"""

import pytest
from mother.delegation import (
    HumanDelegationTask,
    generate_delegation_task,
    format_delegation_markdown,
    format_delegation_json,
    _extract_title,
    _extract_acceptance_criteria,
    _estimate_effort,
)


# === HumanDelegationTask dataclass ===

class TestHumanDelegationTask:
    def test_required_fields(self):
        task = HumanDelegationTask(
            title="Fix the login bug",
            description="Users cannot log in with special chars in password",
            context="Auth module uses bcrypt",
            acceptance_criteria=["Login works with special chars"],
        )
        assert task.title == "Fix the login bug"
        assert task.description == "Users cannot log in with special chars in password"
        assert task.context == "Auth module uses bcrypt"
        assert task.acceptance_criteria == ["Login works with special chars"]

    def test_default_fields(self):
        task = HumanDelegationTask(
            title="T", description="D", context="C",
            acceptance_criteria=["Done"],
        )
        assert task.priority == "normal"
        assert task.deadline == ""
        assert task.estimated_effort == ""
        assert task.related_files == []
        assert task.prior_attempts == []
        assert task.escalation_path == ""

    def test_optional_fields(self):
        task = HumanDelegationTask(
            title="T", description="D", context="C",
            acceptance_criteria=["Done"],
            priority="urgent",
            deadline="2026-03-01",
            estimated_effort="2 hours",
            related_files=["auth.py", "login.py"],
            prior_attempts=["Tried regex, didn't work"],
            escalation_path="Ask team lead",
        )
        assert task.priority == "urgent"
        assert task.deadline == "2026-03-01"
        assert task.related_files == ["auth.py", "login.py"]
        assert task.prior_attempts == ["Tried regex, didn't work"]


# === generate_delegation_task ===

class TestGenerateDelegationTask:
    def test_basic_generation(self):
        task = generate_delegation_task("Fix the authentication bug in login flow")
        assert isinstance(task, HumanDelegationTask)
        assert task.title  # non-empty
        assert task.description == "Fix the authentication bug in login flow"
        assert len(task.acceptance_criteria) > 0
        assert task.priority == "normal"

    def test_software_domain_criteria(self):
        task = generate_delegation_task("Refactor the database module", domain="software")
        criteria_text = " ".join(task.acceptance_criteria)
        assert "compiles" in criteria_text.lower() or "runs" in criteria_text.lower()
        assert "tested" in criteria_text.lower() or "verified" in criteria_text.lower()

    def test_process_domain_criteria(self):
        task = generate_delegation_task("Document the onboarding flow", domain="process")
        criteria_text = " ".join(task.acceptance_criteria)
        assert "documented" in criteria_text.lower() or "validated" in criteria_text.lower()

    def test_api_domain_criteria(self):
        task = generate_delegation_task("Add pagination to /users endpoint", domain="api")
        criteria_text = " ".join(task.acceptance_criteria)
        assert "endpoint" in criteria_text.lower() or "api" in criteria_text.lower()

    def test_priority_forwarded(self):
        task = generate_delegation_task("Urgent fix", priority="urgent")
        assert task.priority == "urgent"

    def test_deadline_forwarded(self):
        task = generate_delegation_task("Ship it", deadline="tomorrow")
        assert task.deadline == "tomorrow"

    def test_context_forwarded(self):
        task = generate_delegation_task("Fix bug", context="The module was refactored last week")
        assert "refactored" in task.context

    def test_context_default_includes_domain(self):
        task = generate_delegation_task("Fix bug", domain="api")
        assert "api" in task.context.lower()

    def test_prior_attempts_forwarded(self):
        task = generate_delegation_task("Fix bug", prior_attempts=["Tried X", "Tried Y"])
        assert task.prior_attempts == ["Tried X", "Tried Y"]

    def test_related_files_forwarded(self):
        task = generate_delegation_task("Fix bug", related_files=["a.py", "b.py"])
        assert task.related_files == ["a.py", "b.py"]

    def test_escalation_path_set(self):
        task = generate_delegation_task("Fix bug")
        assert task.escalation_path  # non-empty default

    def test_always_includes_communicate_back(self):
        task = generate_delegation_task("Any task")
        criteria_text = " ".join(task.acceptance_criteria).lower()
        assert "mother" in criteria_text or "communicated" in criteria_text


# === format_delegation_markdown ===

class TestFormatDelegationMarkdown:
    def test_contains_title(self):
        task = generate_delegation_task("Build the widget")
        md = format_delegation_markdown(task)
        assert "# Task:" in md

    def test_contains_priority(self):
        task = generate_delegation_task("Build the widget", priority="high")
        md = format_delegation_markdown(task)
        assert "high" in md

    def test_contains_deadline(self):
        task = generate_delegation_task("Build it", deadline="2026-03-15")
        md = format_delegation_markdown(task)
        assert "2026-03-15" in md

    def test_contains_acceptance_criteria(self):
        task = generate_delegation_task("Build it")
        md = format_delegation_markdown(task)
        assert "Acceptance Criteria" in md
        assert "- [ ]" in md

    def test_contains_related_files(self):
        task = generate_delegation_task("Fix it", related_files=["core/engine.py"])
        md = format_delegation_markdown(task)
        assert "core/engine.py" in md

    def test_contains_prior_attempts(self):
        task = generate_delegation_task("Fix it", prior_attempts=["Tried regex"])
        md = format_delegation_markdown(task)
        assert "Tried regex" in md

    def test_contains_escalation(self):
        task = generate_delegation_task("Fix it")
        md = format_delegation_markdown(task)
        assert "Escalation" in md


# === format_delegation_json ===

class TestFormatDelegationJson:
    def test_returns_dict(self):
        task = generate_delegation_task("Build widget")
        result = format_delegation_json(task)
        assert isinstance(result, dict)

    def test_all_fields_present(self):
        task = generate_delegation_task("Build widget", priority="high",
                                         deadline="tomorrow", related_files=["a.py"])
        result = format_delegation_json(task)
        assert "title" in result
        assert "description" in result
        assert "context" in result
        assert "acceptance_criteria" in result
        assert "priority" in result
        assert "deadline" in result
        assert "estimated_effort" in result
        assert "related_files" in result
        assert "prior_attempts" in result
        assert "escalation_path" in result

    def test_values_match_task(self):
        task = generate_delegation_task("Build widget", priority="urgent")
        result = format_delegation_json(task)
        assert result["priority"] == "urgent"
        assert result["description"] == "Build widget"


# === Internal helpers ===

class TestExtractTitle:
    def test_short_description(self):
        title = _extract_title("Fix the bug")
        assert title == "Fix the bug"

    def test_long_description_truncates(self):
        desc = "A" * 200
        title = _extract_title(desc)
        assert len(title) <= 80
        assert title.endswith("...")

    def test_multi_sentence_uses_first(self):
        title = _extract_title("Fix the bug. Then refactor the module.")
        assert title == "Fix the bug"


class TestEstimateEffort:
    def test_short_description(self):
        effort = _estimate_effort("Fix bug", "software")
        assert "minute" in effort or "hour" in effort

    def test_medium_description(self):
        desc = " ".join(["word"] * 40)
        effort = _estimate_effort(desc, "software")
        assert "hour" in effort

    def test_long_description(self):
        desc = " ".join(["word"] * 120)
        effort = _estimate_effort(desc, "software")
        assert "day" in effort or "half" in effort
