"""
Tests for mother/context.py — context synthesis (substance over costume).
"""

import pytest

from mother.context import (
    ContextData,
    synthesize_frame,
    synthesize_situation,
    synthesize_context,
)


# --- ContextData construction ---


class TestContextData:
    """Test ContextData frozen dataclass."""

    def test_default_construction(self):
        data = ContextData()
        assert data.name == "Mother"
        assert data.provider == "unknown"
        assert data.cap_chat is True
        assert data.corpus_total == 0
        assert data.total_sessions == 0

    def test_frozen(self):
        data = ContextData()
        with pytest.raises(AttributeError):
            data.name = "changed"

    def test_custom_fields(self):
        data = ContextData(
            name="David",
            provider="grok",
            model="grok-3",
            corpus_total=10,
            total_sessions=5,
        )
        assert data.name == "David"
        assert data.provider == "grok"
        assert data.corpus_total == 10
        assert data.total_sessions == 5

    def test_dict_fields_default_empty(self):
        data = ContextData()
        assert data.corpus_domains == {}
        assert data.tool_domains == {}

    def test_list_fields_default_empty(self):
        data = ContextData()
        assert data.recent_topics == []

    def test_optional_fields_default_none(self):
        data = ContextData()
        assert data.days_since_last is None
        assert data.last_compile_desc is None
        assert data.env_time is None

    def test_capability_flags(self):
        data = ContextData(
            cap_voice=True,
            cap_screen_capture=True,
            cap_microphone=True,
            cap_camera=True,
            cap_perception=True,
        )
        assert data.cap_voice is True
        assert data.cap_screen_capture is True
        assert data.cap_microphone is True
        assert data.cap_camera is True
        assert data.cap_perception is True

    def test_sense_trajectory_defaults(self):
        data = ContextData()
        assert data.rapport_trend == 0.0
        assert data.confidence_trend == 0.0
        assert data.peak_confidence == 0.5
        assert data.peak_rapport == 0.0


# --- synthesize_frame ---


class TestSynthesizeFrame:
    """Test frame synthesis — identity, capabilities, rules."""

    def test_contains_name(self):
        data = ContextData(name="David")
        frame = synthesize_frame(data)
        assert "David" in frame

    def test_contains_provider_model(self):
        data = ContextData(provider="claude", model="claude-sonnet-4-5-20250929")
        frame = synthesize_frame(data)
        assert "claude/claude-sonnet-4-5-20250929" in frame

    def test_contains_motherlabs(self):
        frame = synthesize_frame(ContextData())
        assert "Motherlabs" in frame

    def test_contains_platform(self):
        data = ContextData(platform="darwin")
        frame = synthesize_frame(data)
        assert "darwin" in frame

    def test_auto_detects_platform(self):
        import sys
        data = ContextData(platform="")
        frame = synthesize_frame(data)
        assert sys.platform in frame

    def test_active_capabilities_basic(self):
        data = ContextData()
        frame = synthesize_frame(data)
        assert "chat" in frame
        assert "compile" in frame
        assert "build" in frame

    def test_active_capabilities_voice(self):
        data = ContextData(cap_voice=True)
        frame = synthesize_frame(data)
        assert "voice" in frame

    def test_active_capabilities_screen(self):
        data = ContextData(cap_screen_capture=True)
        frame = synthesize_frame(data)
        assert "screen capture" in frame

    def test_active_capabilities_microphone(self):
        data = ContextData(cap_microphone=True)
        frame = synthesize_frame(data)
        assert "microphone" in frame

    def test_active_capabilities_camera(self):
        data = ContextData(cap_camera=True)
        frame = synthesize_frame(data)
        assert "camera" in frame

    def test_active_capabilities_perception(self):
        data = ContextData(cap_perception=True)
        frame = synthesize_frame(data)
        assert "ambient perception" in frame

    def test_contains_rules(self):
        frame = synthesize_frame(ContextData())
        assert "Rules:" in frame
        assert "preamble" in frame.lower()
        assert "invent" in frame.lower()

    def test_contains_commands(self):
        frame = synthesize_frame(ContextData())
        assert "/compile" in frame
        assert "/build" in frame
        assert "/help" in frame
        assert "/settings" in frame

    def test_contains_time(self):
        data = ContextData(env_time="2:30pm", env_date="Feb 14", env_timezone="GMT")
        frame = synthesize_frame(data)
        assert "2:30pm" in frame
        assert "Feb 14" in frame
        assert "GMT" in frame

    def test_auto_populates_time(self):
        frame = synthesize_frame(ContextData())
        # Should have some time string — not empty
        assert "m " in frame.lower() or ":" in frame  # am/pm or HH:MM

    def test_frame_is_compact(self):
        """Frame should be under 800 chars (~200 tokens)."""
        data = ContextData(
            name="David", provider="claude", model="claude-sonnet-4-5-20250929",
            platform="darwin", cap_voice=True, cap_file_access=True,
            cap_screen_capture=True, cap_microphone=True,
            env_time="2:30pm", env_date="Feb 14", env_timezone="GMT",
        )
        frame = synthesize_frame(data)
        assert len(frame) < 850

    def test_no_costume_language(self):
        """Frame should NOT contain persona costume language."""
        frame = synthesize_frame(ContextData())
        assert "warm the way a good colleague" not in frame
        assert "smart person, not like an AI" not in frame
        assert "corporate warmth" not in frame

    def test_anti_confabulation_rules(self):
        """Frame must contain rules preventing invented experiences."""
        frame = synthesize_frame(ContextData())
        assert "[Context]" in frame  # References the context section
        assert "invent" in frame.lower()
        assert "didn't happen" in frame.lower() or "didn't happen" in frame

    def test_conversation_freedom(self):
        """Frame allows free conversation, doesn't force work redirect."""
        frame = synthesize_frame(ContextData())
        assert "follow the user" in frame.lower() or "conversation is free" in frame.lower()
        assert "focus on the project" not in frame.lower()


# --- synthesize_situation ---


class TestSynthesizeSituation:
    """Test situation synthesis — real accumulated data."""

    def test_new_user_minimal(self):
        data = ContextData()
        situation = synthesize_situation(data)
        assert "First session" in situation
        assert "No prior history" in situation

    def test_new_user_has_session_cost(self):
        data = ContextData(session_cost=0.001, session_cost_limit=5.0)
        situation = synthesize_situation(data)
        assert "$0.001" in situation
        assert "$5.00" in situation

    def test_returning_user_sessions(self):
        data = ContextData(total_sessions=12, total_messages=247)
        situation = synthesize_situation(data)
        assert "12 sessions" in situation
        assert "247 messages" in situation

    def test_returning_user_age(self):
        data = ContextData(total_sessions=5, total_messages=50, instance_age_days=18)
        situation = synthesize_situation(data)
        assert "18 days" in situation

    def test_returning_user_days_since_last(self):
        data = ContextData(total_sessions=5, total_messages=50, days_since_last=2.0)
        situation = synthesize_situation(data)
        assert "2 day" in situation

    def test_returning_user_hours_since_last(self):
        data = ContextData(total_sessions=5, total_messages=50, days_since_last=0.25)
        situation = synthesize_situation(data)
        assert "6h ago" in situation

    def test_recent_topics(self):
        data = ContextData(
            total_sessions=3, total_messages=30,
            recent_topics=["tattoo booking system", "inventory tracker"],
        )
        situation = synthesize_situation(data)
        assert "tattoo booking system" in situation
        assert "inventory tracker" in situation

    def test_topics_capped_at_3(self):
        data = ContextData(
            total_sessions=5, total_messages=50,
            recent_topics=["topic1", "topic2", "topic3", "topic4"],
        )
        situation = synthesize_situation(data)
        assert "topic4" not in situation

    def test_corpus_stats(self):
        data = ContextData(
            corpus_total=8,
            corpus_success_rate=0.875,
            corpus_domains={"software": 6, "api": 2},
            corpus_total_components=47,
            corpus_avg_trust=72.0,
        )
        situation = synthesize_situation(data)
        assert "8 compilations" in situation
        assert "87%" in situation  # 87.5 -> 87
        assert "software" in situation
        assert "47 components" in situation
        assert "72%" in situation

    def test_corpus_absent_when_zero(self):
        data = ContextData(corpus_total=0)
        situation = synthesize_situation(data)
        assert "compilation" not in situation.lower() or "Session" in situation

    def test_tools_shown_when_present(self):
        data = ContextData(tool_count=5, tool_verified_count=3)
        situation = synthesize_situation(data)
        assert "5 tools" in situation
        assert "3 verified" in situation

    def test_tools_absent_when_zero(self):
        data = ContextData(tool_count=0)
        situation = synthesize_situation(data)
        assert "tool" not in situation.lower() or "compilation" in situation.lower()

    def test_last_compile_shown(self):
        data = ContextData(
            last_compile_desc="tattoo booking system",
            last_compile_trust=78.0,
            last_compile_components=12,
            last_compile_weakest="completeness",
        )
        situation = synthesize_situation(data)
        assert "tattoo booking system" in situation
        assert "78%" in situation
        assert "12 components" in situation
        assert "completeness" in situation

    def test_last_compile_absent_when_none(self):
        data = ContextData(last_compile_desc=None)
        situation = synthesize_situation(data)
        assert "Last compile" not in situation

    def test_last_compile_desc_truncated(self):
        data = ContextData(last_compile_desc="a" * 120)
        situation = synthesize_situation(data)
        # The desc in output should be <= 80 chars
        assert "a" * 81 not in situation

    def test_session_cost_shown(self):
        data = ContextData(session_cost=0.003, session_cost_limit=5.0, session_messages=4)
        situation = synthesize_situation(data)
        assert "$0.003" in situation
        assert "$5.00" in situation
        assert "4 messages" in situation

    def test_session_compilations_shown(self):
        data = ContextData(session_compilations=2)
        situation = synthesize_situation(data)
        assert "2 compilations" in situation

    def test_session_errors_shown(self):
        data = ContextData(session_errors=1)
        situation = synthesize_situation(data)
        assert "1 error" in situation

    def test_trajectory_confidence_up(self):
        data = ContextData(confidence_trend=0.15)
        situation = synthesize_situation(data)
        assert "Confidence up" in situation

    def test_trajectory_confidence_down(self):
        data = ContextData(confidence_trend=-0.2)
        situation = synthesize_situation(data)
        assert "Confidence down" in situation

    def test_trajectory_rapport_growing(self):
        data = ContextData(rapport_trend=0.1)
        situation = synthesize_situation(data)
        assert "Rapport growing" in situation

    def test_trajectory_peak(self):
        data = ContextData(peak_confidence=0.82)
        situation = synthesize_situation(data)
        assert "Peak: 0.82" in situation

    def test_no_trajectory_when_flat(self):
        data = ContextData(confidence_trend=0.01, rapport_trend=0.01, peak_confidence=0.5)
        situation = synthesize_situation(data)
        assert "Confidence" not in situation
        assert "Rapport" not in situation
        assert "Peak" not in situation

    def test_situation_compact(self):
        """Full-data situation should be under 2200 chars (~550 tokens)."""
        data = ContextData(
            total_sessions=12, total_messages=247, instance_age_days=18,
            days_since_last=2.0,
            recent_topics=["tattoo booking system", "inventory tracker"],
            corpus_total=8, corpus_success_rate=0.875,
            corpus_domains={"software": 6, "api": 2},
            corpus_total_components=47, corpus_avg_trust=72.0,
            tool_count=5, tool_verified_count=3,
            last_compile_desc="tattoo booking system",
            last_compile_trust=78.0, last_compile_components=12,
            last_compile_weakest="completeness",
            session_messages=4, session_cost=0.003, session_cost_limit=5.0,
            session_compilations=1,
            confidence_trend=0.15, peak_confidence=0.82,
        )
        situation = synthesize_situation(data)
        assert len(situation) < 2200


# --- synthesize_context ---


class TestSynthesizeContext:
    """Test full context synthesis."""

    def test_new_user_context(self):
        data = ContextData(
            name="David", provider="claude", model="claude-sonnet-4-5-20250929",
            platform="darwin",
            env_time="11:05pm", env_date="Feb 13", env_timezone="GMT",
        )
        ctx = synthesize_context(data)
        assert "David" in ctx
        assert "First session" in ctx
        assert "[Context]" in ctx

    def test_returning_user_context(self):
        data = ContextData(
            name="David", provider="claude", model="claude-sonnet-4-5-20250929",
            platform="darwin", total_sessions=12, total_messages=247,
            corpus_total=8, corpus_success_rate=0.875,
            env_time="2:30pm", env_date="Feb 14", env_timezone="GMT",
        )
        ctx = synthesize_context(data)
        assert "12 sessions" in ctx
        assert "8 compilations" in ctx

    def test_includes_sense_block(self):
        data = ContextData()
        sense = "Stance: You have momentum. Push deeper."
        ctx = synthesize_context(data, sense_block=sense)
        assert "Stance:" in ctx
        assert "momentum" in ctx

    def test_no_sense_block_when_none(self):
        data = ContextData()
        ctx = synthesize_context(data, sense_block=None)
        assert "Stance:" not in ctx

    def test_no_costume_in_full_context(self):
        """Full context must not contain old costume language."""
        data = ContextData(
            name="David", provider="grok", model="grok-3",
            total_sessions=5, total_messages=80,
        )
        ctx = synthesize_context(data, sense_block="Stance: Be direct.")
        # None of the old PERSONA_BASE content
        assert "Talk like a smart person" not in ctx
        assert "corporate warmth" not in ctx
        assert "You are Mother. You live on this machine" not in ctx
        assert "local AI entity running as a native application" not in ctx
        assert "Personality:" not in ctx

    def test_full_context_under_limit(self):
        """Full context with max data should be under ~3400 chars."""
        data = ContextData(
            name="David", provider="claude", model="claude-sonnet-4-5-20250929",
            platform="darwin", instance_age_days=18,
            cap_voice=True, cap_screen_capture=True, cap_microphone=True,
            cap_claude_code=True, cap_autonomous=True, cap_perception=True,
            total_sessions=12, total_messages=247, days_since_last=2.0,
            recent_topics=["tattoo booking system", "inventory tracker"],
            corpus_total=8, corpus_success_rate=0.875,
            corpus_domains={"software": 6, "api": 2},
            corpus_total_components=47, corpus_avg_trust=72.0,
            tool_count=5, tool_verified_count=3,
            last_compile_desc="tattoo booking system",
            last_compile_trust=78.0, last_compile_components=12,
            last_compile_weakest="completeness",
            session_messages=4, session_cost=0.003, session_cost_limit=5.0,
            confidence_trend=0.15, peak_confidence=0.82,
            env_time="2:30pm", env_date="Feb 14", env_timezone="GMT",
            autonomous_working=True, autonomous_session_cost=0.50,
            autonomous_actions_count=5, autonomous_budget=2.0,
            goal_details=[
                {"description": "Build autonomic mode", "priority": "high", "health": 0.85},
                {"description": "Fix perception loop", "priority": "normal", "health": 0.6},
            ],
            perception_poll_seconds=10.0, perception_budget_hourly=0.50,
            perception_modes=["screen", "camera"],
            # Body map
            codebase_total_files=91, codebase_total_lines=112000,
            codebase_test_count=4648,
            codebase_modules={"mother": 24, "core": 18, "adapters": 5, "platform": 9},
            codebase_protected=["mother/context.py", "mother/persona.py", "mother/senses.py"],
            codebase_boundary="bridge.py is the only mother→core import path",
            last_build_files_changed=3, last_build_lines_delta="+52/-11",
            last_build_modules_touched=["mother", "core"],
        )
        sense = "Stance: You have momentum. Push deeper. Real rapport exists."
        ctx = synthesize_context(data, sense_block=sense)
        assert len(ctx) < 3400

    def test_single_session_singular(self):
        data = ContextData(total_sessions=1, total_messages=5)
        situation = synthesize_situation(data)
        assert "1 session," in situation

    def test_single_tool_singular(self):
        data = ContextData(tool_count=1)
        situation = synthesize_situation(data)
        assert "1 tool" in situation
        assert "1 tools" not in situation

    def test_single_compilation_singular(self):
        data = ContextData(corpus_total=1, corpus_success_rate=1.0)
        situation = synthesize_situation(data)
        assert "1 compilation" in situation
        assert "1 compilations" not in situation

    def test_single_error_singular(self):
        data = ContextData(session_errors=1)
        situation = synthesize_situation(data)
        assert "1 error" in situation
        assert "1 errors" not in situation

    def test_multiple_errors_plural(self):
        data = ContextData(session_errors=3)
        situation = synthesize_situation(data)
        assert "3 errors" in situation


# ============================================================
# Neurologis Automatica — new field extensions
# ============================================================

class TestNeurologisContextFields:

    def test_temporal_in_situation(self):
        data = ContextData(temporal_context="Idle for 5 minutes. Evening session.")
        situation = synthesize_situation(data)
        assert "Idle for 5 minutes" in situation

    def test_recall_in_context(self):
        data = ContextData(recall_block="[Recalled]\n- user: \"tattoo booking\"")
        ctx = synthesize_context(data)
        assert "[Recalled]" in ctx
        assert "tattoo booking" in ctx

    def test_recall_placement_order(self):
        """Recall block should appear between [Context] and stance."""
        data = ContextData(
            recall_block="[Recalled]\n- user: \"test\"",
        )
        sense = "Stance: Be direct."
        ctx = synthesize_context(data, sense_block=sense)
        recall_pos = ctx.find("[Recalled]")
        stance_pos = ctx.find("Stance:")
        context_pos = ctx.find("[Context]")
        assert context_pos < recall_pos < stance_pos

    def test_empty_recall_omitted(self):
        data = ContextData(recall_block="")
        ctx = synthesize_context(data)
        assert "[Recalled]" not in ctx

    def test_backward_compatibility(self):
        """ContextData without new fields still works."""
        data = ContextData(
            name="David", provider="grok", model="grok-3",
            total_sessions=5, total_messages=80,
        )
        ctx = synthesize_context(data)
        assert "David" in ctx
        assert "[Recalled]" not in ctx


# ============================================================
# Operational awareness (Phase B) — journal + error fields
# ============================================================

class TestOperationalAwarenessContext:

    def test_journal_in_situation(self):
        """Journal data should appear in situation when builds exist."""
        data = ContextData(
            journal_total_builds=7,
            journal_avg_trust=72.5,
            journal_success_streak=3,
            journal_total_cost=1.25,
        )
        situation = synthesize_situation(data)
        assert "7 builds" in situation
        assert "avg trust 72%" in situation
        assert "3 in a row" in situation
        assert "$1.25 total" in situation

    def test_journal_negative_streak(self):
        """Negative streak should show consecutive failures."""
        data = ContextData(
            journal_total_builds=5,
            journal_success_streak=-2,
        )
        situation = synthesize_situation(data)
        assert "2 consecutive failures" in situation

    def test_error_summary_in_situation(self):
        """Error summary should appear in situation."""
        data = ContextData(error_summary="Errors: 2 connection, 1 auth. 2 retriable.")
        situation = synthesize_situation(data)
        assert "Errors: 2 connection" in situation

    def test_journal_omitted_when_zero(self):
        """No journal block when journal_total_builds is 0."""
        data = ContextData(journal_total_builds=0)
        situation = synthesize_situation(data)
        assert "Builds:" not in situation

    def test_operational_backward_compatibility(self):
        """ContextData without Phase B fields still works."""
        data = ContextData(
            name="Mother", provider="grok",
            total_sessions=3, total_messages=40,
        )
        ctx = synthesize_context(data)
        assert "Mother" in ctx
        assert "Builds:" not in ctx


# ============================================================
# Capability flags — grounded capabilities in frame
# ============================================================

class TestCapabilityFlags:
    """New capability flags: claude_code, autonomous, conditional compile/build."""

    def test_claude_code_in_frame(self):
        data = ContextData(cap_claude_code=True)
        frame = synthesize_frame(data)
        assert "self-build" in frame

    def test_autonomous_in_frame(self):
        data = ContextData(cap_autonomous=True)
        frame = synthesize_frame(data)
        assert "autonomic" in frame

    def test_compile_absent_when_disabled(self):
        data = ContextData(cap_compile=False)
        frame = synthesize_frame(data)
        active_line = [l for l in frame.split("\n") if l.startswith("Active:")][0]
        assert "compile" not in active_line

    def test_build_absent_when_disabled(self):
        data = ContextData(cap_build=False)
        frame = synthesize_frame(data)
        active_line = [l for l in frame.split("\n") if l.startswith("Active:")][0]
        assert "build" not in active_line

    def test_compile_build_present_by_default(self):
        data = ContextData()
        frame = synthesize_frame(data)
        active_line = [l for l in frame.split("\n") if l.startswith("Active:")][0]
        assert "compile" in active_line
        assert "build" in active_line


# ============================================================
# Autonomic context — runtime state in situation
# ============================================================

class TestAutonomicContext:
    """Autonomic operating mode state rendering in situation."""

    def test_autonomic_idle(self):
        data = ContextData(
            cap_autonomous=True,
            autonomous_working=False,
            autonomous_actions_count=3,
            autonomous_session_cost=0.15,
            autonomous_budget=1.0,
        )
        situation = synthesize_situation(data)
        assert "Autonomic: idle, 3 actions, $0.15/$1.00." in situation

    def test_autonomic_active(self):
        data = ContextData(
            cap_autonomous=True,
            autonomous_working=True,
            autonomous_actions_count=7,
            autonomous_session_cost=0.42,
            autonomous_budget=2.0,
        )
        situation = synthesize_situation(data)
        assert "Autonomic: ACTIVE" in situation

    def test_autonomic_absent_when_disabled(self):
        data = ContextData(cap_autonomous=False)
        situation = synthesize_situation(data)
        assert "Autonomic:" not in situation

    def test_autonomic_zero_state(self):
        data = ContextData(
            cap_autonomous=True,
            autonomous_working=False,
            autonomous_actions_count=0,
            autonomous_session_cost=0.0,
            autonomous_budget=1.0,
        )
        situation = synthesize_situation(data)
        assert "Autonomic: idle, 0 actions, $0.00/$1.00." in situation


# ============================================================
# Enriched goals — goal_details with priority + health
# ============================================================

class TestEnrichedGoals:
    """Goal details rendering with priority and health."""

    def test_enriched_goals_with_priority_and_health(self):
        data = ContextData(goal_details=[
            {"description": "Build autonomic mode", "priority": "high", "health": 0.85},
            {"description": "Fix perception loop", "priority": "urgent", "health": 0.3},
        ])
        situation = synthesize_situation(data)
        assert "[high] Build autonomic mode (85%)" in situation
        assert "[urgent] Fix perception loop (30%)" in situation

    def test_normal_priority_omits_prefix(self):
        data = ContextData(goal_details=[
            {"description": "Review code", "priority": "normal", "health": 0.5},
        ])
        situation = synthesize_situation(data)
        assert "Active goals:" in situation
        assert "[normal]" not in situation
        assert "Review code (50%)" in situation

    def test_empty_goal_details_falls_through(self):
        """When goal_details is empty, active_goals (plain strings) still render."""
        data = ContextData(
            goal_details=[],
            active_goals=["Build something", "Fix something"],
        )
        situation = synthesize_situation(data)
        assert "Build something" in situation
        assert "Fix something" in situation

    def test_goals_capped_at_5(self):
        data = ContextData(goal_details=[
            {"description": f"Goal {i}", "priority": "normal", "health": 0.5}
            for i in range(7)
        ])
        situation = synthesize_situation(data)
        assert "Goal 4" in situation
        assert "Goal 5" not in situation


# ============================================================
# Perception context — config rendering in situation
# ============================================================

class TestPerceptionContext:
    """Perception config rendering in situation."""

    def test_perception_with_modes_poll_budget(self):
        data = ContextData(
            cap_perception=True,
            perception_modes=["screen", "camera"],
            perception_poll_seconds=10.0,
            perception_budget_hourly=0.50,
        )
        situation = synthesize_situation(data)
        assert "Perception: screen, camera, poll 10s, $0.50/hr." in situation

    def test_perception_absent_when_inactive(self):
        data = ContextData(cap_perception=False, perception_modes=["screen"])
        situation = synthesize_situation(data)
        assert "Perception:" not in situation

    def test_perception_absent_when_no_modes(self):
        data = ContextData(cap_perception=True, perception_modes=[])
        situation = synthesize_situation(data)
        assert "Perception:" not in situation


# ============================================================
# Codebase body map — structural self-knowledge in situation
# ============================================================

class TestCodebaseContext:
    """Codebase topology rendering in situation."""

    def test_codebase_shown_when_populated(self):
        data = ContextData(
            codebase_total_files=91,
            codebase_total_lines=112000,
            codebase_test_count=4648,
            codebase_modules={"mother": 24, "core": 18, "adapters": 5},
        )
        situation = synthesize_situation(data)
        assert "Codebase:" in situation
        assert "91 files" in situation
        assert "112K LOC" in situation
        assert "4648 tests" in situation
        assert "mother (24)" in situation
        assert "core (18)" in situation

    def test_codebase_absent_when_zero(self):
        data = ContextData(codebase_total_files=0)
        situation = synthesize_situation(data)
        assert "Codebase:" not in situation

    def test_protected_short_names(self):
        data = ContextData(
            codebase_total_files=91,
            codebase_total_lines=50000,
            codebase_protected=["mother/context.py", "mother/persona.py", "mother/senses.py"],
            codebase_boundary="bridge.py is the only mother→core import path",
        )
        situation = synthesize_situation(data)
        assert "Protected: context.py, persona.py, senses.py." in situation
        assert "bridge.py is the only" in situation


class TestBuildDeltaContext:
    """Build delta rendering in situation."""

    def test_delta_shown_when_populated(self):
        data = ContextData(
            last_build_files_changed=3,
            last_build_lines_delta="+52/-11",
            last_build_modules_touched=["mother", "core"],
        )
        situation = synthesize_situation(data)
        assert "Last build:" in situation
        assert "3 files" in situation
        assert "[mother, core]" in situation
        assert "+52/-11 lines" in situation

    def test_delta_absent_when_zero(self):
        data = ContextData(last_build_files_changed=0)
        situation = synthesize_situation(data)
        assert "Last build:" not in situation
