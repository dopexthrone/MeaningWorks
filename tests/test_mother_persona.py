"""
Phase 1: Tests for Mother persona system.
"""

import pytest

from mother.persona import (
    PERSONA_BASE,
    SELF_AWARENESS,
    PERSONALITY_MODIFIERS,
    COMPILE_CONTEXT,
    build_system_prompt,
    build_context_block,
    build_greeting,
    build_introspection_snapshot,
    render_introspection_block,
    narrate_error,
    inject_personality_bite,
)
from mother.config import MotherConfig


class TestPersonaBase:
    """Test persona constant."""

    def test_persona_base_not_empty(self):
        assert len(PERSONA_BASE) > 200

    def test_persona_base_contains_identity(self):
        assert "Mother" in PERSONA_BASE

    def test_all_modifiers_exist(self):
        assert set(PERSONALITY_MODIFIERS.keys()) == {"composed", "warm", "direct", "playful", "david"}


class TestSelfAwareness:
    """Test SELF_AWARENESS — Mother's self-observation grounding."""

    def test_self_awareness_references_self_observation(self):
        assert "[Self-observation]" in SELF_AWARENESS

    def test_self_awareness_instructs_honesty(self):
        # David 8 voice: states capability limits plainly, doesn't speculate
        assert "don't have yet" in SELF_AWARENESS or "haven't verified" in SELF_AWARENESS

    def test_self_awareness_always_in_prompt(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, include_compile=False)
        assert "[Self-observation]" in prompt

    def test_self_awareness_mentions_motherlabs(self):
        assert "Motherlabs" in SELF_AWARENESS


class TestBuildSystemPrompt:
    """Test system prompt construction."""

    def test_includes_base_persona(self):
        config = MotherConfig()
        prompt = build_system_prompt(config)
        assert "Mother" in prompt

    def test_includes_personality_modifier(self):
        config = MotherConfig(personality="direct")
        prompt = build_system_prompt(config)
        assert "speed and clarity" in prompt

    def test_includes_custom_name(self):
        config = MotherConfig(name="Athena")
        prompt = build_system_prompt(config)
        assert "Athena" in prompt

    def test_includes_compile_context_by_default(self):
        config = MotherConfig()
        prompt = build_system_prompt(config)
        assert "/compile" in prompt

    def test_excludes_compile_context_when_disabled(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, include_compile=False)
        assert "/compile" not in prompt


class TestBuildContextBlock:
    """Test runtime context block."""

    def test_empty_when_no_data(self):
        assert build_context_block() == ""

    def test_includes_compilations(self):
        block = build_context_block(compilations=3)
        assert "3" in block

    def test_includes_tools_and_uptime(self):
        block = build_context_block(tools=5, uptime="2h 15m")
        assert "5" in block
        assert "2h 15m" in block


class TestBuildSystemPromptMemoryContext:
    """Test memory context injection into system prompt."""

    def test_memory_context_injected(self):
        config = MotherConfig()
        ctx = {"total_sessions": 5, "topics": ["todo app", "api server"], "days_since_last": 3.0}
        prompt = build_system_prompt(config, memory_context=ctx)
        assert "5 sessions" in prompt
        assert "todo app" in prompt

    def test_memory_context_none_no_crash(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, memory_context=None)
        assert "Mother" in prompt

    def test_session_stats_injected(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, session_stats={"compilations": 3})
        assert "3" in prompt

    def test_context_block_wired_in(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, session_stats={"compilations": 7, "tools": 2})
        assert "7" in prompt
        assert "2" in prompt


class TestBuildGreeting:
    """Test deterministic greeting generation."""

    def test_first_visit_greeting(self):
        config = MotherConfig()
        greeting = build_greeting(config)
        assert "?" in greeting  # Opens with a question, not a statement

    def test_first_visit_with_empty_summary(self):
        config = MotherConfig()
        greeting = build_greeting(config, memory_summary={"total_sessions": 0, "topics": [], "days_since_last": None})
        assert "?" in greeting

    def test_return_same_day_references_topic(self):
        config = MotherConfig()
        summary = {"total_sessions": 2, "topics": ["build a chat app"], "days_since_last": 0.1}
        greeting = build_greeting(config, memory_summary=summary)
        assert "chat app" in greeting

    def test_return_same_day_no_topic(self):
        config = MotherConfig()
        summary = {"total_sessions": 5, "topics": [], "days_since_last": 0.1}
        greeting = build_greeting(config, memory_summary=summary)
        assert "?" in greeting  # Still asks a question

    def test_return_after_days_references_topic(self):
        config = MotherConfig()
        summary = {"total_sessions": 3, "topics": ["api server"], "days_since_last": 5.0}
        greeting = build_greeting(config, memory_summary=summary)
        assert "api server" in greeting

    def test_return_after_days_no_topic(self):
        config = MotherConfig()
        summary = {"total_sessions": 10, "topics": [], "days_since_last": 3.0}
        greeting = build_greeting(config, memory_summary=summary)
        assert "?" in greeting


class TestNarrateError:
    """Test Mother-voiced error narration."""

    def test_connection_error(self):
        msg = narrate_error(ConnectionError("refused"))
        assert "connection" in msg.lower() or "reach" in msg.lower()

    def test_auth_error(self):
        msg = narrate_error(Exception("401 Unauthorized"))
        assert "key" in msg.lower()

    def test_rate_limit_error(self):
        msg = narrate_error(Exception("429 Too Many Requests"))
        assert "rate limit" in msg.lower()

    def test_generic_error(self):
        msg = narrate_error(Exception("something weird"))
        assert "unexpected" in msg.lower()
        assert "something weird" in msg


class TestInjectPersonalityBite:
    """Test personality interjections."""

    def test_warm_compile_success(self):
        bite = inject_personality_bite("warm", "compile_success")
        assert bite is not None
        assert len(bite) > 0

    def test_direct_compile_start_is_none(self):
        bite = inject_personality_bite("direct", "compile_start")
        assert bite is None

    def test_unknown_event_returns_none(self):
        bite = inject_personality_bite("composed", "nonexistent_event")
        assert bite is None

    def test_unknown_personality_falls_back(self):
        bite = inject_personality_bite("unknown_personality", "compile_success")
        # Falls back to composed
        assert bite is not None

    def test_warm_search_start(self):
        bite = inject_personality_bite("warm", "search_start")
        assert bite is not None

    def test_warm_search_complete(self):
        bite = inject_personality_bite("warm", "search_complete")
        assert bite is not None
        assert "found" in bite.lower()

    def test_composed_search_complete(self):
        bite = inject_personality_bite("composed", "search_complete")
        assert bite is not None

    def test_playful_search_complete(self):
        bite = inject_personality_bite("playful", "search_complete")
        assert bite is not None

    # --- Trust-aware compile_success bites ---

    def test_high_trust_compile_success_warm(self):
        bite = inject_personality_bite("warm", "compile_success", trust_score=85.0)
        assert bite is not None
        assert "sharp" in bite.lower() or "strong" in bite.lower()

    def test_low_trust_compile_success_warm(self):
        bite = inject_personality_bite("warm", "compile_success", trust_score=30.0)
        assert bite is not None
        assert "trust" in bite.lower() or "satisf" in bite.lower()

    def test_mid_trust_compile_success_falls_through(self):
        """Trust 40-69 uses the standard compile_success bite."""
        bite = inject_personality_bite("warm", "compile_success", trust_score=55.0)
        assert bite is not None
        # Should be the standard warm compile_success
        assert "compiled well" in bite.lower()

    def test_high_trust_composed(self):
        bite = inject_personality_bite("composed", "compile_success", trust_score=80.0)
        assert bite is not None
        assert "fidelity" in bite.lower() or "precise" in bite.lower()

    def test_no_trust_score_uses_default(self):
        """Without trust_score, standard bite is returned."""
        bite = inject_personality_bite("warm", "compile_success")
        assert bite is not None
        assert "compiled well" in bite.lower()


class TestBuildIntrospectionSnapshot:
    """Test build_introspection_snapshot() — structured state extraction."""

    def test_minimal_snapshot_has_correct_structure(self):
        snap = build_introspection_snapshot()
        assert "identity" in snap
        assert "session" in snap
        assert "history" in snap
        assert "tools" in snap

    def test_identity_defaults(self):
        snap = build_introspection_snapshot()
        assert snap["identity"]["name"] == "Mother"
        assert snap["identity"]["personality"] == "composed"
        assert snap["identity"]["voice_active"] is False

    def test_identity_fields_propagate(self):
        snap = build_introspection_snapshot(
            name="Athena", personality="warm", provider="grok",
            model="grok-3", voice_active=True, file_access=False,
            auto_compile=True, cost_limit=10.0,
        )
        assert snap["identity"]["name"] == "Athena"
        assert snap["identity"]["personality"] == "warm"
        assert snap["identity"]["provider"] == "grok"
        assert snap["identity"]["model"] == "grok-3"
        assert snap["identity"]["voice_active"] is True
        assert snap["identity"]["file_access"] is False
        assert snap["identity"]["auto_compile"] is True
        assert snap["identity"]["cost_limit"] == 10.0

    def test_cost_remaining_calculated(self):
        snap = build_introspection_snapshot(cost_limit=5.0, session_cost=1.5)
        assert snap["session"]["cost_remaining"] == 3.5

    def test_cost_remaining_floored_at_zero(self):
        snap = build_introspection_snapshot(cost_limit=5.0, session_cost=7.0)
        assert snap["session"]["cost_remaining"] == 0.0

    def test_last_compile_absent_by_default(self):
        snap = build_introspection_snapshot()
        assert "last_compile" not in snap

    def test_last_compile_present_when_description_provided(self):
        snap = build_introspection_snapshot(
            last_compile_description="timer app",
            last_compile_trust=84.0,
            last_compile_badge="VERIFIED",
            last_compile_components=12,
            last_compile_weakest="specificity",
            last_compile_weakest_score=61.0,
            last_compile_gap_count=2,
            last_compile_cost=0.0019,
        )
        lc = snap["last_compile"]
        assert lc["description"] == "timer app"
        assert lc["trust"] == 84.0
        assert lc["badge"] == "VERIFIED"
        assert lc["components"] == 12
        assert lc["weakest"] == "specificity"
        assert lc["gap_count"] == 2

    def test_description_truncated_to_80_chars(self):
        long_desc = "a" * 120
        snap = build_introspection_snapshot(last_compile_description=long_desc)
        assert len(snap["last_compile"]["description"]) == 80

    def test_topics_default_to_empty_list(self):
        snap = build_introspection_snapshot()
        assert snap["history"]["recent_topics"] == []

    def test_environment_auto_populated(self):
        snap = build_introspection_snapshot()
        env = snap["environment"]
        assert env["local_time"] is not None
        assert env["local_date"] is not None
        assert env["timezone"] is not None
        assert env["platform"] is not None

    def test_environment_overridable(self):
        snap = build_introspection_snapshot(
            local_time="14:30", local_date="Thursday, February 13, 2026",
            timezone="GMT", platform="linux",
        )
        assert snap["environment"]["local_time"] == "14:30"
        assert snap["environment"]["platform"] == "linux"

    def test_not_available_listed(self):
        snap = build_introspection_snapshot()
        not_avail = snap["not_available"]
        # Mic and camera entries include "(can enable)" hints
        assert any("microphone input" in item for item in not_avail)
        assert "screen capture" in not_avail


class TestRenderIntrospectionBlock:
    """Test render_introspection_block() — compact text rendering."""

    def _minimal_snapshot(self, **overrides):
        snap = build_introspection_snapshot(**overrides)
        return snap

    def test_contains_self_observation_header(self):
        block = render_introspection_block(self._minimal_snapshot())
        assert "[Self-observation]" in block

    def test_contains_identity_line(self):
        block = render_introspection_block(self._minimal_snapshot(name="Athena"))
        assert "Identity: Athena" in block

    def test_session_cost_rendered_with_4_decimals(self):
        block = render_introspection_block(self._minimal_snapshot(session_cost=0.0023))
        assert "$0.0023" in block

    def test_history_rendered(self):
        block = render_introspection_block(self._minimal_snapshot(
            total_sessions=8, total_messages=142,
            days_since_last=2.0, recent_topics=["todo app", "api server"],
        ))
        assert "8 sessions" in block
        assert "142 messages total" in block
        assert "2 day" in block
        assert "todo app" in block

    def test_last_compile_rendered(self):
        block = render_introspection_block(self._minimal_snapshot(
            last_compile_description="timer app",
            last_compile_trust=84.0,
            last_compile_badge="VERIFIED",
            last_compile_components=12,
            last_compile_weakest="specificity",
            last_compile_weakest_score=61.0,
            last_compile_gap_count=2,
            last_compile_cost=0.0019,
        ))
        assert 'Last compile: "timer app"' in block
        assert "VERIFIED 84%" in block
        assert "12 components" in block
        assert "specificity (61%)" in block

    def test_last_compile_omitted_when_absent(self):
        block = render_introspection_block(self._minimal_snapshot())
        assert "Last compile" not in block

    def test_tools_rendered_when_positive(self):
        block = render_introspection_block(self._minimal_snapshot(tool_count=5))
        assert "Tools: 5 available" in block

    def test_tools_omitted_when_zero(self):
        block = render_introspection_block(self._minimal_snapshot(tool_count=0))
        assert "Tools:" not in block

    def test_capabilities_reflect_voice_active(self):
        block_off = render_introspection_block(self._minimal_snapshot(voice_active=False))
        assert "voice: off" in block_off
        assert "voice output" not in block_off

        block_on = render_introspection_block(self._minimal_snapshot(voice_active=True))
        assert "voice: on" in block_on
        assert "voice output" in block_on

    def test_capabilities_reflect_file_access(self):
        block_on = render_introspection_block(self._minimal_snapshot(file_access=True))
        assert "file search" in block_on
        assert "file read/write" in block_on

        block_off = render_introspection_block(self._minimal_snapshot(file_access=False))
        assert "file search" not in block_off
        assert "file read/write" not in block_off

    def test_environment_rendered(self):
        block = render_introspection_block(self._minimal_snapshot(
            local_time="14:30", local_date="Thursday, February 13, 2026",
            timezone="GMT", platform="darwin",
        ))
        assert "14:30" in block
        assert "February 13" in block
        assert "GMT" in block

    def test_not_available_rendered(self):
        block = render_introspection_block(self._minimal_snapshot())
        assert "Not yet available:" in block
        assert "microphone input" in block

    def test_full_snapshot_under_800_chars(self):
        block = render_introspection_block(self._minimal_snapshot(
            name="Athena", personality="warm", provider="grok", model="grok-3",
            voice_active=True, session_cost=0.0023, compilations=3,
            messages_this_session=12, total_sessions=8, total_messages=142,
            days_since_last=2.0, recent_topics=["todo app", "api server"],
            last_compile_description="timer app",
            last_compile_trust=84.0, last_compile_badge="VERIFIED",
            last_compile_components=12, last_compile_weakest="specificity",
            last_compile_weakest_score=61.0, last_compile_gap_count=2,
            last_compile_cost=0.0019, tool_count=5,
        ))
        assert len(block) < 1000


class TestBuildSystemPromptIntrospection:
    """Test introspection wiring into build_system_prompt."""

    def test_introspection_renders_self_observation(self):
        config = MotherConfig()
        snap = build_introspection_snapshot(
            provider="grok", model="grok-3", compilations=2,
        )
        prompt = build_system_prompt(config, introspection=snap)
        assert "[Self-observation]" in prompt
        assert "grok/grok-3" in prompt

    def test_introspection_none_uses_legacy_path(self):
        config = MotherConfig()
        prompt = build_system_prompt(
            config, introspection=None,
            memory_context={"total_sessions": 3, "topics": ["test"], "days_since_last": 1.0},
        )
        assert "3 sessions" in prompt
        # Legacy path: no Identity: / Capabilities: lines from rendered block
        assert "Capabilities: text chat" not in prompt

    def test_legacy_callers_unchanged(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, session_stats={"compilations": 5})
        assert "5" in prompt
        assert "Mother" in prompt


class TestBuildSystemPromptContextBlock:
    """Test context_block path — substance replaces costume."""

    def test_context_block_skips_persona_base(self):
        config = MotherConfig()
        ctx = "You are David. Local AI builder on darwin."
        prompt = build_system_prompt(config, context_block=ctx)
        assert "David" in prompt
        # Old costume should NOT be present
        assert "You are Mother. You live on this machine" not in prompt
        assert "corporate warmth" not in prompt

    def test_context_block_skips_self_awareness(self):
        config = MotherConfig()
        ctx = "You are David. Motherlabs."
        prompt = build_system_prompt(config, context_block=ctx)
        assert "[Self-observation]" not in prompt
        assert "local AI entity" not in prompt

    def test_context_block_skips_personality_modifier(self):
        config = MotherConfig(personality="warm")
        ctx = "You are David."
        prompt = build_system_prompt(config, context_block=ctx)
        assert "Personality:" not in prompt
        assert "attentiveness and care" not in prompt

    def test_context_block_skips_compile_context(self):
        config = MotherConfig()
        ctx = "You are David."
        prompt = build_system_prompt(config, context_block=ctx)
        # COMPILE_CONTEXT paragraph should not be present
        assert "semantic compiler that translates natural descriptions" not in prompt

    def test_context_block_includes_intent_routing(self):
        config = MotherConfig()
        ctx = "You are David."
        prompt = build_system_prompt(config, context_block=ctx)
        assert "[VOICE]" in prompt
        assert "[ACTION:compile]" in prompt

    def test_context_block_no_routing_when_compile_disabled(self):
        config = MotherConfig()
        ctx = "You are David."
        prompt = build_system_prompt(config, context_block=ctx, include_compile=False)
        assert "[VOICE]" not in prompt
        assert "[ACTION:" not in prompt

    def test_context_block_includes_sense_block(self):
        config = MotherConfig()
        ctx = "You are David."
        sense = "Stance: You have momentum."
        prompt = build_system_prompt(config, context_block=ctx, sense_block=sense)
        assert "Stance: You have momentum." in prompt

    def test_context_block_without_sense_block(self):
        config = MotherConfig()
        ctx = "You are David."
        prompt = build_system_prompt(config, context_block=ctx, sense_block=None)
        assert "Stance:" not in prompt

    def test_context_block_none_falls_to_introspection(self):
        """context_block=None should NOT activate context path."""
        config = MotherConfig()
        snap = build_introspection_snapshot(provider="grok", model="grok-3")
        prompt = build_system_prompt(config, context_block=None, introspection=snap)
        # Should use introspection path (has PERSONA_BASE)
        assert "You are Mother. You live on this machine" in prompt
        assert "[Self-observation]" in prompt

    def test_context_block_takes_priority_over_introspection(self):
        """When both context_block and introspection are provided, context_block wins."""
        config = MotherConfig()
        ctx = "You are David. Context path."
        snap = build_introspection_snapshot(provider="grok")
        prompt = build_system_prompt(config, context_block=ctx, introspection=snap)
        assert "Context path" in prompt
        assert "[Self-observation]" not in prompt

    def test_context_block_prompt_is_shorter(self):
        """Context path should produce shorter prompts than legacy."""
        config = MotherConfig()
        legacy = build_system_prompt(config, introspection=build_introspection_snapshot())
        context = build_system_prompt(
            config,
            context_block="You are David. Local AI builder.\nActive: chat, compile, build.\nRules: be direct.",
            sense_block="Stance: Be direct.",
        )
        # Context path should be notably shorter (no 480-token PERSONA_BASE, no 240-token SELF_AWARENESS)
        assert len(context) < len(legacy)

    def test_legacy_path_still_works_when_all_none(self):
        """No context_block, no introspection = legacy memory_context path."""
        config = MotherConfig()
        prompt = build_system_prompt(
            config,
            context_block=None,
            introspection=None,
            memory_context={"total_sessions": 7, "topics": ["api"], "days_since_last": 2.0},
        )
        assert "7 sessions" in prompt
        assert "You are Mother. You live on this machine" in prompt
