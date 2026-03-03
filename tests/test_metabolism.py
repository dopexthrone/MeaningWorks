"""Tests for mother/metabolism.py — LEAF module, pure functions."""

import time

import pytest

from mother.metabolism import (
    MetabolicContext,
    MetabolicMode,
    MetabolicState,
    Thought,
    ThoughtDisposition,
    ThoughtType,
    classify_disposition,
    classify_thought_type,
    compute_depth,
    compute_metabolic_mode,
    metabolism_prompt,
    render_metabolism_context,
    should_think,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _ctx(**overrides) -> MetabolicContext:
    """Build MetabolicContext with defaults + overrides."""
    defaults = dict(
        wall_clock_hour=14,
        user_idle_seconds=0.0,
        session_duration_minutes=10.0,
        curiosity=0.3,
        attentiveness=0.5,
        rapport=0.0,
        confidence=0.5,
        vitality=1.0,
        conversation_active=False,
        autonomous_working=False,
        messages_this_session=5,
        unique_topic_count=2,
        session_cost=0.10,
        session_cost_limit=5.0,
        metabolism_session_cost=0.0,
        metabolism_budget=0.30,
        thoughts_this_session=0,
        max_thoughts_per_session=20,
        last_thought_time=0.0,
        current_time=time.time(),
    )
    defaults.update(overrides)
    return MetabolicContext(**defaults)


# ── compute_metabolic_mode ────────────────────────────────────────


class TestComputeMetabolicMode:
    """Mode computation priority tests."""

    def test_default_is_active(self):
        ctx = _ctx()
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_parallel_during_rich_conversation(self):
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=2,
            messages_this_session=3,
            curiosity=0.4,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.PARALLEL

    def test_parallel_requires_enough_topics(self):
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=1,
            messages_this_session=3,
            curiosity=0.4,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_parallel_requires_enough_messages(self):
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=2,
            messages_this_session=2,
            curiosity=0.4,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_parallel_requires_curiosity(self):
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=2,
            messages_this_session=3,
            curiosity=0.3,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_parallel_blocked_by_deep_think(self):
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=3,
            messages_this_session=5,
            curiosity=0.6,
            deep_think_subject="trust failure",
        )
        # deep_think_subject set → skips PARALLEL check, but conversation_active
        # means it won't hit DEEP either (autonomous_working=False, not night)
        # Actually: priority 1 skipped (deep_think_subject set), priority 2 no,
        # priority 3 no (not night), priority 4 → DEEP
        assert compute_metabolic_mode(ctx) == MetabolicMode.DEEP

    def test_autonomous_working_forces_active(self):
        ctx = _ctx(autonomous_working=True, user_idle_seconds=300)
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_sleep_during_night_hours(self):
        ctx = _ctx(
            wall_clock_hour=3,
            user_idle_seconds=1800,
            sleep_start_hour=2,
            sleep_end_hour=7,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.SLEEP

    def test_sleep_requires_idle(self):
        ctx = _ctx(
            wall_clock_hour=3,
            user_idle_seconds=100,
            sleep_start_hour=2,
            sleep_end_hour=7,
        )
        assert compute_metabolic_mode(ctx) != MetabolicMode.SLEEP

    def test_sleep_outside_hours(self):
        ctx = _ctx(
            wall_clock_hour=10,
            user_idle_seconds=3600,
            sleep_start_hour=2,
            sleep_end_hour=7,
        )
        assert compute_metabolic_mode(ctx) != MetabolicMode.SLEEP

    def test_sleep_wrapping_hours(self):
        """Sleep hours that wrap midnight: e.g. 22-6."""
        ctx = _ctx(
            wall_clock_hour=23,
            user_idle_seconds=1800,
            sleep_start_hour=22,
            sleep_end_hour=6,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.SLEEP

    def test_deep_with_subject(self):
        ctx = _ctx(
            deep_think_subject="why did trust score drop",
            user_idle_seconds=300,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.DEEP

    def test_idle_after_user_leaves(self):
        ctx = _ctx(
            user_idle_seconds=120,
            messages_this_session=5,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.IDLE

    def test_idle_requires_content(self):
        ctx = _ctx(
            user_idle_seconds=300,
            messages_this_session=0,
            recall_hit_count=0,
            recent_topics=[],
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE

    def test_idle_with_recall_hits(self):
        ctx = _ctx(
            user_idle_seconds=200,
            messages_this_session=0,
            recall_hit_count=2,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.IDLE

    def test_idle_with_topics(self):
        ctx = _ctx(
            user_idle_seconds=200,
            messages_this_session=0,
            recent_topics=["auth", "caching"],
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.IDLE

    def test_priority_parallel_over_idle(self):
        """Conversation active + rich → PARALLEL, not IDLE."""
        ctx = _ctx(
            conversation_active=True,
            unique_topic_count=3,
            messages_this_session=5,
            curiosity=0.6,
            user_idle_seconds=200,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.PARALLEL

    def test_priority_autonomous_over_sleep(self):
        """Autonomous working blocks everything."""
        ctx = _ctx(
            autonomous_working=True,
            wall_clock_hour=3,
            user_idle_seconds=3600,
        )
        assert compute_metabolic_mode(ctx) == MetabolicMode.ACTIVE


# ── should_think ──────────────────────────────────────────────────


class TestShouldThink:
    """Gating tests for thought generation."""

    def test_active_never_thinks(self):
        ctx = _ctx()
        assert should_think(ctx, MetabolicMode.ACTIVE) is False

    def test_idle_can_think(self):
        ctx = _ctx(user_idle_seconds=300, current_time=1000, last_thought_time=0)
        assert should_think(ctx, MetabolicMode.IDLE) is True

    def test_budget_exhausted_blocks(self):
        ctx = _ctx(metabolism_session_cost=0.30, metabolism_budget=0.30)
        assert should_think(ctx, MetabolicMode.IDLE) is False

    def test_session_cost_80pct_blocks(self):
        ctx = _ctx(session_cost=4.1, session_cost_limit=5.0)
        assert should_think(ctx, MetabolicMode.IDLE) is False

    def test_low_vitality_blocks(self):
        ctx = _ctx(vitality=0.1)
        assert should_think(ctx, MetabolicMode.IDLE) is False

    def test_vitality_threshold(self):
        ctx = _ctx(vitality=0.15)
        assert should_think(ctx, MetabolicMode.IDLE) is True

    def test_max_thoughts_blocks(self):
        ctx = _ctx(thoughts_this_session=20, max_thoughts_per_session=20)
        assert should_think(ctx, MetabolicMode.IDLE) is False

    def test_rate_limit_idle(self):
        now = time.time()
        ctx = _ctx(current_time=now, last_thought_time=now - 100)
        # IDLE min_interval=300s, elapsed=100s → too soon
        assert should_think(ctx, MetabolicMode.IDLE) is False

    def test_rate_limit_ok(self):
        now = time.time()
        ctx = _ctx(current_time=now, last_thought_time=now - 600)
        assert should_think(ctx, MetabolicMode.IDLE) is True

    def test_rate_limit_sleep(self):
        now = time.time()
        ctx = _ctx(current_time=now, last_thought_time=now - 300)
        # SLEEP min_interval=600s → too soon
        assert should_think(ctx, MetabolicMode.SLEEP) is False

    def test_rate_limit_deep(self):
        now = time.time()
        ctx = _ctx(current_time=now, last_thought_time=now - 130)
        # DEEP min_interval=120s → 130s elapsed → ok
        assert should_think(ctx, MetabolicMode.DEEP) is True

    def test_rate_limit_parallel(self):
        now = time.time()
        ctx = _ctx(current_time=now, last_thought_time=now - 100)
        # PARALLEL min_interval=180s → too soon
        assert should_think(ctx, MetabolicMode.PARALLEL) is False

    def test_first_thought_always_ok(self):
        """No last_thought_time → no rate limit applies."""
        ctx = _ctx(current_time=1000, last_thought_time=0)
        assert should_think(ctx, MetabolicMode.IDLE) is True

    def test_zero_cost_limit_no_crash(self):
        ctx = _ctx(session_cost_limit=0.0)
        assert should_think(ctx, MetabolicMode.IDLE) is True


# ── classify_thought_type ─────────────────────────────────────────


class TestClassifyThoughtType:
    """Thought type classification by mode."""

    def test_sleep_with_topics_consolidates(self):
        ctx = _ctx(recent_topics=["a", "b", "c"])
        assert classify_thought_type(MetabolicMode.SLEEP, ctx) == ThoughtType.CONSOLIDATION

    def test_sleep_with_recall_finds_patterns(self):
        ctx = _ctx(recent_topics=["a"], recall_hit_count=3)
        assert classify_thought_type(MetabolicMode.SLEEP, ctx) == ThoughtType.PATTERN

    def test_sleep_default_consolidation(self):
        ctx = _ctx(recent_topics=[])
        assert classify_thought_type(MetabolicMode.SLEEP, ctx) == ThoughtType.CONSOLIDATION

    def test_idle_high_curiosity(self):
        ctx = _ctx(curiosity=0.6)
        assert classify_thought_type(MetabolicMode.IDLE, ctx) == ThoughtType.CURIOSITY

    def test_idle_with_recall_connects(self):
        ctx = _ctx(curiosity=0.3, recall_hit_count=3)
        assert classify_thought_type(MetabolicMode.IDLE, ctx) == ThoughtType.CONNECTION

    def test_idle_failure_streak_questions(self):
        ctx = _ctx(curiosity=0.3, recall_hit_count=0, journal_failure_streak=-3)
        assert classify_thought_type(MetabolicMode.IDLE, ctx) == ThoughtType.QUESTION

    def test_idle_default_curiosity(self):
        ctx = _ctx(curiosity=0.3, recall_hit_count=0, journal_failure_streak=0)
        assert classify_thought_type(MetabolicMode.IDLE, ctx) == ThoughtType.CURIOSITY

    def test_deep_low_trust_questions(self):
        ctx = _ctx(last_compile_trust=30.0)
        assert classify_thought_type(MetabolicMode.DEEP, ctx) == ThoughtType.QUESTION

    def test_deep_default_implication(self):
        ctx = _ctx(last_compile_trust=80.0)
        assert classify_thought_type(MetabolicMode.DEEP, ctx) == ThoughtType.IMPLICATION

    def test_deep_no_compile_implication(self):
        ctx = _ctx(last_compile_trust=None)
        assert classify_thought_type(MetabolicMode.DEEP, ctx) == ThoughtType.IMPLICATION

    def test_parallel_many_topics_connection(self):
        ctx = _ctx(unique_topic_count=3)
        assert classify_thought_type(MetabolicMode.PARALLEL, ctx) == ThoughtType.CONNECTION

    def test_parallel_few_topics_implication(self):
        ctx = _ctx(unique_topic_count=2)
        assert classify_thought_type(MetabolicMode.PARALLEL, ctx) == ThoughtType.IMPLICATION


# ── classify_disposition ──────────────────────────────────────────


class TestClassifyDisposition:
    """Disposition classification tests."""

    def test_sleep_always_journals(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.CONSOLIDATION, MetabolicMode.SLEEP, ctx)
        assert d == ThoughtDisposition.JOURNAL

    def test_deep_always_surfaces(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.IMPLICATION, MetabolicMode.DEEP, ctx)
        assert d == ThoughtDisposition.SURFACE

    def test_connection_high_curiosity_surfaces(self):
        ctx = _ctx(curiosity=0.6)
        d = classify_disposition(ThoughtType.CONNECTION, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.SURFACE

    def test_connection_low_curiosity_internal(self):
        ctx = _ctx(curiosity=0.3)
        d = classify_disposition(ThoughtType.CONNECTION, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.INTERNAL

    def test_parallel_implication_surfaces(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.IMPLICATION, MetabolicMode.PARALLEL, ctx)
        assert d == ThoughtDisposition.SURFACE

    def test_question_always_surfaces(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.QUESTION, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.SURFACE

    def test_consolidation_journals(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.CONSOLIDATION, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.JOURNAL

    def test_pattern_internal(self):
        ctx = _ctx()
        d = classify_disposition(ThoughtType.PATTERN, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.INTERNAL

    def test_low_curiosity_idle_internal(self):
        ctx = _ctx(curiosity=0.2)
        d = classify_disposition(ThoughtType.CURIOSITY, MetabolicMode.IDLE, ctx)
        assert d == ThoughtDisposition.INTERNAL


# ── compute_depth ─────────────────────────────────────────────────


class TestComputeDepth:
    """Depth computation tests."""

    def test_deep_high_curiosity(self):
        ctx = _ctx(curiosity=1.0)
        d = compute_depth(MetabolicMode.DEEP, ctx)
        assert d == 1.0

    def test_deep_low_curiosity(self):
        ctx = _ctx(curiosity=0.0)
        d = compute_depth(MetabolicMode.DEEP, ctx)
        assert d == 0.7

    def test_sleep_constant(self):
        ctx = _ctx()
        assert compute_depth(MetabolicMode.SLEEP, ctx) == 0.6

    def test_idle_moderate(self):
        ctx = _ctx(curiosity=0.5)
        d = compute_depth(MetabolicMode.IDLE, ctx)
        assert 0.3 <= d <= 0.5

    def test_parallel_shallow(self):
        ctx = _ctx()
        assert compute_depth(MetabolicMode.PARALLEL, ctx) == 0.2

    def test_active_zero(self):
        ctx = _ctx()
        assert compute_depth(MetabolicMode.ACTIVE, ctx) == 0.0

    def test_deep_caps_at_one(self):
        ctx = _ctx(curiosity=2.0)  # out of range but should still cap
        d = compute_depth(MetabolicMode.DEEP, ctx)
        assert d <= 1.0


# ── metabolism_prompt ─────────────────────────────────────────────


class TestMetabolismPrompt:
    """Prompt generation tests."""

    def test_active_returns_none(self):
        ctx = _ctx()
        assert metabolism_prompt(MetabolicMode.ACTIVE, ThoughtType.CURIOSITY, ctx) is None

    def test_idle_returns_prompt(self):
        ctx = _ctx(recent_topics=["auth"])
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.CURIOSITY, ctx)
        assert p is not None
        assert "idle" in p.lower()
        assert "auth" in p

    def test_sleep_returns_prompt(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.SLEEP, ThoughtType.CONSOLIDATION, ctx)
        assert p is not None
        assert "sleep" in p.lower()

    def test_deep_includes_subject(self):
        ctx = _ctx(deep_think_subject="trust failure analysis")
        p = metabolism_prompt(MetabolicMode.DEEP, ThoughtType.IMPLICATION, ctx)
        assert "trust failure analysis" in p

    def test_prompt_includes_compile_hint(self):
        ctx = _ctx(last_compile_trust=35.0, last_compile_weakest="consistency")
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.QUESTION, ctx)
        assert "35%" in p
        assert "consistency" in p

    def test_prompt_includes_failure_hint(self):
        ctx = _ctx(journal_failure_streak=-3)
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.QUESTION, ctx)
        assert "3 consecutive failures" in p

    def test_connection_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.CONNECTION, ctx)
        assert "connection" in p.lower()

    def test_question_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.QUESTION, ctx)
        assert "question" in p.lower()

    def test_pattern_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.PATTERN, ctx)
        assert "pattern" in p.lower()

    def test_consolidation_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.SLEEP, ThoughtType.CONSOLIDATION, ctx)
        assert "compress" in p.lower() or "synthesize" in p.lower()

    def test_implication_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.DEEP, ThoughtType.IMPLICATION, ctx)
        assert "consequence" in p.lower() or "implication" in p.lower()

    def test_prompt_concise_instruction(self):
        ctx = _ctx()
        p = metabolism_prompt(MetabolicMode.IDLE, ThoughtType.CURIOSITY, ctx)
        assert "1-2 sentences" in p


# ── render_metabolism_context ─────────────────────────────────────


class TestRenderMetabolismContext:
    """Context rendering tests."""

    def test_empty_active_state(self):
        state = MetabolicState()
        assert render_metabolism_context(state) == ""

    def test_with_narrative(self):
        state = MetabolicState(
            mode=MetabolicMode.IDLE,
            inner_narrative="Processing recent build patterns.",
        )
        result = render_metabolism_context(state)
        assert "Processing recent build patterns." in result

    def test_with_surfaceable_thoughts(self):
        thoughts = [
            Thought(
                thought_type=ThoughtType.CONNECTION,
                disposition=ThoughtDisposition.SURFACE,
                subject="Auth flow connects to the caching layer",
            ),
        ]
        state = MetabolicState(
            mode=MetabolicMode.IDLE,
            thought_count=1,
            recent_thoughts=thoughts,
            surfaceable_count=1,
        )
        result = render_metabolism_context(state)
        assert "Auth flow connects to the caching layer" in result

    def test_multiple_surfaceable(self):
        thoughts = [
            Thought(disposition=ThoughtDisposition.SURFACE, subject="A"),
            Thought(disposition=ThoughtDisposition.SURFACE, subject="B"),
            Thought(disposition=ThoughtDisposition.INTERNAL, subject="C"),
        ]
        state = MetabolicState(
            mode=MetabolicMode.IDLE,
            thought_count=3,
            recent_thoughts=thoughts,
            surfaceable_count=2,
        )
        result = render_metabolism_context(state)
        assert "A" in result
        assert "B" in result
        assert "C" not in result

    def test_max_three_surfaced(self):
        thoughts = [
            Thought(disposition=ThoughtDisposition.SURFACE, subject=f"T{i}")
            for i in range(5)
        ]
        state = MetabolicState(
            mode=MetabolicMode.IDLE,
            thought_count=5,
            recent_thoughts=thoughts,
            surfaceable_count=5,
        )
        result = render_metabolism_context(state)
        assert "T0" in result
        assert "T2" in result
        # T3, T4 truncated
        assert "T3" not in result

    def test_internal_only_no_output(self):
        thoughts = [
            Thought(disposition=ThoughtDisposition.INTERNAL, subject="Hidden"),
        ]
        state = MetabolicState(
            mode=MetabolicMode.IDLE,
            thought_count=1,
            recent_thoughts=thoughts,
            surfaceable_count=0,
        )
        result = render_metabolism_context(state)
        assert result == ""


# ── Dataclass invariants ──────────────────────────────────────────


class TestDataclasses:
    """Frozen dataclass invariant tests."""

    def test_metabolic_context_frozen(self):
        ctx = _ctx()
        with pytest.raises(AttributeError):
            ctx.vitality = 0.5

    def test_thought_frozen(self):
        t = Thought()
        with pytest.raises(AttributeError):
            t.subject = "nope"

    def test_metabolic_state_frozen(self):
        s = MetabolicState()
        with pytest.raises(AttributeError):
            s.mode = MetabolicMode.SLEEP

    def test_thought_defaults(self):
        t = Thought()
        assert t.thought_type == ThoughtType.CURIOSITY
        assert t.disposition == ThoughtDisposition.INTERNAL
        assert t.depth == 0.0

    def test_metabolic_context_defaults(self):
        ctx = MetabolicContext()
        assert ctx.wall_clock_hour == 12
        assert ctx.vitality == 1.0
        assert ctx.metabolism_budget == 0.30
