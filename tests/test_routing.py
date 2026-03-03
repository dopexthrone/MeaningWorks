"""Tests for mother/routing.py — output routing LEAF module."""

import pytest
from mother.routing import (
    Channel, Urgency, Envelope, PresenceContext, RouteDecision,
    classify_urgency, route, adapt_for_whatsapp, adapt_for_voice,
    make_envelope, _is_night, _AWAY_THRESHOLD, _VOICE_MAX_LENGTH,
    _WHATSAPP_MAX_LENGTH,
)


# ===== classify_urgency =====

class TestClassifyUrgency:
    def test_chat_is_realtime(self):
        assert classify_urgency("chat") == Urgency.REALTIME

    def test_impulse_greet_is_prompt(self):
        assert classify_urgency("impulse", impulse_type="greet") == Urgency.PROMPT

    def test_impulse_observe_is_prompt(self):
        assert classify_urgency("impulse", impulse_type="observe") == Urgency.PROMPT

    def test_impulse_speak_is_deferred(self):
        assert classify_urgency("impulse", impulse_type="speak") == Urgency.DEFERRED

    def test_impulse_reflect_is_deferred(self):
        assert classify_urgency("impulse", impulse_type="reflect") == Urgency.DEFERRED

    def test_metabolism_surface_is_deferred(self):
        assert classify_urgency("metabolism", disposition="surface") == Urgency.DEFERRED

    def test_metabolism_journal_consolidation_is_background(self):
        assert classify_urgency(
            "metabolism", disposition="journal", thought_type="consolidation"
        ) == Urgency.BACKGROUND

    def test_metabolism_journal_other_is_background(self):
        assert classify_urgency(
            "metabolism", disposition="journal", thought_type="pattern"
        ) == Urgency.BACKGROUND

    def test_metabolism_internal_is_background(self):
        assert classify_urgency("metabolism", disposition="internal") == Urgency.BACKGROUND

    def test_autonomous_completion_is_prompt(self):
        assert classify_urgency("autonomous", thought_type="completion") == Urgency.PROMPT

    def test_autonomous_status_is_deferred(self):
        assert classify_urgency("autonomous", thought_type="status") == Urgency.DEFERRED

    def test_unknown_source_is_deferred(self):
        assert classify_urgency("unknown") == Urgency.DEFERRED


# ===== _is_night =====

class TestIsNight:
    def test_normal_range_inside(self):
        assert _is_night(3, 2, 7) is True

    def test_normal_range_outside(self):
        assert _is_night(12, 2, 7) is False

    def test_normal_range_boundary_start(self):
        assert _is_night(2, 2, 7) is True

    def test_normal_range_boundary_end(self):
        assert _is_night(7, 2, 7) is False

    def test_wraparound_late(self):
        assert _is_night(23, 23, 7) is True

    def test_wraparound_midnight(self):
        assert _is_night(0, 23, 7) is True

    def test_wraparound_early(self):
        assert _is_night(5, 23, 7) is True

    def test_wraparound_outside(self):
        assert _is_night(12, 23, 7) is False

    def test_wraparound_boundary_end(self):
        assert _is_night(7, 23, 7) is False


# ===== route — decision tree =====

def _presence(**kwargs):
    """Convenience to build PresenceContext with overrides."""
    defaults = dict(
        user_idle_seconds=0.0,
        wall_clock_hour=14,
        session_active=True,
        chat_available=True,
        voice_available=False,
        whatsapp_available=False,
        whatsapp_messages_today=0,
        whatsapp_daily_limit=50,
        night_start_hour=23,
        night_end_hour=7,
        night_digest_enabled=True,
    )
    defaults.update(kwargs)
    return PresenceContext(**defaults)


def _envelope(**kwargs):
    """Convenience to build Envelope with overrides."""
    defaults = dict(
        content="Hello",
        urgency=Urgency.DEFERRED,
        source="impulse",
        length=5,
    )
    defaults.update(kwargs)
    return Envelope(**defaults)


class TestRouteRealtimeUrgency:
    def test_realtime_always_includes_chat(self):
        e = _envelope(urgency=Urgency.REALTIME)
        p = _presence()
        d = route(e, p)
        assert Channel.CHAT in d.channels

    def test_realtime_adds_voice_when_available(self):
        e = _envelope(urgency=Urgency.REALTIME, has_code=False, length=50)
        p = _presence(voice_available=True)
        d = route(e, p)
        assert Channel.VOICE in d.channels

    def test_realtime_skips_voice_with_code(self):
        e = _envelope(urgency=Urgency.REALTIME, has_code=True, length=50)
        p = _presence(voice_available=True)
        d = route(e, p)
        assert Channel.VOICE not in d.channels

    def test_realtime_skips_voice_when_long(self):
        e = _envelope(urgency=Urgency.REALTIME, length=_VOICE_MAX_LENGTH + 1)
        p = _presence(voice_available=True)
        d = route(e, p)
        assert Channel.VOICE not in d.channels

    def test_realtime_adds_whatsapp_when_away(self):
        e = _envelope(urgency=Urgency.REALTIME)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
        )
        d = route(e, p)
        assert Channel.WHATSAPP in d.channels
        assert d.whatsapp_truncate is True

    def test_realtime_no_whatsapp_when_present(self):
        e = _envelope(urgency=Urgency.REALTIME)
        p = _presence(user_idle_seconds=10, whatsapp_available=True)
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels


class TestRouteUserPresent:
    def test_present_chat_only_no_voice(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(user_idle_seconds=10)
        d = route(e, p)
        assert d.channels == (Channel.CHAT,)

    def test_present_chat_plus_voice(self):
        e = _envelope(urgency=Urgency.PROMPT, length=50)
        p = _presence(user_idle_seconds=10, voice_available=True)
        d = route(e, p)
        assert Channel.CHAT in d.channels
        assert Channel.VOICE in d.channels

    def test_present_no_whatsapp(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(user_idle_seconds=10, whatsapp_available=True)
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels


class TestRouteUserAway:
    def test_away_prompt_chat_plus_whatsapp(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
        )
        d = route(e, p)
        assert Channel.CHAT in d.channels
        assert Channel.WHATSAPP in d.channels

    def test_away_prompt_no_whatsapp_at_limit(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
            whatsapp_messages_today=50,
            whatsapp_daily_limit=50,
        )
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels

    def test_away_deferred_chat_only(self):
        e = _envelope(urgency=Urgency.DEFERRED)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
        )
        d = route(e, p)
        assert d.channels == (Channel.CHAT,)

    def test_away_prompt_no_whatsapp_if_disabled(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=False,
        )
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels


class TestRouteBackground:
    def test_background_chat_only_during_day(self):
        e = _envelope(urgency=Urgency.BACKGROUND)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            wall_clock_hour=14,
            whatsapp_available=True,
        )
        d = route(e, p)
        assert d.channels == (Channel.CHAT,)

    def test_background_whatsapp_night_digest(self):
        e = _envelope(urgency=Urgency.BACKGROUND)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            wall_clock_hour=2,
            whatsapp_available=True,
            night_start_hour=23,
            night_end_hour=7,
            night_digest_enabled=True,
        )
        d = route(e, p)
        assert Channel.WHATSAPP in d.channels
        assert Channel.CHAT in d.channels

    def test_background_no_whatsapp_digest_disabled(self):
        e = _envelope(urgency=Urgency.BACKGROUND)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            wall_clock_hour=2,
            whatsapp_available=True,
            night_digest_enabled=False,
        )
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels


# ===== adapt_for_whatsapp =====

class TestAdaptForWhatsapp:
    def test_strips_code_blocks(self):
        text = "Before\n```python\nprint('hi')\n```\nAfter"
        result = adapt_for_whatsapp(text)
        assert "```" not in result
        assert "[code omitted]" in result
        assert "After" in result

    def test_strips_inline_code(self):
        result = adapt_for_whatsapp("Use `print()` to debug")
        assert "`" not in result
        assert "print()" in result

    def test_flattens_markdown_links(self):
        result = adapt_for_whatsapp("See [the docs](https://example.com)")
        assert result == "See the docs"

    def test_strips_bold(self):
        result = adapt_for_whatsapp("This is **bold** text")
        assert result == "This is bold text"

    def test_truncates_long_content(self):
        text = "x" * 2000
        result = adapt_for_whatsapp(text, max_len=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_short_content_unchanged(self):
        result = adapt_for_whatsapp("Hello there")
        assert result == "Hello there"

    def test_collapses_excessive_newlines(self):
        result = adapt_for_whatsapp("A\n\n\n\n\nB")
        assert result == "A\n\nB"

    def test_empty_content(self):
        result = adapt_for_whatsapp("")
        assert result == ""


# ===== adapt_for_voice =====

class TestAdaptForVoice:
    def test_uses_explicit_voice_text(self):
        result = adapt_for_voice("Long content here", voice_text="Short version.")
        assert result == "Short version."

    def test_extracts_first_sentence(self):
        result = adapt_for_voice("First sentence. Second sentence.")
        assert result == "First sentence."

    def test_extracts_first_question(self):
        result = adapt_for_voice("What happened? More details follow.")
        assert result == "What happened?"

    def test_extracts_exclamation(self):
        result = adapt_for_voice("Done! More here.")
        assert result == "Done!"

    def test_strips_code_for_voice(self):
        result = adapt_for_voice("Use `foo` for bar. More info.")
        assert "`" not in result
        assert "Use foo for bar." in result

    def test_strips_code_blocks(self):
        result = adapt_for_voice("Result.\n```python\ncode\n```\nDone.")
        assert result == "Result."

    def test_truncates_long_no_sentence(self):
        text = "x" * 500
        result = adapt_for_voice(text)
        assert len(result) <= _VOICE_MAX_LENGTH

    def test_empty_returns_empty(self):
        result = adapt_for_voice("")
        assert result == ""

    def test_code_only_returns_empty(self):
        result = adapt_for_voice("```\nsome code\n```")
        assert result == ""


# ===== make_envelope =====

class TestMakeEnvelope:
    def test_sets_urgency_from_source(self):
        e = make_envelope("hello", source="chat")
        assert e.urgency == Urgency.REALTIME

    def test_detects_code(self):
        e = make_envelope("Use `foo()` here", source="impulse")
        assert e.has_code is True

    def test_no_code(self):
        e = make_envelope("Just text", source="impulse")
        assert e.has_code is False

    def test_detects_code_blocks(self):
        e = make_envelope("```python\nprint(1)\n```", source="impulse")
        assert e.has_code is True

    def test_sets_length(self):
        e = make_envelope("twelve chars", source="chat")
        assert e.length == 12

    def test_preserves_voice_text(self):
        e = make_envelope("Full content", source="chat", voice_text="Short.")
        assert e.voice_text == "Short."

    def test_sets_timestamp(self):
        e = make_envelope("hi", source="chat")
        assert e.timestamp > 0


# ===== Frozen dataclass immutability =====

class TestFrozenDataclasses:
    def test_envelope_frozen(self):
        e = Envelope()
        with pytest.raises(AttributeError):
            e.content = "changed"

    def test_presence_context_frozen(self):
        p = PresenceContext()
        with pytest.raises(AttributeError):
            p.user_idle_seconds = 999

    def test_route_decision_frozen(self):
        d = RouteDecision()
        with pytest.raises(AttributeError):
            d.channels = (Channel.VOICE,)


# ===== Edge cases =====

class TestEdgeCases:
    def test_empty_envelope_routes_to_chat(self):
        e = _envelope(content="", length=0)
        p = _presence()
        d = route(e, p)
        assert Channel.CHAT in d.channels

    def test_voice_override_set_when_voice_routes(self):
        e = _envelope(
            urgency=Urgency.REALTIME,
            content="Hello there.",
            length=12,
            voice_text="Hello.",
        )
        p = _presence(voice_available=True)
        d = route(e, p)
        assert d.voice_override == "Hello."

    def test_whatsapp_at_exactly_limit_blocked(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
            whatsapp_messages_today=50,
            whatsapp_daily_limit=50,
        )
        d = route(e, p)
        assert Channel.WHATSAPP not in d.channels

    def test_whatsapp_one_under_limit_allowed(self):
        e = _envelope(urgency=Urgency.PROMPT)
        p = _presence(
            user_idle_seconds=_AWAY_THRESHOLD + 1,
            whatsapp_available=True,
            whatsapp_messages_today=49,
            whatsapp_daily_limit=50,
        )
        d = route(e, p)
        assert Channel.WHATSAPP in d.channels
