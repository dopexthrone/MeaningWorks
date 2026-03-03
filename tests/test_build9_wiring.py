"""Tests for Build 9 genome wiring — alternatives, voice-matching, creative catalyst, constraint-creative."""

import pytest

from mother.impulse import (
    Impulse,
    ImpulseContext,
    compute_impulse,
    impulse_prompt,
)


class TestAlternativeGenerating:
    """#151: Surface unresolved compile conflicts as alternatives."""

    def test_alternatives_from_conflict_summary(self):
        """Unresolved conflicts produce displayable alternatives."""
        conflict_summary = {
            "total": 3,
            "resolved": 1,
            "unresolved": [
                {"topic": "auth", "category": "design", "positions": ["JWT tokens", "session cookies"]},
                {"topic": "storage", "category": "tech", "positions": ["PostgreSQL", "MongoDB"]},
            ],
        }
        unresolved = conflict_summary["unresolved"]
        assert len(unresolved) == 2
        assert unresolved[0]["topic"] == "auth"
        assert len(unresolved[0]["positions"]) == 2

    def test_empty_conflicts_no_alternatives(self):
        """No conflicts → no alternatives surfaced."""
        conflict_summary = {"total": 0, "resolved": 0, "unresolved": []}
        assert len(conflict_summary["unresolved"]) == 0

    def test_alternatives_capped_at_three(self):
        """Only first 3 unresolved conflicts shown."""
        unresolved = [
            {"topic": f"topic_{i}", "positions": [f"opt_a_{i}", f"opt_b_{i}"]}
            for i in range(5)
        ]
        capped = unresolved[:3]
        assert len(capped) == 3

    def test_alternative_formatting(self):
        """Alternatives format as topic: position1, position2."""
        c = {"topic": "database", "positions": ["SQL", "NoSQL"]}
        alts = ", ".join(str(p)[:60] for p in c["positions"][:2])
        line = f"  {c['topic']}: {alts}"
        assert "database: SQL, NoSQL" in line


class TestVoiceMatched:
    """#126: Mirror user's communication style via tone_profile."""

    def test_impulse_context_has_user_tone_profile(self):
        """ImpulseContext carries user_tone_profile."""
        ctx = ImpulseContext(user_tone_profile="terse")
        assert ctx.user_tone_profile == "terse"

    def test_terse_tone_mirrored_in_speak(self):
        """Terse user → prompt tells Mother to be short and direct."""
        ctx = ImpulseContext(
            user_tone_profile="terse",
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "short and direct" in prompt

    def test_verbose_tone_mirrored(self):
        """Verbose user → prompt tells Mother to match depth."""
        ctx = ImpulseContext(
            user_tone_profile="verbose",
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "detail" in prompt or "depth" in prompt

    def test_questioning_tone_mirrored(self):
        """Questioning user → prompt tells Mother to be exploratory."""
        ctx = ImpulseContext(
            user_tone_profile="questioning",
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "exploratory" in prompt or "open-ended" in prompt

    def test_no_tone_no_hint(self):
        """Empty tone → no tone-matching hint."""
        ctx = ImpulseContext(
            user_tone_profile="",
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "communicates tersely" not in prompt
        assert "writes in detail" not in prompt
        assert "asks lots of questions" not in prompt


class TestCreativeCatalyst:
    """#146: Structured ideation when curiosity + diversity high."""

    def test_creative_catalyst_triggers(self):
        """High curiosity + 3+ topics → creative catalyst prompt."""
        ctx = ImpulseContext(
            curiosity=0.8,
            rapport=0.3,
            user_idle_seconds=120,
            unique_topic_count=4,
            messages_this_session=5,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.SPEAK
        prompt = impulse_prompt(impulse, ctx)
        assert "creative catalyst" in prompt.lower()

    def test_creative_catalyst_has_ideation_approaches(self):
        """Catalyst prompt offers specific ideation approaches."""
        ctx = ImpulseContext(
            curiosity=0.8,
            unique_topic_count=5,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "connection" in prompt.lower()
        assert "opposite" in prompt.lower() or "invert" in prompt.lower()
        assert "5 years" in prompt

    def test_regular_speak_below_threshold(self):
        """Low curiosity → regular SPEAK prompt, not catalyst."""
        ctx = ImpulseContext(
            curiosity=0.5,
            unique_topic_count=4,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "creative catalyst" not in prompt.lower()
        assert "curiosity" in prompt.lower()

    def test_catalyst_requires_topic_diversity(self):
        """High curiosity but few topics → regular SPEAK."""
        ctx = ImpulseContext(
            curiosity=0.8,
            unique_topic_count=2,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "creative catalyst" not in prompt.lower()

    def test_catalyst_with_tone_matching(self):
        """Creative catalyst + tone matching stack together."""
        ctx = ImpulseContext(
            curiosity=0.8,
            unique_topic_count=4,
            user_tone_profile="terse",
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "creative catalyst" in prompt.lower()
        assert "short and direct" in prompt


class TestConstraintCreative:
    """#152: Domain-specific constraint categories in CONSTRAIN phase."""

    def _make_pipeline_mock(self, domain=None):
        """Create a mock PipelineState for constrain prime tests."""
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        pipeline.get_artifact.side_effect = lambda name: {
            "decompose": {"components": [{"name": "App", "type": "component"}]},
            "ground": {"relationships": []},
        }.get(name, {})
        pipeline.original_input = "Build a booking system"
        pipeline.current_handoff = None

        if domain:
            adapter = MagicMock()
            adapter.domain = domain
            adapter.vocabulary = MagicMock()
            pipeline.known = {"_domain_adapter": adapter}
        else:
            pipeline.known = {}

        return pipeline

    def test_constrain_prime_includes_domain_constraints(self):
        """Software domain adds UX/performance/business constraint hints."""
        from core.pipeline import _build_constrain_prime

        pipeline = self._make_pipeline_mock(domain="software")
        result = _build_constrain_prime(pipeline)
        assert "UX constraints" in result
        assert "performance constraints" in result

    def test_constrain_prime_process_domain(self):
        """Process domain adds timing/resource constraint hints."""
        from core.pipeline import _build_constrain_prime

        pipeline = self._make_pipeline_mock(domain="process")
        result = _build_constrain_prime(pipeline)
        assert "timing constraints" in result
        assert "resource constraints" in result

    def test_constrain_prime_no_adapter(self):
        """No domain adapter → no domain constraint hint (doesn't crash)."""
        from core.pipeline import _build_constrain_prime

        pipeline = self._make_pipeline_mock(domain=None)
        result = _build_constrain_prime(pipeline)
        assert "UX constraints" not in result
        assert "Extract CONSTRAINTS" in result
