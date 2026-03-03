"""Tests for Build 7 genome wiring — communication patterns, challenge calibration."""

import time

import pytest

from mother.relationship import (
    RelationshipInsight,
    _detect_communication_patterns,
    _estimate_user_skill,
    extract_relationship_insights,
    synthesize_relationship_narrative,
)


class TestCommunicationPatternReading:
    """#40: Tone/urgency detection from user messages."""

    def test_urgent_messages_detected(self):
        """Messages with urgency words yield 'urgent' signal."""
        msgs = [
            ("user", "This is urgent, we need it now", 1.0, "s1"),
            ("user", "Please hurry, deadline is today", 2.0, "s1"),
            ("user", "Also quickly fix the bug", 3.0, "s1"),
            ("user", "Can you help with the frontend?", 4.0, "s1"),
            ("user", "Normal message about code", 5.0, "s1"),
        ]
        urgency, _ = _detect_communication_patterns(msgs)
        assert urgency == "urgent"

    def test_exploratory_messages_detected(self):
        """Messages with exploration words yield 'exploratory' signal."""
        msgs = [
            ("user", "Let's explore different approaches", 1.0, "s1"),
            ("user", "What if we tried a new design?", 2.0, "s1"),
            ("user", "I wonder about this pattern", 3.0, "s1"),
            ("user", "Normal message", 4.0, "s1"),
            ("user", "Another normal message", 5.0, "s1"),
        ]
        urgency, _ = _detect_communication_patterns(msgs)
        assert urgency == "exploratory"

    def test_neutral_messages(self):
        """Normal messages yield 'neutral' signal."""
        msgs = [
            ("user", "Please build a login page", 1.0, "s1"),
            ("user", "Add a settings panel", 2.0, "s1"),
            ("user", "Update the color scheme", 3.0, "s1"),
        ]
        urgency, _ = _detect_communication_patterns(msgs)
        assert urgency == "neutral"

    def test_terse_tone(self):
        """Short messages yield 'terse' profile."""
        msgs = [
            ("user", "Fix it", 1.0, "s1"),
            ("user", "Deploy now", 2.0, "s1"),
            ("user", "Add tests", 3.0, "s1"),
        ]
        _, tone = _detect_communication_patterns(msgs)
        assert tone == "terse"

    def test_questioning_tone(self):
        """Many questions yield 'questioning' profile."""
        msgs = [
            ("user", "How does the pipeline work?", 1.0, "s1"),
            ("user", "What about error handling?", 2.0, "s1"),
            ("user", "Is the cache invalidated?", 3.0, "s1"),
            ("user", "Why did the test fail?", 4.0, "s1"),
            ("user", "Where is the config stored?", 5.0, "s1"),
        ]
        _, tone = _detect_communication_patterns(msgs)
        assert tone == "questioning"

    def test_empty_messages(self):
        """No messages return empty strings."""
        urgency, tone = _detect_communication_patterns([])
        assert urgency == ""
        assert tone == ""

    def test_slash_commands_excluded(self):
        """Slash commands are excluded from pattern detection."""
        msgs = [
            ("user", "/compile", 1.0, "s1"),
            ("user", "/status", 2.0, "s1"),
            ("user", "Normal message about work", 3.0, "s1"),
        ]
        urgency, tone = _detect_communication_patterns(msgs)
        assert urgency == "neutral"

    def test_insight_has_fields(self):
        """RelationshipInsight carries urgency_signal and tone_profile."""
        insight = RelationshipInsight(urgency_signal="urgent", tone_profile="terse")
        assert insight.urgency_signal == "urgent"
        assert insight.tone_profile == "terse"

    def test_narrative_includes_urgency(self):
        """Narrative surfaces urgency when detected."""
        insight = RelationshipInsight(
            urgency_signal="urgent",
            relationship_stage="established",
            messages_analyzed=20,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "pressure" in narrative

    def test_narrative_includes_exploration(self):
        """Narrative surfaces exploration mode."""
        insight = RelationshipInsight(
            urgency_signal="exploratory",
            relationship_stage="established",
            messages_analyzed=20,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "exploration" in narrative


class TestChallengeCalibrating:
    """#157: User skill estimation adapts challenge level."""

    def test_expert_detected(self):
        """Technical vocabulary + domain depth → expert."""
        msgs = [
            ("user", "The api endpoint needs oauth middleware for the kubernetes deploy", 1.0, "s1"),
            ("user", "Set up the postgres database schema with graphql", 2.0, "s1"),
            ("user", "Use async websocket callbacks for the pipeline", 3.0, "s1"),
            ("user", "Deploy the microservice with docker and terraform", 4.0, "s1"),
            ("user", "Add redis caching to the typescript compiler", 5.0, "s1"),
        ]
        skill = _estimate_user_skill(msgs, compilation_count=15, domains_explored=4)
        assert skill == "expert"

    def test_beginner_detected(self):
        """Simple vocabulary + no compilations → beginner."""
        msgs = [
            ("user", "I want to build a website", 1.0, "s1"),
            ("user", "Can you make it look nice?", 2.0, "s1"),
            ("user", "Add a button that does something", 3.0, "s1"),
            ("user", "Make it work on phones", 4.0, "s1"),
            ("user", "Change the colors", 5.0, "s1"),
        ]
        skill = _estimate_user_skill(msgs, compilation_count=0, domains_explored=0)
        assert skill == "beginner"

    def test_intermediate_detected(self):
        """Some technical vocab + moderate experience → intermediate."""
        msgs = [
            ("user", "I want to set up the project structure properly", 1.0, "s1"),
            ("user", "Add some database tables for users and products", 2.0, "s1"),
            ("user", "Handle the login and registration flow", 3.0, "s1"),
            ("user", "Fix the page layout and add navigation", 4.0, "s1"),
            ("user", "Update the forms to validate input correctly", 5.0, "s1"),
        ]
        skill = _estimate_user_skill(msgs, compilation_count=4, domains_explored=1)
        assert skill == "intermediate"

    def test_too_few_messages(self):
        """Fewer than 5 messages → empty (insufficient data)."""
        msgs = [
            ("user", "Hello", 1.0, "s1"),
            ("user", "Build something", 2.0, "s1"),
        ]
        skill = _estimate_user_skill(msgs, compilation_count=0, domains_explored=0)
        assert skill == ""

    def test_insight_has_skill_field(self):
        """RelationshipInsight carries user_skill_estimate."""
        insight = RelationshipInsight(user_skill_estimate="expert")
        assert insight.user_skill_estimate == "expert"

    def test_narrative_expert_challenge(self):
        """Expert user narrative says 'challenge, don't simplify'."""
        insight = RelationshipInsight(
            user_skill_estimate="expert",
            relationship_stage="established",
            messages_analyzed=20,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "challenge" in narrative.lower()

    def test_narrative_beginner_explain(self):
        """Beginner user narrative says 'explain more'."""
        insight = RelationshipInsight(
            user_skill_estimate="beginner",
            relationship_stage="building",
            messages_analyzed=20,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "explain" in narrative.lower()

    def test_extract_relationship_insights_includes_patterns(self):
        """Full extract_relationship_insights produces tone + skill fields."""
        now = time.time()
        msgs = [
            ("user", "Urgent: the api endpoint is broken now", now - 100, "s1"),
            ("user", "Fix the database schema quickly", now - 90, "s1"),
            ("user", "Deploy the kubernetes pipeline asap", now - 80, "s1"),
            ("user", "The oauth middleware crashed", now - 70, "s1"),
            ("user", "Check the docker container", now - 60, "s1"),
            ("assistant", "Looking into it.", now - 55, "s1"),
        ]
        sessions = [{"session_id": "s1", "message_count": 6, "first_message": now - 100, "last_message": now - 55}]
        insight = extract_relationship_insights(msgs, sessions)
        assert insight.urgency_signal == "urgent"
        assert insight.user_skill_estimate != ""
