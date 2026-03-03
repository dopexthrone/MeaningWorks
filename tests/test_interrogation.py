"""Tests for pre-dialogue interrogation phase.

Covers: trigger detection, question generation, intent refinement,
and engine integration.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.interrogation import (
    should_interrogate,
    generate_questions,
    refine_intent_from_answers,
    InterrogationQuestion,
    InterrogationRequest,
    InterrogationResponse,
    InterrogationResult,
    _extract_domain_labels,
    _detail_to_question,
)
from core.protocol_spec import PROTOCOL, InterrogationSpec


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clear_intent():
    """Single-domain, well-specified intent."""
    return {
        "core_need": "booking management",
        "domain": "software",
        "actors": ["admin", "customer"],
        "explicit_components": ["Booking", "Calendar", "Payments"],
    }


@pytest.fixture
def ambiguous_intent():
    """Multi-domain, sparse intent."""
    return {
        "core_need": "system",
        "domain": "",
        "actors": [],
        "explicit_components": [],
    }


@pytest.fixture
def multi_domain_text():
    return (
        "I want a tattoo studio booking system with trading dashboards, "
        "CCTV security monitoring, a marketplace for artists, "
        "and a social network for the community."
    )


@pytest.fixture
def single_domain_text():
    return "A booking system for a hair salon with appointments, payments, and reminders."


# =============================================================================
# TRIGGER DETECTION (8 tests)
# =============================================================================

class TestShouldInterrogate:
    def test_multi_domain_triggers(self, ambiguous_intent):
        triggered, reasons = should_interrogate(0.6, ambiguous_intent, 5, [])
        assert triggered
        assert "multi_domain:5" in reasons

    def test_single_domain_no_trigger(self, clear_intent):
        triggered, reasons = should_interrogate(0.8, clear_intent, 1, [])
        assert not triggered
        assert reasons == []

    def test_low_quality_triggers(self, clear_intent):
        triggered, reasons = should_interrogate(0.20, clear_intent, 1, [])
        assert triggered
        assert "quality_warn_zone" in reasons

    def test_acceptable_quality_no_trigger(self, clear_intent):
        triggered, reasons = should_interrogate(0.50, clear_intent, 1, [])
        assert not triggered

    def test_no_components_low_quality_triggers(self):
        intent = {"domain": "software", "explicit_components": []}
        triggered, reasons = should_interrogate(0.40, intent, 1, [])
        assert triggered
        assert "no_explicit_components" in reasons

    def test_no_components_high_quality_no_trigger(self):
        intent = {"domain": "software", "explicit_components": []}
        triggered, reasons = should_interrogate(0.60, intent, 1, [])
        assert not triggered

    def test_unknown_domain_triggers(self):
        intent = {"domain": "unknown", "explicit_components": ["X"]}
        triggered, reasons = should_interrogate(0.8, intent, 1, [])
        assert triggered
        assert "unknown_domain" in reasons

    def test_multiple_reasons_combine(self, ambiguous_intent):
        triggered, reasons = should_interrogate(0.20, ambiguous_intent, 5, [])
        assert triggered
        assert len(reasons) >= 3  # multi_domain + quality + no_components + unknown_domain


# =============================================================================
# QUESTION GENERATION (6 tests)
# =============================================================================

class TestGenerateQuestions:
    def test_multi_domain_produces_domain_scope(self, multi_domain_text):
        questions = generate_questions(
            ["multi_domain:5"], {}, [], 5, multi_domain_text,
        )
        assert len(questions) >= 1
        domain_q = [q for q in questions if q.category == "domain_scope"]
        assert len(domain_q) == 1
        assert domain_q[0].options  # should have domain labels

    def test_quality_warn_converts_details(self):
        details = ["No actors mentioned", "Too short"]
        questions = generate_questions(
            ["quality_warn_zone"], {}, details, 1, "something",
        )
        categories = {q.category for q in questions}
        assert "missing_actors" in categories

    def test_unknown_domain_produces_question(self):
        questions = generate_questions(
            ["unknown_domain"], {}, [], 1, "build a thing",
        )
        domain_q = [q for q in questions if q.category == "unknown_domain"]
        assert len(domain_q) == 1
        assert domain_q[0].options  # common domain options

    def test_max_4_questions(self, multi_domain_text):
        reasons = ["multi_domain:5", "quality_warn_zone", "no_explicit_components", "unknown_domain"]
        details = ["No actors mentioned", "No actions found", "Too short", "Vague description"]
        questions = generate_questions(reasons, {}, details, 5, multi_domain_text)
        assert len(questions) <= 4

    def test_no_duplicate_categories(self, multi_domain_text):
        reasons = ["multi_domain:5", "quality_warn_zone", "no_explicit_components"]
        details = ["No actors mentioned", "Too short"]
        questions = generate_questions(reasons, {}, details, 5, multi_domain_text)
        categories = [q.category for q in questions]
        assert len(categories) == len(set(categories))

    def test_empty_reasons_empty_questions(self):
        questions = generate_questions([], {}, [], 1, "clear input")
        assert questions == []


# =============================================================================
# DOMAIN LABEL EXTRACTION
# =============================================================================

class TestDomainLabels:
    def test_extracts_multiple_domains(self, multi_domain_text):
        labels = _extract_domain_labels(multi_domain_text)
        assert len(labels) >= 4  # tattoo, trading, security, marketplace, social + meta
        assert "All together as one system" in labels
        assert "Separate compilations" in labels

    def test_single_domain_no_meta(self, single_domain_text):
        labels = _extract_domain_labels(single_domain_text)
        # Single domain — no "All together" / "Separate" meta-options
        assert "All together as one system" not in labels


# =============================================================================
# DETAIL TO QUESTION
# =============================================================================

class TestDetailToQuestion:
    def test_no_actors(self):
        q = _detail_to_question("No actors mentioned in description")
        assert q is not None
        assert q.category == "missing_actors"

    def test_too_short(self):
        q = _detail_to_question("Input too short for meaningful analysis")
        assert q is not None
        assert q.category == "missing_components"

    def test_unknown_detail(self):
        q = _detail_to_question("Some random quality detail")
        assert q is None


# =============================================================================
# INTENT REFINEMENT (7 tests)
# =============================================================================

class TestRefineIntent:
    def test_domain_scope_focuses(self):
        questions = [InterrogationQuestion("q1", "Which domain?", "domain_scope", ["Trading", "Security"])]
        response = InterrogationResponse(answers={"q1": "Trading"})
        desc, intent, fracture = refine_intent_from_answers(
            "Build everything", {"domain": ""}, response, questions,
        )
        assert "[FOCUS: Trading]" in desc
        assert intent["domain"] == "Trading"
        assert not fracture

    def test_separate_compilations_fractures(self):
        questions = [InterrogationQuestion("q1", "Which domain?", "domain_scope", ["A", "B", "Separate compilations"])]
        response = InterrogationResponse(answers={"q1": "Separate compilations"})
        desc, intent, fracture = refine_intent_from_answers(
            "Build everything", {"domain": ""}, response, questions,
        )
        assert fracture is True

    def test_unknown_domain_updates_intent(self):
        questions = [InterrogationQuestion("q1", "What industry?", "unknown_domain", ["Finance", "Health"])]
        response = InterrogationResponse(answers={"q1": "Finance"})
        desc, intent, fracture = refine_intent_from_answers(
            "Build a system", {"domain": "unknown"}, response, questions,
        )
        assert intent["domain"] == "Finance"
        assert not fracture

    def test_missing_components_parsed(self):
        questions = [InterrogationQuestion("q1", "Main parts?", "missing_components", [])]
        response = InterrogationResponse(answers={"q1": "booking, payments, and calendar"})
        desc, intent, fracture = refine_intent_from_answers(
            "Build it", {"domain": "software"}, response, questions,
        )
        assert intent["explicit_components"] == ["booking", "payments", "calendar"]
        assert not fracture

    def test_skip_response_no_changes(self):
        questions = [InterrogationQuestion("q1", "Which domain?", "domain_scope", ["A"])]
        response = InterrogationResponse(answers={}, skip=True)
        # skip=True means the caller checks response.skip BEFORE calling refine,
        # but if called anyway with empty answers, nothing changes
        desc, intent, fracture = refine_intent_from_answers(
            "Original", {"domain": "software"}, response, questions,
        )
        assert desc == "Original"
        assert intent["domain"] == "software"
        assert not fracture

    def test_multiple_answers_applied(self):
        questions = [
            InterrogationQuestion("q1", "Domain?", "unknown_domain", []),
            InterrogationQuestion("q2", "Parts?", "missing_components", []),
        ]
        response = InterrogationResponse(answers={"q1": "Health", "q2": "records, billing"})
        desc, intent, fracture = refine_intent_from_answers(
            "Build", {"domain": ""}, response, questions,
        )
        assert intent["domain"] == "Health"
        assert intent["explicit_components"] == ["records", "billing"]

    def test_missing_actors_appends_clarification(self):
        questions = [InterrogationQuestion("q1", "Who uses?", "missing_actors", [])]
        response = InterrogationResponse(answers={"q1": "doctors and patients"})
        desc, intent, fracture = refine_intent_from_answers(
            "A medical system", {"domain": "health"}, response, questions,
        )
        assert "[CLARIFICATION: doctors and patients]" in desc


# =============================================================================
# PROTOCOL SPEC INTEGRATION
# =============================================================================

class TestProtocolSpec:
    def test_interrogation_spec_on_protocol(self):
        assert hasattr(PROTOCOL, "interrogation")
        assert isinstance(PROTOCOL.interrogation, InterrogationSpec)

    def test_default_values(self):
        spec = PROTOCOL.interrogation
        assert spec.multi_domain_threshold == 3
        assert spec.quality_interrogate_threshold == 0.35
        assert spec.max_questions == 4
        assert spec.skip_on_no_callback is True


# =============================================================================
# ENGINE INTEGRATION (4 tests)
# =============================================================================

class TestEngineIntegration:
    """Test interrogation wiring in MotherlabsEngine."""

    def test_clean_input_no_callback(self, tmp_path):
        """When should_interrogate returns False -> on_interrogate never called."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from persistence.corpus import Corpus

        callback = MagicMock()
        corpus = Corpus(tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
            on_interrogate=callback,
        )
        # Patch should_interrogate at source to return no triggers (simulates clear input)
        with patch("core.interrogation.should_interrogate", return_value=(False, [])):
            result = engine.compile(
                "A booking system for a hair salon. Customers book appointments. "
                "Admins manage schedules. The system sends SMS reminders. "
                "Components: Booking, Calendar, Payments, Notifications."
            )
        callback.assert_not_called()

    def test_multi_domain_with_callback(self, tmp_path):
        """Multi-domain input + callback -> callback called, intent refined."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from core.interrogation import InterrogationResponse
        from persistence.corpus import Corpus

        def mock_callback(request):
            # Pick the first domain
            for q in request.questions:
                if q.category == "domain_scope" and q.options:
                    return InterrogationResponse(answers={q.id: q.options[0]})
            return InterrogationResponse(answers={})

        corpus = Corpus(tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
            on_interrogate=mock_callback,
        )
        result = engine.compile(
            "I want a tattoo studio booking system with trading dashboards, "
            "CCTV security monitoring, a marketplace for artists, "
            "and a social network for the community. People book appointments "
            "and artists manage their schedules. The system tracks inventory."
        )
        assert result.interrogation.get("triggered") is True
        assert result.interrogation.get("questions_asked", 0) > 0

    def test_multi_domain_no_callback_proceeds(self, tmp_path):
        """Multi-domain input + no callback -> proceeds with warning."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from persistence.corpus import Corpus

        corpus = Corpus(tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
            on_interrogate=None,  # No callback
        )
        result = engine.compile(
            "I want a tattoo studio booking system with trading dashboards, "
            "CCTV security monitoring, a marketplace for artists, "
            "and a social network for the community. People book appointments "
            "and artists manage their schedules. The system tracks inventory."
        )
        # Should proceed — interrogation skipped
        assert result.interrogation.get("triggered") is True
        assert result.interrogation.get("skipped") is True

    def test_callback_returns_skip(self, tmp_path):
        """Callback returns skip -> proceeds unchanged."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from core.interrogation import InterrogationResponse
        from persistence.corpus import Corpus

        def skip_callback(request):
            return InterrogationResponse(answers={}, skip=True)

        corpus = Corpus(tmp_path / "corpus")
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
            on_interrogate=skip_callback,
        )
        result = engine.compile(
            "I want a tattoo studio booking system with trading dashboards, "
            "CCTV security monitoring, a marketplace for artists, "
            "and a social network for the community. People book appointments "
            "and artists manage their schedules. The system tracks inventory."
        )
        assert result.interrogation.get("triggered") is True
        assert result.interrogation.get("skipped") is True
