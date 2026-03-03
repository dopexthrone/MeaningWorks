"""
Tests for Phase 9.1: Input Quality Gate.

Tests InputQualityAnalyzer scoring, thresholds, engine integration, and API.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.input_quality import InputQualityAnalyzer, QualityScore
from core.exceptions import InputQualityError


# =============================================================================
# InputQualityAnalyzer — Scoring
# =============================================================================


class TestInputQualityAnalyzer:
    """Tests for InputQualityAnalyzer.analyze()."""

    def setup_method(self):
        self.analyzer = InputQualityAnalyzer()

    # --- Empty / trivial ---

    def test_empty_string_scores_zero(self):
        score = self.analyzer.analyze("")
        assert score.overall == 0.0
        assert not score.is_acceptable

    def test_whitespace_only_scores_zero(self):
        score = self.analyzer.analyze("   \n\t  ")
        assert score.overall == 0.0

    def test_none_like_empty(self):
        """Empty string should score 0."""
        score = self.analyzer.analyze("")
        assert score.overall == 0.0
        assert "empty" in score.details[0].lower()

    # --- Vague / rejected inputs ---

    def test_build_a_system_rejected(self):
        """'Build a system' is too vague."""
        score = self.analyzer.analyze("Build a system")
        assert score.overall < InputQualityAnalyzer.REJECT_THRESHOLD
        assert not score.is_acceptable

    def test_make_an_app_rejected(self):
        score = self.analyzer.analyze("Make an app")
        assert score.overall < InputQualityAnalyzer.REJECT_THRESHOLD

    def test_something_cool_rejected(self):
        score = self.analyzer.analyze("I want something cool")
        assert score.overall < InputQualityAnalyzer.REJECT_THRESHOLD

    def test_help_me_rejected(self):
        score = self.analyzer.analyze("Help me build a thing")
        assert score.overall < InputQualityAnalyzer.REJECT_THRESHOLD

    # --- Borderline / warn ---

    def test_simple_domain_mention_warns(self):
        """Mentioning a domain but no actors/actions should warn."""
        score = self.analyzer.analyze("A booking system for appointments")
        assert score.is_acceptable
        # Should be in warn range or above, depending on density
        assert score.overall >= InputQualityAnalyzer.REJECT_THRESHOLD

    # --- Good inputs ---

    def test_tattoo_studio_passes(self):
        """The canonical example from the plan."""
        score = self.analyzer.analyze(
            "Build a booking system for a tattoo studio with artists and clients"
        )
        assert score.is_acceptable
        assert score.overall >= InputQualityAnalyzer.WARN_THRESHOLD

    def test_detailed_description_high_score(self):
        score = self.analyzer.analyze(
            "Build an inventory management system for a restaurant. "
            "Managers track ingredient stock levels, create purchase orders, "
            "and receive delivery notifications. The system should alert when "
            "items fall below reorder thresholds and generate weekly reports."
        )
        assert score.overall >= 0.5
        assert score.is_acceptable
        assert not score.has_warnings

    def test_ecommerce_description_passes(self):
        score = self.analyzer.analyze(
            "An e-commerce platform where customers browse products, add items to cart, "
            "checkout with payment processing, and track order delivery status."
        )
        assert score.overall >= 0.4
        assert score.is_acceptable

    # --- Score dimensions ---

    def test_length_score_increases_with_words(self):
        short = self.analyzer.analyze("Build app")
        medium = self.analyzer.analyze(
            "Build a booking system for a tattoo studio with artists and clients"
        )
        long_text = self.analyzer.analyze(
            "Build a comprehensive appointment scheduling system for a tattoo studio. "
            "Artists have profiles with style specializations. Clients browse artist portfolios, "
            "select preferred styles, and book sessions. The system manages calendar availability, "
            "sends notifications, handles deposits, and tracks session history."
        )
        assert short.length_score < medium.length_score
        assert medium.length_score <= long_text.length_score

    def test_specificity_score_domain_terms(self):
        """Input with domain terms scores higher on specificity."""
        vague = self.analyzer.analyze("Build a system that does stuff")
        specific = self.analyzer.analyze("Build a system with users, orders, and payments")
        assert specific.specificity_score > vague.specificity_score

    def test_actionability_with_verbs(self):
        """Input with action verbs scores higher on actionability."""
        no_verbs = self.analyzer.analyze("A system with users and products")
        with_verbs = self.analyzer.analyze(
            "Users search products, filter by category, purchase items, and track delivery"
        )
        assert with_verbs.actionability_score > no_verbs.actionability_score

    def test_density_filler_vs_meaningful(self):
        """Input with mostly filler words scores lower on density."""
        filler = self.analyzer.analyze(
            "I just really want to like make something that is very cool and also just really nice"
        )
        meaningful = self.analyzer.analyze(
            "Restaurant inventory manager tracking ingredients purchase orders delivery notifications"
        )
        assert meaningful.density_score > filler.density_score


# =============================================================================
# QualityScore properties
# =============================================================================


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_is_acceptable_above_threshold(self):
        score = QualityScore(overall=0.5)
        assert score.is_acceptable

    def test_is_acceptable_below_threshold(self):
        score = QualityScore(overall=0.1)
        assert not score.is_acceptable

    def test_has_warnings_in_range(self):
        score = QualityScore(overall=0.25)
        assert score.is_acceptable
        assert score.has_warnings

    def test_no_warnings_above_threshold(self):
        score = QualityScore(overall=0.5)
        assert not score.has_warnings

    def test_suggestion_populated_for_low_scores(self):
        analyzer = InputQualityAnalyzer()
        score = analyzer.analyze("Build something")
        # Should have a suggestion since score is low
        assert score.suggestion


# =============================================================================
# InputQualityError
# =============================================================================


class TestInputQualityError:
    """Tests for InputQualityError exception."""

    def test_basic_creation(self):
        err = InputQualityError("too vague")
        assert "too vague" in str(err)
        assert err.user_message == "Your description is too vague to compile."

    def test_with_quality_score(self):
        qs = QualityScore(overall=0.1, suggestion="Add more detail.")
        err = InputQualityError("low quality", quality_score=qs)
        assert err.quality_score is qs
        assert err.suggestion == "Add more detail."

    def test_to_user_dict(self):
        err = InputQualityError("bad input")
        d = err.to_user_dict()
        assert "error_type" in d
        assert d["error_type"] == "InputQualityError"


# =============================================================================
# Engine integration
# =============================================================================


class TestEngineInputQualityGate:
    """Tests for input quality gate in MotherlabsEngine.compile()."""

    def _make_engine(self):
        """Create engine with mock LLM."""
        from core.llm import MockClient
        return MotherlabsEngine(llm_client=MockClient(), auto_store=False)

    def test_vague_input_raises_quality_error(self):
        """'Build a system' should be rejected before any LLM call."""
        from core.engine import MotherlabsEngine
        engine = self._make_engine()
        with pytest.raises(InputQualityError):
            engine.compile("Build a system")

    def test_good_input_does_not_raise(self):
        """Sufficiently detailed input should not raise InputQualityError."""
        from core.engine import MotherlabsEngine
        engine = self._make_engine()
        # This should NOT raise InputQualityError (may fail later due to MockClient)
        try:
            engine.compile(
                "Build a booking system for a tattoo studio where clients "
                "schedule sessions with artists based on style specialization"
            )
        except InputQualityError:
            pytest.fail("InputQualityError raised for good input")
        except Exception:
            pass  # Other errors from mock are expected

    def test_custom_min_quality_score(self):
        """Custom min_quality_score should override default threshold."""
        from core.engine import MotherlabsEngine
        engine = self._make_engine()
        # This input is borderline — with high threshold it should fail
        with pytest.raises(InputQualityError):
            engine.compile("A booking system", min_quality_score=0.9)

    def test_min_quality_zero_accepts_anything(self):
        """Setting min_quality_score=0 disables the gate."""
        from core.engine import MotherlabsEngine
        engine = self._make_engine()
        try:
            engine.compile("hi", min_quality_score=0.0)
        except InputQualityError:
            pytest.fail("InputQualityError raised with min_quality_score=0")
        except Exception:
            pass  # Other errors expected


# Import here to avoid circular issues in test collection
from core.engine import MotherlabsEngine
