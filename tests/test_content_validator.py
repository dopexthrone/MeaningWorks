"""Tests for kernel/content_validator.py — content-postcode alignment validation."""

import pytest
from kernel.content_validator import validate_content, ContentFit


class TestDimensionMatch:
    """Content should score higher when it matches its declared dimension."""

    def test_what_dimension_definitional(self):
        """Definitional content matches WHAT dimension."""
        fit = validate_content(
            "User entity represents account holders with attributes like name and email",
            "WHAT",
        )
        assert fit.dimension_match >= 0.3

    def test_how_dimension_procedural(self):
        """Procedural content matches HOW dimension."""
        fit = validate_content(
            "The pipeline executes steps in sequence, transforms input and dispatches results",
            "HOW",
        )
        assert fit.dimension_match >= 0.3

    def test_why_dimension_teleological(self):
        """Purpose-driven content matches WHY dimension."""
        fit = validate_content(
            "The purpose of this enables reliable delivery and ensures data integrity",
            "WHY",
        )
        assert fit.dimension_match >= 0.3

    def test_who_dimension_actors(self):
        """Actor content matches WHO dimension."""
        fit = validate_content(
            "The admin user is responsible for managing operators and service roles",
            "WHO",
        )
        assert fit.dimension_match >= 0.3

    def test_when_dimension_temporal(self):
        """Temporal content matches WHEN dimension."""
        fit = validate_content(
            "Triggers after a timeout interval, then executes in sequence before shutdown",
            "WHEN",
        )
        assert fit.dimension_match >= 0.3

    def test_where_dimension_locational(self):
        """Location content matches WHERE dimension."""
        fit = validate_content(
            "Located in the server module within the api layer at the endpoint path",
            "WHERE",
        )
        assert fit.dimension_match >= 0.3

    def test_if_dimension_conditional(self):
        """Conditional content matches IF dimension."""
        fit = validate_content(
            "Requires validation check unless the permission threshold is met by the guard",
            "IF",
        )
        assert fit.dimension_match >= 0.3

    def test_how_much_dimension_quantitative(self):
        """Quantitative content matches HOW_MUCH dimension."""
        fit = validate_content(
            "Maximum budget of 500 tokens with throughput rate limit and cost threshold",
            "HOW_MUCH",
        )
        assert fit.dimension_match >= 0.3


class TestDimensionMismatch:
    """Mismatched content should score low on dimension."""

    def test_procedural_in_what(self):
        """Procedural content should score low in WHAT dimension."""
        fit = validate_content(
            "The pipeline executes steps in sequence and transforms the input stream",
            "WHAT",
        )
        # HOW content in WHAT dimension should be weaker
        how_fit = validate_content(
            "The pipeline executes steps in sequence and transforms the input stream",
            "HOW",
        )
        assert how_fit.dimension_match > fit.dimension_match

    def test_definitional_in_how(self):
        """Definitional content should score lower in HOW than in WHAT."""
        content = "User entity represents the account model with type schema and attributes"
        what_fit = validate_content(content, "WHAT")
        how_fit = validate_content(content, "HOW")
        assert what_fit.dimension_match > how_fit.dimension_match

    def test_nonsense_scores_low(self):
        """Random text should score low on any dimension."""
        fit = validate_content(
            "I like rainbows and butterflies in the sunshine meadow",
            "WHAT",
        )
        assert fit.score < 0.3
        assert len(fit.warnings) > 0


class TestConcernInfluence:
    """Concern axis should influence the composite score."""

    def test_entity_concern_boosts_entity_content(self):
        """ENT concern boosts entity-related content."""
        fit = validate_content(
            "User entity defines the record schema with model type",
            "WHAT",
            concern="ENT",
        )
        assert fit.concern_match >= 0.3

    def test_behavior_concern_boosts_behavior_content(self):
        """BHV concern boosts behavior-related content."""
        fit = validate_content(
            "Process flow triggers transition events and behavior actions",
            "HOW",
            concern="BHV",
        )
        assert fit.concern_match >= 0.3

    def test_unknown_concern_neutral(self):
        """Unknown concern code gives neutral score."""
        fit = validate_content(
            "Some generic text about things",
            "WHAT",
            concern="XYZ",
        )
        assert fit.concern_match == 0.5  # neutral for unknown

    def test_concern_lowers_mismatched(self):
        """Wrong concern should produce warnings."""
        fit = validate_content(
            "User entity record schema model type attribute",
            "WHAT",
            concern="MET",  # metric concern for entity content
        )
        # Entity content in MET concern should warn
        assert fit.concern_match < fit.dimension_match


class TestEdgeCases:
    """Edge cases and backward compatibility."""

    def test_empty_content(self):
        """Empty content returns zero score with warning."""
        fit = validate_content("", "WHAT")
        assert fit.score == 0.0
        assert "empty content" in fit.warnings

    def test_whitespace_only(self):
        """Whitespace-only content treated as empty."""
        fit = validate_content("   \n  ", "HOW")
        assert fit.score == 0.0

    def test_no_concern(self):
        """No concern falls back to dimension-only scoring."""
        fit = validate_content(
            "Entity defines the data model schema type",
            "WHAT",
        )
        assert fit.score == fit.dimension_match

    def test_returns_frozen_dataclass(self):
        """ContentFit is frozen."""
        fit = validate_content("test content", "WHAT")
        with pytest.raises(AttributeError):
            fit.score = 0.5

    def test_score_bounded(self):
        """Score always between 0.0 and 1.0."""
        fit = validate_content(
            "entity object class model record schema type struct data field property interface",
            "WHAT",
            concern="ENT",
        )
        assert 0.0 <= fit.score <= 1.0
        assert 0.0 <= fit.dimension_match <= 1.0
        assert 0.0 <= fit.concern_match <= 1.0

    def test_single_word_content(self):
        """Single word content doesn't crash."""
        fit = validate_content("entity", "WHAT")
        assert isinstance(fit, ContentFit)

    def test_long_content(self):
        """Long content doesn't crash."""
        fit = validate_content("entity " * 500, "WHAT")
        assert isinstance(fit, ContentFit)
