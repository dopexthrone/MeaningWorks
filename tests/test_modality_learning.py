"""Tests for mother/modality_learning.py — multimodal pattern learning."""

import tempfile
from pathlib import Path

import pytest

from mother.modality_learning import (
    InteractionRecord,
    ModalityInsight,
    LearningReport,
    ModalityPatternDetector,
    save_modality_insights,
    load_modality_insights,
    format_learning_context,
)


# ---------------------------------------------------------------------------
# InteractionRecord dataclass
# ---------------------------------------------------------------------------

class TestInteractionRecord:
    def test_frozen(self):
        r = InteractionRecord(patterns_active=("focused",), modalities_active=("screen",),
                              response_quality=0.8, timestamp=100.0)
        with pytest.raises(AttributeError):
            r.response_quality = 0.5

    def test_fields(self):
        r = InteractionRecord(patterns_active=("presenting", "focused"),
                              modalities_active=("screen", "speech"),
                              response_quality=0.9, timestamp=100.0)
        assert r.patterns_active == ("presenting", "focused")
        assert r.modalities_active == ("screen", "speech")
        assert r.response_quality == 0.9


# ---------------------------------------------------------------------------
# ModalityInsight dataclass
# ---------------------------------------------------------------------------

class TestModalityInsight:
    def test_frozen(self):
        i = ModalityInsight(pattern="focused", correlation=0.3, sample_count=15,
                            recommendation="boost")
        with pytest.raises(AttributeError):
            i.correlation = 0.0


# ---------------------------------------------------------------------------
# LearningReport dataclass
# ---------------------------------------------------------------------------

class TestLearningReport:
    def test_frozen(self):
        r = LearningReport(insights=(), total_interactions=0,
                           most_useful_pattern="", least_useful_pattern="")
        with pytest.raises(AttributeError):
            r.total_interactions = 5

    def test_empty(self):
        r = LearningReport(insights=(), total_interactions=0,
                           most_useful_pattern="", least_useful_pattern="")
        assert r.insights == ()
        assert r.most_useful_pattern == ""


# ---------------------------------------------------------------------------
# ModalityPatternDetector — basics
# ---------------------------------------------------------------------------

class TestDetectorBasics:
    def test_empty_detector(self):
        d = ModalityPatternDetector()
        assert d.record_count() == 0

    def test_record_interaction(self):
        d = ModalityPatternDetector()
        d.record_interaction(("focused",), ("screen",), 0.8, timestamp=100.0)
        assert d.record_count() == 1

    def test_record_clamps_quality(self):
        d = ModalityPatternDetector()
        d.record_interaction(("focused",), ("screen",), 1.5, timestamp=100.0)
        d.record_interaction(("focused",), ("screen",), -0.5, timestamp=101.0)
        # Should not crash, values clamped

    def test_clear(self):
        d = ModalityPatternDetector()
        d.record_interaction(("focused",), ("screen",), 0.8, timestamp=100.0)
        d.clear()
        assert d.record_count() == 0

    def test_empty_analysis(self):
        d = ModalityPatternDetector()
        report = d.analyze()
        assert report.total_interactions == 0
        assert report.insights == ()
        assert report.most_useful_pattern == ""


# ---------------------------------------------------------------------------
# ModalityPatternDetector — correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_positive_correlation(self):
        d = ModalityPatternDetector(min_samples=5)
        # "focused" pattern active → high quality
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        # No pattern → low quality
        for i in range(10):
            d.record_interaction((), (), 0.3, timestamp=float(i + 10))
        report = d.analyze()
        focused = [ins for ins in report.insights if ins.pattern == "focused"][0]
        assert focused.correlation > 0
        assert focused.recommendation == "boost"

    def test_negative_correlation(self):
        d = ModalityPatternDetector(min_samples=5)
        # "multitasking" pattern active → low quality
        for i in range(10):
            d.record_interaction(("multitasking",), ("screen", "speech"), 0.2, timestamp=float(i))
        # No pattern → high quality
        for i in range(10):
            d.record_interaction((), (), 0.9, timestamp=float(i + 10))
        report = d.analyze()
        mt = [ins for ins in report.insights if ins.pattern == "multitasking"][0]
        assert mt.correlation < 0
        assert mt.recommendation == "suppress"

    def test_zero_variance_maintain(self):
        d = ModalityPatternDetector(min_samples=5)
        # Same quality with and without pattern
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.5, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.5, timestamp=float(i + 10))
        report = d.analyze()
        focused = [ins for ins in report.insights if ins.pattern == "focused"][0]
        assert abs(focused.correlation) < 0.05
        assert focused.recommendation == "maintain"

    def test_below_min_samples_maintain(self):
        d = ModalityPatternDetector(min_samples=10)
        # Only 5 samples with pattern
        for i in range(5):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(5):
            d.record_interaction((), (), 0.3, timestamp=float(i + 5))
        report = d.analyze()
        focused = [ins for ins in report.insights if ins.pattern == "focused"][0]
        assert focused.recommendation == "maintain"
        assert focused.correlation == 0.0  # no correlation computed below threshold

    def test_multiple_patterns(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(10):
            d.record_interaction(("presenting",), ("screen", "speech"), 0.4, timestamp=float(i + 10))
        for i in range(10):
            d.record_interaction((), (), 0.5, timestamp=float(i + 20))
        report = d.analyze()
        assert len(report.insights) == 2
        focused = [ins for ins in report.insights if ins.pattern == "focused"][0]
        presenting = [ins for ins in report.insights if ins.pattern == "presenting"][0]
        assert focused.correlation > presenting.correlation

    def test_most_and_least_useful(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(10):
            d.record_interaction(("multitasking",), ("screen", "speech"), 0.2, timestamp=float(i + 10))
        for i in range(10):
            d.record_interaction((), (), 0.5, timestamp=float(i + 20))
        report = d.analyze()
        assert report.most_useful_pattern == "focused"
        assert report.least_useful_pattern == "multitasking"

    def test_no_patterns_empty_insights(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(20):
            d.record_interaction((), (), 0.5, timestamp=float(i))
        report = d.analyze()
        assert report.insights == ()
        assert report.total_interactions == 20

    def test_correlation_bounded(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 1.0, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.0, timestamp=float(i + 10))
        report = d.analyze()
        focused = [ins for ins in report.insights if ins.pattern == "focused"][0]
        assert -1.0 <= focused.correlation <= 1.0


# ---------------------------------------------------------------------------
# ModalityPatternDetector — recommend_weights
# ---------------------------------------------------------------------------

class TestRecommendWeights:
    def test_positive_pattern_boosted(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.3, timestamp=float(i + 10))
        weights = d.recommend_weights()
        assert weights["focused"] > 1.0

    def test_negative_pattern_suppressed(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("multitasking",), ("screen", "speech"), 0.2, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.9, timestamp=float(i + 10))
        weights = d.recommend_weights()
        assert weights["multitasking"] < 1.0

    def test_below_min_samples_neutral(self):
        d = ModalityPatternDetector(min_samples=20)
        for i in range(5):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        weights = d.recommend_weights()
        assert weights["focused"] == 1.0

    def test_weight_bounds(self):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 1.0, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.0, timestamp=float(i + 10))
        weights = d.recommend_weights()
        assert 0.5 <= weights["focused"] <= 2.0

    def test_empty_returns_empty(self):
        d = ModalityPatternDetector()
        weights = d.recommend_weights()
        assert weights == {}


# ---------------------------------------------------------------------------
# Persistence — save/load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.3, timestamp=float(i + 10))
        report = d.analyze()
        save_modality_insights(report, db_dir=tmp_path)
        loaded = load_modality_insights(db_dir=tmp_path)
        assert len(loaded.insights) == len(report.insights)
        loaded_patterns = {i.pattern for i in loaded.insights}
        assert "focused" in loaded_patterns

    def test_load_empty_db(self, tmp_path):
        report = load_modality_insights(db_dir=tmp_path)
        assert report.insights == ()
        assert report.total_interactions == 0

    def test_load_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        report = load_modality_insights(db_dir=nonexistent)
        assert report.insights == ()

    def test_save_preserves_recommendation(self, tmp_path):
        d = ModalityPatternDetector(min_samples=5)
        for i in range(10):
            d.record_interaction(("focused",), ("screen",), 0.9, timestamp=float(i))
        for i in range(10):
            d.record_interaction((), (), 0.3, timestamp=float(i + 10))
        report = d.analyze()
        save_modality_insights(report, db_dir=tmp_path)
        loaded = load_modality_insights(db_dir=tmp_path)
        focused = [i for i in loaded.insights if i.pattern == "focused"][0]
        assert focused.recommendation == "boost"

    def test_multiple_saves_append(self, tmp_path):
        report1 = LearningReport(
            insights=(ModalityInsight("a", 0.5, 10, "boost"),),
            total_interactions=10, most_useful_pattern="a", least_useful_pattern="a")
        report2 = LearningReport(
            insights=(ModalityInsight("b", -0.3, 8, "suppress"),),
            total_interactions=8, most_useful_pattern="b", least_useful_pattern="b")
        save_modality_insights(report1, db_dir=tmp_path)
        save_modality_insights(report2, db_dir=tmp_path)
        loaded = load_modality_insights(db_dir=tmp_path, limit=50)
        patterns = {i.pattern for i in loaded.insights}
        assert "a" in patterns
        assert "b" in patterns

    def test_load_respects_limit(self, tmp_path):
        insights = tuple(
            ModalityInsight(f"p{i}", 0.1 * i, 10, "maintain")
            for i in range(20)
        )
        report = LearningReport(insights=insights, total_interactions=200,
                                most_useful_pattern="p19", least_useful_pattern="p0")
        save_modality_insights(report, db_dir=tmp_path)
        loaded = load_modality_insights(db_dir=tmp_path, limit=5)
        assert len(loaded.insights) == 5


# ---------------------------------------------------------------------------
# format_learning_context
# ---------------------------------------------------------------------------

class TestFormatLearningContext:
    def test_empty_returns_empty_string(self):
        report = LearningReport(insights=(), total_interactions=0,
                                most_useful_pattern="", least_useful_pattern="")
        assert format_learning_context(report) == ""

    def test_contains_header(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "[Modality Learning]" in result

    def test_contains_interactions_count(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "20" in result

    def test_boost_shown(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "boost" in result
        assert "focused" in result

    def test_suppress_shown(self):
        report = LearningReport(
            insights=(ModalityInsight("multitasking", -0.4, 12, "suppress"),),
            total_interactions=20, most_useful_pattern="", least_useful_pattern="multitasking")
        result = format_learning_context(report)
        assert "suppress" in result

    def test_maintain_skipped(self):
        report = LearningReport(
            insights=(ModalityInsight("idle", 0.01, 8, "maintain"),),
            total_interactions=20, most_useful_pattern="", least_useful_pattern="")
        result = format_learning_context(report)
        # "idle" line should be skipped (maintain)
        assert "idle" not in result or "maintain" not in result

    def test_most_useful_shown(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "Most useful: focused" in result

    def test_least_useful_shown_if_different(self):
        report = LearningReport(
            insights=(
                ModalityInsight("focused", 0.3, 15, "boost"),
                ModalityInsight("multitasking", -0.4, 12, "suppress"),
            ),
            total_interactions=30, most_useful_pattern="focused", least_useful_pattern="multitasking")
        result = format_learning_context(report)
        assert "Most useful: focused" in result
        assert "Least useful: multitasking" in result

    def test_least_useful_hidden_if_same(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "Least useful" not in result

    def test_positive_correlation_plus_sign(self):
        report = LearningReport(
            insights=(ModalityInsight("focused", 0.3, 15, "boost"),),
            total_interactions=20, most_useful_pattern="focused", least_useful_pattern="focused")
        result = format_learning_context(report)
        assert "+0.30" in result

    def test_negative_correlation_minus_sign(self):
        report = LearningReport(
            insights=(ModalityInsight("multitasking", -0.4, 12, "suppress"),),
            total_interactions=20, most_useful_pattern="", least_useful_pattern="multitasking")
        result = format_learning_context(report)
        assert "-0.40" in result
