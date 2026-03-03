"""Tests for mother.journal_patterns — L2 operational pattern compiler."""

import json

import pytest

from mother.journal_patterns import (
    JournalPatterns,
    _DIMENSIONS,
    _FAILING_THRESHOLD,
    _WEAK_THRESHOLD,
    extract_patterns,
)


# --- Helpers ---

def _make_entry(
    success=True,
    trust_score=75.0,
    domain="software",
    scores=None,
    timestamp=0.0,
):
    """Create a journal entry dict."""
    return {
        "success": success,
        "trust_score": trust_score,
        "domain": domain,
        "dimension_scores": json.dumps(scores) if scores else "",
        "timestamp": timestamp,
    }


def _uniform_scores(value=70):
    """All 7 dimensions at the same value."""
    return {d: value for d in _DIMENSIONS}


def _varied_scores(**overrides):
    """All dimensions at 70, with overrides."""
    base = _uniform_scores(70)
    base.update(overrides)
    return base


# --- Zero entries ---

class TestEmptyEntries:
    def test_empty_list_returns_default(self):
        p = extract_patterns([])
        assert p == JournalPatterns()
        assert p.trends_line == ""
        assert p.failure_line == ""
        assert p.dimension_averages == {}
        assert p.chronic_weak == []

    def test_entries_without_dimension_scores(self):
        entries = [_make_entry(scores=None) for _ in range(5)]
        p = extract_patterns(entries)
        assert p.dimension_averages == {}
        assert p.trends_line == ""


# --- Single entry ---

class TestSingleEntry:
    def test_single_entry_averages(self):
        scores = _varied_scores(traceability=42, completeness=90)
        p = extract_patterns([_make_entry(scores=scores)])
        assert p.dimension_averages["traceability"] == 42.0
        assert p.dimension_averages["completeness"] == 90.0

    def test_single_entry_no_trajectory(self):
        """Trajectory needs 3+ entries."""
        scores = _varied_scores(traceability=42)
        p = extract_patterns([_make_entry(scores=scores)])
        assert p.dimension_trajectories == {}

    def test_single_entry_chronic_weak(self):
        scores = _varied_scores(traceability=42, actionability=51)
        p = extract_patterns([_make_entry(scores=scores)])
        assert "traceability" in p.chronic_weak
        assert "actionability" in p.chronic_weak
        # Sorted weakest first
        assert p.chronic_weak.index("traceability") < p.chronic_weak.index("actionability")

    def test_single_entry_trends_line(self):
        scores = _varied_scores(traceability=42)
        p = extract_patterns([_make_entry(scores=scores)])
        assert "Weak: traceability (42%)" in p.trends_line


# --- Two entries ---

class TestTwoEntries:
    def test_two_entries_averages(self):
        e1 = _make_entry(scores=_varied_scores(traceability=40), timestamp=1.0)
        e2 = _make_entry(scores=_varied_scores(traceability=60), timestamp=2.0)
        # newest-first (as journal.recent() returns)
        p = extract_patterns([e2, e1])
        assert p.dimension_averages["traceability"] == 50.0

    def test_two_entries_no_trajectory(self):
        """Still needs 3+ for trajectory."""
        e1 = _make_entry(scores=_varied_scores(traceability=40), timestamp=1.0)
        e2 = _make_entry(scores=_varied_scores(traceability=80), timestamp=2.0)
        p = extract_patterns([e2, e1])
        assert p.dimension_trajectories == {}


# --- Trajectories (3+ entries) ---

class TestTrajectories:
    def test_declining_dimension(self):
        """5 entries where traceability declines from 80 to 40."""
        entries = []
        for i, val in enumerate([80, 75, 60, 50, 40]):
            entries.append(
                _make_entry(scores=_varied_scores(traceability=val), timestamp=float(i))
            )
        # newest-first
        p = extract_patterns(list(reversed(entries)))
        assert "traceability" in p.dimension_trajectories
        assert p.dimension_trajectories["traceability"] < -5
        assert "Declining: traceability" in p.trends_line

    def test_improving_dimension(self):
        """5 entries where completeness improves from 50 to 90."""
        entries = []
        for i, val in enumerate([50, 55, 70, 80, 90]):
            entries.append(
                _make_entry(scores=_varied_scores(completeness=val), timestamp=float(i))
            )
        p = extract_patterns(list(reversed(entries)))
        assert "completeness" in p.dimension_trajectories
        assert p.dimension_trajectories["completeness"] > 5
        assert "Improving: completeness" in p.trends_line

    def test_flat_dimension_not_reported(self):
        """Dimension with delta < threshold not in trajectories."""
        entries = []
        for i, val in enumerate([70, 71, 72, 73, 74]):
            entries.append(
                _make_entry(scores=_varied_scores(traceability=val), timestamp=float(i))
            )
        p = extract_patterns(list(reversed(entries)))
        assert "traceability" not in p.dimension_trajectories

    def test_three_entries_minimum_for_trajectory(self):
        entries = []
        for i, val in enumerate([40, 60, 90]):
            entries.append(
                _make_entry(scores=_varied_scores(traceability=val), timestamp=float(i))
            )
        p = extract_patterns(list(reversed(entries)))
        # With 3 entries: first_half=[40], second_half=[60,90] -> delta = 75-40 = 35
        assert "traceability" in p.dimension_trajectories


# --- Failure co-occurrences ---

class TestFailureCoOccurrences:
    def test_co_occurring_weak_dims_in_failures(self):
        """3 failures where actionability + specificity both drop below 50."""
        entries = []
        for i in range(3):
            entries.append(
                _make_entry(
                    success=False,
                    trust_score=40.0,
                    scores=_varied_scores(actionability=30, specificity=35),
                    timestamp=float(i),
                )
            )
        # Add 2 successes
        for i in range(3, 5):
            entries.append(
                _make_entry(
                    success=True,
                    trust_score=80.0,
                    scores=_uniform_scores(80),
                    timestamp=float(i),
                )
            )
        p = extract_patterns(list(reversed(entries)))
        assert len(p.failure_co_occurrences) > 0
        top = p.failure_co_occurrences[0]
        assert "actionability" in (top[0], top[1])
        assert "specificity" in (top[0], top[1])
        assert top[2] == 3  # co-occurred in all 3 failures
        assert "Low-trust pattern" in p.failure_line

    def test_no_failures_no_co_occurrences(self):
        entries = [
            _make_entry(success=True, trust_score=80.0, scores=_uniform_scores(80), timestamp=float(i))
            for i in range(5)
        ]
        p = extract_patterns(list(reversed(entries)))
        assert p.failure_co_occurrences == []
        assert p.failure_line == ""

    def test_single_failure_insufficient(self):
        """Need 2+ failures for co-occurrence pattern."""
        entries = [
            _make_entry(
                success=False, trust_score=30.0,
                scores=_varied_scores(actionability=20, specificity=25),
                timestamp=0.0,
            ),
            _make_entry(
                success=True, trust_score=80.0,
                scores=_uniform_scores(80),
                timestamp=1.0,
            ),
        ]
        p = extract_patterns(list(reversed(entries)))
        assert p.failure_co_occurrences == []

    def test_low_trust_counts_as_failure(self):
        """Entries with trust_score < 55 are treated as failures even if success=True."""
        entries = []
        for i in range(3):
            entries.append(
                _make_entry(
                    success=True,
                    trust_score=45.0,  # Low trust
                    scores=_varied_scores(actionability=30, specificity=35),
                    timestamp=float(i),
                )
            )
        p = extract_patterns(list(reversed(entries)))
        assert len(p.failure_co_occurrences) > 0


# --- Domain weaknesses ---

class TestDomainWeaknesses:
    def test_domain_with_3_entries(self):
        entries = []
        for i in range(3):
            entries.append(
                _make_entry(
                    domain="software",
                    scores=_varied_scores(traceability=40, modularity=45),
                    timestamp=float(i),
                )
            )
        p = extract_patterns(list(reversed(entries)))
        assert "software" in p.domain_weaknesses
        assert "traceability" in p.domain_weaknesses["software"]

    def test_domain_with_fewer_than_3_not_reported(self):
        entries = [
            _make_entry(
                domain="api",
                scores=_varied_scores(traceability=30),
                timestamp=float(i),
            )
            for i in range(2)
        ]
        p = extract_patterns(list(reversed(entries)))
        assert "api" not in p.domain_weaknesses

    def test_multiple_domains(self):
        entries = []
        for i in range(3):
            entries.append(
                _make_entry(
                    domain="software",
                    scores=_varied_scores(traceability=40),
                    timestamp=float(i),
                )
            )
        for i in range(3, 6):
            entries.append(
                _make_entry(
                    domain="api",
                    scores=_varied_scores(modularity=35),
                    timestamp=float(i),
                )
            )
        p = extract_patterns(list(reversed(entries)))
        assert "software" in p.domain_weaknesses
        assert "api" in p.domain_weaknesses

    def test_strong_domain_not_reported(self):
        entries = [
            _make_entry(
                domain="software",
                scores=_uniform_scores(80),
                timestamp=float(i),
            )
            for i in range(3)
        ]
        p = extract_patterns(list(reversed(entries)))
        assert p.domain_weaknesses.get("software") is None


# --- Formatting ---

class TestFormatting:
    def test_trends_line_structure(self):
        scores = _varied_scores(traceability=42, actionability=51)
        p = extract_patterns([_make_entry(scores=scores)])
        assert p.trends_line.startswith("Weak:")
        assert p.trends_line.endswith(".")

    def test_trends_line_cap_3_weak(self):
        scores = {
            "completeness": 40,
            "consistency": 41,
            "specificity": 42,
            "actionability": 43,  # This is 4th — should be capped
            "traceability": 70,
            "modularity": 70,
            "testability": 70,
        }
        p = extract_patterns([_make_entry(scores=scores)])
        # Count mentions of % in the Weak: portion
        weak_part = p.trends_line.split(".")[0]  # "Weak: ..."
        assert weak_part.count("%") == 3  # Capped at 3

    def test_failure_line_format(self):
        entries = []
        for i in range(3):
            entries.append(
                _make_entry(
                    success=False, trust_score=30.0,
                    scores=_varied_scores(actionability=30, specificity=35),
                    timestamp=float(i),
                )
            )
        p = extract_patterns(list(reversed(entries)))
        assert p.failure_line.startswith("Low-trust pattern:")
        assert p.failure_line.endswith(".")

    def test_empty_trends_when_all_strong(self):
        p = extract_patterns([_make_entry(scores=_uniform_scores(80))])
        assert p.trends_line == ""

    def test_empty_failure_when_all_succeed(self):
        entries = [
            _make_entry(success=True, trust_score=80.0, scores=_uniform_scores(80), timestamp=float(i))
            for i in range(5)
        ]
        p = extract_patterns(list(reversed(entries)))
        assert p.failure_line == ""


# --- Graceful degradation ---

class TestGracefulDegradation:
    def test_malformed_json_skipped(self):
        entries = [
            {"success": True, "trust_score": 80, "domain": "sw",
             "dimension_scores": "not-json", "timestamp": 1.0},
        ]
        p = extract_patterns(entries)
        assert p.dimension_averages == {}

    def test_mixed_valid_and_invalid(self):
        entries = [
            _make_entry(scores=_varied_scores(traceability=42), timestamp=2.0),
            {"success": True, "trust_score": 80, "domain": "sw",
             "dimension_scores": "", "timestamp": 1.0},
        ]
        p = extract_patterns(entries)
        assert p.dimension_averages["traceability"] == 42.0

    def test_missing_keys_dont_crash(self):
        entries = [{"timestamp": 1.0}]
        p = extract_patterns(entries)
        assert p.dimension_averages == {}

    def test_empty_scores_dict(self):
        entries = [
            {"success": True, "trust_score": 80, "domain": "sw",
             "dimension_scores": "{}", "timestamp": 1.0},
        ]
        p = extract_patterns(entries)
        assert p.dimension_averages == {}


# --- Frozen dataclass ---

class TestFrozenDataclass:
    def test_cannot_mutate(self):
        p = extract_patterns([])
        with pytest.raises(AttributeError):
            p.trends_line = "hacked"

    def test_is_frozen(self):
        p = JournalPatterns()
        assert p.__dataclass_params__.frozen


# --- Integration-style ---

class TestIntegrationScenario:
    def test_realistic_20_entries(self):
        """Simulate 20 builds: early ones weak on traceability, improving over time."""
        entries = []
        for i in range(20):
            trace_val = 40 + i * 2  # 40 -> 78
            act_val = 70 if i >= 10 else 45  # drops then recovers
            success = trace_val > 50
            trust = max(40, trace_val - 5)
            entries.append(
                _make_entry(
                    success=success,
                    trust_score=trust,
                    domain="software",
                    scores=_varied_scores(traceability=trace_val, actionability=act_val),
                    timestamp=float(i),
                )
            )
        # newest-first
        p = extract_patterns(list(reversed(entries)))

        # Should have averages
        assert len(p.dimension_averages) == 7

        # Traceability should be improving
        assert p.dimension_trajectories.get("traceability", 0) > 0

        # Some failures should create co-occurrence patterns
        # (early entries: low trust + low traceability + low actionability)
        # Verify lines are non-empty
        assert "traceability" in p.trends_line
        assert "Improving:" in p.trends_line

    def test_newest_first_ordering(self):
        """Verify that newest-first input (journal.recent() format) works correctly."""
        # Explicitly: newest first = [t=4, t=3, t=2, t=1, t=0]
        # Traceability declining: 80, 70, 60, 50, 40 (chronologically)
        entries = []
        vals = [40, 50, 60, 70, 80]  # newest first (80 is newest -> reverse of chrono)
        # Wait, journal.recent() returns newest first. So entry at index 0 is most recent.
        # For declining: chronologically 80,70,60,50,40 -> newest-first: 40,50,60,70,80
        for i, val in enumerate([40, 50, 60, 70, 80]):
            entries.append(
                _make_entry(
                    scores=_varied_scores(traceability=val),
                    timestamp=float(20 - i),  # newest has highest timestamp
                )
            )
        p = extract_patterns(entries)
        # After reversing to chrono: [80, 70, 60, 50, 40] -> declining
        assert p.dimension_trajectories.get("traceability", 0) < -5
        assert "Declining: traceability" in p.trends_line
