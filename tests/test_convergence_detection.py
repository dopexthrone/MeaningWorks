"""Tests for convergence detection module (core/convergence.py).

Tests cover:
- blueprint_delta: Jaccard-based comparison of blueprint snapshots
- estimate_turn_budget: input complexity → turn budget scaling
- ConvergenceTracker: plateau detection, minimum turns, convergence signals
- _count_domains: domain detection from input text
- take_snapshot: message → snapshot extraction
"""

import pytest

from core.convergence import (
    blueprint_delta,
    estimate_turn_budget,
    take_snapshot,
    ConvergenceTracker,
    _count_domains,
    _extract_component_names,
    _extract_relationships,
    _extract_insights,
)


# =============================================================================
# blueprint_delta tests
# =============================================================================


class TestBlueprintDelta:
    """Tests for blueprint_delta comparison."""

    def test_identical_snapshots_delta_zero(self):
        """Identical snapshots produce delta 0.0."""
        snap = {
            "components": {"User", "Auth", "Database"},
            "relationships": {("User", "Auth"), ("Auth", "Database")},
            "insights": {"user needs login"},
        }
        assert blueprint_delta(snap, snap) == 0.0

    def test_completely_different_snapshots_delta_one(self):
        """Completely disjoint snapshots produce delta 1.0."""
        snap_a = {
            "components": {"User", "Auth"},
            "relationships": {("User", "Auth")},
            "insights": {"user login"},
        }
        snap_b = {
            "components": {"Trading", "Portfolio"},
            "relationships": {("Trading", "Portfolio")},
            "insights": {"trading system"},
        }
        assert blueprint_delta(snap_a, snap_b) == 1.0

    def test_empty_snapshots_delta_zero(self):
        """Two empty snapshots produce delta 0.0."""
        empty = {"components": set(), "relationships": set(), "insights": set()}
        assert blueprint_delta(empty, empty) == 0.0

    def test_partial_overlap_components(self):
        """Partial component overlap produces intermediate delta."""
        snap_a = {
            "components": {"User", "Auth", "Database"},
            "relationships": set(),
            "insights": set(),
        }
        snap_b = {
            "components": {"User", "Auth", "Cache"},
            "relationships": set(),
            "insights": set(),
        }
        delta = blueprint_delta(snap_a, snap_b)
        # Jaccard: intersection=2, union=4, distance=0.5, weighted 0.5*0.5=0.25
        assert 0.2 < delta < 0.3

    def test_added_component_increases_delta(self):
        """Adding a component increases delta."""
        snap_a = {
            "components": {"User", "Auth"},
            "relationships": set(),
            "insights": set(),
        }
        snap_b = {
            "components": {"User", "Auth", "Database"},
            "relationships": set(),
            "insights": set(),
        }
        delta = blueprint_delta(snap_a, snap_b)
        assert delta > 0.0

    def test_relationship_changes_contribute(self):
        """Relationship changes contribute to delta."""
        snap_a = {
            "components": {"User", "Auth"},
            "relationships": {("User", "Auth")},
            "insights": set(),
        }
        snap_b = {
            "components": {"User", "Auth"},
            "relationships": {("User", "Auth"), ("Auth", "User")},
            "insights": set(),
        }
        delta = blueprint_delta(snap_a, snap_b)
        assert delta > 0.0  # Relationship change

    def test_insight_changes_contribute(self):
        """New insights contribute to delta."""
        snap_a = {
            "components": set(),
            "relationships": set(),
            "insights": {"first insight"},
        }
        snap_b = {
            "components": set(),
            "relationships": set(),
            "insights": {"first insight", "second insight"},
        }
        delta = blueprint_delta(snap_a, snap_b)
        assert delta > 0.0

    def test_delta_bounded_zero_one(self):
        """Delta is always between 0.0 and 1.0."""
        for _ in range(10):
            snap_a = {
                "components": {"A", "B"},
                "relationships": {("A", "B")},
                "insights": {"x"},
            }
            snap_b = {
                "components": {"C", "D", "E"},
                "relationships": {("C", "D"), ("D", "E")},
                "insights": {"y", "z"},
            }
            delta = blueprint_delta(snap_a, snap_b)
            assert 0.0 <= delta <= 1.0

    def test_component_weight_dominant(self):
        """Component changes have more weight than insight changes."""
        base = {
            "components": {"User"},
            "relationships": set(),
            "insights": {"x"},
        }
        comp_change = {
            "components": {"Trading"},
            "relationships": set(),
            "insights": {"x"},
        }
        insight_change = {
            "components": {"User"},
            "relationships": set(),
            "insights": {"y"},
        }
        comp_delta = blueprint_delta(base, comp_change)
        insight_delta = blueprint_delta(base, insight_change)
        assert comp_delta > insight_delta


# =============================================================================
# estimate_turn_budget tests
# =============================================================================


class TestEstimateTurnBudget:
    """Tests for turn budget estimation."""

    def test_short_input_low_budget(self):
        """Short input gets base budget."""
        min_t, rec_t, max_t = estimate_turn_budget("Build a login system")
        assert min_t >= 6
        assert rec_t <= 12
        assert max_t <= 25

    def test_long_input_higher_budget(self):
        """Long multi-domain input gets higher budget."""
        long_input = """
        Build a tattoo studio management system with walk-in scheduling,
        artist portfolio management, CCTV integration for security,
        email automation for booking confirmations, accounting tools
        for financial tracking, SEO research for marketing, and a
        marketplace for selling tattoo designs. Also integrate with
        smart home devices for lighting control in the studio.
        """ * 3  # ~500 words
        min_t, rec_t, max_t = estimate_turn_budget(long_input)
        assert rec_t > 12  # More than base
        assert max_t > 20  # Extended ceiling

    def test_corpus_feedback_adjusts_budget(self):
        """Corpus history influences turn budget."""
        text = "Build a system"
        # Without corpus
        _, rec_no_corpus, _ = estimate_turn_budget(text)
        # With corpus saying 20 turns typical
        _, rec_with_corpus, _ = estimate_turn_budget(
            text, corpus_avg_turns=20.0, corpus_sample_size=10
        )
        assert rec_with_corpus > rec_no_corpus

    def test_corpus_needs_minimum_samples(self):
        """Corpus with too few samples is ignored."""
        text = "Build a system"
        _, rec_no_corpus, _ = estimate_turn_budget(text)
        _, rec_tiny_corpus, _ = estimate_turn_budget(
            text, corpus_avg_turns=50.0, corpus_sample_size=1
        )
        # With only 1 sample, corpus is ignored
        assert rec_no_corpus == rec_tiny_corpus

    def test_max_turns_has_ceiling(self):
        """Max turns never exceeds hard ceiling."""
        huge_input = "tattoo trading security home CCTV email " * 500
        _, _, max_t = estimate_turn_budget(huge_input)
        assert max_t <= 60

    def test_min_turns_at_least_six(self):
        """Minimum turns is always at least 6."""
        min_t, _, _ = estimate_turn_budget("x")
        assert min_t >= 6

    def test_multi_domain_input_scales(self):
        """Input with many domain signals gets more turns."""
        single_domain = "Build a trading dashboard"
        multi_domain = (
            "Build a tattoo studio with CCTV security, email automation, "
            "SEO marketing, accounting tools, crypto marketplace, "
            "home IoT integration, voice assistant, calendar sync"
        )
        _, rec_single, _ = estimate_turn_budget(single_domain)
        _, rec_multi, _ = estimate_turn_budget(multi_domain)
        assert rec_multi > rec_single


# =============================================================================
# _count_domains tests
# =============================================================================


class TestCountDomains:
    """Tests for domain counting."""

    def test_single_domain(self):
        """Single domain input returns 1."""
        assert _count_domains("Build a simple login page") >= 1

    def test_multi_domain_counted(self):
        """Multiple domain signals are counted."""
        text = "tattoo studio with CCTV security and email marketing and crypto trading"
        count = _count_domains(text)
        assert count >= 3

    def test_capped_at_15(self):
        """Domain count capped at 15."""
        text = " ".join([
            "tattoo", "studio", "trading", "home", "CCTV", "security",
            "email", "SEO", "accounting", "financial", "crypto",
            "marketplace", "device", "IoT", "voice", "calendar", "social",
            "health", "fitness", "shopping",
        ])
        assert _count_domains(text) <= 15


# =============================================================================
# take_snapshot tests
# =============================================================================


class TestTakeSnapshot:
    """Tests for snapshot extraction from messages."""

    def test_extracts_component_names(self):
        """Extracts capitalized component names from messages."""
        messages = [
            {"content": "The User entity has email and password fields.", "insight": None},
            {"content": "The Authentication flow validates credentials.", "insight": None},
        ]
        snap = take_snapshot(messages)
        assert "User" in snap["components"]
        assert "Authentication" in snap["components"]

    def test_filters_noise_words(self):
        """Noise words like 'The', 'This' are filtered out."""
        messages = [
            {"content": "The system should handle This case.", "insight": None},
        ]
        snap = take_snapshot(messages)
        assert "The" not in snap["components"]
        assert "This" not in snap["components"]

    def test_extracts_insights(self):
        """Extracts insight strings from messages."""
        messages = [
            {"content": "Analysis", "insight": "User needs authentication"},
            {"content": "More analysis", "insight": "Database stores credentials"},
        ]
        snap = take_snapshot(messages)
        assert len(snap["insights"]) == 2

    def test_empty_messages(self):
        """Empty message list produces empty snapshot."""
        snap = take_snapshot([])
        assert len(snap["components"]) == 0
        assert len(snap["relationships"]) == 0
        assert len(snap["insights"]) == 0

    def test_extracts_relationships(self):
        """Extracts arrow-notation relationships."""
        messages = [
            {"content": "User -> Auth means the user authenticates.", "insight": None},
            {"content": "Auth depends on Database for storage.", "insight": None},
        ]
        snap = take_snapshot(messages)
        assert ("User", "Auth") in snap["relationships"]
        assert ("Auth", "Database") in snap["relationships"]


# =============================================================================
# ConvergenceTracker tests
# =============================================================================


class TestConvergenceTracker:
    """Tests for convergence tracking."""

    def test_first_update_returns_high_delta(self):
        """First update always returns 1.0 (no previous to compare)."""
        tracker = ConvergenceTracker()
        messages = [
            {"content": "The User entity exists.", "insight": "user found"},
        ]
        delta = tracker.update(messages)
        assert delta == 1.0

    def test_identical_updates_low_delta(self):
        """Updating with same messages produces low delta."""
        tracker = ConvergenceTracker(min_turns_before_convergence=2)
        messages = [
            {"content": "The User entity exists.", "insight": "user found"},
        ]
        tracker.update(messages, total_turns=1)
        delta = tracker.update(messages, total_turns=2)
        assert delta == 0.0

    def test_convergence_after_plateau(self):
        """Convergence detected after plateau_window consecutive low deltas."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=3,
        )
        messages = [
            {"content": "The User entity has email.", "insight": "user exists"},
        ]
        tracker.update(messages, total_turns=1)
        tracker.update(messages, total_turns=2)
        tracker.update(messages, total_turns=3)
        # After 3 identical updates, last 2 deltas should be 0.0
        assert tracker.has_converged()

    def test_no_convergence_before_min_turns(self):
        """No convergence before min_turns_before_convergence."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=10,
        )
        messages = [
            {"content": "The User entity has email.", "insight": "user exists"},
        ]
        for i in range(5):
            tracker.update(messages, total_turns=i + 1)
        # 5 identical updates but min_turns is 10
        assert not tracker.has_converged()

    def test_no_convergence_with_changing_content(self):
        """No convergence when content keeps changing."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=3,
        )
        for i in range(6):
            messages = [
                {"content": f"Component{i} entity exists.", "insight": f"insight {i}"},
            ]
            tracker.update(messages, total_turns=i + 1)
        # Each update adds new components, delta stays high
        assert not tracker.has_converged()

    def test_should_continue_respects_max_turns(self):
        """should_continue returns False at max_turns."""
        tracker = ConvergenceTracker()
        assert not tracker.should_continue(current_turn=10, max_turns=10)
        assert tracker.should_continue(current_turn=5, max_turns=10)

    def test_should_continue_false_when_converged(self):
        """should_continue returns False when converged."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=2,
        )
        messages = [{"content": "User exists.", "insight": "user"}]
        tracker.update(messages, total_turns=1)
        tracker.update(messages, total_turns=2)
        tracker.update(messages, total_turns=3)
        assert not tracker.should_continue(current_turn=3, max_turns=30)

    def test_component_count_tracks(self):
        """component_count reflects discovered components."""
        tracker = ConvergenceTracker()
        messages = [
            {"content": "User and Auth and Database are entities.", "insight": None},
        ]
        tracker.update(messages)
        assert tracker.component_count >= 2  # User, Auth, Database (minus noise)

    def test_convergence_summary_structure(self):
        """convergence_summary returns expected keys."""
        tracker = ConvergenceTracker()
        messages = [{"content": "Test", "insight": None}]
        tracker.update(messages)
        summary = tracker.convergence_summary
        assert "turns" in summary
        assert "deltas" in summary
        assert "converged" in summary
        assert "component_count" in summary
        assert "final_delta" in summary

    def test_last_delta_property(self):
        """last_delta returns most recent delta."""
        tracker = ConvergenceTracker()
        assert tracker.last_delta == 1.0  # Default
        messages = [{"content": "User entity.", "insight": None}]
        tracker.update(messages)
        assert tracker.last_delta == 1.0  # First is always 1.0
        tracker.update(messages)
        assert tracker.last_delta == 0.0  # Same content

    def test_gradual_convergence(self):
        """Track convergence through gradual stabilization."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.1,
            min_turns_before_convergence=4,
        )
        # Round 1: discover User, Auth
        tracker.update([
            {"content": "User and Auth entities.", "insight": "user auth"},
        ], total_turns=3)

        # Round 2: discover Database too — still changing
        tracker.update([
            {"content": "User and Auth and Database entities.", "insight": "user auth db"},
        ], total_turns=6)

        assert not tracker.has_converged()

        # Round 3: same components — stabilizing
        tracker.update([
            {"content": "User and Auth and Database entities.", "insight": "user auth db"},
        ], total_turns=9)

        # Now last 2 deltas are [change, 0.0] — need one more
        # Actually the content matches round 2 exactly, so delta=0.0
        # But plateau_window=2 needs 2 consecutive < threshold
        # deltas are [1.0, change, 0.0] — not yet

        # Round 4: same again
        tracker.update([
            {"content": "User and Auth and Database entities.", "insight": "user auth db"},
        ], total_turns=12)

        # Now last 2 deltas should be [0.0, 0.0]
        assert tracker.has_converged()


# =============================================================================
# Integration: convergence in engine dialogue flow
# =============================================================================


class TestConvergenceInDialogue:
    """Integration tests for convergence detection in dialogue context."""

    def test_simple_input_converges_in_few_rounds(self):
        """Simple input with repeated content converges quickly."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=6,
        )

        # Simulate 3 rounds of dialogue with same components
        for round_num in range(5):
            messages = [
                {"content": "The User entity has email and password_hash.", "insight": "user fields"},
                {"content": "The Login process validates credentials.", "insight": "login flow"},
            ]
            tracker.update(messages, total_turns=(round_num + 1) * 3)

        # Should converge — same content repeated
        assert tracker.has_converged()
        assert tracker.convergence_summary["converged"] is True

    def test_complex_input_needs_more_rounds(self):
        """Complex input that keeps discovering new components doesn't converge early."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=6,
        )

        components_per_round = [
            ["User", "Auth"],
            ["User", "Auth", "Database", "Cache"],
            ["User", "Auth", "Database", "Cache", "Studio", "Artist"],
            ["User", "Auth", "Database", "Cache", "Studio", "Artist", "Booking"],
        ]

        for i, comps in enumerate(components_per_round):
            messages = [
                {"content": f"Entities: {', '.join(comps)}.", "insight": f"round {i}"},
            ]
            tracker.update(messages, total_turns=(i + 1) * 3)

        # Should NOT converge — new components each round
        assert not tracker.has_converged()


# =============================================================================
# Grid convergence summary
# =============================================================================


class TestGridConvergenceSummary:
    """Tests for grid_convergence_summary function."""

    def test_grid_convergence_summary_fields(self):
        """grid_convergence_summary returns expected keys."""
        from core.convergence import grid_convergence_summary
        from kernel.grid import Grid

        grid = Grid()
        grid.set_intent(
            intent_text="Test",
            postcode_key="INT.SEM.ECO.WHY.SFT",
            primitive="intent_contract",
        )

        summary = grid_convergence_summary(grid)
        assert "fill_rate" in summary
        assert "total_cells" in summary
        assert "filled" in summary
        assert "empty" in summary
        assert "unfilled_connections" in summary
        assert "converged" in summary

    def test_grid_convergence_aligns_with_navigator(self):
        """Grid convergence summary matches is_converged from navigator."""
        from core.convergence import grid_convergence_summary
        from kernel.grid import Grid
        from kernel.navigator import is_converged
        from kernel.ops import fill as grid_fill

        # Converged grid: filled root, no gaps
        grid = Grid()
        grid.set_intent(
            intent_text="Simple app",
            postcode_key="INT.SEM.ECO.WHY.SFT",
            primitive="intent_contract",
        )
        summary = grid_convergence_summary(grid)
        assert summary["converged"] == is_converged(grid)

        # Unconverged grid: unfilled connections
        grid2 = Grid()
        grid2.set_intent(
            intent_text="Complex app",
            postcode_key="INT.SEM.ECO.WHY.SFT",
            primitive="intent_contract",
        )
        grid_fill(
            grid2,
            "INT.SEM.ECO.WHY.SFT",
            primitive="root",
            content="Complex app",
            confidence=0.95,
            connections=("STR.ENT.ECO.WHAT.SFT",),
            source=("__intent_contract__",),
        )
        summary2 = grid_convergence_summary(grid2)
        assert summary2["converged"] == is_converged(grid2)
        assert summary2["unfilled_connections"] > 0

    def test_turn_budget_scales_with_experiential_spec(self):
        """The experiential spec input gets a large turn budget."""
        spec_text = (
            "I need something that is more than siri, alexa, smart home and AI tools. "
            "I am after something that follows me around, does tasks for me. "
            "What my business needs is something that lives inside the business in AI level. "
            "the agent has full perception of operations, assists and manages internal operations. "
            "Mother helping front desk setup their Square POS efficiently. "
            "Mother was neutral and greeted customers via studio speakers, while seeing them on CCTV. "
            "Artists were able to use the system that Mother created for studio. "
            "We have our TVs, lights, speakers, CCTV, printers, spotify, emails, whatsapp, phone, "
            "web chatbot, calendar, Square, crypto bank, API keys, all wired up. "
            "She even built our accounting tool. "
            "For my friend who is super into Trading. "
            "I also connected Mother to our CCTV and Ring doorbell for security. "
            "I speak to Motherlabs via phone, rayban glasses, smart watch. "
            "Motherlabs connects with my AI fridge, knows what to shop. "
            "Motherlabs becomes a network for professional business AI entities. "
            "Its like a mycelium, fractal recursive self-improving system."
        )
        min_t, rec_t, max_t = estimate_turn_budget(spec_text)
        # Multi-domain input should get significantly more than base 6
        assert rec_t >= 12
        assert max_t >= 20


# =============================================================================
# Dialogue time optimization tests (convergence speed tuning)
# =============================================================================


class TestConvergenceTimeOptimization:
    """Tests for dialogue time optimization changes.

    These verify:
    - Relaxed plateau_window past recommended_turns
    - Raised delta_threshold (0.05 → 0.08)
    - Lowered convergence_min threshold (min_turns <= 8 → early check)
    """

    def test_convergence_relaxed_window_after_recommended(self):
        """Past recommended turns, single low delta is sufficient to converge."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=3,
            recommended_turns=9,
        )
        messages = [
            {"content": "User and Auth and Database entities.", "insight": "core system"},
        ]
        # Rounds 1-3 (turns 3, 6, 9) — build up history
        tracker.update(messages, total_turns=3)
        tracker.update(messages, total_turns=6)
        # Turn 9 = recommended_turns: single low delta should converge
        tracker.update(messages, total_turns=9)
        # At turn 9, we have deltas [1.0, 0.0, 0.0]
        # With recommended_turns=9, effective_window=1, last delta=0.0 < 0.08
        assert tracker.has_converged()

    def test_convergence_strict_window_before_recommended(self):
        """Before recommended turns, still requires full plateau_window=2."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=3,
            recommended_turns=12,
        )
        # Build changing content then one stable round
        tracker.update([
            {"content": "User entity.", "insight": "user"},
        ], total_turns=3)
        tracker.update([
            {"content": "User and Auth entities.", "insight": "user auth"},
        ], total_turns=6)
        # Turn 9 is before recommended_turns=12, so plateau_window=2 still required
        tracker.update([
            {"content": "User and Auth entities.", "insight": "user auth"},
        ], total_turns=9)
        # deltas: [1.0, high, 0.0] — only 1 low delta, need 2
        assert not tracker.has_converged()

    def test_convergence_threshold_08(self):
        """Delta of 0.07 counts as converged with threshold 0.08 (was missed at 0.05)."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=3,
            recommended_turns=9,
        )
        # Simulate deltas that bounce in the 0.05-0.08 band
        tracker._turn_count = 9
        tracker._deltas = [1.0, 0.38, 0.52, 0.07, 0.03]
        # Last 2 deltas: [0.07, 0.03] — both < 0.08
        # With recommended_turns=9 and turn_count=9, effective_window=1
        # But even with window=2 (if before recommended), 0.07 < 0.08 passes
        assert tracker.has_converged()

        # Verify the same would NOT converge at threshold 0.05
        tracker_strict = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=3,
        )
        tracker_strict._turn_count = 9
        tracker_strict._deltas = [1.0, 0.38, 0.52, 0.07, 0.03]
        # 0.07 > 0.05, so last 2 deltas [0.07, 0.03] would NOT pass
        assert not tracker_strict.has_converged()

    def test_convergence_min_8_gets_early_check(self):
        """min_turns=8 (from estimate_turn_budget) gets convergence_min=3.

        This means convergence can be checked after round 1 (turn 3) instead
        of waiting until turn 8.
        """
        # Simulate what engine.py does: convergence_min = 3 if min_turns <= 8
        min_turns = 8  # Typical for 600-2000 char single-domain intent
        convergence_min = 3 if min_turns <= 8 else min_turns
        assert convergence_min == 3, "min_turns=8 should get convergence_min=3"

        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=convergence_min,
        )
        messages = [
            {"content": "User and Auth and Database.", "insight": "entities"},
        ]
        # 3 updates with same content = converges at turn 3
        tracker.update(messages, total_turns=1)
        tracker.update(messages, total_turns=2)
        tracker.update(messages, total_turns=3)
        assert tracker.has_converged()
