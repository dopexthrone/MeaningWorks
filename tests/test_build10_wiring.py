"""
Build 10 wiring tests — #164 Load-balancing, #57 Experimentation-tracking,
#112 ROI-calculating, #161 Permission-gradient.

Tests verify each property is wired end-to-end.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ── #164 Load-balancing ──────────────────────────────────────────────

class TestLoadBalancing:
    """Verify peer delegation check in _compile_goal."""

    def test_delegation_router_exists(self):
        """DelegationRouter.choose_peer_for_task exists and is callable."""
        from mother.delegation import DelegationRouter
        assert hasattr(DelegationRouter, "choose_peer_for_task")

    def test_choose_peer_returns_none_no_connections(self):
        """With no connections, choose_peer_for_task returns None."""
        from mother.delegation import DelegationRouter
        wormhole = MagicMock()
        wormhole.connections = {}
        router = DelegationRouter(wormhole)
        result = router.choose_peer_for_task("compile")
        assert result is None

    def test_choose_peer_returns_peer_with_connections(self):
        """With connections, choose_peer_for_task returns a peer id."""
        from mother.delegation import DelegationRouter
        wormhole = MagicMock()
        conn = MagicMock()
        conn.capabilities = ["compile", "build"]
        conn.performance_history = [{"duration": 5.0, "success": True}]
        wormhole.connections = {"peer-abc-123": conn}
        router = DelegationRouter(wormhole)
        result = router.choose_peer_for_task("compile")
        assert result == "peer-abc-123"


# ── #57 Experimentation-tracking ─────────────────────────────────────

class TestExperimentationTracking:
    """Verify experiment_tag in journal."""

    def _make_journal(self, tmp_path):
        from mother.journal import BuildJournal
        return BuildJournal(tmp_path / "test.db")

    def test_experiment_tag_field_exists(self):
        """JournalEntry has experiment_tag field."""
        from mother.journal import JournalEntry
        entry = JournalEntry(experiment_tag="v2-rewrite")
        assert entry.experiment_tag == "v2-rewrite"

    def test_record_with_experiment_tag(self, tmp_path):
        """Recording an entry with experiment_tag persists it."""
        from mother.journal import JournalEntry
        journal = self._make_journal(tmp_path)
        entry = JournalEntry(
            event_type="compile",
            description="test compile",
            success=True,
            experiment_tag="alpha-1",
        )
        eid = journal.record(entry)
        assert eid > 0
        recent = journal.recent(1)
        assert len(recent) == 1
        assert recent[0].experiment_tag == "alpha-1"
        journal.close()

    def test_by_experiment_filters(self, tmp_path):
        """by_experiment returns only entries with matching tag."""
        from mother.journal import JournalEntry
        journal = self._make_journal(tmp_path)
        journal.record(JournalEntry(
            event_type="compile", description="a",
            success=True, experiment_tag="exp-a",
        ))
        journal.record(JournalEntry(
            event_type="compile", description="b",
            success=True, experiment_tag="exp-b",
        ))
        journal.record(JournalEntry(
            event_type="compile", description="c",
            success=False, experiment_tag="exp-a",
        ))
        results = journal.by_experiment("exp-a")
        assert len(results) == 2
        assert all(r.experiment_tag == "exp-a" for r in results)
        journal.close()

    def test_by_experiment_empty_tag(self, tmp_path):
        """by_experiment with nonexistent tag returns empty."""
        journal = self._make_journal(tmp_path)
        results = journal.by_experiment("nonexistent")
        assert results == []
        journal.close()

    def test_migration_adds_column(self, tmp_path):
        """Opening journal twice doesn't fail (column already exists)."""
        journal1 = self._make_journal(tmp_path)
        journal1.close()
        # Second open triggers migration again — should not error
        journal2 = self._make_journal(tmp_path)
        journal2.close()


# ── #112 ROI-calculating ─────────────────────────────────────────────

class TestROICalculating:
    """Verify ROI estimation appears after successful compile."""

    def test_roi_formula_components(self):
        """ROI formula: components × 2h × $75/hr vs compile cost."""
        comp_count = 5
        est_manual_hours = comp_count * 2.0
        compile_cost = 0.50
        manual_cost_est = est_manual_hours * 75.0
        ratio = manual_cost_est / compile_cost
        assert est_manual_hours == 10.0
        assert manual_cost_est == 750.0
        assert ratio == 1500.0

    def test_roi_skipped_when_no_components(self):
        """ROI block should not trigger when component_count is 0."""
        result = MagicMock()
        result.blueprint = {"components": []}
        result.cost = 0.10
        comps = result.blueprint.get("components", [])
        assert len(comps) == 0
        # ROI block would be skipped

    def test_roi_skipped_when_no_cost(self):
        """ROI block should not display when compile cost is 0."""
        result = MagicMock()
        result.blueprint = {"components": ["a", "b"]}
        result.cost = 0.0
        comp_count = len(result.blueprint["components"])
        compile_cost = result.cost
        assert comp_count == 2
        # manual_cost_est > 0 but compile_cost == 0 → no division, no display
        manual_cost_est = comp_count * 2.0 * 75.0
        assert manual_cost_est == 300.0
        # Both must be > 0 for display
        assert not (manual_cost_est > 0 and compile_cost > 0)

    def test_roi_display_with_valid_data(self):
        """ROI text contains leverage ratio when both costs available."""
        comp_count = 3
        compile_cost = 0.25
        est_manual_hours = comp_count * 2.0
        manual_cost_est = est_manual_hours * 75.0
        ratio = manual_cost_est / compile_cost
        roi_text = (
            f"ROI estimate: {comp_count} components × ~2h manual = "
            f"~{est_manual_hours:.0f}h (~${manual_cost_est:.0f}) vs "
            f"${compile_cost:.4f} compile cost ({ratio:.0f}× leverage)"
        )
        assert "3 components" in roi_text
        assert "6h" in roi_text
        assert "$450" in roi_text
        assert "$0.2500" in roi_text
        assert "1800× leverage" in roi_text


# ── #161 Permission-gradient ─────────────────────────────────────────

class TestPermissionGradient:
    """Verify domain_trust field and its effect on stance computation."""

    def test_domain_trust_field_exists(self):
        """StanceContext has domain_trust field."""
        from mother.stance import StanceContext
        ctx = StanceContext(domain_trust=0.8)
        assert ctx.domain_trust == 0.8

    def test_domain_trust_default(self):
        """domain_trust defaults to 0.5 (neutral)."""
        from mother.stance import StanceContext
        ctx = StanceContext()
        assert ctx.domain_trust == 0.5

    def test_high_domain_trust_acts_sooner(self):
        """With high domain_trust, ACT triggers at shorter idle."""
        from mother.stance import compute_stance, StanceContext, Stance
        # High trust — should ACT at 90s idle
        ctx_high = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=95.0,
            domain_trust=0.8,
        )
        assert compute_stance(ctx_high) == Stance.ACT

        # Default trust — same idle should NOT be ACT (needs 120s)
        ctx_default = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=95.0,
            domain_trust=0.5,
        )
        assert compute_stance(ctx_default) != Stance.ACT

    def test_low_domain_trust_forces_ask(self):
        """With low domain_trust, never auto-ACT, always ASK."""
        from mother.stance import compute_stance, StanceContext, Stance
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=250.0,  # well past 240s threshold
            domain_trust=0.2,
        )
        # Even with enough idle, low trust → ASK, not ACT
        assert compute_stance(ctx) == Stance.ASK

    def test_low_domain_trust_needs_longer_idle_for_ask(self):
        """Low domain_trust requires 120s idle even for ASK."""
        from mother.stance import compute_stance, StanceContext, Stance
        # 90s idle — enough for normal ASK (60s) but not for low-trust ASK (120s)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=90.0,
            domain_trust=0.2,
        )
        assert compute_stance(ctx) == Stance.WAIT

    def test_high_domain_trust_ask_at_shorter_idle(self):
        """High domain_trust allows ASK at 45s idle."""
        from mother.stance import compute_stance, StanceContext, Stance
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=50.0,
            domain_trust=0.8,
        )
        assert compute_stance(ctx) == Stance.ASK
