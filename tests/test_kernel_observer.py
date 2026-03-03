"""
Tests for kernel/observer.py — observation loop that writes deltas back to grid.

AX3 FEEDBACK: every run writes a delta. The grid evolves from real usage.
"""

import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.ops import fill
from kernel.observer import (
    ObservationDelta,
    ObservationBatch,
    record_observation,
    apply_observation,
    apply_batch,
    compute_confidence_drift,
    find_low_confidence_cells,
    find_anomalous_cells,
    _adjust_confidence,
    _should_transition,
    _CONFIRM_BOOST,
    _CONTRADICT_DECAY,
    _ANOMALY_DECAY,
    _PROMOTE_THRESHOLD,
    _DEMOTE_THRESHOLD,
    _QUARANTINE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grid_with_cell(postcode: str = "INT.SEM.ECO.WHY.SFT",
                         fill_state: FillState = FillState.P,
                         confidence: float = 0.70) -> Grid:
    """Create a grid with one cell at given state."""
    grid = Grid()
    grid.set_intent("test intent", postcode, "test_primitive")
    pc = parse_postcode(postcode)
    cell = Cell(
        postcode=pc,
        primitive="test_primitive",
        content="test content",
        fill=fill_state,
        confidence=confidence,
        connections=frozenset(),
        parent=None,
        source=("test",),
        revisions=(),
    )
    grid.cells[postcode] = cell
    return grid


def _make_grid_multi() -> Grid:
    """Grid with several cells at different states."""
    grid = Grid()
    grid.set_intent("multi test", "INT.SEM.ECO.WHY.SFT", "intent")

    cells = [
        ("INT.SEM.ECO.WHY.SFT", FillState.F, 0.90),
        ("ORG.ENT.APP.WHAT.SFT", FillState.P, 0.60),
        ("COG.BHV.DOM.HOW.SFT", FillState.P, 0.30),
        ("STR.FNC.CMP.HOW.SFT", FillState.F, 0.55),
        ("AGN.AGT.FET.WHO.SFT", FillState.P, 0.80),
    ]
    for pk, fs, conf in cells:
        pc = parse_postcode(pk)
        grid.cells[pk] = Cell(
            postcode=pc, primitive="p", content="c",
            fill=fs, confidence=conf,
            connections=frozenset(), parent=None,
            source=("test",), revisions=(),
        )
        grid.activated_layers.add(pc.layer)
    return grid


# ---------------------------------------------------------------------------
# ObservationDelta dataclass
# ---------------------------------------------------------------------------

class TestObservationDelta:
    def test_frozen(self):
        d = ObservationDelta(
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="compilation",
            expected="output",
            actual="output",
            confidence_before=0.7,
            confidence_after=0.73,
        )
        with pytest.raises(AttributeError):
            d.postcode = "OTHER"

    def test_defaults(self):
        d = ObservationDelta(
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="compilation",
            expected="a",
            actual="b",
            confidence_before=0.5,
            confidence_after=0.42,
        )
        assert d.anomaly is False
        assert d.anomaly_detail == ""

    def test_with_anomaly(self):
        d = ObservationDelta(
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="execution",
            expected="success",
            actual="crash",
            confidence_before=0.8,
            confidence_after=0.68,
            anomaly=True,
            anomaly_detail="Process crashed unexpectedly",
        )
        assert d.anomaly is True
        assert "crashed" in d.anomaly_detail


# ---------------------------------------------------------------------------
# ObservationBatch dataclass
# ---------------------------------------------------------------------------

class TestObservationBatch:
    def test_defaults(self):
        b = ObservationBatch(run_id="test-1")
        assert b.deltas == []
        assert b.cells_touched == 0
        assert b.cells_improved == 0
        assert b.cells_degraded == 0
        assert b.transitions == []

    def test_mutable(self):
        b = ObservationBatch(run_id="test-2")
        b.cells_touched = 5
        assert b.cells_touched == 5


# ---------------------------------------------------------------------------
# Confidence adjustment
# ---------------------------------------------------------------------------

class TestConfidenceAdjustment:
    def test_confirm_boosts(self):
        result = _adjust_confidence(0.70, confirmed=True)
        assert result == pytest.approx(0.70 + _CONFIRM_BOOST)

    def test_contradict_decays(self):
        result = _adjust_confidence(0.70, confirmed=False)
        assert result == pytest.approx(0.70 - _CONTRADICT_DECAY)

    def test_anomaly_decays_more(self):
        result = _adjust_confidence(0.70, confirmed=False, anomaly=True)
        assert result == pytest.approx(0.70 - _ANOMALY_DECAY)

    def test_clamp_max(self):
        result = _adjust_confidence(0.99, confirmed=True)
        assert result <= 1.0

    def test_clamp_min(self):
        result = _adjust_confidence(0.05, confirmed=False, anomaly=True)
        assert result >= 0.0

    def test_at_zero(self):
        result = _adjust_confidence(0.0, confirmed=False)
        assert result == 0.0

    def test_at_one(self):
        result = _adjust_confidence(1.0, confirmed=True)
        assert result == 1.0

    def test_anomaly_overrides_confirmed(self):
        # anomaly=True takes priority regardless of confirmed
        result = _adjust_confidence(0.70, confirmed=True, anomaly=True)
        assert result == pytest.approx(0.70 - _ANOMALY_DECAY)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestShouldTransition:
    def test_promote_p_to_f(self):
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, _PROMOTE_THRESHOLD)
        assert result == FillState.F

    def test_no_promote_below_threshold(self):
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.84)
        assert result is None

    def test_demote_f_to_p(self):
        grid = _make_grid_with_cell(confidence=0.80, fill_state=FillState.F)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.49)
        assert result == FillState.P

    def test_no_demote_above_threshold(self):
        grid = _make_grid_with_cell(confidence=0.80, fill_state=FillState.F)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.51)
        assert result is None

    def test_quarantine_p_to_q(self):
        grid = _make_grid_with_cell(confidence=0.30, fill_state=FillState.P)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.19)
        assert result == FillState.Q

    def test_no_quarantine_above_threshold(self):
        grid = _make_grid_with_cell(confidence=0.30, fill_state=FillState.P)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.21)
        assert result is None

    def test_f_stays_f_above_demote(self):
        grid = _make_grid_with_cell(confidence=0.90, fill_state=FillState.F)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        result = _should_transition(cell, 0.88)
        assert result is None


# ---------------------------------------------------------------------------
# record_observation
# ---------------------------------------------------------------------------

class TestRecordObservation:
    def test_record_existing_cell_confirmed(self):
        grid = _make_grid_with_cell(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "output A", "output A",
            confirmed=True,
        )
        assert delta.confidence_before == 0.70
        assert delta.confidence_after == pytest.approx(0.70 + _CONFIRM_BOOST)
        assert delta.anomaly is False

    def test_record_existing_cell_contradiction(self):
        grid = _make_grid_with_cell(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "verification", "pass", "fail",
            confirmed=False,
        )
        assert delta.confidence_after == pytest.approx(0.70 - _CONTRADICT_DECAY)

    def test_record_existing_cell_anomaly(self):
        grid = _make_grid_with_cell(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "execution", "success", "crash",
            confirmed=False, anomaly=True,
            anomaly_detail="segfault",
        )
        assert delta.anomaly is True
        assert delta.confidence_after == pytest.approx(0.70 - _ANOMALY_DECAY)

    def test_record_nonexistent_cell(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        delta = record_observation(
            grid, "ORG.ENT.APP.WHAT.SFT",
            "compilation", "exist", "missing",
        )
        assert delta.anomaly is True
        assert "does not exist" in delta.anomaly_detail
        assert delta.confidence_before == 0.0
        assert delta.confidence_after == 0.0

    def test_record_does_not_mutate_grid(self):
        grid = _make_grid_with_cell(confidence=0.70)
        record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "a", "b", confirmed=True,
        )
        # Grid cell unchanged
        assert grid.cells["INT.SEM.ECO.WHY.SFT"].confidence == 0.70


# ---------------------------------------------------------------------------
# apply_observation
# ---------------------------------------------------------------------------

class TestApplyObservation:
    def test_apply_updates_confidence(self):
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "a", "a", confirmed=True,
        )
        transition = apply_observation(grid, delta)
        assert grid.cells["INT.SEM.ECO.WHY.SFT"].confidence == pytest.approx(0.73)
        assert transition is None  # no state change at 0.73

    def test_apply_triggers_promotion(self):
        grid = _make_grid_with_cell(confidence=0.84, fill_state=FillState.P)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "verification", "pass", "pass", confirmed=True,
        )
        transition = apply_observation(grid, delta)
        assert transition is not None
        assert transition == ("P", "F")
        assert grid.cells["INT.SEM.ECO.WHY.SFT"].fill == FillState.F

    def test_apply_triggers_demotion(self):
        grid = _make_grid_with_cell(confidence=0.55, fill_state=FillState.F)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "execution", "success", "fail",
            confirmed=False, anomaly=True,
        )
        transition = apply_observation(grid, delta)
        # 0.55 - 0.12 = 0.43 < 0.50 → demote
        assert transition == ("F", "P")
        assert grid.cells["INT.SEM.ECO.WHY.SFT"].fill == FillState.P

    def test_apply_triggers_quarantine(self):
        grid = _make_grid_with_cell(confidence=0.25, fill_state=FillState.P)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "execution", "ok", "crash",
            confirmed=False, anomaly=True,
        )
        transition = apply_observation(grid, delta)
        # 0.25 - 0.12 = 0.13 < 0.20 → quarantine
        assert transition == ("P", "Q")
        assert grid.cells["INT.SEM.ECO.WHY.SFT"].fill == FillState.Q

    def test_apply_records_revision(self):
        grid = _make_grid_with_cell(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "a", "a", confirmed=True,
        )
        apply_observation(grid, delta)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert len(cell.revisions) == 1
        assert cell.revisions[0] == ("compilation", 0.70)

    def test_apply_appends_source(self):
        grid = _make_grid_with_cell(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "verification", "a", "a", confirmed=True,
        )
        apply_observation(grid, delta)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert "obs:verification" in cell.source

    def test_apply_nonexistent_cell(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        delta = ObservationDelta(
            postcode="ORG.ENT.APP.WHAT.SFT",
            event_type="compilation",
            expected="exist",
            actual="missing",
            confidence_before=0.0,
            confidence_after=0.0,
            anomaly=True,
        )
        result = apply_observation(grid, delta)
        assert result is None

    def test_multiple_observations_accumulate(self):
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)
        for _ in range(5):
            delta = record_observation(
                grid, "INT.SEM.ECO.WHY.SFT",
                "compilation", "a", "a", confirmed=True,
            )
            apply_observation(grid, delta)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(0.70 + 5 * _CONFIRM_BOOST)
        assert len(cell.revisions) == 5


# ---------------------------------------------------------------------------
# apply_batch
# ---------------------------------------------------------------------------

class TestApplyBatch:
    def test_batch_basic(self):
        grid = _make_grid_multi()
        deltas = []
        for pk in ["INT.SEM.ECO.WHY.SFT", "ORG.ENT.APP.WHAT.SFT"]:
            d = record_observation(grid, pk, "compilation", "a", "a", confirmed=True)
            deltas.append(d)
        batch = apply_batch(grid, deltas)
        assert batch.cells_touched == 2
        assert batch.cells_improved == 2
        assert batch.cells_degraded == 0
        assert len(batch.deltas) == 2

    def test_batch_mixed_outcomes(self):
        grid = _make_grid_multi()
        deltas = [
            record_observation(grid, "INT.SEM.ECO.WHY.SFT",
                             "compilation", "a", "a", confirmed=True),
            record_observation(grid, "ORG.ENT.APP.WHAT.SFT",
                             "verification", "pass", "fail", confirmed=False),
        ]
        batch = apply_batch(grid, deltas)
        assert batch.cells_improved == 1
        assert batch.cells_degraded == 1

    def test_batch_tracks_transitions(self):
        grid = _make_grid_with_cell(confidence=0.84, fill_state=FillState.P)
        deltas = [
            record_observation(grid, "INT.SEM.ECO.WHY.SFT",
                             "verification", "pass", "pass", confirmed=True),
        ]
        batch = apply_batch(grid, deltas)
        assert len(batch.transitions) == 1
        assert batch.transitions[0] == ("INT.SEM.ECO.WHY.SFT", "P", "F")

    def test_batch_empty(self):
        grid = _make_grid_multi()
        batch = apply_batch(grid, [])
        assert batch.cells_touched == 0
        assert batch.deltas == []

    def test_batch_run_id(self):
        grid = _make_grid_multi()
        deltas = [
            record_observation(grid, "INT.SEM.ECO.WHY.SFT",
                             "compilation", "a", "a", confirmed=True),
        ]
        batch = apply_batch(grid, deltas)
        assert batch.run_id == "obs-1"


# ---------------------------------------------------------------------------
# compute_confidence_drift
# ---------------------------------------------------------------------------

class TestConfidenceDrift:
    def test_no_previous_grid(self):
        grid = _make_grid_multi()
        result = compute_confidence_drift(grid)
        assert "mean" in result
        assert "min" in result
        assert "max" in result
        assert result["drift"] == 0.0
        assert result["count"] == 5

    def test_empty_grid(self):
        grid = Grid()
        # No cells at all (no set_intent, which creates a root cell)
        result = compute_confidence_drift(grid)
        assert result["count"] == 0
        assert result["mean"] == 0.0

    def test_drift_with_previous(self):
        grid1 = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)
        grid2 = _make_grid_with_cell(confidence=0.80, fill_state=FillState.P)
        result = compute_confidence_drift(grid2, grid1)
        assert result["drift"] == pytest.approx(0.10)
        assert result["improved"] == 1
        assert result["degraded"] == 0
        assert result["stable"] == 0

    def test_drift_negative(self):
        grid1 = _make_grid_with_cell(confidence=0.80, fill_state=FillState.F)
        grid2 = _make_grid_with_cell(confidence=0.60, fill_state=FillState.F)
        result = compute_confidence_drift(grid2, grid1)
        assert result["drift"] == pytest.approx(-0.20)
        assert result["degraded"] == 1

    def test_drift_no_overlap(self):
        grid1 = Grid()
        grid1.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        pc1 = parse_postcode("INT.SEM.ECO.WHY.SFT")
        grid1.cells["INT.SEM.ECO.WHY.SFT"] = Cell(
            postcode=pc1, primitive="p", content="c",
            fill=FillState.F, confidence=0.90,
            connections=frozenset(), parent=None,
            source=("test",), revisions=(),
        )

        grid2 = Grid()
        grid2.set_intent("test", "ORG.ENT.APP.WHAT.SFT", "intent")
        pc2 = parse_postcode("ORG.ENT.APP.WHAT.SFT")
        grid2.cells["ORG.ENT.APP.WHAT.SFT"] = Cell(
            postcode=pc2, primitive="p", content="c",
            fill=FillState.F, confidence=0.80,
            connections=frozenset(), parent=None,
            source=("test",), revisions=(),
        )
        result = compute_confidence_drift(grid2, grid1)
        assert result["drift"] == 0.0  # no overlapping cells

    def test_excludes_empty_cells(self):
        grid = _make_grid_with_cell(confidence=0.0, fill_state=FillState.E)
        result = compute_confidence_drift(grid)
        # Empty cells excluded — only root cell (set_intent) might count
        # depending on what set_intent creates
        assert result["count"] <= 1


# ---------------------------------------------------------------------------
# find_low_confidence_cells
# ---------------------------------------------------------------------------

class TestFindLowConfidence:
    def test_finds_low_cells(self):
        grid = _make_grid_multi()
        results = find_low_confidence_cells(grid, threshold=0.60)
        # COG.BHV.DOM.HOW.SFT at 0.30, STR.FNC.CMP.HOW.SFT at 0.55 (F)
        postcodes = [r[0] for r in results]
        assert "COG.BHV.DOM.HOW.SFT" in postcodes
        assert "STR.FNC.CMP.HOW.SFT" in postcodes

    def test_sorted_by_confidence(self):
        grid = _make_grid_multi()
        results = find_low_confidence_cells(grid, threshold=0.60)
        confidences = [r[1] for r in results]
        assert confidences == sorted(confidences)

    def test_empty_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        results = find_low_confidence_cells(grid)
        assert len(results) == 0

    def test_high_threshold_catches_more(self):
        grid = _make_grid_multi()
        results = find_low_confidence_cells(grid, threshold=0.95)
        # All F/P cells below 0.95
        assert len(results) == 5

    def test_low_threshold_catches_fewer(self):
        grid = _make_grid_multi()
        results = find_low_confidence_cells(grid, threshold=0.31)
        # Only COG.BHV at 0.30
        assert len(results) == 1

    def test_excludes_empty_and_quarantined(self):
        grid = _make_grid_with_cell(confidence=0.10, fill_state=FillState.Q)
        results = find_low_confidence_cells(grid, threshold=0.50)
        # Q cells excluded
        assert len(results) == 0


# ---------------------------------------------------------------------------
# find_anomalous_cells
# ---------------------------------------------------------------------------

class TestFindAnomalous:
    def test_finds_anomalies(self):
        deltas = [
            ObservationDelta("A", "compilation", "a", "a", 0.7, 0.73, anomaly=False),
            ObservationDelta("B", "execution", "ok", "crash", 0.8, 0.68, anomaly=True),
            ObservationDelta("C", "verification", "pass", "fail", 0.6, 0.52, anomaly=True),
        ]
        results = find_anomalous_cells(deltas)
        assert len(results) == 2
        assert results[0].postcode == "B"
        assert results[1].postcode == "C"

    def test_no_anomalies(self):
        deltas = [
            ObservationDelta("A", "compilation", "a", "a", 0.7, 0.73),
        ]
        results = find_anomalous_cells(deltas)
        assert len(results) == 0

    def test_empty_list(self):
        assert find_anomalous_cells([]) == []


# ---------------------------------------------------------------------------
# Integration: observation → apply → drift
# ---------------------------------------------------------------------------

class TestObservationIntegration:
    def test_full_lifecycle(self):
        """Record observations, apply them, check drift."""
        grid = _make_grid_multi()

        # Snapshot before
        before_drift = compute_confidence_drift(grid)

        # All confirmed
        deltas = []
        for pk in grid.cells:
            d = record_observation(grid, pk, "compilation", "a", "a", confirmed=True)
            deltas.append(d)

        batch = apply_batch(grid, deltas)
        assert batch.cells_improved == 5
        assert batch.cells_degraded == 0

        # Check drift against before
        # (need a separate "before" grid to compare)
        grid_before = _make_grid_multi()
        after_drift = compute_confidence_drift(grid, grid_before)
        assert after_drift["drift"] > 0
        assert after_drift["improved"] == 5

    def test_repeated_failures_quarantine(self):
        """Repeated contradictions push P cell to Q."""
        grid = _make_grid_with_cell(confidence=0.50, fill_state=FillState.P)

        # Each contradiction drops by 0.08
        # 0.50 → 0.42 → 0.34 → 0.26 → 0.18 (< 0.20 → Q)
        for i in range(4):
            delta = record_observation(
                grid, "INT.SEM.ECO.WHY.SFT",
                "verification", "pass", "fail",
                confirmed=False,
            )
            apply_observation(grid, delta)

        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.fill == FillState.Q
        assert cell.confidence == pytest.approx(0.50 - 4 * _CONTRADICT_DECAY)

    def test_repeated_confirms_promote(self):
        """Repeated confirmations push P cell to F."""
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)

        # Each confirm boosts by 0.03
        # Need 0.85 - 0.70 = 0.15, so 5 confirms → 0.85
        for i in range(5):
            delta = record_observation(
                grid, "INT.SEM.ECO.WHY.SFT",
                "compilation", "a", "a", confirmed=True,
            )
            apply_observation(grid, delta)

        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.fill == FillState.F
        assert cell.confidence == pytest.approx(0.70 + 5 * _CONFIRM_BOOST)

    def test_mixed_observations_settle(self):
        """Mix of confirms and contradictions reaches equilibrium."""
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)

        # Alternate confirm/contradict
        for i in range(10):
            confirmed = i % 2 == 0
            delta = record_observation(
                grid, "INT.SEM.ECO.WHY.SFT",
                "compilation", "a", "b" if not confirmed else "a",
                confirmed=confirmed,
            )
            apply_observation(grid, delta)

        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        # 5 confirms (+0.15) + 5 contradicts (-0.40) = net -0.25
        assert cell.confidence == pytest.approx(0.70 + 5 * _CONFIRM_BOOST - 5 * _CONTRADICT_DECAY)
        assert len(cell.revisions) == 10

    def test_low_confidence_after_anomalies(self):
        """Anomalies surface cells for human escalation."""
        grid = _make_grid_with_cell(confidence=0.70, fill_state=FillState.P)

        # 3 anomalies: 0.70 → 0.58 → 0.46 → 0.34
        for _ in range(3):
            delta = record_observation(
                grid, "INT.SEM.ECO.WHY.SFT",
                "execution", "ok", "crash",
                anomaly=True, anomaly_detail="unexpected",
            )
            apply_observation(grid, delta)

        low = find_low_confidence_cells(grid, threshold=0.60)
        assert len(low) >= 1
        assert low[0][0] == "INT.SEM.ECO.WHY.SFT"
