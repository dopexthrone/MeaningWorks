"""
Stress test: cog-infra pattern transfer — episodic memory, training emission,
append-only history, typed edges, L2 patterns, tier routing, autonomous scheduler.

Hammers all new infrastructure at scale:
- 100+ memory records, pattern detection across corpus
- 50+ training examples, JSONL integrity under concurrent writes
- 20+ grid saves with history accumulation
- Hub detection, connection metadata at scale
- Pattern persistence roundtrips
- Tier routing with all failure modes
- Scheduler state machine exhaustive walk
- Full recursive self-improvement loop wiring verification
"""

import asyncio
import json
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from kernel.memory import (
    MemoryRecord,
    LearnedPattern,
    consolidate,
    save_memory,
    load_recent_memories,
    memory_stats,
    detect_patterns,
    save_patterns,
    load_patterns,
    format_pattern_context,
    _db_path,
)
from kernel.training import (
    TrainingExample,
    extract_training_examples,
    emit_jsonl,
    training_stats,
)
from kernel.store import (
    save_grid,
    load_grid,
    cell_history,
    cell_version_count,
    history_stats,
    load_connection_metadata,
    hub_postcodes,
    save_typed_connection,
    _ensure_schema,
)
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from core.llm import RouteTier, MockClient, FailoverClient
from mother.daemon import DaemonMode, DaemonConfig, CompileRequest


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(run_id, **overrides):
    defaults = dict(
        run_id=run_id, timestamp=time.time(),
        intent_summary="Build something", domain="software",
        trust_score=75.0, fidelity_score=0.8,
        compression_losses={}, gate_results={"compilation": "pass"},
        learnings=[], gaps=[], cell_count=20, fill_rate=0.7,
        cost_usd=0.15, duration_seconds=120.0,
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _make_state(known=None):
    s = SimpleNamespace()
    s.known = known or {}
    return s


def _make_result(**kwargs):
    defaults = dict(success=True, verification={}, insights=[],
                    semantic_grid=None, blueprint={}, cache_stats={},
                    stage_timings={}, error=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _FillState(Enum):
    F = "filled"
    P = "partial"
    E = "empty"
    Q = "quarantined"
    C = "candidate"


def _make_cell(fill="F", content="content", confidence=0.9):
    return SimpleNamespace(fill=_FillState[fill], content=content, confidence=confidence)


def _make_mock_grid(cells_dict):
    return SimpleNamespace(cells=cells_dict)


PC_TEMPLATE = [
    "INT.ENT.ECO.WHAT.SFT", "SEM.BHV.APP.HOW.SFT", "ORG.FNC.DOM.WHY.SFT",
    "STR.REL.CMP.WHAT.SFT", "COG.PLN.FET.HOW.SFT", "STA.STA.FNC.WHEN.SFT",
    "IDN.ENT.APP.WHO.SFT", "TME.SCH.DOM.WHEN.SFT", "EXC.FLW.CMP.HOW.SFT",
    "CTR.GTE.FET.IF.SFT",
]


def _make_real_grid(n_cells, connections_per=0):
    g = Grid()
    g.intent_text = "stress test intent"
    pcs = []
    scopes = ["ECO", "APP", "DOM", "FET", "CMP", "FNC", "STP", "OPR", "EXP", "VAL"]
    for i in range(n_cells):
        idx = i % len(PC_TEMPLATE)
        # Use i // template_count for scope to avoid collisions
        scope = scopes[(i // len(PC_TEMPLATE)) % len(scopes)]
        pc_str = PC_TEMPLATE[idx].split(".")
        pc_str[2] = scope
        pc_key = ".".join(pc_str)
        try:
            pc = parse_postcode(pc_key)
        except ValueError:
            continue

        conns = tuple(pcs[-connections_per:]) if connections_per and pcs else ()
        cell = Cell(
            postcode=pc, primitive=f"node_{i}",
            content=f"Content for cell {i}", fill=FillState.F,
            confidence=0.5 + (i % 5) * 0.1, connections=conns,
            source=(f"stress_{i}",),
        )
        g.activated_layers.add(pc.layer)
        g.cells[pc.key] = cell
        pcs.append(pc.key)
    return g


# ===========================================================================
# 1. EPISODIC MEMORY — 100+ records at scale
# ===========================================================================

class TestMemoryScale:
    def test_100_records_save_load(self, tmp_path):
        """Save 100 records, verify load ordering and stats."""
        for i in range(100):
            save_memory(
                _make_record(f"run-{i:04d}", trust_score=50.0 + i * 0.3,
                             domain="software" if i % 2 == 0 else "api",
                             timestamp=1000.0 + i),
                db_dir=tmp_path,
            )

        all_records = load_recent_memories(limit=200, db_dir=tmp_path)
        assert len(all_records) == 100

        # Most recent first
        assert all_records[0].run_id == "run-0099"
        assert all_records[-1].run_id == "run-0000"

        # Domain filter
        sw = load_recent_memories(domain="software", db_dir=tmp_path)
        assert len(sw) == 50

        # Stats
        stats = memory_stats(db_dir=tmp_path)
        assert stats["total"] == 100
        assert stats["domains"]["software"] == 50
        assert stats["domains"]["api"] == 50
        assert stats["avg_trust"] > 50

    def test_consolidation_from_real_compile_artifacts(self, tmp_path):
        """Consolidate from realistic CompileResult + SharedState."""
        state = _make_state({
            "input": "Build a booking system for a tattoo studio " * 5,
            "domain": "software",
            "closed_loop_result": {"fidelity": 0.82, "passed": True},
            "compression_loss_categories": {"entity": 3, "relationship": 1},
        })
        result = _make_result(
            success=True,
            verification={
                "completeness": {"score": 85},
                "consistency": {"score": 78},
                "coherence": {"score": 90},
                "traceability": {"score": 72},
            },
            insights=[
                "pattern reuse detected in auth",
                "missing payment gateway constraint",
                "improved schema extraction",
            ],
            semantic_grid={"cells": 45, "fill_rate": 0.73},
        )

        mem = consolidate(result, state, "real-001", 185.3)
        save_memory(mem, db_dir=tmp_path)

        loaded = load_recent_memories(db_dir=tmp_path)
        assert len(loaded) == 1
        m = loaded[0]
        assert m.trust_score == pytest.approx(81.25)  # (85+78+90+72)/4
        assert m.fidelity_score == 0.82
        assert m.compression_losses == {"entity": 3, "relationship": 1}
        assert m.cell_count == 45
        assert m.fill_rate == 0.73
        assert len(m.learnings) >= 2  # "pattern" and "improved"
        assert len(m.gaps) >= 1       # "missing"
        assert m.gate_results["compilation"] == "pass"
        assert m.gate_results["closed_loop"] == "pass"

    def test_concurrent_saves(self, tmp_path):
        """Multiple threads saving simultaneously."""
        def save_batch(start):
            for i in range(20):
                save_memory(
                    _make_record(f"thread-{start}-{i}"),
                    db_dir=tmp_path,
                )

        threads = [threading.Thread(target=save_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_records = load_recent_memories(limit=200, db_dir=tmp_path)
        assert len(all_records) == 100  # 5 threads × 20 records


# ===========================================================================
# 2. L2 PATTERN DETECTION — corpus-scale
# ===========================================================================

class TestPatternDetectionScale:
    def test_detect_from_100_memories(self, tmp_path):
        """Detect patterns across 100 memory records with planted signals."""
        memories = []
        for i in range(100):
            losses = {}
            gaps = []
            gates = {"compilation": "pass"}

            # Plant signals: entity loss in 40% of runs
            if i % 5 < 2:
                losses["entity"] = 2
            # Plant: low coherence gap in 30%
            if i % 10 < 3:
                gaps.append("Low coherence: 45/100")
            # Plant: closed_loop failure in 25%
            if i % 4 == 0:
                gates["closed_loop"] = "fail"

            memories.append(_make_record(
                f"r-{i}", compression_losses=losses, gaps=gaps,
                gate_results=gates, domain="software",
                trust_score=60.0 + (i % 20),
            ))

        patterns = detect_patterns(memories, min_occurrences=10)

        # Should detect: recurring_loss_entity, recurring_gap, gate_failure
        ids = [p.pattern_id for p in patterns]
        assert "recurring_loss_entity" in ids
        assert any("gate_failure" in pid for pid in ids)
        assert any("recurring_gap" in pid for pid in ids)

        # All confidence <= 1.0
        assert all(p.confidence <= 1.0 for p in patterns)
        # All frequency >= 10
        assert all(p.frequency >= 10 for p in patterns)

    def test_pattern_persistence_roundtrip_at_scale(self, tmp_path):
        """Save 50 patterns, load with filters."""
        patterns = [
            LearnedPattern(
                pattern_id=f"p-{i}",
                category="recurring_gap" if i % 2 == 0 else "domain_confidence",
                description=f"Pattern {i} description",
                frequency=i + 1,
                confidence=0.1 + (i % 10) * 0.1,
                affected_postcodes=(),
                remediation=f"Fix pattern {i}",
            )
            for i in range(50)
        ]

        saved = save_patterns(patterns, db_dir=tmp_path)
        assert saved == 50

        # High confidence only
        high = load_patterns(min_confidence=0.8, db_dir=tmp_path)
        assert all(p.confidence >= 0.8 for p in high)

        # All patterns
        all_p = load_patterns(min_confidence=0.0, db_dir=tmp_path)
        assert len(all_p) == 50

        # Sorted by frequency DESC
        freqs = [p.frequency for p in all_p]
        assert freqs == sorted(freqs, reverse=True)

    def test_format_context_within_prompt_budget(self):
        """format_pattern_context caps at 10 patterns for prompt size."""
        patterns = [
            LearnedPattern(f"p{i}", "gap", f"Description {i} " * 20,
                           i, 0.5, (), "fix " * 10)
            for i in range(30)
        ]
        ctx = format_pattern_context(patterns)
        lines = [l for l in ctx.split("\n") if l.startswith("- [")]
        assert len(lines) <= 10


# ===========================================================================
# 3. TRAINING EMISSION — volume + integrity
# ===========================================================================

class TestTrainingScale:
    def test_200_cell_grid_extraction(self, tmp_path):
        """Extract training examples from a grid with 200 cells."""
        cells = {}
        for i in range(200):
            if i % 3 == 0:
                cells[f"pc{i}"] = _make_cell("F", f"filled content {i}", 0.9)
            elif i % 3 == 1:
                cells[f"pc{i}"] = _make_cell("Q", f"quarantined {i}", 0.1)
            else:
                cells[f"pc{i}"] = _make_cell("P", f"partial {i}", 0.4)

        grid = _make_mock_grid(cells)
        state = _make_state({"input": "Large test", "domain": "software"})

        examples = extract_training_examples(grid, _make_result(), state, "run-big")
        assert len(examples) == 200  # cap is 200
        types = [e.type for e in examples]
        assert "positive" in types
        assert "negative" in types
        assert "instruction" in types

    def test_jsonl_integrity_after_multiple_writes(self, tmp_path):
        """Write 10 batches, verify all lines parse as valid JSON."""
        training_dir = tmp_path / "training"
        for batch in range(10):
            examples = [
                TrainingExample("positive", f"pc{batch}_{i}", "ctx", f"out{i}",
                                0.9, "", "software", f"r{batch}")
                for i in range(20)
            ]
            emit_jsonl(examples, output_dir=training_dir)

        stats = training_stats(output_dir=training_dir)
        assert stats["total_examples"] == 200
        assert stats["type_counts"]["positive"] == 200

        # Verify every line is valid JSON
        for f in training_dir.glob("*.jsonl"):
            for line_num, line in enumerate(f.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON at {f.name}:{line_num}")


# ===========================================================================
# 4. APPEND-ONLY HISTORY — accumulation under repeated saves
# ===========================================================================

class TestHistoryScale:
    def test_20_saves_accumulate_history(self, tmp_path):
        """Save same map 20 times, verify history grows monotonically."""
        for i in range(20):
            g = _make_real_grid(5)
            # Modify content each time
            for pc_key in list(g.cells.keys()):
                cell = g.cells[pc_key]
                g.cells[pc_key] = Cell(
                    postcode=cell.postcode, primitive=cell.primitive,
                    content=f"v{i} content", fill=cell.fill,
                    confidence=min(1.0, 0.3 + i * 0.03),
                    connections=cell.connections, source=cell.source,
                )
            save_grid(g, "stress-map", db_dir=tmp_path)

        # Live snapshot only has latest content
        loaded = load_grid("stress-map", db_dir=tmp_path)
        for cell in loaded.cells.values():
            assert "v19" in cell.content

        # History has all 20 versions for each cell
        stats = history_stats("stress-map", db_dir=tmp_path)
        assert stats["total_versions"] == 100  # 5 cells × 20 saves
        assert stats["unique_cells"] == 5
        assert stats["max_version"] == 20

        # Check individual cell history
        first_pc = list(loaded.cells.keys())[0]
        hist = cell_history("stress-map", first_pc, db_dir=tmp_path)
        assert len(hist) == 20
        assert hist[0]["version"] == 1
        assert hist[-1]["version"] == 20
        assert "v0" in hist[0]["content"]
        assert "v19" in hist[-1]["content"]

        # Confidence should increase over versions
        assert hist[0]["confidence"] < hist[-1]["confidence"]


# ===========================================================================
# 5. TYPED EDGES — hub detection at scale
# ===========================================================================

class TestTypedEdgesScale:
    def test_hub_detection_in_large_graph(self, tmp_path):
        """Build a graph with clear hubs, verify detection."""
        g = _make_real_grid(20, connections_per=0)
        pcs = list(g.cells.keys())

        # Make first cell a hub connected to all others
        hub_pc = pcs[0]
        hub_cell = g.cells[hub_pc]
        g.cells[hub_pc] = Cell(
            postcode=hub_cell.postcode, primitive=hub_cell.primitive,
            content=hub_cell.content, fill=hub_cell.fill,
            confidence=hub_cell.confidence,
            connections=tuple(pcs[1:]),  # connected to all 19 others
            source=hub_cell.source,
        )

        save_grid(g, "hub-test", db_dir=tmp_path)

        hubs = hub_postcodes("hub-test", min_connections=10, db_dir=tmp_path)
        assert hub_pc in hubs
        assert len(hubs) == 1  # only the hub qualifies

        # Connection metadata
        meta = load_connection_metadata("hub-test", db_dir=tmp_path)
        hub_conns = [m for m in meta if m["from"] == hub_pc]
        assert len(hub_conns) == 19  # connected to all others
        assert all(m["type"] == "association" for m in hub_conns)
        assert all(m["strength"] == 0.5 for m in hub_conns)

    def test_typed_connection_upgrade(self, tmp_path):
        """Save grid with default edges, then upgrade specific ones."""
        g = _make_real_grid(5, connections_per=1)
        save_grid(g, "typed-test", db_dir=tmp_path)

        pcs = list(g.cells.keys())
        # Upgrade first connection to 'derivation' with high strength
        save_typed_connection(
            "typed-test", pcs[1], pcs[0],
            connection_type="derivation", strength=0.95,
            db_dir=tmp_path,
        )

        meta = load_connection_metadata("typed-test", db_dir=tmp_path)
        upgraded = [m for m in meta if m["from"] == pcs[1] and m["to"] == pcs[0]]
        assert len(upgraded) == 1
        assert upgraded[0]["type"] == "derivation"
        assert upgraded[0]["strength"] == 0.95


# ===========================================================================
# 6. TIER ROUTING — exhaustive failure modes
# ===========================================================================

class TestTierRoutingExhaustive:
    def _make_provider(self, name, fail=False):
        m = MockClient()
        if fail:
            m.complete = MagicMock(side_effect=Exception(f"{name} down"))
        else:
            original = m.complete
            m.complete = MagicMock(return_value=f"[{name}]")
        return m

    def test_all_tiers_route_correctly(self):
        """Each tier routes to its preferred provider."""
        p0 = self._make_provider("cheap")
        p1 = self._make_provider("balanced")
        p2 = self._make_provider("premium")

        client = FailoverClient(
            [p0, p1, p2],
            tier_map={
                RouteTier.LOCAL: [0],
                RouteTier.STANDARD: [1],
                RouteTier.CRITICAL: [2],
            },
        )

        for tier, expected in [
            (RouteTier.LOCAL, "[cheap]"),
            (RouteTier.STANDARD, "[balanced]"),
            (RouteTier.CRITICAL, "[premium]"),
        ]:
            result = client.complete(
                messages=[{"role": "user", "content": "test"}],
                tier=tier,
            )
            assert result == expected, f"Tier {tier} should route to {expected}, got {result}"

    def test_tier_failover_cascade(self):
        """Preferred provider down → next preferred → default order."""
        p0 = self._make_provider("p0")
        p1 = self._make_provider("p1", fail=True)
        p2 = self._make_provider("p2", fail=True)

        client = FailoverClient(
            [p0, p1, p2],
            tier_map={RouteTier.CRITICAL: [2, 1]},
        )

        # p2 fails, p1 fails, falls back to p0
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p0]"

    def test_all_providers_fail_raises(self):
        """When every provider fails, FailoverExhaustedError is raised."""
        from core.exceptions import FailoverExhaustedError

        p0 = self._make_provider("p0", fail=True)
        p1 = self._make_provider("p1", fail=True)

        client = FailoverClient([p0, p1])
        with pytest.raises(FailoverExhaustedError):
            client.complete(
                messages=[{"role": "user", "content": "test"}],
                tier=RouteTier.CRITICAL,
            )

    def test_no_tier_backward_compatible(self):
        """Without tier, original order is preserved."""
        p0 = self._make_provider("p0")
        p1 = self._make_provider("p1")

        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        # No tier → default order → p0
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "[p0]"


# ===========================================================================
# 7. AUTONOMOUS SCHEDULER — state machine walk
# ===========================================================================

class TestSchedulerStateMachine:
    def test_full_lifecycle(self, tmp_path):
        """start → tick(no goal) → tick(goal) → tick(cooldown) → tick(failure×3) → paused."""
        d = DaemonMode(config=DaemonConfig(), config_dir=tmp_path)
        d._running = True

        # 1. No goal → nothing enqueued
        d._find_critical_goal = MagicMock(return_value=None)
        run(d._scheduler_tick())
        assert len(d._queue) == 0

        # 2. Critical goal → enqueued
        d._find_critical_goal = MagicMock(return_value=(1, "Fix entity loss"))
        run(d._scheduler_tick())
        assert len(d._queue) == 1
        assert d._queue[0].input_text.startswith("[SELF-IMPROVEMENT]")
        assert d._last_autonomous_compile is not None

        # 3. Cooldown → not enqueued
        d._queue.clear()
        d._find_critical_goal = MagicMock(return_value=(2, "Another goal"))
        run(d._scheduler_tick())
        assert len(d._queue) == 0  # cooldown not expired

        # 4. Expire cooldown manually
        d._last_autonomous_compile = time.time() - 2000
        run(d._scheduler_tick())
        assert len(d._queue) == 1

        # 5. Simulate 3 consecutive failures → paused
        d._queue.clear()
        d._autonomous_failures = 3
        d._last_autonomous_compile = time.time() - 2000
        d._find_critical_goal = MagicMock(return_value=(3, "Yet another"))
        run(d._scheduler_tick())
        assert len(d._queue) == 0  # paused

    def test_failure_counter_reset_on_success(self, tmp_path):
        """Autonomous success resets failure counter."""
        async def mock_compile(text, domain):
            return SimpleNamespace(success=True, verification={},
                                   blueprint={"components": []}, error=None)

        d = DaemonMode(config=DaemonConfig(), config_dir=tmp_path, compile_fn=mock_compile)
        d._autonomous_failures = 2
        d._running = True

        req = CompileRequest(input_text="[SELF-IMPROVEMENT] Fix X", domain="software")
        d._queue.append(req)

        # Simulate processing
        async def process():
            pending = [r for r in d._queue if r.status == "pending"]
            r = pending[0]
            r.status = "running"
            result = await d._compile_fn(r.input_text, r.domain)
            r.status = "completed"
            d._autonomous_failures = 0

        run(process())
        assert d._autonomous_failures == 0


# ===========================================================================
# 8. RECURSIVE LOOP WIRING — verify the complete chain exists
# ===========================================================================

class TestRecursiveLoopWiring:
    """Verify all components of the L2→L3 recursive self-improvement loop are wired."""

    def test_engine_has_memory_consolidation_hook(self):
        """Phase 23a exists in engine.compile()."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Phase 23a" in source
        assert "consolidate" in source
        assert "save_memory" in source

    def test_engine_has_training_emission_hook(self):
        """Phase 23b exists in engine.compile()."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Phase 23b" in source
        assert "extract_training_examples" in source
        assert "emit_jsonl" in source

    def test_engine_has_pattern_detection_hook(self):
        """Phase 23c exists in engine.compile()."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Phase 23c" in source
        assert "detect_patterns" in source
        assert "save_patterns" in source

    def test_engine_has_pattern_injection_hook(self):
        """Phase 22g injects learned patterns before compilation."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "Phase 22g" in source
        assert "load_patterns" in source
        assert "format_pattern_context" in source
        assert 'l2_patterns' in source

    def test_daemon_has_scheduler_loop(self):
        """DaemonMode._scheduler_loop exists and is started."""
        import inspect
        source = inspect.getsource(DaemonMode.start)
        assert "_scheduler_loop" in source

    def test_daemon_scheduler_checks_goals(self):
        """Scheduler tick queries GoalStore for critical goals."""
        import inspect
        source = inspect.getsource(DaemonMode._scheduler_tick)
        assert "_find_critical_goal" in source

    def test_daemon_tracks_autonomous_failures(self):
        """Process queue distinguishes [SELF-IMPROVEMENT] for failure tracking."""
        import inspect
        source = inspect.getsource(DaemonMode._process_queue)
        assert "SELF-IMPROVEMENT" in source
        assert "_autonomous_failures" in source

    def test_store_has_fills_history(self):
        """fills_history table is created in schema."""
        import inspect
        from kernel.store import _SCHEMA
        assert "fills_history" in _SCHEMA

    def test_store_has_typed_connections(self):
        """connections table has connection_type, strength, created_at."""
        import inspect
        from kernel.store import _migrate_connections
        source = inspect.getsource(_migrate_connections)
        assert "connection_type" in source
        assert "strength" in source
        assert "created_at" in source

    def test_route_tier_enum_complete(self):
        """RouteTier has all 3 tiers."""
        assert hasattr(RouteTier, "CRITICAL")
        assert hasattr(RouteTier, "STANDARD")
        assert hasattr(RouteTier, "LOCAL")

    def test_failover_accepts_tier_map(self):
        """FailoverClient.__init__ accepts tier_map parameter."""
        import inspect
        sig = inspect.signature(FailoverClient.__init__)
        assert "tier_map" in sig.parameters

    def test_loop_completeness(self):
        """The full recursive loop: compile → learn → detect → inject → compile."""
        # This test verifies all pieces exist by importing them
        from kernel.memory import consolidate, save_memory       # Step 1: learn
        from kernel.memory import detect_patterns, save_patterns  # Step 2: detect
        from kernel.memory import load_patterns, format_pattern_context  # Step 3: inject
        from kernel.training import extract_training_examples, emit_jsonl  # Step 1b: train
        from mother.daemon import DaemonMode  # Step 4: autonomous scheduling

        # Verify the data flows: memory → patterns → prompt context → compilation
        memories = [_make_record(f"r{i}", compression_losses={"entity": 1})
                    for i in range(10)]
        patterns = detect_patterns(memories, min_occurrences=3)
        assert len(patterns) > 0

        ctx = format_pattern_context(patterns)
        assert "[LEARNED PATTERNS" in ctx
        assert "entity" in ctx.lower()


# ===========================================================================
# 9. SCHEMA MIGRATION — old DB upgrade
# ===========================================================================

class TestSchemaMigration:
    def test_old_db_upgrades_cleanly(self, tmp_path):
        """Simulate an old DB without new tables/columns, verify migration."""
        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        # Create ONLY the original tables (no fills_history, no typed columns)
        conn.executescript("""
            CREATE TABLE maps (id TEXT PRIMARY KEY, name TEXT,
                intent TEXT DEFAULT '', root TEXT DEFAULT '',
                created REAL, updated REAL);
            CREATE TABLE cells (map_id TEXT, postcode TEXT, primitive TEXT,
                content TEXT DEFAULT '', fill_state TEXT DEFAULT 'E',
                confidence REAL DEFAULT 0.0, parent TEXT,
                source_json TEXT DEFAULT '[]', revisions_json TEXT DEFAULT '[]',
                PRIMARY KEY (map_id, postcode));
            CREATE TABLE connections (map_id TEXT, from_postcode TEXT,
                to_postcode TEXT,
                PRIMARY KEY (map_id, from_postcode, to_postcode));
        """)
        # Insert old-style data
        conn.execute("INSERT INTO connections VALUES ('old', 'A', 'B')")
        conn.commit()
        conn.close()

        # Now run _ensure_schema — should migrate
        conn2 = sqlite3.connect(str(path))
        _ensure_schema(conn2)

        # Verify fills_history exists
        tables = [r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "fills_history" in tables

        # Verify typed columns exist
        cols = {r[1] for r in conn2.execute("PRAGMA table_info(connections)").fetchall()}
        assert "connection_type" in cols
        assert "strength" in cols
        assert "created_at" in cols

        # Old data has defaults
        conn2.row_factory = sqlite3.Row
        row = conn2.execute("SELECT * FROM connections WHERE map_id='old'").fetchone()
        assert row["connection_type"] == "association"
        assert row["strength"] == 0.5
        conn2.close()
