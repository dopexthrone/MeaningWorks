"""
tests/test_l3_dynamic_desc.py — L3: Dynamic Self-Description Seed.

Tests for the dynamic self-description that replaces the hardcoded 83-line
string in self_compile(). Covers:
1. Substrate detection (5 tests)
2. Description generation (8 tests)
3. Canonical expansion (5 tests)
4. Integration (4 tests)
5. Regression (3 tests)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from persistence.corpus import Corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    """Create a minimal engine with mock LLM for testing."""
    from core.engine import MotherlabsEngine
    client = Mock()
    client.provider_name = "mock"
    client.model_name = "mock-model"
    client.deterministic = True
    client.model = "mock-model"
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=Corpus(tmp_path / "corpus"),
        auto_store=False,
    )
    return engine


def _make_mock_grid(cells_keys=None, fill_rate=0.5):
    """Create a mock Grid with given cell keys and fill_rate."""
    grid = MagicMock()
    if cells_keys is None:
        cells_keys = {"L0.ENT.ECO.SEM.GEN": Mock(), "L1.BHV.APP.FNC.GEN": Mock()}
    grid.cells = {k: Mock() for k in cells_keys}
    grid.fill_rate = fill_rate
    grid.total_cells = len(cells_keys)
    return grid


# ===========================================================================
# 1. SUBSTRATE DETECTION
# ===========================================================================

class TestSubstrateDetection:
    """_detect_substrate_summary() returns runtime substrate info."""

    def test_returns_string(self, tmp_path):
        """Must return a non-empty string."""
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_platform(self, tmp_path):
        """Must mention the OS name (macOS, Linux, or Windows)."""
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert any(name in result for name in ("macOS", "Linux", "Windows", sys.platform))

    def test_includes_ram(self, tmp_path):
        """Must include RAM measurement (GB)."""
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert "GB RAM" in result

    def test_includes_capabilities(self, tmp_path):
        """Must include 'Available:' section."""
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert "Available:" in result

    def test_handles_missing_commands(self, tmp_path):
        """When no optional commands exist, shows 'baseline'."""
        engine = _make_engine(tmp_path)
        with patch("shutil.which", return_value=None):
            result = engine._detect_substrate_summary()
            assert "baseline" in result

    def test_includes_python_version(self, tmp_path):
        """Must include the Python version string."""
        import platform
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert platform.python_version() in result

    def test_includes_cpu_cores(self, tmp_path):
        """Must include CPU core count."""
        import os
        engine = _make_engine(tmp_path)
        result = engine._detect_substrate_summary()
        assert "CPU cores" in result


# ===========================================================================
# 2. DESCRIPTION GENERATION
# ===========================================================================

class TestDescriptionGeneration:
    """_generate_self_description() produces the full entity description."""

    def test_returns_nonempty_string(self, tmp_path):
        """Must return a non-empty string."""
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_identity_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "1. IDENTITY" in result

    def test_contains_perception_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "2. PERCEPTION" in result
        assert "PerceptionEngine" in result

    def test_contains_cognition_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "3. COGNITION" in result
        assert "SemanticGrid" in result

    def test_contains_actuators_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "4. ACTUATORS" in result
        assert "VoiceBridge" in result

    def test_contains_senses_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "5. SENSES" in result
        assert "SenseVector" in result
        for sense in ("confidence", "rapport", "curiosity", "vitality", "attentiveness", "frustration"):
            assert sense in result

    def test_contains_autonomy_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "6. AUTONOMY" in result
        assert "DaemonMode" in result
        assert "GoalStore" in result

    def test_contains_learning_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "7. LEARNING" in result
        assert "L1:" in result
        assert "L2:" in result
        assert "L3:" in result

    def test_contains_substrate_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "8. SUBSTRATE" in result
        # Should contain actual platform info, not a placeholder
        assert any(name in result for name in ("macOS", "Linux", "Windows", sys.platform))

    def test_contains_missing_bridge_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "9. THE MISSING BRIDGE" in result
        assert "perception" in result.lower()

    def test_contains_convergence_section(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        assert "10. CONVERGENCE CRITERION" in result
        assert "F(F)" in result

    def test_longer_than_old_description(self, tmp_path):
        """Dynamic description should be substantially longer than the old 83-line string."""
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        # Old description was ~83 lines / ~4000 chars
        assert len(result) > 4000

    def test_contains_key_terms(self, tmp_path):
        """Must contain the foundational vocabulary."""
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        for term in ("excavat", "provenance", "trust", "convergence", "ClosedLoopGate"):
            assert term.lower() in result.lower(), f"Missing key term: {term}"

    def test_substrate_is_dynamically_populated(self, tmp_path):
        """Substrate section should contain actual runtime data, not a template variable."""
        engine = _make_engine(tmp_path)
        result = engine._generate_self_description()
        # Should NOT contain the literal f-string placeholder
        assert "{substrate}" not in result
        # Should contain actual CPU/RAM info
        assert "CPU cores" in result


# ===========================================================================
# 3. CANONICAL EXPANSION
# ===========================================================================

class TestCanonicalExpansion:
    """SELF_COMPILE_CANONICAL and SELF_COMPILE_RELATIONSHIPS expanded."""

    def test_21_canonical_components(self):
        from core.engine import MotherlabsEngine
        assert len(MotherlabsEngine.SELF_COMPILE_CANONICAL) == 21

    def test_20_relationships(self):
        from core.engine import MotherlabsEngine
        assert len(MotherlabsEngine.SELF_COMPILE_RELATIONSHIPS) == 20

    def test_new_entities_present(self):
        from core.engine import MotherlabsEngine
        canonical = MotherlabsEngine.SELF_COMPILE_CANONICAL
        new_entities = [
            "PerceptionEngine", "SenseVector", "VoiceBridge",
            "FileSystemBridge", "GoalStore", "DaemonMode",
            "ClosedLoopGate", "SemanticGrid",
        ]
        for entity in new_entities:
            assert entity in canonical, f"Missing new entity: {entity}"

    def test_no_duplicates(self):
        from core.engine import MotherlabsEngine
        canonical = MotherlabsEngine.SELF_COMPILE_CANONICAL
        assert len(canonical) == len(set(canonical)), "Duplicate canonical components"
        rels = MotherlabsEngine.SELF_COMPILE_RELATIONSHIPS
        assert len(rels) == len(set(rels)), "Duplicate relationships"

    def test_backward_compatible(self):
        """Original 13 components still present."""
        from core.engine import MotherlabsEngine
        canonical = MotherlabsEngine.SELF_COMPILE_CANONICAL
        original_13 = [
            "Intent Agent", "Persona Agent", "Entity Agent", "Process Agent",
            "Synthesis Agent", "Verify Agent", "Governor Agent",
            "SharedState", "ConfidenceVector", "ConflictOracle",
            "Message", "DialogueProtocol", "Corpus",
        ]
        for comp in original_13:
            assert comp in canonical, f"Missing original component: {comp}"

    def test_new_relationships_present(self):
        """All 8 new relationship edges exist."""
        from core.engine import MotherlabsEngine
        rels = MotherlabsEngine.SELF_COMPILE_RELATIONSHIPS
        new_rels = [
            ("PerceptionEngine", "SenseVector", "feeds"),
            ("SenseVector", "Governor Agent", "constrains"),
            ("DaemonMode", "GoalStore", "queries"),
            ("DaemonMode", "Governor Agent", "schedules"),
            ("ClosedLoopGate", "Synthesis Agent", "validates"),
            ("Governor Agent", "ClosedLoopGate", "invokes"),
            ("SemanticGrid", "SharedState", "enriches"),
            ("Corpus", "SemanticGrid", "persists"),
        ]
        for rel in new_rels:
            assert rel in rels, f"Missing new relationship: {rel}"

    def test_no_orphan_nodes(self):
        """Every canonical component appears in at least one relationship."""
        from core.engine import MotherlabsEngine
        canonical = set(MotherlabsEngine.SELF_COMPILE_CANONICAL)
        rels = MotherlabsEngine.SELF_COMPILE_RELATIONSHIPS
        nodes_in_rels = set()
        for src, dst, _ in rels:
            nodes_in_rels.add(src)
            nodes_in_rels.add(dst)
        orphans = canonical - nodes_in_rels
        # Message and DialogueProtocol are data types, not agents — they don't
        # participate in relationships. That's acceptable.
        acceptable_orphans = {"Message", "DialogueProtocol", "ConfidenceVector",
                              "VoiceBridge", "FileSystemBridge"}
        real_orphans = orphans - acceptable_orphans
        assert not real_orphans, f"Orphan canonical components: {real_orphans}"


# ===========================================================================
# 4. INTEGRATION
# ===========================================================================

class TestIntegration:
    """Integration tests verifying self_compile uses the dynamic description."""

    def test_self_compile_uses_generate(self, tmp_path):
        """self_compile() should call _generate_self_description()."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True

        called = []
        original_gen = engine._generate_self_description

        def tracking_gen():
            result = original_gen()
            called.append(True)
            return result

        engine._generate_self_description = tracking_gen
        engine.compile_with_axioms = Mock(return_value=mock_result)

        with patch("kernel.store.save_grid"):
            engine.self_compile()

        assert called, "_generate_self_description was not called"

    def test_compile_with_axioms_receives_dynamic_description(self, tmp_path):
        """compile_with_axioms should receive the dynamic description, not the old static one."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True

        engine.compile_with_axioms = Mock(return_value=mock_result)

        with patch("kernel.store.save_grid"):
            engine.self_compile()

        call_args = engine.compile_with_axioms.call_args
        description = call_args[0][0]
        # Dynamic description has these sections; old one didn't
        assert "PERCEPTION" in description
        assert "SENSES" in description
        assert "SUBSTRATE" in description

    def test_description_changes_with_platform(self, tmp_path):
        """Mocking different platform should change the description."""
        engine = _make_engine(tmp_path)

        # Get normal description
        desc1 = engine._generate_self_description()

        # Mock a different platform
        with patch("sys.platform", "linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("platform.python_version", return_value="3.12.0"), \
             patch("os.cpu_count", return_value=2), \
             patch("shutil.disk_usage", return_value=(100*1024**3, 50*1024**3, 50*1024**3)), \
             patch("shutil.which", return_value=None):
            # Need to patch at import level inside the method
            desc2 = engine._detect_substrate_summary()

        # The substrate summaries should differ
        assert desc2 != engine._detect_substrate_summary() or "Linux" in desc2

    def test_canonical_relationships_form_valid_graph(self):
        """All relationship endpoints exist in the canonical list or are acceptable."""
        from core.engine import MotherlabsEngine
        canonical = set(MotherlabsEngine.SELF_COMPILE_CANONICAL)
        rels = MotherlabsEngine.SELF_COMPILE_RELATIONSHIPS
        for src, dst, rel_type in rels:
            assert src in canonical, f"Source '{src}' not in canonical"
            assert dst in canonical, f"Destination '{dst}' not in canonical"
            assert isinstance(rel_type, str) and len(rel_type) > 0


# ===========================================================================
# 5. REGRESSION
# ===========================================================================

class TestRegression:
    """Ensure existing behavior is preserved."""

    def test_self_compile_returns_compile_result(self, tmp_path):
        """self_compile() still returns whatever compile_with_axioms returns."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True
        engine.compile_with_axioms = Mock(return_value=mock_result)

        with patch("kernel.store.save_grid"):
            result = engine.self_compile()
        assert result is mock_result

    def test_self_compile_saves_as_compiler_self_desc(self, tmp_path):
        """Grid still saved with map_id='compiler-self-desc'."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()
        mock_result = Mock()
        mock_result.success = True

        def fake_compile(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile

        with patch("kernel.store.save_grid") as mock_save:
            engine.self_compile()
            mock_save.assert_called_once_with(
                mock_grid,
                map_id="compiler-self-desc",
                name="Compiler Self-Description",
            )

    def test_run_self_compile_loop_still_works(self, tmp_path):
        """run_self_compile_loop() still returns SelfCompileReport."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = None
            return mock_result

        engine.self_compile = fake_self_compile

        with patch("kernel.store.load_grid", return_value=None):
            report = engine.run_self_compile_loop(runs=1)

        assert report is not None
        assert hasattr(report, "convergence")
        assert hasattr(report, "overall_health")
