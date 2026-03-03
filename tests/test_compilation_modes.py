"""Tests for core/compilation_modes.py — CompilationMode enum + ModeConfig."""

import pytest
from core.compilation_modes import (
    CompilationMode,
    ModeConfig,
    mode_config,
    parse_mode,
)


# ============================================================
# CompilationMode enum
# ============================================================

class TestCompilationMode:
    """Enum membership and value tests."""

    def test_build_value(self):
        assert CompilationMode.BUILD.value == "build"

    def test_context_value(self):
        assert CompilationMode.CONTEXT.value == "context"

    def test_explore_value(self):
        assert CompilationMode.EXPLORE.value == "explore"

    def test_self_value(self):
        assert CompilationMode.SELF.value == "self"

    def test_four_members(self):
        assert len(CompilationMode) == 4

    def test_members_unique(self):
        values = [m.value for m in CompilationMode]
        assert len(values) == len(set(values))

    def test_iteration_order(self):
        names = [m.name for m in CompilationMode]
        assert names == ["BUILD", "CONTEXT", "EXPLORE", "SELF"]

    def test_from_value(self):
        assert CompilationMode("build") is CompilationMode.BUILD

    def test_from_invalid_value(self):
        with pytest.raises(ValueError):
            CompilationMode("invalid")

    def test_equality(self):
        assert CompilationMode.BUILD == CompilationMode.BUILD
        assert CompilationMode.BUILD != CompilationMode.CONTEXT

    def test_hashable(self):
        d = {CompilationMode.BUILD: 1, CompilationMode.CONTEXT: 2}
        assert d[CompilationMode.BUILD] == 1

    def test_string_representation(self):
        assert "BUILD" in repr(CompilationMode.BUILD)


# ============================================================
# ModeConfig frozen dataclass
# ============================================================

class TestModeConfig:
    """ModeConfig structural constraints."""

    def test_frozen(self):
        cfg = mode_config(CompilationMode.BUILD)
        with pytest.raises(AttributeError):
            cfg.agent_posture = "something_else"

    def test_all_fields_present(self):
        """Every ModeConfig has all required fields."""
        expected_fields = {
            "agent_posture", "dialogue_max_turns", "convergence_threshold",
            "skip_synthesis", "skip_verification", "verification_label",
            "actionability_checks", "grid_domain_hint",
            "persist_to_world_grid", "persist_to_memory", "posture_preamble",
        }
        for mode in CompilationMode:
            cfg = mode_config(mode)
            actual = set(cfg.__dataclass_fields__.keys())
            assert expected_fields == actual, f"{mode.name} has wrong fields: {actual}"

    def test_every_mode_has_config(self):
        for mode in CompilationMode:
            cfg = mode_config(mode)
            assert isinstance(cfg, ModeConfig)


# ============================================================
# BUILD mode config
# ============================================================

class TestBuildConfig:
    """BUILD mode: current default behavior."""

    @pytest.fixture
    def cfg(self):
        return mode_config(CompilationMode.BUILD)

    def test_posture(self, cfg):
        assert cfg.agent_posture == "converge"

    def test_max_turns(self, cfg):
        assert cfg.dialogue_max_turns == 64

    def test_convergence_threshold(self, cfg):
        assert cfg.convergence_threshold == 0.80

    def test_no_skip_synthesis(self, cfg):
        assert cfg.skip_synthesis is False

    def test_no_skip_verification(self, cfg):
        assert cfg.skip_verification is False

    def test_verification_label(self, cfg):
        assert cfg.verification_label == "codegen_readiness"

    def test_actionability_checks(self, cfg):
        assert cfg.actionability_checks == ("methods",)

    def test_grid_domain_hint(self, cfg):
        assert cfg.grid_domain_hint == "SFT"

    def test_no_persist_world_grid(self, cfg):
        assert cfg.persist_to_world_grid is False

    def test_no_persist_memory(self, cfg):
        assert cfg.persist_to_memory is False

    def test_empty_preamble(self, cfg):
        assert cfg.posture_preamble == ""


# ============================================================
# CONTEXT mode config
# ============================================================

class TestContextConfig:
    """CONTEXT mode: deep understanding."""

    @pytest.fixture
    def cfg(self):
        return mode_config(CompilationMode.CONTEXT)

    def test_posture(self, cfg):
        assert cfg.agent_posture == "excavate"

    def test_max_turns(self, cfg):
        assert cfg.dialogue_max_turns == 32

    def test_convergence_threshold(self, cfg):
        assert cfg.convergence_threshold == 0.70

    def test_no_skip_synthesis(self, cfg):
        assert cfg.skip_synthesis is False

    def test_no_skip_verification(self, cfg):
        assert cfg.skip_verification is False

    def test_verification_label(self, cfg):
        assert cfg.verification_label == "analytical_completeness"

    def test_actionability_checks_context(self, cfg):
        assert cfg.actionability_checks == ("description", "relationships")

    def test_grid_domain_hint(self, cfg):
        assert cfg.grid_domain_hint == "ORG"

    def test_persist_world_grid(self, cfg):
        assert cfg.persist_to_world_grid is True

    def test_persist_memory(self, cfg):
        assert cfg.persist_to_memory is True

    def test_preamble_has_context_directive(self, cfg):
        assert "CONTEXT UNDERSTANDING" in cfg.posture_preamble

    def test_preamble_no_components(self, cfg):
        assert "Do NOT propose components" in cfg.posture_preamble


# ============================================================
# EXPLORE mode config
# ============================================================

class TestExploreConfig:
    """EXPLORE mode: divergent exploration."""

    @pytest.fixture
    def cfg(self):
        return mode_config(CompilationMode.EXPLORE)

    def test_posture(self, cfg):
        assert cfg.agent_posture == "diverge"

    def test_max_turns(self, cfg):
        assert cfg.dialogue_max_turns == 48

    def test_convergence_threshold(self, cfg):
        assert cfg.convergence_threshold == 0.50

    def test_skip_synthesis(self, cfg):
        assert cfg.skip_synthesis is True

    def test_skip_verification(self, cfg):
        assert cfg.skip_verification is True

    def test_no_actionability_checks(self, cfg):
        assert cfg.actionability_checks == ()

    def test_grid_domain_hint(self, cfg):
        assert cfg.grid_domain_hint == "ORG"

    def test_no_persist(self, cfg):
        assert cfg.persist_to_world_grid is False
        assert cfg.persist_to_memory is False

    def test_preamble_has_explore_directive(self, cfg):
        assert "DIVERGENT EXPLORATION" in cfg.posture_preamble

    def test_preamble_no_converge(self, cfg):
        assert "Do NOT converge" in cfg.posture_preamble


# ============================================================
# SELF mode config
# ============================================================

class TestSelfConfig:
    """SELF mode: compiler self-audit."""

    @pytest.fixture
    def cfg(self):
        return mode_config(CompilationMode.SELF)

    def test_posture(self, cfg):
        assert cfg.agent_posture == "audit"

    def test_max_turns(self, cfg):
        assert cfg.dialogue_max_turns == 16

    def test_convergence_threshold(self, cfg):
        assert cfg.convergence_threshold == 0.90

    def test_no_skip_synthesis(self, cfg):
        assert cfg.skip_synthesis is False

    def test_no_skip_verification(self, cfg):
        assert cfg.skip_verification is False

    def test_persist_world_grid(self, cfg):
        assert cfg.persist_to_world_grid is True

    def test_no_persist_memory(self, cfg):
        assert cfg.persist_to_memory is False

    def test_preamble_has_audit_directive(self, cfg):
        assert "SELF-AUDIT" in cfg.posture_preamble


# ============================================================
# mode_config() factory
# ============================================================

class TestModeConfigFactory:
    """mode_config() factory function."""

    def test_returns_modeconfig(self):
        for mode in CompilationMode:
            assert isinstance(mode_config(mode), ModeConfig)

    def test_same_instance_on_repeat(self):
        """Config objects are singletons — same reference."""
        a = mode_config(CompilationMode.BUILD)
        b = mode_config(CompilationMode.BUILD)
        assert a is b

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Expected CompilationMode"):
            mode_config("build")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            mode_config(None)


# ============================================================
# parse_mode()
# ============================================================

class TestParseMode:
    """parse_mode() string → enum conversion."""

    def test_build(self):
        assert parse_mode("build") is CompilationMode.BUILD

    def test_context(self):
        assert parse_mode("context") is CompilationMode.CONTEXT

    def test_explore(self):
        assert parse_mode("explore") is CompilationMode.EXPLORE

    def test_self(self):
        assert parse_mode("self") is CompilationMode.SELF

    def test_case_insensitive(self):
        assert parse_mode("BUILD") is CompilationMode.BUILD
        assert parse_mode("Context") is CompilationMode.CONTEXT
        assert parse_mode("EXPLORE") is CompilationMode.EXPLORE

    def test_whitespace_stripped(self):
        assert parse_mode("  build  ") is CompilationMode.BUILD

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown compilation mode"):
            parse_mode("invalid")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_mode("")

    def test_error_lists_valid(self):
        with pytest.raises(ValueError, match="build"):
            parse_mode("wrong")


# ============================================================
# Cross-mode constraints
# ============================================================

class TestCrossModeConstraints:
    """Invariants that must hold across all modes."""

    def test_all_postures_unique(self):
        postures = [mode_config(m).agent_posture for m in CompilationMode]
        assert len(postures) == len(set(postures))

    def test_all_max_turns_positive(self):
        for mode in CompilationMode:
            assert mode_config(mode).dialogue_max_turns > 0

    def test_convergence_between_0_and_1(self):
        for mode in CompilationMode:
            t = mode_config(mode).convergence_threshold
            assert 0.0 < t <= 1.0, f"{mode.name}: threshold {t} out of range"

    def test_skip_synthesis_implies_skip_verification(self):
        """If synthesis is skipped, verification must also be skipped."""
        for mode in CompilationMode:
            cfg = mode_config(mode)
            if cfg.skip_synthesis:
                assert cfg.skip_verification, (
                    f"{mode.name}: skip_synthesis=True but skip_verification=False"
                )

    def test_build_is_default_behavior(self):
        """BUILD mode should not change any default pipeline behavior."""
        cfg = mode_config(CompilationMode.BUILD)
        assert cfg.posture_preamble == ""
        assert cfg.skip_synthesis is False
        assert cfg.skip_verification is False

    def test_verification_labels_all_strings(self):
        for mode in CompilationMode:
            label = mode_config(mode).verification_label
            assert isinstance(label, str) and len(label) > 0

    def test_grid_domain_hints_valid(self):
        """Grid domain hints should be recognized kernel layer prefixes."""
        valid_hints = {"SFT", "ORG", "DOM", "APP", "ECO", "CMP", "FET", "FNC"}
        for mode in CompilationMode:
            hint = mode_config(mode).grid_domain_hint
            assert hint in valid_hints, f"{mode.name}: invalid grid_domain_hint {hint!r}"

    def test_actionability_checks_are_tuples(self):
        for mode in CompilationMode:
            checks = mode_config(mode).actionability_checks
            assert isinstance(checks, tuple)
