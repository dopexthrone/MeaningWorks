"""
Phase 17.2: Blueprint Health Gate Tests.

Tests for:
- check_blueprint_health() — all check conditions
- check_input_size() — truncation guard
- HealthReport properties
- Leaf module constraint (no engine/protocol imports)
"""

import pytest
from core.blueprint_health import check_blueprint_health, check_input_size, HealthReport


# =============================================================================
# Helpers
# =============================================================================

def _bp(components=None, relationships=None, constraints=None):
    """Build a minimal blueprint dict."""
    return {
        "components": components or [],
        "relationships": relationships or [],
        "constraints": constraints or [],
    }


def _comp(name, comp_type="entity", description="test"):
    return {"name": name, "type": comp_type, "description": description}


def _rel(from_c, to_c, rel_type="depends_on"):
    return {"from": from_c, "to": to_c, "type": rel_type}


# =============================================================================
# check_blueprint_health
# =============================================================================

class TestCheckBlueprintHealth:
    def test_empty_blueprint_unhealthy(self):
        report = check_blueprint_health(_bp())
        assert report.healthy is False
        assert any("0 components" in e for e in report.errors)

    def test_single_component_healthy(self):
        report = check_blueprint_health(_bp([_comp("UserService")]))
        assert report.healthy is True
        assert report.errors == ()

    def test_unnamed_component_error(self):
        report = check_blueprint_health(_bp([_comp("")]))
        assert report.healthy is False
        assert any("no name" in e for e in report.errors)

    def test_whitespace_only_name_is_unnamed(self):
        report = check_blueprint_health(_bp([_comp("  ")]))
        assert report.healthy is False
        assert any("no name" in e for e in report.errors)

    def test_case_insensitive_collision(self):
        """'UserAuth' and 'User Auth' collide (stripped of spaces, case-insensitive)."""
        report = check_blueprint_health(_bp([
            _comp("UserAuth"),
            _comp("User Auth"),
        ]))
        assert report.healthy is False
        assert any("collision" in e.lower() for e in report.errors)

    def test_case_sensitive_distinct_no_collision(self):
        """'UserAuth' and 'OrderService' are distinct."""
        report = check_blueprint_health(_bp([
            _comp("UserAuth"),
            _comp("OrderService"),
        ]))
        assert report.healthy is True
        assert report.stats["collision_count"] == 0

    def test_orphan_ratio_warning(self):
        """3 of 4 components are orphans (ratio=0.75) -> warning."""
        report = check_blueprint_health(_bp(
            components=[_comp("A"), _comp("B"), _comp("C"), _comp("D")],
            relationships=[_rel("A", "B")],
        ))
        assert report.healthy is True  # warnings don't make unhealthy
        assert any("orphan" in w.lower() for w in report.warnings)

    def test_no_orphan_warning_when_all_connected(self):
        report = check_blueprint_health(_bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B")],
        ))
        assert not any("orphan" in w.lower() for w in report.warnings)

    def test_component_count_over_50_warning(self):
        comps = [_comp(f"C{i}") for i in range(55)]
        report = check_blueprint_health(_bp(comps))
        assert report.healthy is True
        assert any("55 components" in w for w in report.warnings)

    def test_component_count_over_100_error(self):
        comps = [_comp(f"C{i}") for i in range(105)]
        report = check_blueprint_health(_bp(comps))
        assert report.healthy is False
        assert any("105" in e for e in report.errors)

    def test_dangling_relationship_warning(self):
        """Relationship references a component not in the blueprint."""
        report = check_blueprint_health(_bp(
            components=[_comp("A")],
            relationships=[_rel("A", "NonExistent")],
        ))
        assert any("dangling" in w.lower() for w in report.warnings)

    def test_no_dangling_when_all_refs_valid(self):
        report = check_blueprint_health(_bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B")],
        ))
        assert not any("dangling" in w.lower() for w in report.warnings)

    def test_score_1_for_healthy_blueprint(self):
        report = check_blueprint_health(_bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B")],
        ))
        assert report.score == 1.0

    def test_score_drops_with_errors(self):
        report = check_blueprint_health(_bp())
        assert report.score < 1.0

    def test_healthy_is_true_iff_no_errors(self):
        good = check_blueprint_health(_bp([_comp("X")]))
        assert good.healthy is True
        assert good.errors == ()

        bad = check_blueprint_health(_bp())
        assert bad.healthy is False
        assert len(bad.errors) > 0

    def test_stats_populated(self):
        report = check_blueprint_health(_bp([_comp("A")]))
        assert "component_count" in report.stats
        assert report.stats["component_count"] == 1


# =============================================================================
# check_input_size
# =============================================================================

class TestCheckInputSize:
    def test_small_input_ok(self):
        text = " ".join(["word"] * 5000)
        ok, result = check_input_size(text)
        assert ok is True
        assert result == text

    def test_large_input_truncated(self):
        text = " ".join(["word"] * 15000)
        ok, result = check_input_size(text)
        assert ok is False
        assert len(result.split()) == 10000

    def test_exact_limit_ok(self):
        text = " ".join(["word"] * 10000)
        ok, result = check_input_size(text)
        assert ok is True

    def test_empty_input_ok(self):
        ok, result = check_input_size("")
        assert ok is True
        assert result == ""

    def test_custom_max_words(self):
        text = " ".join(["word"] * 200)
        ok, result = check_input_size(text, max_words=100)
        assert ok is False
        assert len(result.split()) == 100


# =============================================================================
# HealthReport frozen dataclass
# =============================================================================

class TestHealthReport:
    def test_frozen(self):
        report = HealthReport(
            healthy=True, score=1.0, errors=(), warnings=(), stats={}
        )
        with pytest.raises(AttributeError):
            report.healthy = False

    def test_fields(self):
        report = HealthReport(
            healthy=False, score=0.5,
            errors=("no components",),
            warnings=("large",),
            stats={"component_count": 0},
        )
        assert report.healthy is False
        assert report.score == 0.5
        assert len(report.errors) == 1
        assert len(report.warnings) == 1


# =============================================================================
# Leaf module constraint
# =============================================================================

class TestLeafModule:
    def test_no_engine_import(self):
        import core.blueprint_health as mod
        source = open(mod.__file__).read()
        assert "from core.engine" not in source
        assert "from core.protocol" not in source
        assert "from core.pipeline" not in source
