import pytest
from datetime import datetime, timezone
from core.consistency import (
    ConsistencyDimension, ConsistencyMetrics, System, ConsistencyFlow, ConsistencyImprovement,
    ErrorSeverity, ThresholdLevel, ImpactLevel, ConsistencyError
)
from uuid import uuid4

INSTANCE_ID = "TASK: strengthen implementation and add test coverage for consistency dimension at 22.5% needs targeted improvement."

@pytest.fixture
def dimension():
    return ConsistencyDimension(INSTANCE_ID)

@pytest.fixture
def metrics(dimension):
    return ConsistencyMetrics(dimension)

@pytest.fixture
def system():
    return System(INSTANCE_ID)

@pytest.fixture
def flow(system):
    return ConsistencyFlow(system)

@pytest.fixture
def improvement(metrics):
    return ConsistencyImprovement(metrics)

class TestConsistencyDimension:
    def test_get_current_level(self, dimension):
        assert dimension.get_current_level() == 22.5

    def test_set_target_improvement(self, dimension):
        dimension.set_target_improvement(90.0)
        assert dimension.target_improvement == 90.0
        dimension.set_target_improvement(150.0)
        assert dimension.target_improvement == 100.0
        dimension.set_target_improvement(-10.0)
        assert dimension.target_improvement == 0.0

    def test_update_level(self, dimension):
        dimension.update_level(50.0)
        assert dimension.current_level == 50.0
        assert len(dimension.history) == 1
        assert dimension.history[0] == 50.0

    def test_get_progress(self, dimension):
        dimension.current_level = 40.0
        dimension.target_improvement = 50.0
        assert dimension.get_progress() == 80.0
        dimension.target_improvement = 0.0
        assert dimension.get_progress() == 100.0

    def test_reset_history(self, dimension):
        dimension.update_level(30.0)
        dimension.reset_history()
        assert len(dimension.history) == 0

    def test_initialize(self, dimension):
        dimension.initialize(10.0)
        assert dimension.current_level == 10.0
        assert len(dimension.history) == 1

class TestConsistencyMetrics:
    def test_track_error(self, metrics):
        error = metrics.track_error("syntax_mismatch", ErrorSeverity.HIGH, "Test error")
        assert isinstance(error, ConsistencyError)
        assert len(metrics.errors) == 1
        assert metrics.current_value < 22.5  # decreased

    def test_update_target(self, metrics):
        metrics.update_target(95.0)
        assert metrics.target_level == 95.0
        assert metrics.dimension.target_improvement == 95.0

    def test_check_threshold(self, metrics):
        assert metrics.check_threshold() == ThresholdLevel.WARNING  # 22.5 <30
        metrics.current_value = 35.0
        assert metrics.check_threshold() == ThresholdLevel.OK
        metrics.current_value = 15.0
        assert metrics.check_threshold() == ThresholdLevel.CRITICAL

    def test_compute_benchmark_score(self, metrics):
        score = metrics.compute_benchmark_score()
        assert 0.0 <= score <= 100.0

    def test_clear_errors(self, metrics):
        metrics.track_error("test", ErrorSeverity.LOW, "test")
        count = metrics.clear_errors()
        assert count == 1
        assert len(metrics.errors) == 0

    def test_get_error_summary(self, metrics):
        metrics.track_error("e1", ErrorSeverity.HIGH, "e1")
        metrics.track_error("e2", ErrorSeverity.LOW, "e2")
        summary = metrics.get_error_summary()
        assert summary["high"] == 1
        assert summary["low"] == 1

class TestSystem:
    def test_assess_user_impact(self, system):
        impact = system.assess_user_impact(85.0)
        assert impact["impact_level"] == "severe"
        assert isinstance(impact["affected_users"], int)
        assert len(impact["mitigation_steps"]) == 3

    def test_add_user_profile(self, system):
        user_id = str(uuid4())
        system.add_user_profile(user_id, ImpactLevel.HIGH)
        assert len(system.user_profiles) == 1

    def test_monitor_consistency(self, system):
        monitor = system.monitor_consistency()
        assert "current_level" in monitor
        assert "threshold_status" in monitor

    def test_improve_consistency(self, system):
        old = system.dimension.current_level
        new = system.improve_consistency(20.0)
        assert new > old
        assert new <= 100.0

    def test_get_user_impact_summary(self, system):
        system.add_user_profile(str(uuid4()), ImpactLevel.LOW)
        summary = system.get_user_impact_summary()
        assert "low" in summary

    def test_initialize(self, system):
        system.initialize()
        assert len(system.dimension.history) == 1

class TestConsistencyFlow:
    def test_synchronize(self, flow):
        nodes = ["node1", "node2"]
        success = flow.synchronize(nodes)
        assert success
        assert len(flow.nodes) == 2

    def test_detect_divergence(self, flow):
        flow.synchronize(["n1", "n2"])
        flow.state_versions["n1"] = 5
        divergent = flow.detect_divergence()
        assert len(divergent) == 1  # n2 at 1 vs 5

    def test_resolve_conflict(self, flow):
        conflicts = ["node1"]
        success = flow.resolve_conflict(conflicts)
        assert success

    def test_achieve_consensus(self, flow):
        flow.synchronize(["n1"])
        ratio = flow.achieve_consensus()
        assert ratio == 1.0

    def test_add_node(self, flow):
        flow.add_node("new_node")
        assert "new_node" in flow.nodes

    def test_check_consensus(self, flow):
        flow.synchronize(["n1"])
        assert flow.check_consensus()

class TestConsistencyImprovement:
    def test_monitor(self, improvement):
        mon = improvement.monitor()
        assert "level" in mon
        assert mon["status"] in ["ok", "warning", "critical"]

    def test_identify(self, improvement):
        issues = improvement.identify()
        assert isinstance(issues, list)

    def test_correct(self, improvement):
        success = improvement.correct("Low consistency level")
        assert success
        assert len(improvement.improvement_history) == 1

    def test_validate(self, improvement):
        improvement.metrics.current_value = 90.0
        assert improvement.validate()

    def test_run_full_cycle(self, improvement):
        cycle = improvement.run_full_cycle()
        assert "before" in cycle
        assert "after" in cycle

    def test_get_history_summary(self, improvement):
        improvement.correct("test")
        summary = improvement.get_history_summary()
        assert "total_improvements" in summary