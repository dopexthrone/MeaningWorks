"""Tests for expanded method inference in _infer_component_methods.

Validates:
- All 10 type categories produce appropriate methods
- Name-based fallback detects type from component name
- Catch-all produces methods for unknown types
- Gap-fill runs after dialogue backfill
- Components with existing methods are preserved
- _classify_component_type works for type and name matching
"""
from __future__ import annotations

import pytest

from unittest.mock import Mock

from core.engine import MotherlabsEngine
from core.llm import BaseLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock"
    return MotherlabsEngine(llm_client=client)


def _make_blueprint(*components):
    return {"components": list(components)}


def _make_component(name, comp_type="", methods=None):
    comp = {"name": name, "type": comp_type, "description": f"Test {name}"}
    if methods is not None:
        comp["methods"] = methods
    return comp


def _make_state(extracted_methods=None, extracted_state_machines=None):
    class FakeState:
        def __init__(self):
            self.known = {}
            if extracted_methods:
                self.known["extracted_methods"] = extracted_methods
            if extracted_state_machines:
                self.known["extracted_state_machines"] = extracted_state_machines
    return FakeState()


# ---------------------------------------------------------------------------
# Type category tests
# ---------------------------------------------------------------------------

class TestTypeCategoryInference:
    """Each type category produces appropriate methods."""

    @pytest.mark.parametrize("comp_type", [
        "entity", "data", "model", "store", "storage",
        "record", "document", "table", "schema", "resource",
    ])
    def test_entity_types_get_crud(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Item", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "create_item" in names
        assert "get_item" in names
        assert "update_item" in names
        assert "delete_item" in names

    @pytest.mark.parametrize("comp_type", [
        "process", "service", "handler", "workflow", "pipeline",
        "engine", "processor", "worker", "executor", "task",
    ])
    def test_process_types_get_lifecycle(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Runner", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "execute" in names
        assert "get_status" in names
        assert "validate" in names

    @pytest.mark.parametrize("comp_type", [
        "interface", "api", "controller", "gateway", "endpoint",
        "route", "view", "page", "screen", "form",
    ])
    def test_interface_types_get_handler(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Portal", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "handle_request" in names
        assert "validate_input" in names

    @pytest.mark.parametrize("comp_type", [
        "manager", "system", "module", "component", "subsystem",
        "platform", "application", "framework", "core", "hub",
    ])
    def test_manager_types_get_operations(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Central", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "initialize" in names
        assert "process" in names
        assert "configure" in names

    @pytest.mark.parametrize("comp_type", [
        "monitor", "tracker", "dashboard", "observer", "watcher",
        "meter", "profiler", "logger", "auditor", "analyzer",
    ])
    def test_monitor_types_get_observation(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Health", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "check" in names
        assert "report" in names
        assert "track" in names

    @pytest.mark.parametrize("comp_type", [
        "scheduler", "queue", "dispatcher", "planner", "cron",
        "timer", "trigger", "orchestrator",
    ])
    def test_scheduler_types_get_scheduling(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Jobs", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "schedule" in names
        assert "cancel" in names
        assert "get_next" in names

    @pytest.mark.parametrize("comp_type", [
        "registry", "repository", "catalog", "index", "cache",
        "pool", "collection", "directory", "lookup",
    ])
    def test_registry_types_get_registry(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Items", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "register" in names
        assert "lookup" in names
        assert "list_all" in names

    @pytest.mark.parametrize("comp_type", [
        "validator", "guard", "checker", "verifier", "filter",
        "sanitizer", "policy", "rule",
    ])
    def test_validator_types_get_validation(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Input", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "validate" in names
        assert "check_rules" in names

    @pytest.mark.parametrize("comp_type", [
        "notifier", "messenger", "mailer", "alerter", "broadcaster",
        "publisher", "emitter", "sender",
    ])
    def test_notifier_types_get_notification(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Alerts", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "send" in names
        assert "subscribe" in names

    @pytest.mark.parametrize("comp_type", [
        "connector", "adapter", "bridge", "integration", "client",
        "provider", "driver", "proxy", "wrapper",
    ])
    def test_connector_types_get_integration(self, comp_type):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("External", comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "connect" in names
        assert "transform" in names
        assert "sync" in names


class TestCatchAll:
    """Unknown types get default methods."""

    def test_unknown_type_gets_defaults(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Widget", "foobar"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "initialize" in names
        assert "process" in names
        assert len(methods) == 2

    def test_empty_type_gets_defaults(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("Thing", ""))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        assert len(methods) == 2


class TestNameBasedFallback:
    """Name-based detection when type doesn't match."""

    def test_name_contains_manager(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("ProjectManager", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "initialize" in names
        assert "configure" in names

    def test_name_contains_service(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("PaymentService", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "execute" in names

    def test_name_contains_tracker(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("ProgressTracker", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "track" in names

    def test_name_contains_scheduler(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("JobScheduler", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "schedule" in names

    def test_name_contains_repository(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("UserRepository", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "register" in names
        assert "lookup" in names

    def test_name_contains_validator(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("InputValidator", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "validate" in names

    def test_name_contains_notifier(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("EmailNotifier", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "send" in names

    def test_name_contains_adapter(self):
        engine = _make_engine()
        bp = _make_blueprint(_make_component("StripeAdapter", "unknown"))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert "connect" in names


class TestPreserveExisting:
    """Components with existing methods are not overwritten."""

    def test_existing_methods_preserved(self):
        engine = _make_engine()
        existing = [{"name": "custom_method", "parameters": [], "return_type": "str"}]
        bp = _make_blueprint(_make_component("Item", "entity", methods=existing))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        assert len(methods) == 1
        assert methods[0]["name"] == "custom_method"


class TestGapFill:
    """Gap-fill runs after dialogue backfill."""

    def test_gap_fill_after_dialogue_backfill(self):
        """Components not matched by dialogue methods still get inferred methods."""
        engine = _make_engine()
        bp = _make_blueprint(
            _make_component("TaskManager", "entity"),
            _make_component("NotificationService", "service"),
        )
        # Dialogue only has a method for TaskManager
        methods = [
            {"component": "TaskManager", "name": "assign_task",
             "parameters": [], "return_type": "None", "derived_from": "dialogue"},
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(bp, state)

        tm = result["components"][0]
        ns = result["components"][1]

        # TaskManager got dialogue method
        tm_names = [m["name"] for m in tm["methods"]]
        assert "assign_task" in tm_names

        # NotificationService got gap-fill inferred methods
        assert ns.get("methods"), "NotificationService should have gap-fill methods"
        ns_names = [m["name"] for m in ns["methods"]]
        assert "execute" in ns_names


class TestStressTestScenarios:
    """Simulate the component types that caused low coverage in stress test."""

    @pytest.mark.parametrize("name,comp_type,expected_method", [
        # From stress test #2 (tattoo booking): typical service components
        ("BookingEngine", "service", "execute"),
        ("ArtistPortfolio", "entity", "get_artistportfolio"),
        ("PaymentProcessor", "processor", "execute"),
        ("DepositManager", "manager", "initialize"),
        # From stress test #7 (crypto portfolio): financial components
        ("PriceFeedConnector", "connector", "connect"),
        ("PortfolioTracker", "tracker", "track"),
        ("TaxReportGenerator", "service", "execute"),
        ("AlertRuleEngine", "engine", "execute"),
        # From stress test #8 (medical clinic): complex system
        ("AppointmentScheduler", "scheduler", "schedule"),
        ("PatientRecordStore", "store", "create_patientrecordstore"),
        ("InsuranceVerifier", "verifier", "validate"),
        ("PrescriptionTracker", "tracker", "track"),
        ("TelehealthGateway", "gateway", "handle_request"),
        # From stress test #9 (course platform): education
        ("CourseManager", "manager", "initialize"),
        ("QuizEngine", "engine", "execute"),
        ("ProgressDashboard", "dashboard", "check"),
        ("CertificateGenerator", "service", "execute"),
        ("ForumModerator", "unknown", "initialize"),  # catch-all
    ])
    def test_stress_test_component(self, name, comp_type, expected_method):
        engine = _make_engine()
        bp = _make_blueprint(_make_component(name, comp_type))
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = {m["name"] for m in methods}
        assert expected_method in names, (
            f"{name} (type={comp_type}) should have method '{expected_method}', "
            f"got: {names}"
        )

    def test_full_coverage_on_mixed_blueprint(self):
        """A blueprint with 10 diverse components should have 100% method coverage."""
        engine = _make_engine()
        bp = _make_blueprint(
            _make_component("UserStore", "store"),
            _make_component("PaymentService", "service"),
            _make_component("APIGateway", "gateway"),
            _make_component("TaskManager", "manager"),
            _make_component("MetricsDashboard", "dashboard"),
            _make_component("JobScheduler", "scheduler"),
            _make_component("PluginRegistry", "registry"),
            _make_component("InputValidator", "validator"),
            _make_component("EmailNotifier", "notifier"),
            _make_component("StripeAdapter", "adapter"),
        )
        result = engine._infer_component_methods(bp)

        total = len(result["components"])
        with_methods = sum(1 for c in result["components"] if c.get("methods"))
        coverage = with_methods / total * 100

        assert coverage == 100.0, f"Coverage {coverage}% — some components have no methods"


class TestResynthesisGapFill:
    """Re-synthesized components get method inference."""

    def test_infer_called_after_resynthesis(self):
        """Engine source calls _infer_component_methods after re-synthesis."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        # Find the re-synthesis block
        resynth_idx = source.find("_targeted_resynthesis")
        assert resynth_idx > 0, "_targeted_resynthesis not found in compile()"
        # After re-synthesis, _infer_component_methods should be called
        after_resynth = source[resynth_idx:]
        assert "_infer_component_methods" in after_resynth, (
            "_infer_component_methods not called after re-synthesis"
        )

    def test_resynthesized_components_get_methods(self):
        """Components added by re-synthesis get gap-filled methods."""
        mock_llm = Mock(spec=BaseLLMClient)
        mock_llm.model = "mock"
        engine = MotherlabsEngine(llm_client=mock_llm)

        # Simulate a blueprint after re-synthesis: mix of components
        # with and without methods
        bp = {
            "components": [
                {"name": "User", "type": "entity", "methods": [
                    {"name": "create", "description": "Create user"}
                ]},
                # These simulate re-synthesized components with no methods
                {"name": "AuditLog", "type": "entity"},
                {"name": "NotificationService", "type": "service"},
                {"name": "PaymentGateway", "type": "gateway"},
            ],
            "relationships": [],
        }
        result = engine._infer_component_methods(bp)
        for comp in result["components"]:
            assert comp.get("methods"), f"{comp['name']} has no methods after gap-fill"
