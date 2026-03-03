import pytest

from subsystems.compression_improvement_system import (
    CompressionMetrics,
    CompressionCore,
    EvaluationProcess,
    EntityDimensionImprovement,
    ImprovementProcess,
    AICompressionResearcher,
    InferenceOptimizationEngineer,
    EntityModelingSpecialist,
    SpecialistPersonas,
    PersonaAttributes
)


class TestCompressionMetrics:
    def test_init(self):
        cm = CompressionMetrics()
        assert cm.dimensions["entity"] == 1.1
        assert cm.get_compression_entity() == 1.1

    def test_specific_methods(self):
        cm = CompressionMetrics()
        cm.create_compression_entity(2.0)
        assert cm.get_compression_entity() == 2.0
        cm.update_compression_entity(3.0)
        assert cm.get_compression_entity() == 3.0
        cm.delete_compression_entity()
        assert "entity" not in cm.dimensions

    def test_general_methods(self):
        cm = CompressionMetrics()
        cm.reset_dimensions()
        cm.update_dimension("other", 10.0)
        assert cm.get_dimension("other") == 10.0
        assert cm.get_dimension("entity") == 1.1


class TestCompressionCore:
    def test_init_and_hierarchy(self):
        ms = CompressionCore()
        assert ms.compression_metrics.get_compression_entity() == 1.1
        ms.initialize_metrics()
        assert ms.compression_metrics.get_compression_entity() == 1.1

    def test_track_entity_dimension(self):
        ms = CompressionCore()
        ms.track_entity_dimension(5.0)
        assert ms.compression_metrics.get_compression_entity() == 5.0


class TestEvaluationProcess:
    def test_write_and_access(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 10.0)
        assert ep.access_dimension("entity") == 10.0
        ep.write_performance("other", 15.0)
        assert ep.access_dimension("other") == 15.0
        assert ep.access_dimension("missing") == 0.0


class TestEntityDimensionImprovement:
    def test_execute(self):
        ep = EvaluationProcess()
        edi = EntityDimensionImprovement(ep)
        assert edi.get_status() == "idle"
        edi.execute()
        assert edi.get_status() == "completed"
        assert ep.access_dimension("entity") > 20.0

    def test_validate(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 25.0)
        edi = EntityDimensionImprovement(ep)
        assert edi.validate() == True

        ep.write_performance("entity", 15.0)
        assert edi.validate() == False  # but status idle, but validate checks current


class TestImprovementProcess:
    def test_check_precondition_low(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 1.1)
        ip = ImprovementProcess(ep)
        assert ip.check_precondition() == True

    def test_check_precondition_high(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 25.0)
        ip = ImprovementProcess(ep)
        assert ip.check_precondition() == False

    def test_execute_improvement_needed(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 1.1)
        ip = ImprovementProcess(ep)
        result = ip.execute_improvement()
        assert result == "Improvement executed"
        assert ep.access_dimension("entity") > 20.0
        assert ip.check_precondition() == False

    def test_execute_no_improvement_needed(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 25.0)
        ip = ImprovementProcess(ep)
        result = ip.execute_improvement()
        assert result == "No improvement needed"
        assert ep.access_dimension("entity") == 25.0

class TestSpecialistPersonas:
    def test_researcher_propose_low(self):
        CompressionMetrics().reset_dimensions()
        ep = EvaluationProcess()
        researcher = AICompressionResearcher(ep)
        assert researcher.propose_update() > 7.5

    def test_researcher_propose_high(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 25.0)
        researcher = AICompressionResearcher(ep)
        assert researcher.propose_update() == 0.0

    def test_engineer_propose_low(self):
        CompressionMetrics().reset_dimensions()
        ep = EvaluationProcess()
        engineer = InferenceOptimizationEngineer(ep)
        assert engineer.propose_update() > 6.5

    def test_modeler_propose_low(self):
        CompressionMetrics().reset_dimensions()
        ep = EvaluationProcess()
        modeler = EntityModelingSpecialist(ep)
        assert modeler.propose_update() > 9.0

    def test_read_metrics(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 15.0)
        sp = SpecialistPersonas(ep)
        assert sp.read_metrics()["entity"] == 15.0

    def test_write_updates(self):
        ep = EvaluationProcess()
        sp = SpecialistPersonas(ep)
        sp.write_updates({"entity": 22.0})
        assert ep.access_dimension("entity") == 22.0

    def test_activate_parallel_improves(self):
        ep = EvaluationProcess()
        sp = SpecialistPersonas(ep)
        sp.activate_parallel()
        assert ep.access_dimension("entity") > 20.0

    def test_activate_parallel_no_change(self):
        ep = EvaluationProcess()
        ep.write_performance("entity", 25.0)
        sp = SpecialistPersonas(ep)
        sp.activate_parallel()
        assert ep.access_dimension("entity") == 25.0

class TestPersonaAttributes:
    def test_crystallize_agent_interfaces(self):
        pa = PersonaAttributes()
        interfaces = pa.crystallize_agent_interfaces()
        assert len(interfaces) == 3
        expected_keys = [
            "Compression Metrics Engineer",
            "Entity Pipeline Architect",
            "Infrastructure Cost Optimizer"
        ]
        assert list(interfaces.keys()) == expected_keys
        for key in interfaces:
            assert interfaces[key]["priority"] == "entity"
            assert interfaces[key]["gap"] == 0.0
            assert interfaces[key]["probe"] in ["track_losses", "missing_entities"]
  # unchanged
