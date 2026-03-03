"""
Tests for core/domain_adapter.py — DomainAdapter frozen dataclass hierarchy.

Phase A: Extract Domain-Specific Code into Adapter
"""

import pytest
from core.domain_adapter import (
    DomainAdapter,
    VocabularyMap,
    PromptTemplates,
    ClassificationConfig,
    VerificationOverrides,
    MaterializationConfig,
)


# =============================================================================
# VocabularyMap
# =============================================================================

class TestVocabularyMap:
    def test_default_creation(self):
        v = VocabularyMap()
        assert v.type_keywords == {}
        assert v.relationship_flows == {}
        assert v.type_hints == {}
        assert v.entity_types == frozenset()
        assert v.process_types == frozenset()

    def test_with_data(self):
        v = VocabularyMap(
            type_keywords={"agent": frozenset({"service", "handler"})},
            entity_types=frozenset({"entity", "model"}),
        )
        assert "agent" in v.type_keywords
        assert "service" in v.type_keywords["agent"]
        assert "entity" in v.entity_types

    def test_frozen(self):
        v = VocabularyMap()
        with pytest.raises(AttributeError):
            v.type_keywords = {}


# =============================================================================
# PromptTemplates
# =============================================================================

class TestPromptTemplates:
    def test_default_empty(self):
        p = PromptTemplates()
        assert p.intent_system_prompt == ""
        assert p.persona_system_prompt == ""
        assert p.entity_system_prompt == ""
        assert p.process_system_prompt == ""
        assert p.synthesis_system_prompt == ""
        assert p.emission_preamble == ""

    def test_with_prompts(self):
        p = PromptTemplates(intent_system_prompt="Extract intent.")
        assert p.intent_system_prompt == "Extract intent."

    def test_frozen(self):
        p = PromptTemplates()
        with pytest.raises(AttributeError):
            p.intent_system_prompt = "changed"


# =============================================================================
# ClassificationConfig
# =============================================================================

class TestClassificationConfig:
    def test_defaults(self):
        c = ClassificationConfig()
        assert c.subject_patterns == ()
        assert c.object_patterns == ()
        assert c.generic_terms == frozenset()
        assert c.min_name_length == 3

    def test_with_patterns(self):
        c = ClassificationConfig(
            subject_patterns=(r'\b{}\s+handles?\b',),
            generic_terms=frozenset({"data", "input"}),
        )
        assert len(c.subject_patterns) == 1
        assert "data" in c.generic_terms

    def test_frozen(self):
        c = ClassificationConfig()
        with pytest.raises(AttributeError):
            c.min_name_length = 5


# =============================================================================
# VerificationOverrides
# =============================================================================

class TestVerificationOverrides:
    def test_defaults(self):
        v = VerificationOverrides()
        assert v.actionability_checks == ("methods",)
        assert v.readiness_label == "codegen_readiness"
        assert len(v.dimension_weights) == 7
        assert abs(sum(v.dimension_weights) - 1.0) < 0.01

    def test_custom_checks(self):
        v = VerificationOverrides(
            actionability_checks=("decision_points", "activities"),
            readiness_label="process_readiness",
        )
        assert "decision_points" in v.actionability_checks
        assert v.readiness_label == "process_readiness"

    def test_frozen(self):
        v = VerificationOverrides()
        with pytest.raises(AttributeError):
            v.readiness_label = "changed"


# =============================================================================
# MaterializationConfig
# =============================================================================

class TestMaterializationConfig:
    def test_defaults(self):
        m = MaterializationConfig()
        assert m.output_format == "python"
        assert m.file_extension == ".py"
        assert m.syntax_validator == "ast.parse"

    def test_yaml_output(self):
        m = MaterializationConfig(
            output_format="yaml",
            file_extension=".yaml",
            syntax_validator="yaml.safe_load",
        )
        assert m.output_format == "yaml"

    def test_frozen(self):
        m = MaterializationConfig()
        with pytest.raises(AttributeError):
            m.output_format = "rust"


# =============================================================================
# DomainAdapter
# =============================================================================

class TestDomainAdapter:
    def test_minimal_creation(self):
        a = DomainAdapter(name="test", version="0.1")
        assert a.name == "test"
        assert a.version == "0.1"
        assert isinstance(a.vocabulary, VocabularyMap)
        assert isinstance(a.prompts, PromptTemplates)
        assert isinstance(a.classification, ClassificationConfig)
        assert isinstance(a.verification, VerificationOverrides)
        assert isinstance(a.materialization, MaterializationConfig)

    def test_full_creation(self):
        a = DomainAdapter(
            name="software",
            version="1.0",
            vocabulary=VocabularyMap(
                type_keywords={"agent": frozenset({"handler"})},
                entity_types=frozenset({"entity"}),
            ),
            prompts=PromptTemplates(intent_system_prompt="Extract."),
            classification=ClassificationConfig(min_name_length=4),
            verification=VerificationOverrides(readiness_label="custom"),
            materialization=MaterializationConfig(output_format="yaml"),
        )
        assert a.name == "software"
        assert "handler" in a.vocabulary.type_keywords["agent"]
        assert a.classification.min_name_length == 4
        assert a.materialization.output_format == "yaml"

    def test_frozen(self):
        a = DomainAdapter(name="test", version="0.1")
        with pytest.raises(AttributeError):
            a.name = "changed"

    def test_different_adapters_not_equal(self):
        a1 = DomainAdapter(name="software", version="1.0")
        a2 = DomainAdapter(name="process", version="1.0")
        assert a1 != a2

    def test_same_adapter_equal(self):
        a1 = DomainAdapter(name="test", version="1.0")
        a2 = DomainAdapter(name="test", version="1.0")
        assert a1 == a2


# =============================================================================
# Software adapter extraction equivalence
# =============================================================================

class TestSoftwareAdapterEquivalence:
    """Verify the software adapter contains exact values from hardcoded sources."""

    def test_software_adapter_imports(self):
        from adapters.software import SOFTWARE_ADAPTER
        assert SOFTWARE_ADAPTER.name == "software"
        assert SOFTWARE_ADAPTER.version == "1.0"

    def test_type_keywords_match_classification(self):
        from adapters.software import SOFTWARE_VOCABULARY
        from core.classification import _TYPE_KEYWORDS
        for comp_type, keywords in _TYPE_KEYWORDS.items():
            assert comp_type in SOFTWARE_VOCABULARY.type_keywords
            assert SOFTWARE_VOCABULARY.type_keywords[comp_type] == frozenset(keywords)

    def test_relationship_flows_match_extractor(self):
        from adapters.software import SOFTWARE_VOCABULARY
        from core.interface_extractor import _RELATIONSHIP_TO_FLOW
        assert SOFTWARE_VOCABULARY.relationship_flows == _RELATIONSHIP_TO_FLOW

    def test_type_hints_match_extractor(self):
        from adapters.software import SOFTWARE_VOCABULARY
        from core.interface_extractor import _TYPE_HINTS_BY_COMPONENT
        assert SOFTWARE_VOCABULARY.type_hints == _TYPE_HINTS_BY_COMPONENT

    def test_entity_types_match_project_writer(self):
        from adapters.software import SOFTWARE_VOCABULARY
        from core.project_writer import _ENTITY_TYPES
        assert SOFTWARE_VOCABULARY.entity_types == _ENTITY_TYPES

    def test_process_types_match_project_writer(self):
        from adapters.software import SOFTWARE_VOCABULARY
        from core.project_writer import _PROCESS_TYPES
        assert SOFTWARE_VOCABULARY.process_types == _PROCESS_TYPES

    def test_generic_terms_match_classification(self):
        from adapters.software import SOFTWARE_CLASSIFICATION
        from core.classification import _GENERIC
        assert SOFTWARE_CLASSIFICATION.generic_terms == _GENERIC

    def test_actionability_checks_default(self):
        from adapters.software import SOFTWARE_VERIFICATION
        assert SOFTWARE_VERIFICATION.actionability_checks == ("methods",)

    def test_dimension_weights_sum_to_one(self):
        from adapters.software import SOFTWARE_VERIFICATION
        total = sum(SOFTWARE_VERIFICATION.dimension_weights)
        assert abs(total - 1.0) < 0.01

    def test_materialization_python(self):
        from adapters.software import SOFTWARE_MATERIALIZATION
        assert SOFTWARE_MATERIALIZATION.output_format == "python"
        assert SOFTWARE_MATERIALIZATION.file_extension == ".py"

    def test_adapter_auto_registered(self):
        import adapters  # noqa: F401 — triggers auto-registration
        from core.adapter_registry import get_adapter
        adapter = get_adapter("software")
        assert adapter.name == "software"
