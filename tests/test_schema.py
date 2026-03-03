"""
Tests for core/schema.py - Blueprint schema validation.

Phase 5.4: Test Coverage

Tests all validators, graph operations, and canonical coverage checking.
"""

import pytest
from core.schema import (
    # Enums
    ComponentType,
    RelationshipType,
    # Dataclasses
    Parameter,
    MethodSpec,
    Transition,
    StateSpec,
    ComponentSchema,
    RelationshipSchema,
    ConstraintSchema,
    BlueprintSchema,
    # Functions
    validate_blueprint,
    check_canonical_coverage,
    check_canonical_relationships,
    compare_blueprints,
    # Graph functions
    check_reachability,
    find_orphan_components,
    validate_relationship_types,
    detect_cycles,
    validate_graph,
    # Nested blueprint functions
    check_nesting_depth,
    resolve_component_path,
    validate_cross_references,
    validate_nested_blueprint,
    # Version
    BLUEPRINT_VERSION,
    add_version,
    # Canonicals
    CANONICAL_COMPONENTS,
    CANONICAL_RELATIONSHIPS,
    # Phase 8.2: Deduplication
    deduplicate_blueprint,
    # Phase 6.3/20: Constraint parsing and code generation
    ConstraintType,
    FormalConstraint,
    parse_constraint,
    generate_validator_code,
)


class TestBlueprintVersion:
    """Test blueprint versioning (Phase 5.3)."""

    def test_blueprint_version_exists(self):
        """Version constant should be defined."""
        assert BLUEPRINT_VERSION == "3.0"

    def test_add_version_to_blueprint(self):
        """add_version should add version field."""
        blueprint = {"components": []}
        result = add_version(blueprint)

        assert result["version"] == "3.0"
        assert result is blueprint  # Modifies in place

    def test_v30_blueprint_with_dimensions_validates(self):
        """v3.0 blueprint with dimensional metadata validates."""
        blueprint = {
            "version": "3.0",
            "components": [
                {"name": "A", "type": "entity", "description": "Component A", "derived_from": "Test input describes A"},
                {"name": "B", "type": "process", "description": "Component B", "derived_from": "Test input describes B"},
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B"},
            ],
            "constraints": [],
            "unresolved": [],
            "dimensions": {
                "axes": [{"name": "structural", "range_low": "low", "range_high": "high", "exploration_depth": 0.8}],
                "node_positions": {"A": {"dimension_values": {"structural": 0.9}, "confidence": 0.85}},
                "fragile_edges": [],
                "silence_zones": [],
                "confidence_trajectory": [0.1, 0.5, 0.8],
                "dialogue_depth": 10,
            },
        }
        result = validate_blueprint(blueprint)
        assert result["valid"]

    def test_v20_blueprint_without_dimensions_validates(self):
        """v2.0 blueprint without dimensional metadata still validates."""
        blueprint = {
            "version": "2.0",
            "components": [
                {"name": "A", "type": "entity", "description": "Component A", "derived_from": "Test input describes A"},
                {"name": "B", "type": "process", "description": "Component B", "derived_from": "Test input describes B"},
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B"},
            ],
            "constraints": [],
            "unresolved": [],
        }
        result = validate_blueprint(blueprint)
        assert result["valid"]

    def test_dimensional_fields_ignored_by_validate(self):
        """validate_blueprint ignores dimensional fields (validates flat structure only)."""
        blueprint = {
            "components": [
                {"name": "X", "type": "entity", "description": "X desc", "derived_from": "Test mentions X"},
            ],
            "relationships": [],
            "constraints": [],
            "unresolved": [],
            "dimensions": {"axes": [], "node_positions": {}},
        }
        result = validate_blueprint(blueprint)
        assert result["valid"]

    def test_missing_dimensional_fields_no_failure(self):
        """Blueprint without dimensional fields does not fail validation."""
        blueprint = {
            "components": [
                {"name": "Y", "type": "entity", "description": "Y desc", "derived_from": "Test mentions Y"},
            ],
            "relationships": [],
            "constraints": [],
            "unresolved": [],
        }
        result = validate_blueprint(blueprint)
        assert result["valid"]


class TestEnums:
    """Test component and relationship type enums."""

    def test_component_types(self):
        """All component types should be valid."""
        assert ComponentType.ENTITY.value == "entity"
        assert ComponentType.PROCESS.value == "process"
        assert ComponentType.INTERFACE.value == "interface"
        assert ComponentType.EVENT.value == "event"
        assert ComponentType.CONSTRAINT.value == "constraint"
        assert ComponentType.SUBSYSTEM.value == "subsystem"

    def test_relationship_types(self):
        """All relationship types should be valid."""
        assert RelationshipType.CONTAINS.value == "contains"
        assert RelationshipType.TRIGGERS.value == "triggers"
        assert RelationshipType.ACCESSES.value == "accesses"
        assert RelationshipType.DEPENDS_ON.value == "depends_on"
        assert RelationshipType.FLOWS_TO.value == "flows_to"


class TestMethodSpec:
    """Test method specification dataclass."""

    def test_method_spec_validation_passes(self):
        """Valid MethodSpec should pass validation."""
        method = MethodSpec(
            name="login",
            parameters=[Parameter(name="email", type_hint="str")],
            return_type="Session",
            description="Authenticate user",
            derived_from="INSIGHT: login method"
        )
        errors = method.validate()
        assert len(errors) == 0

    def test_method_spec_validation_fails_without_name(self):
        """MethodSpec without name should fail."""
        method = MethodSpec(
            name="",
            derived_from="test"
        )
        errors = method.validate()
        assert any("missing name" in e for e in errors)

    def test_method_spec_validation_fails_without_derived_from(self):
        """MethodSpec without derived_from should fail."""
        method = MethodSpec(
            name="test_method",
            derived_from=""
        )
        errors = method.validate()
        assert any("missing derived_from" in e for e in errors)


class TestComponentSchema:
    """Test component schema validation."""

    def test_valid_component(self):
        """Valid component should pass validation."""
        comp = ComponentSchema(
            name="User",
            type=ComponentType.ENTITY,
            description="User account entity",
            derived_from="INSIGHT: User entity with email field"
        )
        errors = comp.validate()
        assert len(errors) == 0

    def test_component_missing_name(self):
        """Component without name should fail."""
        comp = ComponentSchema(
            name="",
            type=ComponentType.ENTITY,
            description="Test",
            derived_from="Test derivation source"
        )
        errors = comp.validate()
        assert any("name is required" in e for e in errors)

    def test_component_missing_description(self):
        """Component without description should fail."""
        comp = ComponentSchema(
            name="Test",
            type=ComponentType.ENTITY,
            description="",
            derived_from="Test derivation source"
        )
        errors = comp.validate()
        assert any("missing description" in e for e in errors)

    def test_component_weak_derivation(self):
        """Component with weak derivation should fail."""
        comp = ComponentSchema(
            name="Test",
            type=ComponentType.ENTITY,
            description="Test component",
            derived_from="short"  # Less than 10 chars
        )
        errors = comp.validate()
        assert any("weak derivation" in e for e in errors)


class TestBlueprintValidation:
    """Test full blueprint validation."""

    def test_valid_blueprint(self, sample_blueprint):
        """Valid blueprint should pass validation."""
        result = validate_blueprint(sample_blueprint)
        # May have warnings but should not have critical errors
        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result

    def test_blueprint_with_orphans(self):
        """Blueprint with orphan components should generate warnings."""
        blueprint = {
            "components": [
                {
                    "name": "Orphan",
                    "type": "entity",
                    "description": "Orphan component",
                    "derived_from": "Test derivation"
                }
            ],
            "relationships": [],
            "constraints": [],
            "unresolved": []
        }
        result = validate_blueprint(blueprint)
        assert any("orphan" in w.lower() for w in result.get("warnings", []))


class TestGraphValidation:
    """Test graph validation functions."""

    def test_check_reachability_all_connected(self, sample_blueprint):
        """All components should be reachable in connected graph."""
        result = check_reachability(sample_blueprint)
        assert result["coverage"] > 0
        assert isinstance(result["reachable"], list)

    def test_find_orphans(self):
        """Should find orphan components."""
        blueprint = {
            "components": [
                {"name": "A"},
                {"name": "Orphan"}
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "depends_on"}
            ]
        }
        result = find_orphan_components(blueprint)
        assert "Orphan" in result["orphans"]

    def test_detect_cycles(self, blueprint_with_cycles):
        """Should detect dependency cycles."""
        result = detect_cycles(blueprint_with_cycles)
        assert result["has_cycles"] is True
        assert len(result["cycles"]) > 0

    def test_no_cycles_in_valid_blueprint(self, sample_blueprint):
        """Valid blueprint should have no cycles."""
        result = detect_cycles(sample_blueprint)
        assert result["has_cycles"] is False

    def test_validate_relationship_types(self, sample_blueprint):
        """Valid relationship types should pass."""
        result = validate_relationship_types(sample_blueprint)
        assert result["all_valid"] is True

    def test_validate_graph_comprehensive(self, sample_blueprint):
        """Comprehensive graph validation."""
        result = validate_graph(sample_blueprint)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "reachability" in result
        assert "orphans" in result
        assert "cycles" in result


class TestCanonicalCoverage:
    """Test canonical component and relationship coverage."""

    def test_canonical_components_defined(self):
        """Canonical components should be defined."""
        assert len(CANONICAL_COMPONENTS) > 0
        assert "SharedState" in CANONICAL_COMPONENTS
        assert "Intent Agent" in CANONICAL_COMPONENTS

    def test_check_coverage_empty_blueprint(self):
        """Empty blueprint should have 0% coverage."""
        blueprint = {"components": [], "relationships": []}
        result = check_canonical_coverage(blueprint)
        assert result["coverage"] == 0.0

    def test_check_coverage_with_matches(self):
        """Blueprint with matching components should have coverage."""
        blueprint = {
            "components": [
                {"name": "SharedState", "type": "entity"},
                {"name": "Intent Agent", "type": "process"}
            ],
            "relationships": []
        }
        result = check_canonical_coverage(blueprint)
        assert result["coverage"] > 0
        assert "SharedState" in result["found"]

    def test_check_canonical_relationships_empty(self):
        """Empty blueprint should have 0% relationship coverage."""
        blueprint = {"components": [], "relationships": []}
        result = check_canonical_relationships(blueprint)
        assert result["coverage"] == 0.0

    def test_check_canonical_relationships_with_matches(self):
        """Blueprint with matching relationships should have coverage."""
        blueprint = {
            "components": [
                {"name": "Governor Agent"},
                {"name": "Intent Agent"}
            ],
            "relationships": [
                {"from": "Governor Agent", "to": "Intent Agent", "type": "triggers"}
            ]
        }
        result = check_canonical_relationships(blueprint)
        assert result["coverage"] > 0


class TestNestedBlueprints:
    """Test nested blueprint validation."""

    def test_check_nesting_depth_valid(self, nested_blueprint):
        """Valid nested blueprint should pass depth check."""
        errors = check_nesting_depth(nested_blueprint, 0, 3)
        assert len(errors) == 0

    def test_check_nesting_depth_exceeded(self):
        """Exceeding max depth should fail."""
        # Create deeply nested blueprint
        deep = {
            "components": [
                {
                    "name": "Level1",
                    "type": "subsystem",
                    "sub_blueprint": {
                        "components": [
                            {
                                "name": "Level2",
                                "type": "subsystem",
                                "sub_blueprint": {
                                    "components": [
                                        {
                                            "name": "Level3",
                                            "type": "subsystem",
                                            "sub_blueprint": {
                                                "components": [
                                                    {
                                                        "name": "Level4",
                                                        "type": "subsystem",
                                                        "sub_blueprint": {"components": []}
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            ]
        }
        errors = check_nesting_depth(deep, 0, 2)  # Max depth 2
        assert len(errors) > 0

    def test_resolve_component_path(self, nested_blueprint):
        """Should resolve dotted component paths."""
        # Test resolving a path in nested blueprint
        comp = resolve_component_path(nested_blueprint, "App.UserService")
        if comp:
            assert comp["name"] == "UserService"

    def test_validate_nested_blueprint(self, nested_blueprint):
        """Nested blueprint validation should run without error."""
        result = validate_nested_blueprint(nested_blueprint)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result


class TestBlueprintComparison:
    """Test blueprint comparison for DDC verification."""

    def test_compare_identical_blueprints(self, sample_blueprint):
        """Identical blueprints should be equivalent."""
        result = compare_blueprints(sample_blueprint, sample_blueprint)
        assert result["equivalent"] is True
        assert result["component_overlap"] == 1.0

    def test_compare_different_blueprints(self, sample_blueprint, invalid_blueprint):
        """Different blueprints should not be equivalent."""
        result = compare_blueprints(sample_blueprint, invalid_blueprint)
        assert result["component_overlap"] < 1.0

    def test_compare_partial_overlap(self):
        """Blueprints with partial overlap should report correctly."""
        bp1 = {
            "components": [
                {"name": "A", "type": "entity"},
                {"name": "B", "type": "entity"}
            ],
            "relationships": []
        }
        bp2 = {
            "components": [
                {"name": "A", "type": "entity"},
                {"name": "C", "type": "entity"}
            ],
            "relationships": []
        }
        result = compare_blueprints(bp1, bp2)
        assert result["component_overlap"] > 0
        assert result["component_overlap"] < 1.0
        assert "B" in result["missing_in_bp2"]
        assert "C" in result["missing_in_bp1"]


class TestBlueprintSchemaConversion:
    """Test BlueprintSchema conversion methods."""

    def test_from_dict_basic(self, sample_blueprint):
        """Should convert dict to BlueprintSchema."""
        schema = BlueprintSchema.from_dict(sample_blueprint)
        assert len(schema.components) == len(sample_blueprint["components"])
        assert len(schema.relationships) == len(sample_blueprint["relationships"])

    def test_to_dict_roundtrip(self, sample_blueprint):
        """Converting to dict and back should preserve data."""
        schema = BlueprintSchema.from_dict(sample_blueprint)
        result = schema.to_dict()

        assert len(result["components"]) == len(sample_blueprint["components"])
        # Check component names preserved
        original_names = {c["name"] for c in sample_blueprint["components"]}
        result_names = {c["name"] for c in result["components"]}
        assert original_names == result_names


class TestFromDictStringCoercion:
    """Test that from_dict handles LLM output where params/methods are strings."""

    def test_string_parameters_coerced(self):
        """Parameters as bare strings should be coerced to Parameter objects."""
        bp = {
            "domain": "process",
            "components": [{
                "name": "OnboardingActivity",
                "type": "activity",
                "description": "Onboarding step",
                "methods": [{
                    "name": "execute",
                    "parameters": ["employee_name", "start_date"],
                    "return_type": "None",
                }],
            }],
            "relationships": [],
        }
        schema = BlueprintSchema.from_dict(bp)
        methods = schema.components[0].methods
        assert len(methods) == 1
        assert len(methods[0].parameters) == 2
        assert methods[0].parameters[0].name == "employee_name"
        assert methods[0].parameters[1].name == "start_date"
        assert methods[0].parameters[0].type_hint == "Any"

    def test_string_methods_coerced(self):
        """Methods as bare strings should be coerced to MethodSpec objects."""
        bp = {
            "domain": "process",
            "components": [{
                "name": "ApprovalGateway",
                "type": "gateway",
                "description": "Approval decision",
                "methods": ["approve", "reject", "escalate"],
            }],
            "relationships": [],
        }
        schema = BlueprintSchema.from_dict(bp)
        methods = schema.components[0].methods
        assert len(methods) == 3
        assert methods[0].name == "approve"
        assert methods[1].name == "reject"
        assert methods[2].name == "escalate"

    def test_mixed_parameters(self):
        """Mix of dict and string parameters should both work."""
        bp = {
            "domain": "test",
            "components": [{
                "name": "Foo",
                "type": "entity",
                "description": "Test",
                "methods": [{
                    "name": "bar",
                    "parameters": [
                        {"name": "x", "type_hint": "int"},
                        "y",
                    ],
                }],
            }],
            "relationships": [],
        }
        schema = BlueprintSchema.from_dict(bp)
        params = schema.components[0].methods[0].parameters
        assert len(params) == 2
        assert params[0].name == "x"
        assert params[0].type_hint == "int"
        assert params[1].name == "y"
        assert params[1].type_hint == "Any"


class TestStateSpec:
    """Test state machine specification."""

    def test_valid_state_spec(self):
        """Valid StateSpec should pass validation."""
        spec = StateSpec(
            states=["INIT", "ACTIVE", "HALTED"],
            initial_state="INIT",
            transitions=[
                Transition(
                    from_state="INIT",
                    to_state="ACTIVE",
                    trigger="start",
                    derived_from="test"
                )
            ],
            derived_from="State machine from spec"
        )
        errors = spec.validate()
        assert len(errors) == 0

    def test_state_spec_no_states(self):
        """StateSpec without states should fail."""
        spec = StateSpec(
            states=[],
            derived_from="test derivation"
        )
        errors = spec.validate()
        assert any("no states" in e for e in errors)

    def test_state_spec_invalid_initial(self):
        """StateSpec with invalid initial state should fail."""
        spec = StateSpec(
            states=["A", "B"],
            initial_state="C",  # Not in states
            derived_from="test derivation"
        )
        errors = spec.validate()
        assert any("not in states" in e for e in errors)


# =============================================================================
# PHASE 8.2: DEDUPLICATION TESTS
# =============================================================================


class TestDeduplicateBlueprint:
    """Tests for deduplicate_blueprint() - Phase 8.2."""

    def test_no_op_when_no_dupes(self):
        """Blueprint with no duplicates should be unchanged."""
        bp = {
            "components": [
                {"name": "User", "type": "entity", "description": "User entity", "derived_from": "input"},
                {"name": "Session", "type": "entity", "description": "Session entity", "derived_from": "input"},
            ],
            "relationships": [
                {"from": "User", "to": "Session", "type": "accesses", "description": "user session"},
            ],
            "constraints": [],
            "unresolved": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 2
        assert len(cleaned["relationships"]) == 1
        assert report["total_removed"] == 0

    def test_exact_name_dedup(self):
        """Two components with exact same name: keep richer one."""
        bp = {
            "components": [
                {"name": "User", "type": "entity", "description": "Short", "derived_from": "input"},
                {"name": "User", "type": "entity", "description": "Longer description of the user entity with more detail", "derived_from": "input"},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        assert "Longer description" in cleaned["components"][0]["description"]
        assert len(report["name_dupes_removed"]) == 1

    def test_case_insensitive_dedup(self):
        """Components with same name different case should dedup."""
        bp = {
            "components": [
                {"name": "intent agent", "type": "entity", "description": "short", "derived_from": "input"},
                {"name": "Intent Agent", "type": "entity", "description": "Intent Agent extracts core need from input text", "derived_from": "input"},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        assert len(report["name_dupes_removed"]) == 1

    def test_subsystem_containment_dedup(self):
        """Component in subsystem's sub_blueprint should be removed from top-level."""
        bp = {
            "components": [
                {"name": "UserService", "type": "subsystem", "description": "User service",
                 "derived_from": "input",
                 "sub_blueprint": {
                     "components": [
                         {"name": "Profile", "type": "entity", "description": "User profile", "derived_from": "input"},
                     ],
                     "relationships": [],
                 }},
                {"name": "Profile", "type": "entity", "description": "User profile entity", "derived_from": "input"},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        assert cleaned["components"][0]["name"] == "UserService"
        assert len(report["containment_dupes_removed"]) == 1
        assert "Profile" in report["containment_dupes_removed"]

    def test_subsystem_containment_dedup_fuzzy(self):
        """'Browser' at top-level matches 'Browser Service' inside sub_blueprint."""
        bp = {
            "components": [
                {"name": "Browser", "type": "service", "description": "Web browser", "derived_from": "input"},
                {"name": "Networking", "type": "subsystem", "description": "Network layer",
                 "derived_from": "input",
                 "sub_blueprint": {
                     "components": [
                         {"name": "Browser Service", "type": "service", "description": "Browser svc", "derived_from": "input"},
                     ],
                     "relationships": [],
                 }},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        assert cleaned["components"][0]["name"] == "Networking"
        assert len(report["containment_dupes_removed"]) == 1

    def test_keeps_richer_component(self):
        """When deduping, keep the component with more content."""
        bp = {
            "components": [
                {"name": "Auth", "type": "entity", "description": "Auth",
                 "derived_from": "input", "methods": [{"name": "login"}, {"name": "logout"}]},
                {"name": "Auth", "type": "entity", "description": "Authentication module", "derived_from": "input"},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        # Should keep the one with methods (richer)
        assert cleaned["components"][0].get("methods") is not None
        assert len(cleaned["components"][0]["methods"]) == 2

    def test_relationship_dedup(self):
        """Duplicate relationships (same from, to, type) should be removed."""
        bp = {
            "components": [
                {"name": "A", "type": "entity", "description": "A", "derived_from": "input"},
                {"name": "B", "type": "entity", "description": "B", "derived_from": "input"},
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B"},
                {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B again"},
                {"from": "A", "to": "B", "type": "accesses", "description": "A accesses B"},
            ],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["relationships"]) == 2  # 1 duplicate removed
        assert report["relationship_dupes_removed"] == 1

    def test_preserves_derived_from(self):
        """Dedup should not lose derived_from on kept components."""
        bp = {
            "components": [
                {"name": "User", "type": "entity", "description": "User entity with details",
                 "derived_from": "INSIGHT: User entity from dialogue turn 3"},
                {"name": "User", "type": "entity", "description": "User", "derived_from": "input"},
            ],
            "relationships": [],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 1
        assert "INSIGHT" in cleaned["components"][0]["derived_from"]

    def test_preserves_constraints_and_unresolved(self):
        """Dedup should preserve constraints and unresolved lists."""
        bp = {
            "components": [],
            "relationships": [],
            "constraints": [{"description": "Must be secure", "applies_to": ["User"]}],
            "unresolved": ["Token format TBD"],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["constraints"]) == 1
        assert len(cleaned["unresolved"]) == 1

    def test_empty_blueprint(self):
        """Empty blueprint should work without error."""
        bp = {"components": [], "relationships": []}
        cleaned, report = deduplicate_blueprint(bp)
        assert len(cleaned["components"]) == 0
        assert report["total_removed"] == 0

    def test_total_removed_count(self):
        """total_removed should be sum of all dedup types."""
        bp = {
            "components": [
                {"name": "A", "type": "entity", "description": "A entity longer desc", "derived_from": "input"},
                {"name": "A", "type": "entity", "description": "A", "derived_from": "input"},
            ],
            "relationships": [
                {"from": "A", "to": "B", "type": "triggers", "description": "t1"},
                {"from": "A", "to": "B", "type": "triggers", "description": "t2"},
            ],
        }
        cleaned, report = deduplicate_blueprint(bp)
        assert report["total_removed"] == 2  # 1 name + 1 relationship


# =============================================================================
# Phase 20: Enhanced Validator Code Generation
# =============================================================================


class TestValidatorCodeEnhanced:
    """Phase 20: Test that generate_validator_code() includes field values in messages."""

    def test_validator_code_includes_field_value_in_message(self):
        """Range validator error message should include actual field value."""
        c = FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="score",
            params={"min": 0.0, "max": 1.0},
        )
        code = generate_validator_code(c)
        assert "self.score" in code
        # Should have f-string with actual value
        assert "got {self.score}" in code

    def test_validate_method_raises_not_returns(self):
        """Generated validator code should produce assertions that lead to raises."""
        c = FormalConstraint(
            constraint_type=ConstraintType.POSITIVE,
            field="amount",
            params={},
        )
        code = generate_validator_code(c)
        # The assert-style code will be converted to raise ValueError by templates.py
        assert "assert" in code
        assert "amount" in code
        # f-string includes actual value
        assert "got {self.amount}" in code

    def test_validator_code_all_types_have_value_info(self):
        """All constraint types (except NOT_NULL and custom) should include actual value in message."""
        test_cases = [
            (ConstraintType.RANGE, {"min": 0, "max": 100}, "got {self.val}"),
            (ConstraintType.POSITIVE, {}, "got {self.val}"),
            (ConstraintType.NON_NEGATIVE, {}, "got {self.val}"),
            (ConstraintType.ENUM, {"values": ["a", "b"]}, "got {self.val!r}"),
            (ConstraintType.LENGTH, {"min": 1, "max": 10}, "got {len(self.val)}"),
            (ConstraintType.UNIQUE, {}, "duplicates"),
        ]
        for ctype, params, expected_fragment in test_cases:
            c = FormalConstraint(
                constraint_type=ctype,
                field="val",
                params=params,
            )
            code = generate_validator_code(c)
            assert expected_fragment in code, f"{ctype.value} validator should contain '{expected_fragment}', got: {code}"
