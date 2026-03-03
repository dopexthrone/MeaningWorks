"""
Tests for Phase 9.2: Constraint-Aware Code Generation.

Tests that blueprint constraints flow into generated code as:
- Constraint comments in method docstrings
- Inline CONSTRAINT comments before NotImplementedError
- Field description comments
- Validation rules in entity validate() methods
- Constraint-aware test generation
"""

import pytest
from codegen.generator import BlueprintCodeGenerator, to_python_name, to_pascal_case, to_snake_case


# =============================================================================
# Test Fixtures
# =============================================================================


def make_blueprint(components=None, relationships=None, constraints=None):
    """Helper to create a minimal blueprint."""
    return {
        "components": components or [],
        "relationships": relationships or [],
        "constraints": constraints or [],
    }


# =============================================================================
# Constraint-Aware Method Generation (9.2)
# =============================================================================


class TestConstraintAwareMethodGeneration:
    """Test that constraints appear in generated methods."""

    def test_method_with_matching_constraint_has_docstring(self):
        """Method should include constraint in docstring when constraint mentions the method."""
        bp = make_blueprint(
            components=[{
                "name": "BookingService",
                "type": "process",
                "description": "Handles session booking",
                "methods": [{
                    "name": "book_session",
                    "parameters": [{"name": "session_id", "type_hint": "str"}],
                    "return_type": "bool",
                    "description": "Book a session",
                }],
                "state_machine": {"states": ["IDLE"], "initial_state": "IDLE"},
            }],
            constraints=[{
                "description": "session duration must be 30-480 minutes",
                "applies_to": ["BookingService"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "CONSTRAINT: session duration must be 30-480 minutes" in code

    def test_method_with_constraint_in_docstring(self):
        """Constraint should appear in the Constraints: section of the docstring."""
        bp = make_blueprint(
            components=[{
                "name": "Validator",
                "type": "entity",
                "description": "Validates data",
                "methods": [{
                    "name": "check_age",
                    "parameters": [{"name": "age", "type_hint": "int"}],
                    "return_type": "bool",
                    "description": "Check age validity",
                }],
            }],
            constraints=[{
                "description": "age must be between 18 and 120",
                "applies_to": ["Validator"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "Constraints:" in code
        assert "age must be between 18 and 120" in code

    def test_method_without_matching_constraint_has_no_constraint_comment(self):
        """Methods with no matching constraints should not have CONSTRAINT comments."""
        bp = make_blueprint(
            components=[{
                "name": "Logger",
                "type": "entity",
                "description": "Logs events",
                "methods": [{
                    "name": "log_event",
                    "parameters": [{"name": "event", "type_hint": "str"}],
                    "return_type": "None",
                    "description": "Log an event",
                }],
            }],
            constraints=[{
                "description": "session duration must be 30-480 minutes",
                "applies_to": ["BookingService"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "CONSTRAINT:" not in code

    def test_method_has_todo_not_implemented(self):
        """Phase 9.2: NotImplementedError should include method name."""
        bp = make_blueprint(
            components=[{
                "name": "Service",
                "type": "entity",
                "description": "A service",
                "methods": [{
                    "name": "process_order",
                    "parameters": [],
                    "return_type": "None",
                    "description": "Process an order",
                }],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert 'raise NotImplementedError("TODO: Implement process_order")' in code

    def test_method_derived_from_quoted(self):
        """derived_from should be quoted in docstring."""
        bp = make_blueprint(
            components=[{
                "name": "Cart",
                "type": "entity",
                "description": "Shopping cart",
                "methods": [{
                    "name": "add_item",
                    "parameters": [{"name": "item", "type_hint": "str"}],
                    "return_type": "None",
                    "description": "Add an item to cart",
                    "derived_from": "users can add items to their cart",
                }],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert 'Derived from: "users can add items to their cart"' in code

    def test_multiple_constraints_for_one_method(self):
        """Multiple matching constraints should all appear."""
        bp = make_blueprint(
            components=[{
                "name": "Session",
                "type": "entity",
                "description": "A session",
                "methods": [{
                    "name": "validate_session",
                    "parameters": [],
                    "return_type": "bool",
                    "description": "Validate session",
                }],
            }],
            constraints=[
                {"description": "session duration must be 30-480 minutes", "applies_to": ["Session"]},
                {"description": "session must have an assigned artist", "applies_to": ["Session"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "session duration must be 30-480 minutes" in code
        assert "session must have an assigned artist" in code


# =============================================================================
# Entity Validation Rules (constraint -> validate())
# =============================================================================


class TestEntityValidationRules:
    """Test that constraints become validation rules in entity validate() methods."""

    def test_range_constraint_generates_validation(self):
        """RANGE constraint should generate raise ValueError validation."""
        bp = make_blueprint(
            components=[{
                "name": "Score",
                "type": "entity",
                "description": "A score",
            }],
            constraints=[{
                "description": "score must be between 0 and 100",
                "applies_to": ["Score"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code
        assert "0" in code and "100" in code

    def test_not_null_constraint_generates_validation(self):
        """NOT_NULL constraint should generate validation."""
        bp = make_blueprint(
            components=[{
                "name": "User",
                "type": "entity",
                "description": "A user",
            }],
            constraints=[{
                "description": "name must not be null",
                "applies_to": ["User"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code

    def test_no_constraints_uses_simple_dataclass(self):
        """Entity with no constraints should use simple dataclass (no validate)."""
        bp = make_blueprint(
            components=[{
                "name": "Item",
                "type": "entity",
                "description": "An item",
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        # Simple dataclass — no validate() method, no constraint comments
        assert "@dataclass" in code
        assert "class Item:" in code
        assert "CONSTRAINT:" not in code


# =============================================================================
# Field Description Comments (9.2)
# =============================================================================


class TestFieldDescriptionComments:
    """Test that field descriptions from blueprint appear as inline comments."""

    def test_field_with_description_has_comment(self):
        """Fields with descriptions should get inline comments."""
        bp = make_blueprint(
            components=[{
                "name": "Artist",
                "type": "entity",
                "description": "A tattoo artist",
            }],
            relationships=[
                {"from": "Artist", "to": "Specialty", "type": "contains"},
            ],
        )
        # Add specialty as a contained component with description
        bp["components"].append({
            "name": "Specialty",
            "type": "entity",
            "description": "The artist's area of expertise (e.g. traditional, realism)",
        })
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        # The field should have a comment from the description
        assert "expertise" in code.lower() or "specialty" in code.lower()

    def test_field_with_type_hint_preserved(self):
        """Fields with explicit type_hint should be generated correctly."""
        bp = make_blueprint(
            components=[{
                "name": "Config",
                "type": "entity",
                "description": "Configuration",
            }],
            relationships=[
                {"from": "Config", "to": "MaxRetries", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "MaxRetries",
            "type": "entity",
            "description": "Maximum number of retries",
            "type_hint": "int",
            "default_value": "3",
        })
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "int" in code
        assert "3" in code


# =============================================================================
# Constraint-Aware Test Generation (9.2)
# =============================================================================


class TestConstraintAwareTestGeneration:
    """Test that generated tests reference actual constraint values."""

    def test_range_constraint_test_uses_values(self):
        """Generated test for RANGE constraint should use actual min/max values."""
        bp = make_blueprint(
            components=[{
                "name": "Metric",
                "type": "entity",
                "description": "A metric",
            }],
            constraints=[{
                "description": "value must be between 0 and 100",
                "applies_to": ["Metric"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        assert "0" in tests and "100" in tests

    def test_enum_constraint_test_uses_values(self):
        """Generated test for ENUM constraint should reference actual values."""
        bp = make_blueprint(
            components=[{
                "name": "Order",
                "type": "entity",
                "description": "An order",
            }],
            constraints=[{
                "description": "status must be one of: pending, active, complete",
                "applies_to": ["Order"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        assert "pending" in tests

    def test_constraint_test_has_docstring(self):
        """Generated constraint test should have the constraint text in docstring."""
        bp = make_blueprint(
            components=[{
                "name": "Session",
                "type": "entity",
                "description": "A session",
            }],
            constraints=[{
                "description": "duration must be positive",
                "applies_to": ["Session"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        assert "duration must be positive" in tests


# =============================================================================
# _get_constraints_for_component (9.2)
# =============================================================================


class TestGetConstraintsForComponent:
    """Test the constraint lookup method."""

    def test_finds_constraint_by_applies_to(self):
        bp = make_blueprint(
            components=[{"name": "Session", "type": "entity", "description": "A session"}],
            constraints=[
                {"description": "duration in range [30, 480]", "applies_to": ["Session"]},
                {"description": "name not null", "applies_to": ["User"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_component("Session")
        assert len(constraints) == 1
        assert constraints[0].field == "duration"

    def test_global_constraint_matches_all(self):
        """Constraint with empty applies_to should match any component."""
        bp = make_blueprint(
            components=[{"name": "Anything", "type": "entity", "description": "Something"}],
            constraints=[
                {"description": "id must not be null", "applies_to": []},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_component("Anything")
        assert len(constraints) == 1

    def test_no_matching_constraints(self):
        bp = make_blueprint(
            components=[{"name": "Logger", "type": "entity", "description": "Logs"}],
            constraints=[
                {"description": "duration in range [30, 480]", "applies_to": ["Session"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_component("Logger")
        assert len(constraints) == 0


# =============================================================================
# _get_constraints_for_method (9.2)
# =============================================================================


class TestGetConstraintsForMethod:
    """Test method-level constraint lookup."""

    def test_finds_constraint_mentioning_method(self):
        bp = make_blueprint(
            constraints=[
                {"description": "book_session duration must be 30-480 minutes", "applies_to": []},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_method("book_session", "BookingService")
        assert len(constraints) == 1

    def test_finds_constraint_mentioning_component(self):
        bp = make_blueprint(
            constraints=[
                {"description": "total must be positive", "applies_to": ["Cart"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_method("checkout", "Cart")
        assert len(constraints) == 1

    def test_no_match_returns_empty(self):
        bp = make_blueprint(
            constraints=[
                {"description": "age must be positive", "applies_to": ["User"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        constraints = gen._get_constraints_for_method("log", "Logger")
        assert len(constraints) == 0


# =============================================================================
# Full Pipeline: Blueprint -> Constraint-Aware Code
# =============================================================================


class TestFullConstraintAwarePipeline:
    """End-to-end test: a realistic blueprint produces constraint-aware code."""

    def test_tattoo_booking_blueprint(self):
        """A booking system blueprint should generate code with constraints."""
        bp = make_blueprint(
            components=[
                {
                    "name": "Session",
                    "type": "entity",
                    "description": "A tattoo session between artist and client",
                    "methods": [
                        {
                            "name": "book",
                            "parameters": [
                                {"name": "artist_id", "type_hint": "str"},
                                {"name": "duration", "type_hint": "int"},
                            ],
                            "return_type": "bool",
                            "description": "Book a session with an artist",
                            "derived_from": "clients can schedule sessions with artists",
                        },
                    ],
                },
                {
                    "name": "Artist",
                    "type": "entity",
                    "description": "A tattoo artist with style specialization",
                },
            ],
            relationships=[
                {"from": "Session", "to": "Artist", "type": "accesses"},
            ],
            constraints=[
                {
                    "description": "session duration must be between 30 and 480 minutes",
                    "applies_to": ["Session"],
                },
                {
                    "description": "artist must have matching style specialization",
                    "applies_to": ["Session", "Artist"],
                },
            ],
        )

        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()

        # Constraint comments in method
        assert "CONSTRAINT: session duration must be between 30 and 480 minutes" in code
        assert "CONSTRAINT: artist must have matching style specialization" in code

        # Derived from in docstring
        assert 'Derived from: "clients can schedule sessions with artists"' in code

        # TODO in NotImplementedError
        assert 'raise NotImplementedError("TODO: Implement book")' in code

        # Validation rules in entity — Phase 20: raises ValueError instead of return False
        assert "raise ValueError" in code  # from RANGE constraint

    def test_generated_code_is_valid_python(self):
        """Generated code should be syntactically valid."""
        bp = make_blueprint(
            components=[
                {
                    "name": "Order",
                    "type": "entity",
                    "description": "A customer order",
                    "methods": [
                        {
                            "name": "submit",
                            "parameters": [],
                            "return_type": "bool",
                            "description": "Submit the order for processing",
                        },
                    ],
                },
            ],
            constraints=[
                {"description": "total must be positive", "applies_to": ["Order"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        # Should compile without syntax errors
        compile(code, "<test>", "exec")

    def test_generated_tests_reference_constraints(self):
        """Generated tests should reference actual constraint values."""
        bp = make_blueprint(
            components=[
                {"name": "Score", "type": "entity", "description": "A score"},
            ],
            constraints=[
                {"description": "value must be between 0 and 100", "applies_to": ["Score"]},
            ],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        # The test should reference the actual values
        assert "0" in tests
        assert "100" in tests


# =============================================================================
# Phase 20: Constraint Enforcement
# =============================================================================


class TestConstraintEnforcement:
    """Phase 20: Test that entities enforce constraints on construction."""

    def test_entity_has_post_init(self):
        """Entity template should generate __post_init__ method."""
        bp = make_blueprint(
            components=[{
                "name": "Score",
                "type": "entity",
                "description": "A score value",
            }],
            constraints=[{
                "description": "value must be between 0 and 100",
                "applies_to": ["Score"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "__post_init__" in code

    def test_entity_post_init_calls_validate(self):
        """__post_init__ should call self.validate()."""
        bp = make_blueprint(
            components=[{
                "name": "Metric",
                "type": "entity",
                "description": "A metric",
            }],
            constraints=[{
                "description": "score must be between 0 and 100",
                "applies_to": ["Metric"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "self.validate()" in code

    def test_validation_raises_valueerror_not_returns_false(self):
        """validate() should raise ValueError, not return False."""
        bp = make_blueprint(
            components=[{
                "name": "Item",
                "type": "entity",
                "description": "An item",
            }],
            constraints=[{
                "description": "price must be between 0 and 10000",
                "applies_to": ["Item"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code
        assert "return False" not in code

    def test_validation_error_includes_field_name(self):
        """ValueError message should include the field name."""
        bp = make_blueprint(
            components=[{
                "name": "Account",
                "type": "entity",
                "description": "A bank account",
            }],
            constraints=[{
                "description": "balance in range [0, 1000000]",
                "applies_to": ["Account"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert 'balance' in code

    def test_validation_passes_with_valid_data_returns_true(self):
        """validate() should return True when all constraints pass."""
        bp = make_blueprint(
            components=[{
                "name": "Score",
                "type": "entity",
                "description": "A score",
            }],
            constraints=[{
                "description": "value must be between 0 and 100",
                "applies_to": ["Score"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "return True" in code

    def test_range_enforcement_generates_raise(self):
        """RANGE constraint should generate raise ValueError."""
        bp = make_blueprint(
            components=[{
                "name": "Temperature",
                "type": "entity",
                "description": "Temperature reading",
            }],
            constraints=[{
                "description": "celsius in range [0, 1000]",
                "applies_to": ["Temperature"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code

    def test_enum_enforcement_generates_raise(self):
        """ENUM constraint should generate raise ValueError."""
        bp = make_blueprint(
            components=[{
                "name": "Light",
                "type": "entity",
                "description": "A traffic light",
            }],
            constraints=[{
                "description": "color must be one of: red, yellow, green",
                "applies_to": ["Light"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code

    def test_not_null_enforcement_generates_raise(self):
        """NOT_NULL constraint should generate raise ValueError."""
        bp = make_blueprint(
            components=[{
                "name": "Person",
                "type": "entity",
                "description": "A person",
            }],
            constraints=[{
                "description": "name must not be null",
                "applies_to": ["Person"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "raise ValueError" in code


class TestProcessConstraintEnforcement:
    """Phase 20: Test that process classes get constraint validation."""

    def test_process_has_validate_constraints(self):
        """Process with constraints should have validate_constraints() method."""
        bp = make_blueprint(
            components=[{
                "name": "OrderFlow",
                "type": "process",
                "description": "Order processing flow",
                "state_machine": {
                    "states": ["PENDING", "ACTIVE", "COMPLETE"],
                    "initial_state": "PENDING",
                },
            }],
            constraints=[{
                "description": "count in range [1, 100]",
                "applies_to": ["OrderFlow"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "validate_constraints" in code

    def test_process_init_calls_validate_constraints(self):
        """Process __init__ should call validate_constraints() when constraints exist."""
        bp = make_blueprint(
            components=[{
                "name": "Workflow",
                "type": "process",
                "description": "A workflow",
                "state_machine": {
                    "states": ["INIT", "DONE"],
                    "initial_state": "INIT",
                },
            }],
            constraints=[{
                "description": "priority in range [1, 10]",
                "applies_to": ["Workflow"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "self.validate_constraints()" in code

    def test_process_without_constraints_has_no_validate(self):
        """Process without constraints should not have validate_constraints()."""
        bp = make_blueprint(
            components=[{
                "name": "SimpleFlow",
                "type": "process",
                "description": "A simple flow",
                "state_machine": {
                    "states": ["START", "END"],
                    "initial_state": "START",
                },
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "validate_constraints" not in code

    def test_generated_test_verifies_constraint_violation(self):
        """Generated tests should include enforcement tests with pytest.raises."""
        bp = make_blueprint(
            components=[{
                "name": "Metric",
                "type": "entity",
                "description": "A metric",
            }],
            constraints=[{
                "description": "score in range [0, 100]",
                "applies_to": ["Metric"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        assert "pytest.raises(ValueError)" in tests

    def test_generated_test_verifies_valid_construction(self):
        """Generated tests should verify valid values work."""
        bp = make_blueprint(
            components=[{
                "name": "Score",
                "type": "entity",
                "description": "A score",
            }],
            constraints=[{
                "description": "value in range [0, 100]",
                "applies_to": ["Score"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        tests = gen.generate_tests()
        # Should have the valid value assertion
        assert "0" in tests and "100" in tests

    def test_generated_code_entity_enforces_on_init(self):
        """Generated entity code with constraints should have __post_init__."""
        bp = make_blueprint(
            components=[{
                "name": "Config",
                "type": "entity",
                "description": "Configuration",
            }],
            constraints=[{
                "description": "timeout in range [1, 3600]",
                "applies_to": ["Config"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "__post_init__" in code
        assert "self.validate()" in code

    def test_generated_code_process_enforces_on_init(self):
        """Generated process code with constraints should call validate_constraints() in __init__."""
        bp = make_blueprint(
            components=[{
                "name": "Pipeline",
                "type": "process",
                "description": "Data pipeline",
                "state_machine": {
                    "states": ["IDLE", "RUNNING"],
                    "initial_state": "IDLE",
                },
            }],
            constraints=[{
                "description": "batch_size in range [1, 1000]",
                "applies_to": ["Pipeline"],
            }],
        )
        gen = BlueprintCodeGenerator(bp)
        code = gen.generate()
        assert "self.validate_constraints()" in code
        assert "raise ValueError" in code
