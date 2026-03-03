"""
Tests for Constraint Parsing - Phase 6.3.

Tests the constraint parsing and code generation functionality.
"""

import pytest
from core.schema import (
    ConstraintType,
    FormalConstraint,
    parse_constraint,
    parse_blueprint_constraints,
    generate_validator_code,
    generate_validate_method,
)


class TestConstraintType:
    """Test ConstraintType enum."""

    def test_all_types_exist(self):
        """All expected constraint types should exist."""
        expected = [
            "range", "not_null", "not_empty", "regex", "type",
            "enum", "length", "positive", "non_negative", "unique", "custom"
        ]
        for t in expected:
            assert ConstraintType(t) is not None

    def test_constraint_type_values(self):
        """Constraint types should have string values."""
        assert ConstraintType.RANGE.value == "range"
        assert ConstraintType.NOT_NULL.value == "not_null"
        assert ConstraintType.REGEX.value == "regex"


class TestFormalConstraint:
    """Test FormalConstraint dataclass."""

    def test_create_range_constraint(self):
        """Should create range constraint."""
        c = FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="confidence",
            params={"min": 0, "max": 1},
            applies_to=["Message"],
            description="Confidence must be between 0 and 1"
        )
        assert c.constraint_type == ConstraintType.RANGE
        assert c.field == "confidence"
        assert c.params["min"] == 0
        assert c.params["max"] == 1

    def test_to_dict(self):
        """Should export to dictionary."""
        c = FormalConstraint(
            constraint_type=ConstraintType.NOT_NULL,
            field="name",
            params={},
            applies_to=["User"],
            description="Name is required"
        )
        d = c.to_dict()
        assert d["type"] == "not_null"
        assert d["field"] == "name"
        assert d["applies_to"] == ["User"]

    def test_from_dict(self):
        """Should parse from dictionary."""
        data = {
            "type": "range",
            "field": "age",
            "params": {"min": 0, "max": 150},
            "applies_to": ["Person"],
            "description": "Age must be valid"
        }
        c = FormalConstraint.from_dict(data)
        assert c.constraint_type == ConstraintType.RANGE
        assert c.field == "age"
        assert c.params["min"] == 0

    def test_from_dict_unknown_type(self):
        """Unknown type should default to CUSTOM."""
        data = {"type": "unknown_type", "field": "x"}
        c = FormalConstraint.from_dict(data)
        assert c.constraint_type == ConstraintType.CUSTOM


class TestParseConstraintRange:
    """Test parsing range constraints."""

    def test_range_bracket_notation(self):
        """Should parse 'value in range [min, max]'."""
        c = parse_constraint("confidence in range [0, 1]")
        assert c is not None
        assert c.constraint_type == ConstraintType.RANGE
        assert c.field == "confidence"
        assert c.params["min"] == 0
        assert c.params["max"] == 1

    def test_range_between_notation(self):
        """Should parse 'value between min and max'."""
        c = parse_constraint("age between 0 and 150")
        assert c is not None
        assert c.constraint_type == ConstraintType.RANGE
        assert c.field == "age"
        assert c.params["min"] == 0
        assert c.params["max"] == 150

    def test_range_comparison_notation(self):
        """Should parse '0 <= value <= 100'."""
        c = parse_constraint("0 <= score <= 100")
        assert c is not None
        assert c.constraint_type == ConstraintType.RANGE
        assert c.field == "score"
        assert c.params["min"] == 0
        assert c.params["max"] == 100

    def test_range_with_floats(self):
        """Should parse ranges with float values."""
        c = parse_constraint("rating in [0.0, 5.0]")
        assert c is not None
        assert c.params["min"] == 0.0
        assert c.params["max"] == 5.0


class TestParseConstraintNotNull:
    """Test parsing not-null constraints."""

    def test_not_null(self):
        """Should parse 'field must not be null'."""
        c = parse_constraint("name must not be null")
        assert c is not None
        assert c.constraint_type == ConstraintType.NOT_NULL
        assert c.field == "name"

    def test_required(self):
        """Should parse 'field is required'."""
        c = parse_constraint("email is required")
        assert c is not None
        assert c.constraint_type == ConstraintType.NOT_NULL
        assert c.field == "email"

    def test_cannot_be_none(self):
        """Should parse 'field cannot be none'."""
        c = parse_constraint("id cannot be none")
        assert c is not None
        assert c.constraint_type == ConstraintType.NOT_NULL
        assert c.field == "id"


class TestParseConstraintRegex:
    """Test parsing regex constraints."""

    def test_regex_slash_notation(self):
        """Should parse 'field matches /pattern/'."""
        c = parse_constraint("email matches /^[a-zA-Z0-9._%+-]+@/")
        assert c is not None
        assert c.constraint_type == ConstraintType.REGEX
        assert c.field == "email"
        assert "^[a-zA-Z0-9._%+-]+@" in c.params["pattern"]

    def test_regex_quote_notation(self):
        """Should parse 'field matches \"pattern\"'."""
        c = parse_constraint("phone must match '^[0-9]{10}$'")
        assert c is not None
        assert c.constraint_type == ConstraintType.REGEX
        assert c.field == "phone"


class TestParseConstraintEnum:
    """Test parsing enum constraints."""

    def test_enum_one_of(self):
        """Should parse 'field must be one of: a, b, c'."""
        c = parse_constraint("status must be one of: active, inactive, pending")
        assert c is not None
        assert c.constraint_type == ConstraintType.ENUM
        assert c.field == "status"
        assert "active" in c.params["values"]
        assert "inactive" in c.params["values"]
        assert "pending" in c.params["values"]

    def test_enum_in_brackets(self):
        """Should parse 'field in [a, b, c]'."""
        c = parse_constraint("color in [red, green, blue]")
        assert c is not None
        assert c.constraint_type == ConstraintType.ENUM
        assert c.field == "color"
        assert len(c.params["values"]) == 3


class TestParseConstraintNumeric:
    """Test parsing numeric constraints."""

    def test_positive(self):
        """Should parse 'field must be positive'."""
        c = parse_constraint("count must be positive")
        assert c is not None
        assert c.constraint_type == ConstraintType.POSITIVE
        assert c.field == "count"

    def test_positive_greater_than_zero(self):
        """Should parse 'field > 0'."""
        c = parse_constraint("amount > 0")
        assert c is not None
        assert c.constraint_type == ConstraintType.POSITIVE
        assert c.field == "amount"

    def test_non_negative(self):
        """Should parse 'field must be non-negative'."""
        c = parse_constraint("index must be non-negative")
        assert c is not None
        assert c.constraint_type == ConstraintType.NON_NEGATIVE
        assert c.field == "index"

    def test_non_negative_gte_zero(self):
        """Should parse 'field >= 0'."""
        c = parse_constraint("balance >= 0")
        assert c is not None
        assert c.constraint_type == ConstraintType.NON_NEGATIVE
        assert c.field == "balance"


class TestParseConstraintLength:
    """Test parsing length constraints."""

    def test_length_between(self):
        """Should parse 'field length between min and max'."""
        c = parse_constraint("name length between 1 and 100")
        assert c is not None
        assert c.constraint_type == ConstraintType.LENGTH
        assert c.field == "name"
        assert c.params["min"] == 1
        assert c.params["max"] == 100

    def test_max_length(self):
        """Should parse 'field max length N'."""
        c = parse_constraint("description max length 500")
        assert c is not None
        assert c.constraint_type == ConstraintType.LENGTH
        assert c.field == "description"
        assert c.params["max"] == 500


class TestParseConstraintUnique:
    """Test parsing unique constraints."""

    def test_unique(self):
        """Should parse 'field must be unique'."""
        c = parse_constraint("ids must be unique")
        assert c is not None
        assert c.constraint_type == ConstraintType.UNIQUE
        assert c.field == "ids"

    def test_no_duplicates(self):
        """Should parse 'no duplicates in field'."""
        c = parse_constraint("no duplicates in names")
        assert c is not None
        assert c.constraint_type == ConstraintType.UNIQUE
        assert c.field == "names"


class TestParseConstraintCustom:
    """Test parsing custom/fallback constraints."""

    def test_unparseable_returns_custom(self):
        """Unparseable constraint should return CUSTOM type."""
        c = parse_constraint("this is some arbitrary constraint text")
        assert c is not None
        assert c.constraint_type == ConstraintType.CUSTOM
        assert "arbitrary" in c.params.get("expression", "")

    def test_applies_to_passed_through(self):
        """applies_to should be passed through."""
        c = parse_constraint("count > 0", applies_to=["Entity", "Process"])
        assert c is not None
        assert "Entity" in c.applies_to
        assert "Process" in c.applies_to


class TestParseBlueprintConstraints:
    """Test parsing constraints from blueprint."""

    def test_parse_multiple_constraints(self):
        """Should parse all constraints in blueprint."""
        blueprint = {
            "components": [],
            "relationships": [],
            "constraints": [
                {"description": "confidence in range [0, 1]", "applies_to": ["Message"]},
                {"description": "name is required", "applies_to": ["User"]},
                {"description": "count >= 0", "applies_to": ["Counter"]},
            ]
        }
        constraints = parse_blueprint_constraints(blueprint)
        assert len(constraints) == 3
        assert constraints[0].constraint_type == ConstraintType.RANGE
        assert constraints[1].constraint_type == ConstraintType.NOT_NULL
        assert constraints[2].constraint_type == ConstraintType.NON_NEGATIVE

    def test_empty_blueprint_returns_empty(self):
        """Empty constraints should return empty list."""
        blueprint = {"components": [], "relationships": [], "constraints": []}
        constraints = parse_blueprint_constraints(blueprint)
        assert constraints == []


class TestGenerateValidatorCode:
    """Test validator code generation."""

    def test_generate_range_validator(self):
        """Should generate range assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="confidence",
            params={"min": 0, "max": 1}
        )
        code = generate_validator_code(c)
        assert "assert 0 <= self.confidence <= 1" in code
        assert "confidence must be in range" in code

    def test_generate_not_null_validator(self):
        """Should generate not-null assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.NOT_NULL,
            field="name"
        )
        code = generate_validator_code(c)
        assert "assert self.name is not None" in code

    def test_generate_positive_validator(self):
        """Should generate positive assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.POSITIVE,
            field="count"
        )
        code = generate_validator_code(c)
        assert "assert self.count > 0" in code

    def test_generate_non_negative_validator(self):
        """Should generate non-negative assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.NON_NEGATIVE,
            field="index"
        )
        code = generate_validator_code(c)
        assert "assert self.index >= 0" in code

    def test_generate_enum_validator(self):
        """Should generate enum assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.ENUM,
            field="status",
            params={"values": ["active", "inactive"]}
        )
        code = generate_validator_code(c)
        assert "assert self.status in" in code
        assert "active" in code
        assert "inactive" in code

    def test_generate_unique_validator(self):
        """Should generate unique assertion."""
        c = FormalConstraint(
            constraint_type=ConstraintType.UNIQUE,
            field="ids"
        )
        code = generate_validator_code(c)
        assert "len(self.ids) == len(set(self.ids))" in code

    def test_generate_length_validator_both(self):
        """Should generate length assertion with min and max."""
        c = FormalConstraint(
            constraint_type=ConstraintType.LENGTH,
            field="name",
            params={"min": 1, "max": 100}
        )
        code = generate_validator_code(c)
        assert "1 <= len(self.name) <= 100" in code

    def test_generate_length_validator_max_only(self):
        """Should generate length assertion with max only."""
        c = FormalConstraint(
            constraint_type=ConstraintType.LENGTH,
            field="description",
            params={"max": 500}
        )
        code = generate_validator_code(c)
        assert "len(self.description) <= 500" in code


class TestGenerateValidateMethod:
    """Test validate method generation."""

    def test_generate_empty_method(self):
        """Empty constraints should generate simple return True."""
        code = generate_validate_method([])
        assert "def validate(self)" in code
        assert "return True" in code

    def test_generate_method_with_constraints(self):
        """Should generate method with all constraints."""
        constraints = [
            FormalConstraint(
                constraint_type=ConstraintType.RANGE,
                field="confidence",
                params={"min": 0, "max": 1}
            ),
            FormalConstraint(
                constraint_type=ConstraintType.NOT_NULL,
                field="name"
            ),
        ]
        code = generate_validate_method(constraints, class_name="Message")
        assert "def validate(self)" in code
        assert "self.confidence" in code
        assert "self.name is not None" in code
        assert "errors" in code

    def test_generated_code_is_valid_python(self):
        """Generated code should be syntactically valid Python."""
        constraints = [
            FormalConstraint(
                constraint_type=ConstraintType.POSITIVE,
                field="count"
            ),
        ]
        code = generate_validate_method(constraints)
        # This should not raise SyntaxError
        compile(code, "<string>", "exec")
